from __future__ import annotations

import contextlib
import io
import json
import os
import re
import sys
import tempfile
import types
import unittest
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import httpx

from astrbot_plugin_game_recommender.clients.rawg import RawgClient
from astrbot_plugin_game_recommender.clients.steam import SteamClient
from astrbot_plugin_game_recommender.services.formatter import (
    format_recommendation_messages_with_llm,
)
from astrbot_plugin_game_recommender.services.message_delivery import (
    build_forward_message_chain,
)
from astrbot_plugin_game_recommender.services.recommender import GameRecommender
from astrbot_plugin_game_recommender.services.steam_price_bridge import SteamPriceBridge
from astrbot_plugin_game_recommender.storage.repository import SQLiteCacheRepository


ROOT = Path(__file__).resolve().parents[1]
PROJECT_ROOT = ROOT.parent


def load_test_env(path: Path) -> None:
    if not path.exists():
        return
    for raw_line in path.read_text(encoding="utf-8-sig").splitlines():
        parsed = parse_env_line(raw_line)
        if parsed is None:
            continue
        key, value = parsed
        os.environ.setdefault(key, value)


def parse_env_line(line: str) -> tuple[str, str] | None:
    text = line.strip()
    if not text or text.startswith("#") or "=" not in text:
        return None
    key, value = text.split("=", 1)
    key = key.strip()
    if not key:
        return None
    return key, unquote_env_value(value.strip())


def unquote_env_value(value: str) -> str:
    if len(value) >= 2 and value[0] == value[-1] and value[0] in {"'", '"'}:
        return value[1:-1]
    return value


def normalize_secret(value: Any, name: str) -> str:
    if isinstance(value, str):
        text = value.strip()
        if text:
            return text
    elif isinstance(value, list):
        for item in value:
            if isinstance(item, str) and item.strip():
                return item.strip()
    raise AssertionError(f"{name} is missing or invalid")


def live_timeout_seconds(config: dict[str, Any]) -> int:
    override = os.getenv("GAME_RECOMMENDER_LIVE_TIMEOUT_SECONDS")
    if override:
        return max(int(override), 1)
    try:
        configured = int(config.get("timeout_seconds") or 0)
    except (TypeError, ValueError):
        configured = 0
    return max(configured, 60)


def restore_env(key: str, value: str | None) -> None:
    if value is None:
        os.environ.pop(key, None)
    else:
        os.environ[key] = value


load_test_env(ROOT / "tests/.env")
LIVE_ENABLED = os.getenv("GAME_RECOMMENDER_LIVE_TESTS") == "1"

LIVE_INPUTS = [
    "推荐几个适合 Switch 和 Steam 的双人游戏，不要恐怖，最好支持中文，预算 100 以内，类似双人成行但别太难。",
    "想找 3 款 PC/Steam 上可以和朋友线上合作的轻松解谜游戏，像 It Takes Two，不要血腥恐怖。",
    "给我推荐几款 Switch 本地同屏合作的休闲游戏，最好有中文，不要恐怖，预算别太高。",
    "预算 50 以内 Steam 双人联机，不要魂like、不要恐怖，偏休闲一点。",
    "Steam 上有没有类似星露谷物语的双人或多人种田经营游戏，中文优先，不要恐怖和高难。",
    "推荐 Switch 上适合情侣一起玩的平台跳跃或轻解谜游戏，像 Unravel Two，别太难。",
    "我想找 PC 和 Xbox 都能玩的合作射击游戏，但不要血腥恐怖，最好不是 PVP 为主。",
    "PS5 或 Steam 上有没有适合两个人慢慢玩的冒险游戏，参考 Portal 2，中文优先。",
    "找几款类似胡闹厨房的派对合作游戏，Switch 优先，能本地多人，不要太硬核。",
    "Steam Deck 上适合两人玩的休闲合作游戏，预算 80 以内，不要肉鸽。",
    "推荐一些支持中文的双人竞速/赛车游戏，Switch 或 Steam 都行，不要拟真太难。",
    "想要合作生存建造类，PC 上玩，像 Don't Starve Together，但不要太恐怖，预算 100 左右。",
    "有没有适合四个人一起玩的轻松派对游戏，Steam，别推荐恐怖、魂类或纯单人。",
    "推荐类似 Moving Out 的搬家/协作闯关游戏，Switch 和 PC 都可，最好本地同屏。",
    "想找亲子能玩的 Switch 合作游戏，中文优先，不要恐怖、暴力和复杂操作。",
    "推荐类似 Terraria 的多人探索建造游戏，Steam，别太贵，不要恐怖。",
    "Mac 或 Steam 上可以远程同乐的双人游戏，偏解谜或合作冒险，最好支持中文。",
    "给我几款非恐怖的合作策略游戏，PC，能两个人打，不要纯竞技对战。",
]


class LiveTestHelperTest(unittest.TestCase):
    def test_normalize_secret_accepts_string_and_key_pool(self) -> None:
        self.assertEqual(normalize_secret(" sk-direct ", "LLM API key"), "sk-direct")
        self.assertEqual(normalize_secret(["", " sk-from-list "], "LLM API key"), "sk-from-list")

    def test_normalize_secret_rejects_empty_or_invalid_values(self) -> None:
        for value in ("", [], ["", "  "], [123], None):
            with self.subTest(value=value):
                with self.assertRaisesRegex(AssertionError, "LLM API key"):
                    normalize_secret(value, "LLM API key")

    def test_load_test_env_reads_values_without_overriding_existing_environment(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / ".env"
            path.write_text(
                "\n".join(
                    [
                        "# local live test settings",
                        "GAME_RECOMMENDER_CONFIG=/tmp/game.json",
                        "EXISTING_VALUE=from_file",
                        "QUOTED_VALUE=\"quoted text\"",
                        "",
                    ]
                ),
                encoding="utf-8",
            )
            old_config = os.environ.pop("GAME_RECOMMENDER_CONFIG", None)
            old_existing = os.environ.get("EXISTING_VALUE")
            old_quoted = os.environ.pop("QUOTED_VALUE", None)
            os.environ["EXISTING_VALUE"] = "from_env"
            try:
                load_test_env(path)
                self.assertEqual(os.environ["GAME_RECOMMENDER_CONFIG"], "/tmp/game.json")
                self.assertEqual(os.environ["EXISTING_VALUE"], "from_env")
                self.assertEqual(os.environ["QUOTED_VALUE"], "quoted text")
            finally:
                restore_env("GAME_RECOMMENDER_CONFIG", old_config)
                restore_env("EXISTING_VALUE", old_existing)
                restore_env("QUOTED_VALUE", old_quoted)

    def test_live_timeout_uses_longer_default_and_env_override(self) -> None:
        old_timeout = os.environ.pop("GAME_RECOMMENDER_LIVE_TIMEOUT_SECONDS", None)
        try:
            self.assertEqual(live_timeout_seconds({"timeout_seconds": 15}), 60)
            os.environ["GAME_RECOMMENDER_LIVE_TIMEOUT_SECONDS"] = "90"
            self.assertEqual(live_timeout_seconds({"timeout_seconds": 15}), 90)
        finally:
            restore_env("GAME_RECOMMENDER_LIVE_TIMEOUT_SECONDS", old_timeout)

    def test_emit_live_result_prints_one_scenario_messages(self) -> None:
        result = LiveScenarioResult(
            text="测试输入",
            messages=["总体说明", "1. 《测试游戏》\n层级：推荐"],
            chain=None,
            rawg_calls=[],
            steam_calls=[],
            llm_calls=1,
        )
        buffer = io.StringIO()
        with contextlib.redirect_stdout(buffer):
            emit_live_result(2, result)

        output = buffer.getvalue()
        self.assertIn("Live 场景 2", output)
        self.assertIn("测试输入", output)
        self.assertIn("总体说明", output)
        self.assertIn("《测试游戏》", output)


@unittest.skipUnless(LIVE_ENABLED, "set GAME_RECOMMENDER_LIVE_TESTS=1 to run live I/O tests")
class LiveRecommendationOutputTest(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self) -> None:
        install_astrbot_test_modules()
        from astrbot_plugin_game_recommender.services.preference_parser import PreferenceParser

        self.PreferenceParser = PreferenceParser
        self.plugin_config = load_json(config_path())
        self.cmd_config = load_json(cmd_config_path())
        self.provider_id = (
            str(self.plugin_config.get("llm_provider_id") or "").strip()
            or str(self.cmd_config.get("provider_settings", {}).get("default_provider_id") or "").strip()
        )
        if not str(self.plugin_config.get("rawg_api_key") or "").strip():
            raise AssertionError(f"RAWG API Key is missing in {config_path()}")
        if not self.provider_id:
            raise AssertionError(f"LLM provider id is missing in {config_path()} and {cmd_config_path()}")

        self.temp_dir = tempfile.TemporaryDirectory()
        self.http_client = httpx.AsyncClient(
            timeout=httpx.Timeout(live_timeout_seconds(self.plugin_config)),
            follow_redirects=True,
            headers={"User-Agent": "astrbot_plugin_game_recommender/live-test"},
        )
        cache = SQLiteCacheRepository(Path(self.temp_dir.name) / "live_cache.sqlite3")
        self.rawg = RecordingGameSource(
            RawgClient(
                client=self.http_client,
                api_key=str(self.plugin_config.get("rawg_api_key") or ""),
                cache=cache,
                cache_ttl_hours=int(self.plugin_config.get("cache_ttl_hours") or 24),
            )
        )
        self.steam = RecordingGameSource(
            SteamClient(
                client=self.http_client,
                cache=cache,
                cache_ttl_hours=int(self.plugin_config.get("cache_ttl_hours") or 24),
                default_country=str(self.plugin_config.get("default_region") or "CN"),
                language="schinese",
            )
        )
        self.context = OpenAICompatibleLiveContext(self.cmd_config, self.provider_id, self.http_client)
        await self.context.assert_provider_works()
        self.event = LiveEvent()
        self.price_bridge = SteamPriceBridge(self.http_client, self.plugin_config)
        self.price_bridge.lookup_limit = min(
            self.price_bridge.lookup_limit,
            int(os.getenv("GAME_RECOMMENDER_LIVE_PRICE_LOOKUP_LIMIT", "2")),
        )

    async def asyncTearDown(self) -> None:
        await self.http_client.aclose()
        self.temp_dir.cleanup()

    async def test_live_natural_language_inputs_generate_forward_records(self) -> None:
        first_titles: list[str] = []
        for index, text in enumerate(LIVE_INPUTS, start=1):
            try:
                result = await self.run_live_scenario(text)
            except Exception as exc:
                emit_live_failure(index, text, exc)
                raise
            emit_live_result(index, result)
            assert_forward_record_shape(self, result)
            assert_no_unbounded_rawg_queries(self, result.rawg_calls)
            if index == 1:
                first_titles = recommendation_titles(result.messages)[:5]

        self.assertGreaterEqual(self.context.llm_call_count, len(LIVE_INPUTS))
        joined_titles = "\n".join(first_titles).lower()
        for banned in ("witcher", "batman", "persona"):
            self.assertNotIn(banned, joined_titles)
        expected_similar = ("split fiction", "unravel two", "overcooked", "moving out", "keywe")
        self.assertTrue(
            any(name in joined_titles for name in expected_similar),
            f"first scenario titles={first_titles!r}",
        )

    async def run_live_scenario(self, text: str) -> "LiveScenarioResult":
        self.rawg.calls.clear()
        self.steam.calls.clear()
        llm_calls_before = self.context.llm_call_count

        parser = self.PreferenceParser(self.context, self.provider_id)
        preference = await parser.parse_preference(self.event, text)
        max_results = min(max(int(self.plugin_config.get("max_results") or 5), 1), 5)
        recommender = GameRecommender(self.rawg, max_results=max_results, steam_source=self.steam)
        candidate_pool_size = (
            max(max_results * 3, preference.result_count or max_results)
            if preference.budget is not None or self.price_bridge.is_available()
            else None
        )
        ranked = await recommender.recommend(preference, candidate_pool_size=candidate_pool_size)
        ranked = await self.price_bridge.enrich_ranked_games(ranked, preference)
        messages = await format_recommendation_messages_with_llm(
            self.context,
            self.event,
            self.provider_id,
            preference,
            ranked,
            limit=max_results,
        )
        chain = build_forward_message_chain(messages, components=FakeForwardComponents)
        return LiveScenarioResult(
            text=text,
            messages=messages,
            chain=chain,
            rawg_calls=list(self.rawg.calls),
            steam_calls=list(self.steam.calls),
            llm_calls=self.context.llm_call_count - llm_calls_before,
        )


@dataclass
class LiveScenarioResult:
    text: str
    messages: list[str]
    chain: list[Any] | None
    rawg_calls: list[dict[str, Any]]
    steam_calls: list[dict[str, Any]]
    llm_calls: int

    def summary(self) -> str:
        return (
            f"input={self.text!r}\n"
            f"messages={self.messages[:2]!r}\n"
            f"rawg_calls={redacted_calls(self.rawg_calls)!r}\n"
            f"steam_calls={redacted_calls(self.steam_calls)!r}\n"
            f"llm_calls={self.llm_calls}"
        )


def emit_live_result(index: int, result: LiveScenarioResult) -> None:
    print(f"\n=== Live 场景 {index}: {result.text} ===", flush=True)
    for node_index, message in enumerate(result.messages, start=1):
        print(f"\n--- 节点 {node_index} ---\n{message}", flush=True)
    print(
        "\n--- 调用摘要 ---\n"
        f"LLM 调用：{result.llm_calls}；RAWG 查询：{len(result.rawg_calls)}；"
        f"Steam 查询：{len(result.steam_calls)}",
        flush=True,
    )


def emit_live_failure(index: int, text: str, exc: Exception) -> None:
    print(
        f"\n=== Live 场景 {index}: {text} ===\n"
        f"执行失败：{type(exc).__name__}: {exc}",
        flush=True,
    )


class RecordingGameSource:
    def __init__(self, wrapped: Any) -> None:
        self.wrapped = wrapped
        self.calls: list[dict[str, Any]] = []

    async def search_games(self, **kwargs: Any) -> Any:
        self.calls.append(dict(kwargs))
        return await self.wrapped.search_games(**kwargs)


@dataclass(frozen=True)
class LiveLLMResponse:
    completion_text: str


class OpenAICompatibleLiveContext:
    def __init__(self, cmd_config: dict[str, Any], provider_id: str, client: httpx.AsyncClient) -> None:
        self.provider_id = provider_id
        self.client = client
        self.provider = enabled_provider(cmd_config, provider_id)
        self.source = provider_source(cmd_config, self.provider)
        self.model = (
            os.getenv("GAME_RECOMMENDER_LIVE_LLM_MODEL", "").strip()
            or str(self.provider.get("model") or "").strip()
        )
        self.api_base = (
            os.getenv("GAME_RECOMMENDER_LIVE_LLM_API_BASE", "").strip()
            or str(self.source.get("api_base") or "https://api.openai.com/v1")
        ).rstrip("/")
        self.api_key = normalize_secret(
            os.getenv("GAME_RECOMMENDER_LIVE_LLM_API_KEY")
            or self.source.get("key"),
            "LLM API key",
        )
        self.llm_call_count = 0
        self.llm_attempt_count = 0
        self.llm_failures: list[str] = []
        if not self.model:
            raise AssertionError(f"LLM model is missing for provider {provider_id}")
        if not self.api_key:
            raise AssertionError(f"LLM API key is missing for provider {provider_id}")

    async def get_current_chat_provider_id(self, umo: Any | None = None) -> str:
        del umo
        return self.provider_id

    async def assert_provider_works(self) -> None:
        try:
            await self.llm_generate(
                prompt='请只返回 JSON：{"ok": true}',
                system_prompt="你是测试探针，只返回用户要求的 JSON。",
            )
        except Exception as exc:
            raise AssertionError(
                f"LLM provider preflight failed for {self.provider_id} "
                f"model={self.model} api_base={self.api_base}: {exc}"
            ) from exc

    async def llm_generate(self, **kwargs: Any) -> LiveLLMResponse:
        prompt = str(kwargs.get("prompt") or "")
        system_prompt = str(kwargs.get("system_prompt") or "")
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})
        payload: dict[str, Any] = {
            "model": self.model,
            "messages": messages,
        }
        extra_body = self.provider.get("custom_extra_body")
        if isinstance(extra_body, dict):
            payload.update(extra_body)
        self.llm_attempt_count += 1
        try:
            response = await self.client.post(
                f"{self.api_base}/chat/completions",
                headers={
                    "Authorization": f"Bearer {self.api_key}",
                    "Content-Type": "application/json",
                },
                json=payload,
            )
            response.raise_for_status()
        except Exception as exc:
            self.llm_failures.append(f"{type(exc).__name__}: {exc}")
            raise
        data = response.json()
        choices = data.get("choices") if isinstance(data, dict) else None
        if not choices:
            raise AssertionError(f"LLM provider {self.provider_id} returned no choices")
        message = choices[0].get("message") if isinstance(choices[0], dict) else None
        content = message.get("content") if isinstance(message, dict) else None
        self.llm_call_count += 1
        return LiveLLMResponse(completion_text=str(content or ""))


class LiveEvent:
    unified_msg_origin = "live-test"


class FakePlain:
    def __init__(self, text: str) -> None:
        self.text = text


class FakeNode:
    def __init__(self, name: str, content: list[FakePlain]) -> None:
        self.name = name
        self.content = content


class FakeNodes:
    def __init__(self, nodes: list[FakeNode]) -> None:
        self.nodes = nodes


class FakeForwardComponents:
    Plain = FakePlain
    Node = FakeNode
    Nodes = FakeNodes


def install_astrbot_test_modules() -> None:
    if "astrbot.api" in sys.modules:
        return
    astrbot = types.ModuleType("astrbot")
    api = types.ModuleType("astrbot.api")
    event = types.ModuleType("astrbot.api.event")
    star = types.ModuleType("astrbot.api.star")
    api.logger = TestLogger()
    event.AstrMessageEvent = LiveEvent
    star.Context = OpenAICompatibleLiveContext
    sys.modules.setdefault("astrbot", astrbot)
    sys.modules.setdefault("astrbot.api", api)
    sys.modules.setdefault("astrbot.api.event", event)
    sys.modules.setdefault("astrbot.api.star", star)


class TestLogger:
    def warning(self, *_args: Any, **_kwargs: Any) -> None:
        return None

    def debug(self, *_args: Any, **_kwargs: Any) -> None:
        return None

    def info(self, *_args: Any, **_kwargs: Any) -> None:
        return None

    def exception(self, *_args: Any, **_kwargs: Any) -> None:
        return None


def config_path() -> Path:
    return Path(
        os.getenv(
            "GAME_RECOMMENDER_CONFIG",
            str(PROJECT_ROOT / "Astrbot/data/config/astrbot_plugin_game_recommender_config.json"),
        )
    )


def cmd_config_path() -> Path:
    return Path(
        os.getenv(
            "ASTRBOT_CMD_CONFIG",
            str(PROJECT_ROOT / "Astrbot/data/cmd_config.json"),
        )
    )


def load_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise AssertionError(f"required config does not exist: {path}")
    return json.loads(path.read_text(encoding="utf-8-sig"))


def enabled_provider(cmd_config: dict[str, Any], provider_id: str) -> dict[str, Any]:
    for provider in cmd_config.get("provider", []):
        if provider.get("id") == provider_id:
            if provider.get("enable") is False:
                raise AssertionError(f"LLM provider is disabled: {provider_id}")
            return provider
    raise AssertionError(f"LLM provider was not found: {provider_id}")


def provider_source(cmd_config: dict[str, Any], provider: dict[str, Any]) -> dict[str, Any]:
    source_id = provider.get("provider_source_id")
    for source in cmd_config.get("provider_sources", []):
        if source.get("id") == source_id:
            return source
    raise AssertionError(f"provider source was not found: {source_id}")


def assert_forward_record_shape(
    test_case: unittest.TestCase,
    result: LiveScenarioResult,
) -> None:
    test_case.assertIsNotNone(result.chain, result.summary())
    assert result.chain is not None
    test_case.assertEqual(len(result.chain), 1, result.summary())
    nodes = result.chain[0]
    test_case.assertIsInstance(nodes, FakeNodes, result.summary())
    test_case.assertEqual(len(nodes.nodes), len(result.messages), result.summary())
    test_case.assertGreater(len(nodes.nodes), 1, result.summary())
    test_case.assertIn("一句话结论", nodes.nodes[0].content[0].text, result.summary())
    test_case.assertGreater(result.llm_calls, 0, result.summary())
    output = "\n\n".join(result.messages)
    test_case.assertNotIn("暂未发现明显不适合点", output, result.summary())
    for message in result.messages[1:]:
        test_case.assertIn("层级：", message, result.summary())
        test_case.assertIn("推荐理由：", message, result.summary())
        test_case.assertIn("可能不适合的点：", message, result.summary())
        test_case.assertIn("平台：", message, result.summary())
        test_case.assertTrue(
            any(marker in message for marker in ("价格：", "购买 / 平台建议：", "购买链接：", "数据来源：")),
            result.summary(),
        )


def assert_no_unbounded_rawg_queries(
    test_case: unittest.TestCase,
    rawg_calls: list[dict[str, Any]],
) -> None:
    test_case.assertTrue(rawg_calls)
    for call in rawg_calls:
        search = call.get("search")
        test_case.assertNotEqual(search, "popular games", redacted_calls(rawg_calls))
        test_case.assertTrue(
            search or call.get("platforms") or call.get("genres") or call.get("tags"),
            redacted_calls(rawg_calls),
        )


def recommendation_titles(messages: list[str]) -> list[str]:
    titles: list[str] = []
    for message in messages[1:]:
        match = re.search(r"《([^》]+)》", message)
        if match:
            titles.append(match.group(1))
    return titles


def redacted_calls(calls: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [
        {
            key: value
            for key, value in call.items()
            if "key" not in key.lower() and "token" not in key.lower()
        }
        for call in calls
    ]


if __name__ == "__main__":
    unittest.main()
