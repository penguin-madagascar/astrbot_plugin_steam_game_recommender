from __future__ import annotations

import ast
import asyncio
import json
import sys
import tempfile
import unittest
from decimal import Decimal
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT.parent))

from astrbot_plugin_game_recommender.services.formatter import (  # noqa: E402
    format_game_block,
    format_game_detail,
    format_recommendation_messages,
    format_recommendation_messages_with_llm,
)
from astrbot_plugin_game_recommender.services.steam_price_bridge import (  # noqa: E402
    SteamPriceBridge,
    attach_missing_price_warning,
    attach_price_summary,
    load_price_plugin_symbols,
)
from astrbot_plugin_game_recommender.storage.models import (  # noqa: E402
    GameCandidate,
    GamePreference,
    GamePriceSummary,
    RankedGame,
)
from astrbot_plugin_steam_price_heybox.models import (  # noqa: E402
    GameIdentity,
    RegionPrice,
    SteamGameDetails,
    SteamPrice,
)
from astrbot_plugin_steam_price_heybox.price_analysis import parse_price_history  # noqa: E402


class CommandRegistrationTest(unittest.TestCase):
    def test_english_commands_keep_chinese_aliases(self) -> None:
        tree = ast.parse((ROOT / "main.py").read_text(encoding="utf-8"))
        commands: dict[str, set[str]] = {}

        for node in ast.walk(tree):
            if not isinstance(node, ast.AsyncFunctionDef):
                continue
            for decorator in node.decorator_list:
                if not (
                    isinstance(decorator, ast.Call)
                    and isinstance(decorator.func, ast.Attribute)
                    and decorator.func.attr == "command"
                    and decorator.args
                    and isinstance(decorator.args[0], ast.Constant)
                ):
                    continue
                aliases: set[str] = set()
                for keyword in decorator.keywords:
                    if keyword.arg == "alias" and isinstance(keyword.value, ast.Set):
                        aliases = {
                            item.value
                            for item in keyword.value.elts
                            if isinstance(item, ast.Constant) and isinstance(item.value, str)
                        }
                commands[str(decorator.args[0].value)] = aliases

        self.assertIn("gamerec", commands)
        self.assertIn("游戏推荐", commands["gamerec"])
        self.assertNotIn("gamedesc", commands)
        self.assertIn("accountbind", commands)
        self.assertIn("账号绑定", commands["accountbind"])
        self.assertIn("unplayedrec", commands)
        self.assertIn("未玩推荐", commands["unplayedrec"])
        self.assertIn("gamerec_retry", commands)
        self.assertIn("重新推荐", commands["gamerec_retry"])
        self.assertIn("换一批", commands["gamerec_retry"])

    def test_game_data_api_key_is_not_required_in_config_schema(self) -> None:
        schema = json.loads((ROOT / "_conf_schema.json").read_text(encoding="utf-8"))
        legacy_key = "ra" + "wg_api_key"

        self.assertNotIn(legacy_key, schema)

    def test_dashboard_hides_price_plugin_runtime_settings(self) -> None:
        schema = json.loads((ROOT / "_conf_schema.json").read_text(encoding="utf-8"))

        self.assertIn("steam_price_heybox_notice", schema)
        for removed_key in (
            "itad" + "_api_key",
            "enable_steam_price_enrichment",
            "steam_price_country",
            "steam_price_history_days",
            "steam_price_lookup_limit",
        ):
            self.assertNotIn(removed_key, schema)

    def test_readme_omits_plugin_market_release_section(self) -> None:
        readme = (ROOT / "README.md").read_text(encoding="utf-8")

        self.assertNotIn("插件市场发布信息", readme)
        self.assertIn("## 示例", readme)

    def test_dashboard_schema_copy_and_order(self) -> None:
        schema = json.loads((ROOT / "_conf_schema.json").read_text(encoding="utf-8"))

        self.assertEqual(next(iter(schema)), "llm_provider_id")
        for key, config in schema.items():
            with self.subTest(key=key):
                self.assertTrue(str(config.get("hint") or "").strip())
                self.assertFalse(str(config.get("description") or "").endswith("。"))

    def test_price_notice_is_readonly_obvious_text(self) -> None:
        schema = json.loads((ROOT / "_conf_schema.json").read_text(encoding="utf-8"))
        notice = schema["steam_price_heybox_notice"]

        self.assertEqual(notice["type"], "text")
        self.assertIs(notice["_readonly"], True)
        self.assertIs(notice["obvious_hint"], True)
        self.assertIn("无需配置", notice["default"])
        self.assertIn("自动启用", notice["default"])
        self.assertIn("未安装", notice["default"])

    def test_repository_no_longer_mentions_itad(self) -> None:
        ignored_parts = {".git", ".venv", "__pycache__"}
        searched_suffixes = {".py", ".json", ".md", ".yaml", ".toml", ".txt"}
        markers = ("Itad" + "Client", "IsThereAny" + "Deal", "itad" + "_api_key")
        offenders = []

        for path in ROOT.rglob("*"):
            if any(part in ignored_parts for part in path.parts):
                continue
            if path.is_file() and path.suffix in searched_suffixes:
                text = path.read_text(encoding="utf-8")
                for marker in markers:
                    if marker in text:
                        offenders.append(f"{path.relative_to(ROOT)}:{marker}")

        self.assertEqual(offenders, [])


class PriceFormattingTest(unittest.TestCase):
    def test_price_summary_is_json_serializable(self) -> None:
        game = RankedGame(
            title="Test Game",
            score=10,
            price_summary=price_summary(current_cny=60, lowest_cny=50),
        )

        payload = json.dumps(game.model_dump(), ensure_ascii=False)

        self.assertIn("current_price", payload)
        self.assertIn("¥60", payload)
        self.assertIn("lowest_cny", payload)

    def test_recommendation_block_includes_price_and_links(self) -> None:
        game = RankedGame(
            title="Test Game",
            platforms=["PC"],
            stores=["Steam"],
            score=10,
            price_summary=price_summary(current_cny=60, lowest_cny=50),
        )

        text = "\n".join(format_game_block(1, game))

        self.assertIn("价格：Steam 当前价 ¥60", text)
        self.assertIn("史低 ¥50", text)
        self.assertIn("Steam：https://store.steampowered.com/app/123/", text)
        self.assertIn("小黑盒：https://www.xiaoheihe.cn/app/topic/game/pc/123", text)

    def test_recommendation_reasons_keep_review_ratio_text(self) -> None:
        game = RankedGame(
            title="Rated Game",
            platforms=["PC"],
            stores=["Steam"],
            score=10,
            reasons=["Steam 好评率 95%", "Metacritic 92"],
        )

        text = "\n".join(format_game_block(1, game))

        self.assertIn("Steam 好评率 95%", text)
        self.assertNotIn("Steam 好评率 95；", text)

    def test_recommendations_can_be_split_into_intro_and_game_messages(self) -> None:
        games = [
            RankedGame(
                title="Split Fiction",
                platforms=["PC", "Nintendo Switch 2"],
                stores=["Steam"],
                score=20,
                reasons=["双人合作核心玩法"],
                warnings=["Nintendo 侧为 Switch 2，不是原版 Switch"],
            ),
            RankedGame(
                title="Unravel Two",
                platforms=["PC", "Nintendo Switch"],
                stores=["Steam", "Nintendo Store"],
                score=19,
                reasons=["同屏合作解谜平台玩法"],
                warnings=["Steam 价格未获取到"],
            ),
        ]

        messages = format_recommendation_messages(
            GamePreference(result_count=2, budget=100),
            games,
            limit=2,
        )

        self.assertEqual(len(messages), 3)
        self.assertNotIn("一句话结论：", messages[0])
        self.assertNotIn("1. 《Split Fiction》", messages[0])
        self.assertTrue(messages[1].startswith("1. 《Split Fiction》"))
        self.assertTrue(messages[2].startswith("2. 《Unravel Two》"))

    def test_empty_recommendations_do_not_start_with_intro_prefix(self) -> None:
        messages = format_recommendation_messages(GamePreference(result_count=2), [], limit=2)

        self.assertEqual(len(messages), 1)
        self.assertFalse(messages[0].startswith("一句话结论："))
        self.assertIn("当前条件", messages[0])

    def test_empty_warnings_do_not_emit_vague_no_issue_placeholder(self) -> None:
        game = RankedGame(
            title="Test Game",
            platforms=["PC"],
            stores=["Steam"],
            score=10,
            reasons=["双人合作核心玩法"],
        )

        text = "\n".join(format_game_block(1, game))

        self.assertNotIn("暂未发现明显不适合点", text)
        self.assertIn("仍需以商店页面确认", text)

    def test_game_detail_appends_price_summary_when_available(self) -> None:
        game = GameCandidate(title="Test Game", platforms=["PC"], stores=["Steam"])

        text = format_game_detail(game, price_summary(current_cny=60, lowest_cny=50))

        self.assertIn("《Test Game》", text)
        self.assertIn("Steam 价格：Steam 当前价 ¥60", text)
        self.assertIn("史低 ¥50", text)


class EmptyFallbackFormattingTest(unittest.IsolatedAsyncioTestCase):
    async def test_non_empty_recommendations_never_call_llm_polishing(self) -> None:
        context = FakeLlmContext(AssertionError("per-game LLM polishing must stay disabled"))

        messages = await format_recommendation_messages_with_llm(
            context,
            FakeEvent(),
            "provider-1",
            GamePreference(result_count=1),
            [RankedGame(title="Verified Game", score=80, fit_points=["Steam 标签匹配"])],
            limit=1,
        )

        self.assertEqual(context.calls, [])
        self.assertIn("Verified Game", messages[1])

    async def test_empty_recommendations_do_not_call_llm_when_fallback_is_disabled(self) -> None:
        context = FakeLlmContext(
            "LLM 兜底建议（未经过 Steam 索引验证）\n1. 《Mario Kart 8 Deluxe》：适合聚会。"
        )

        messages = await format_recommendation_messages_with_llm(
            context,
            FakeEvent(),
            "provider-1",
            GamePreference(result_count=2),
            [],
            limit=2,
            enable_empty_fallback=False,
            raw_query="Switch 聚会游戏",
        )

        self.assertEqual(context.calls, [])
        self.assertEqual(len(messages), 1)
        self.assertIn("暂时没有找到满足当前条件的游戏", messages[0])

    async def test_empty_recommendations_use_llm_when_fallback_is_enabled(self) -> None:
        context = FakeLlmContext(
            "LLM 兜底建议（未经过 Steam 索引验证）\n1. 《Mario Kart 8 Deluxe》：适合轻松多人竞速。"
        )

        messages = await format_recommendation_messages_with_llm(
            context,
            FakeEvent(),
            "provider-1",
            GamePreference(platforms=["nintendo switch"], result_count=2),
            [],
            limit=2,
            enable_empty_fallback=True,
            raw_query="Switch 聚会游戏",
        )

        self.assertEqual(len(context.calls), 1)
        self.assertEqual(context.calls[0]["chat_provider_id"], "provider-1")
        self.assertEqual(len(messages), 1)
        self.assertIn("LLM 兜底建议（未经过 Steam 索引验证）", messages[0])
        self.assertIn("Mario Kart 8 Deluxe", messages[0])

    async def test_empty_recommendations_fall_back_when_llm_returns_blank_text(self) -> None:
        messages = await format_recommendation_messages_with_llm(
            FakeLlmContext(""),
            FakeEvent(),
            "provider-1",
            GamePreference(result_count=2),
            [],
            limit=2,
            enable_empty_fallback=True,
            raw_query="Switch 聚会游戏",
        )

        self.assertEqual(len(messages), 1)
        self.assertIn("暂时没有找到满足当前条件的游戏", messages[0])

    async def test_empty_recommendations_fall_back_when_llm_raises(self) -> None:
        with patch("astrbot_plugin_game_recommender.services.formatter.logger.warning"):
            messages = await format_recommendation_messages_with_llm(
                FakeLlmContext(RuntimeError("provider unavailable")),
                FakeEvent(),
                "provider-1",
                GamePreference(result_count=2),
                [],
                limit=2,
                enable_empty_fallback=True,
                raw_query="Switch 聚会游戏",
            )

        self.assertEqual(len(messages), 1)
        self.assertIn("暂时没有找到满足当前条件的游戏", messages[0])


class PriceBridgeTest(unittest.IsolatedAsyncioTestCase):
    async def test_missing_http_client_leaves_games_unchanged(self) -> None:
        bridge = SteamPriceBridge(client=None, config={})
        games = [RankedGame(title="Test Game", score=10)]

        enriched = await bridge.enrich_ranked_games(games, GamePreference(budget=100))

        self.assertFalse(bridge.is_available())
        self.assertEqual(enriched[0].title, "Test Game")
        self.assertIsNone(enriched[0].price_summary)

    async def test_bridge_uses_default_region_and_internal_price_defaults(self) -> None:
        captured_config = {}

        def factory(price_config, _client):
            captured_config.update(price_config)
            return FakePriceService()

        bridge = SteamPriceBridge(
            client=object(),
            config={"default_region": "us"},
            service_factory=factory,
        )

        self.assertTrue(bridge.is_available())
        self.assertEqual(bridge.lookup_limit, 10)
        self.assertEqual(captured_config["default_country"], "US")
        self.assertEqual(captured_config["default_history_country"], "US")
        self.assertEqual(captured_config["default_language"], "schinese")
        self.assertEqual(captured_config["history_days"], 720)
        self.assertEqual(captured_config["global_price_limit"], 10)
        self.assertFalse(captured_config["show_api_links"])
        self.assertEqual(captured_config["llm_name_retry_count"], 0)

    async def test_lookup_builds_summary_from_price_service(self) -> None:
        bridge = SteamPriceBridge(
            client=object(),
            config={"default_region": "CN"},
            service_factory=lambda _config, _client: FakePriceService(),
        )

        summary = await bridge.lookup("Test Game")

        self.assertIsNotNone(summary)
        assert summary is not None
        self.assertEqual(summary.appid, 123)
        self.assertEqual(summary.current_price, "¥60")
        self.assertEqual(summary.lowest_price, "¥50")
        self.assertEqual(summary.lowest_date, "2026-01-10")
        self.assertEqual(summary.current_cny, 60)
        self.assertEqual(summary.lowest_cny, 50)
        self.assertIn("当前促销", summary.sale_status or "")
        self.assertIn("乌克兰 / UA", summary.region_summary or "")

    async def test_budget_enrichment_softly_penalizes_games_over_budget(self) -> None:
        bridge = FixedPriceBridge(
            {
                "Expensive Game": price_summary(current_cny=120, lowest_cny=110),
                "Budget Game": price_summary(current_cny=40, lowest_cny=30),
            }
        )
        games = [
            RankedGame(title="Expensive Game", score=100, platforms=["PC"], stores=["Steam"]),
            RankedGame(title="Budget Game", score=90, platforms=["PC"], stores=["Steam"]),
        ]

        enriched = await bridge.enrich_ranked_games(games, GamePreference(budget=50))

        self.assertEqual(
            [game.title for game in enriched],
            ["Expensive Game", "Budget Game"],
        )
        self.assertEqual(enriched[0].score, 95)
        self.assertEqual(enriched[1].score, 95)

    async def test_budget_enrichment_keeps_unknown_prices_with_small_penalty(self) -> None:
        bridge = FixedPriceBridge({"Budget Game": price_summary(current_cny=40, lowest_cny=30)})
        games = [
            RankedGame(title="Unknown Price Game", score=100, platforms=["PC"], stores=["Steam"]),
            RankedGame(title="Budget Game", score=90, platforms=["PC"], stores=["Steam"]),
        ]

        enriched = await bridge.enrich_ranked_games(games, GamePreference(budget=50))

        self.assertEqual(
            [game.title for game in enriched],
            ["Unknown Price Game", "Budget Game"],
        )
        self.assertEqual(enriched[0].score, 98)
        self.assertEqual(enriched[1].score, 95)

    async def test_no_budget_price_lookup_preserves_final_order(self) -> None:
        bridge = FixedPriceBridge(
            {
                "First": price_summary(current_cny=100, lowest_cny=80),
                "Second": price_summary(current_cny=10, lowest_cny=5),
            }
        )
        games = [
            RankedGame(title="First", score=90, platforms=["PC"], stores=["Steam"]),
            RankedGame(title="Second", score=80, platforms=["PC"], stores=["Steam"]),
        ]

        enriched = await bridge.enrich_ranked_games(games, GamePreference())

        self.assertEqual([game.title for game in enriched], ["First", "Second"])

    async def test_price_lookup_concurrency_is_capped_at_four(self) -> None:
        bridge = ConcurrentPriceBridge()
        games = [
            RankedGame(
                title=f"Game {index}",
                score=100 - index,
                platforms=["PC"],
                stores=["Steam"],
            )
            for index in range(8)
        ]

        await bridge.enrich_ranked_games(games, GamePreference())

        self.assertGreater(bridge.max_active, 1)
        self.assertLessEqual(bridge.max_active, 4)

    async def test_price_plugin_can_load_from_sibling_plugin_directory(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            plugin_dir = root / "astrbot_plugin_steam_price_heybox"
            plugin_dir.mkdir()
            (plugin_dir / "__init__.py").write_text("", encoding="utf-8")
            (plugin_dir / "models.py").write_text(
                "\n".join(
                    [
                        "class PriceHistory: pass",
                        "class RegionPrice: pass",
                        "class SteamGameDetails: pass",
                    ]
                ),
                encoding="utf-8",
            )
            (plugin_dir / "steam_price.py").write_text(
                "\n".join(
                    [
                        "class PriceLookupError(RuntimeError): pass",
                        "class SteamPriceService:",
                        "    marker = 'fake-sibling-plugin'",
                        "    @classmethod",
                        "    def from_config(cls, config, client):",
                        "        return cls()",
                        "def format_region_summary(regions): return ''",
                        "def format_sale_status(history, today): return []",
                        "def money_text(value, currency): return str(value)",
                        "def parse_country(value): return value.upper() if value else ''",
                    ]
                ),
                encoding="utf-8",
            )

            removed = {
                name: sys.modules.pop(name)
                for name in list(sys.modules)
                if name == "astrbot_plugin_steam_price_heybox"
                or name.startswith("astrbot_plugin_steam_price_heybox.")
            }
            parent_path = str(ROOT.parent)
            removed_paths = [path for path in sys.path if path == parent_path]
            sys.path[:] = [path for path in sys.path if path != parent_path]
            try:
                symbols = load_price_plugin_symbols(search_roots=[root])
            finally:
                for name in list(sys.modules):
                    if name == "astrbot_plugin_steam_price_heybox" or name.startswith(
                        "astrbot_plugin_steam_price_heybox."
                    ):
                        sys.modules.pop(name, None)
                sys.modules.update(removed)
                sys.path[:0] = removed_paths

        self.assertIsNotNone(symbols.service_class)
        self.assertEqual(symbols.service_class.marker, "fake-sibling-plugin")
        self.assertEqual(symbols.parse_country("cn"), "CN")


class BudgetScoringTest(unittest.TestCase):
    def test_current_price_inside_budget_adds_score(self) -> None:
        game = RankedGame(title="Budget Game", score=10)
        enriched = attach_price_summary(
            game,
            price_summary(current_cny=60, lowest_cny=50),
            GamePreference(budget=100),
        )

        self.assertGreater(enriched.score, game.score)
        self.assertTrue(any("预算" in reason for reason in enriched.reasons))

    def test_lowest_price_inside_budget_warns_but_keeps_game(self) -> None:
        game = RankedGame(title="Sale Game", score=10)
        enriched = attach_price_summary(
            game,
            price_summary(current_cny=120, lowest_cny=80),
            GamePreference(budget=100),
        )

        self.assertGreaterEqual(enriched.score, game.score)
        self.assertTrue(
            any("史低" in warning and "预算" in warning for warning in enriched.warnings)
        )

    def test_price_over_budget_penalizes_without_filtering(self) -> None:
        game = RankedGame(title="Expensive Game", score=10)
        enriched = attach_price_summary(
            game,
            price_summary(current_cny=120, lowest_cny=110),
            GamePreference(budget=100),
        )

        self.assertEqual(enriched.title, "Expensive Game")
        self.assertLess(enriched.score, game.score)
        self.assertTrue(any("高于预算" in warning for warning in enriched.warnings))

    def test_missing_price_for_budget_request_lowers_score(self) -> None:
        game = RankedGame(title="Unknown Price Game", score=10)

        enriched = attach_missing_price_warning(game)

        self.assertLess(enriched.score, game.score)
        self.assertTrue(any("价格未获取" in warning for warning in enriched.warnings))


def price_summary(current_cny: float, lowest_cny: float) -> GamePriceSummary:
    return GamePriceSummary(
        source="steam_price_heybox",
        appid=123,
        country="CN",
        current_price=f"¥{current_cny:g}",
        lowest_price=f"¥{lowest_cny:g}",
        lowest_date="2026-01-10",
        lowest_discount=50,
        sale_status="当前促销：2026-01-10 开始，最低 ¥50",
        region_summary="最低价区服：乌克兰 / UA，约 ¥40",
        store_url="https://store.steampowered.com/app/123/",
        heybox_url="https://www.xiaoheihe.cn/app/topic/game/pc/123",
        current_cny=current_cny,
        lowest_cny=lowest_cny,
    )


class FixedPriceBridge(SteamPriceBridge):
    def __init__(self, summaries: dict[str, GamePriceSummary]) -> None:
        super().__init__(
            client=object(),
            config={},
            service_factory=lambda _config, _client: object(),
        )
        self.summaries = summaries

    async def lookup(self, title: str, country: str | None = None) -> GamePriceSummary | None:
        del country
        return self.summaries.get(title)


class ConcurrentPriceBridge(FixedPriceBridge):
    def __init__(self) -> None:
        super().__init__({})
        self.active = 0
        self.max_active = 0

    async def lookup(self, title: str, country: str | None = None) -> GamePriceSummary | None:
        del title, country
        self.active += 1
        self.max_active = max(self.max_active, self.active)
        await asyncio.sleep(0)
        self.active -= 1
        return None


class FakeEvent:
    unified_msg_origin = "qq:test"


class FakeLlmResponse:
    def __init__(self, text: str) -> None:
        self.completion_text = text


class FakeLlmContext:
    def __init__(self, response: str | Exception) -> None:
        self.response = response
        self.calls: list[dict] = []

    async def llm_generate(self, **kwargs):
        self.calls.append(kwargs)
        if isinstance(self.response, Exception):
            raise self.response
        return FakeLlmResponse(self.response)


class FakeSteamClient:
    async def details(self, _appid: int, _country: str, _language: str) -> SteamGameDetails:
        return SteamGameDetails(
            appid=123,
            name="Test Game",
            game_type="game",
            is_free=False,
            coming_soon=False,
            release_date="2026 年 1 月 1 日",
            price=SteamPrice(Decimal("60"), Decimal("100"), "CNY", 40),
            developers=("Developer",),
            publishers=("Publisher",),
            platforms=("windows",),
            genres=("动作",),
            categories=("单人",),
            languages=("简体中文",),
            controller_support="full",
            achievement_count=10,
            dlc_count=0,
            metacritic_score=80,
            recommendation_count=1000,
            required_age="",
            content_notes="",
            website="",
        )


class FakeHeyboxClient:
    async def global_prices(self, _appid: int) -> list[RegionPrice]:
        return [
            RegionPrice("CN", "中国", Decimal("60"), Decimal("100"), 40),
            RegionPrice("UA", "乌克兰", Decimal("40"), Decimal("80"), 50),
        ]


class FakePriceService:
    default_language = "schinese"
    steam_client = FakeSteamClient()
    heybox_client = FakeHeyboxClient()

    async def resolve_game(self, _title: str, country: str) -> tuple[GameIdentity, str]:
        return GameIdentity(123, "Test Game / appid=123"), country

    async def load_history(self, _appid: int, _country: str):
        return parse_price_history(
            {
                "prices": [
                    history_point("2026-01-01", "100", 0),
                    history_point("2026-01-10", "50", 50),
                ],
                "lowest_info": {"date": "2026-01-10", "price": "50", "discount": 50},
                "lowest_info_v2": {"currency": "CNY"},
            }
        )


def history_point(recorded_on: str, price: str, discount: int) -> dict:
    return {
        "date": recorded_on,
        "price": price,
        "rmb_price": price,
        "currency": "CNY",
        "discount": discount,
    }


if __name__ == "__main__":
    unittest.main()
