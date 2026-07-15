from __future__ import annotations

import importlib
import json
import unittest
from types import SimpleNamespace

from astrbot_plugin_steam_game_recommender.services.formatter import (
    format_recommendation_messages,
)
from astrbot_plugin_steam_game_recommender.services.run_notices import RunNotice
from astrbot_plugin_steam_game_recommender.storage.models import GamePreference, RankedGame


def fallback_module():
    try:
        return importlib.import_module(
            "astrbot_plugin_steam_game_recommender.services.llm_fallback"
        )
    except ModuleNotFoundError as exc:
        raise AssertionError("LLM fallback service module is missing") from exc


def response_payload(*suggestions: dict[str, object]) -> str:
    return json.dumps({"suggestions": list(suggestions)}, ensure_ascii=False)


def suggestion(title: object = "Game A", reason: object = "匹配合作解谜偏好。") -> dict:
    return {"title": title, "reason": reason}


class UnverifiedSuggestionContractTest(unittest.TestCase):
    def test_accepts_exact_json_contract(self) -> None:
        service = fallback_module()

        parsed = service.parse_unverified_suggestions(
            response_payload(
                suggestion("Game A", "适合轻松合作。"),
                suggestion("Game B", "符合解谜偏好。"),
            ),
            result_limit=2,
        )

        self.assertEqual(
            parsed,
            (
                service.UnverifiedGameSuggestion("Game A", "适合轻松合作。"),
                service.UnverifiedGameSuggestion("Game B", "符合解谜偏好。"),
            ),
        )

    def test_accepts_one_json_object_in_fence_or_peripheral_explanation(self) -> None:
        service = fallback_module()
        payload = response_payload(suggestion())
        cases = [
            f"```json\n{payload}\n```",
            f"以下是结构化结果：\n{payload}\n请查收。",
        ]

        for raw in cases:
            with self.subTest(raw=raw):
                parsed = service.parse_unverified_suggestions(raw, result_limit=1)

                self.assertEqual(parsed[0].title, "Game A")

    def test_rejects_blank_missing_extra_or_wrong_typed_fields(self) -> None:
        service = fallback_module()
        invalid = [
            "",
            "not json",
            "{}",
            '{"suggestions":[]}',
            '{"suggestions":"not-an-array"}',
            '{"suggestions":[{"title":"Game A"}]}',
            '{"suggestions":[{"title":"Game A","reason":"理由","extra":1}]}',
            '{"suggestions":[{"title":1,"reason":"理由"}]}',
            '{"suggestions":[{"title":"Game A","reason":false}]}',
            '{"suggestions":[{"title":"Game A","reason":"理由"}],"extra":1}',
            (
                '{"suggestions":[{"title":"Game A","reason":"理由"}]}'
                '{"suggestions":[{"title":"Game B","reason":"理由"}]}'
            ),
        ]

        for raw in invalid:
            with self.subTest(raw=raw):
                with self.assertRaises(service.LlmFallbackContractError):
                    service.parse_unverified_suggestions(raw, result_limit=2)

    def test_folds_whitespace_deduplicates_normalized_titles_and_truncates(self) -> None:
        service = fallback_module()
        raw = response_payload(
            suggestion("  Ｇａｍｅ\nＡ  ", "  第一条\n理由。  "),
            suggestion("game a", "重复项不应覆盖第一条。"),
            suggestion("Game B", "第二款理由。"),
            suggestion("Game C", "超量但合同合法。"),
        )

        parsed = service.parse_unverified_suggestions(raw, result_limit=2)

        self.assertEqual(
            parsed,
            (
                service.UnverifiedGameSuggestion("Ｇａｍｅ Ａ", "第一条 理由。"),
                service.UnverifiedGameSuggestion("Game B", "第二款理由。"),
            ),
        )

    def test_rejects_blank_or_overlong_normalized_text(self) -> None:
        service = fallback_module()
        cases = [
            suggestion(" \n ", "理由"),
            suggestion("Game A", " \n "),
            suggestion("x" * 121, "理由"),
            suggestion("Game A", "x" * 181),
        ]

        for item in cases:
            with self.subTest(item=item):
                with self.assertRaises(service.LlmFallbackContractError):
                    service.parse_unverified_suggestions(
                        response_payload(item),
                        result_limit=1,
                    )

    def test_rejects_prohibited_claims_in_items_or_peripheral_response(self) -> None:
        service = fallback_module()
        prohibited = [
            "访问 https://example.com 查看详情。",
            "使用 steam://run/123 启动。",
            "对应 App ID 123。",
            "对应游戏AppID：123。",
            "详情见 store.steampowered.com/app/123。",
            "购买链接可在商店找到。",
            "推荐分 90/100。",
            "评分：9。",
            "玩家评分为 9 分。",
            "匹配度达到 95%。",
            "当前价格为 68 元。",
            "售价为 USD 20。",
            "仅需USD20。",
            "Steam 好评率 90%。",
            "评测数量超过 1000 条。",
            "该条目已验证为 Steam 游戏。",
        ]

        for text in prohibited:
            with self.subTest(text=text):
                with self.assertRaises(service.LlmFallbackContractError):
                    service.parse_unverified_suggestions(
                        response_payload(suggestion(reason=text)),
                        result_limit=1,
                    )

        raw = "详情见 https://example.com\n" + response_payload(suggestion())
        with self.assertRaises(service.LlmFallbackContractError):
            service.parse_unverified_suggestions(raw, result_limit=1)

    def test_validates_every_item_before_truncating(self) -> None:
        service = fallback_module()
        raw = response_payload(
            suggestion("Game A", "第一款理由。"),
            suggestion("Game B", "第二款理由。"),
            suggestion("Game C", "访问 https://example.com。"),
        )

        with self.assertRaises(service.LlmFallbackContractError):
            service.parse_unverified_suggestions(raw, result_limit=2)

    def test_allows_explicit_unverified_disclaimer(self) -> None:
        service = fallback_module()

        parsed = service.parse_unverified_suggestions(
            response_payload(
                suggestion(reason="仅按需求匹配，未经过 Steam 数据验证。")
            ),
            result_limit=1,
        )

        self.assertEqual(parsed[0].reason, "仅按需求匹配，未经过 Steam 数据验证。")


class LlmFallbackGenerationTest(unittest.IsolatedAsyncioTestCase):
    async def test_contract_failure_regenerates_once_from_same_original_requirement(self) -> None:
        service = fallback_module()
        context = SequencedContext(
            "not json",
            response_payload(suggestion("Game A", "符合合作偏好。")),
        )

        parsed = await service.generate_unverified_game_suggestions(
            context,
            "provider/explicit-fallback",
            raw_query="合作解谜",
            preference=GamePreference(platforms=["steam"], genres_like=["co-op"]),
            result_limit=2,
        )

        self.assertEqual(parsed[0].title, "Game A")
        self.assertEqual(len(context.calls), 2)
        self.assertEqual(
            [call["chat_provider_id"] for call in context.calls],
            ["provider/explicit-fallback", "provider/explicit-fallback"],
        )
        self.assertEqual(
            context.calls[0]["prompt"].split("INPUT=", 1)[1],
            context.calls[1]["prompt"].split("INPUT=", 1)[1],
        )
        self.assertIn("合作解谜", context.calls[0]["prompt"])
        self.assertEqual(context.current_provider_calls, 0)

    async def test_two_contract_failures_raise_after_exactly_two_calls(self) -> None:
        service = fallback_module()
        context = SequencedContext("not json", '{"suggestions":[]}')

        with self.assertRaises(service.LlmFallbackContractError):
            await service.generate_unverified_game_suggestions(
                context,
                "provider/explicit-fallback",
                raw_query="合作解谜",
                preference=GamePreference(),
                result_limit=2,
            )

        self.assertEqual(len(context.calls), 2)

    async def test_provider_failure_does_not_repair(self) -> None:
        service = fallback_module()
        context = SequencedContext(RuntimeError("provider unavailable"))

        with self.assertRaises(service.LlmFallbackProviderError):
            await service.generate_unverified_game_suggestions(
                context,
                "provider/explicit-fallback",
                raw_query="合作解谜",
                preference=GamePreference(),
                result_limit=2,
            )

        self.assertEqual(len(context.calls), 1)

    async def test_empty_provider_is_rejected_without_using_current_session_model(self) -> None:
        service = fallback_module()
        context = SequencedContext(response_payload(suggestion()))

        with self.assertRaises(service.LlmFallbackProviderError):
            await service.generate_unverified_game_suggestions(
                context,
                " ",
                raw_query="合作解谜",
                preference=GamePreference(),
                result_limit=1,
            )

        self.assertEqual(context.calls, [])
        self.assertEqual(context.current_provider_calls, 0)

    async def test_repeated_identical_requests_are_not_cached(self) -> None:
        service = fallback_module()
        payload = response_payload(suggestion())
        context = SequencedContext(payload, payload)

        for _ in range(2):
            parsed = await service.generate_unverified_game_suggestions(
                context,
                "provider/explicit-fallback",
                raw_query="合作解谜",
                preference=GamePreference(),
                result_limit=1,
            )
            self.assertEqual(parsed[0].title, "Game A")

        self.assertEqual(len(context.calls), 2)

    async def test_preference_payload_uses_json_serializable_model_dump_mode(
        self,
    ) -> None:
        service = fallback_module()
        context = SequencedContext(response_payload(suggestion()))

        await service.generate_unverified_game_suggestions(
            context,
            "provider/explicit-fallback",
            raw_query="合作解谜",
            preference=JsonModePreference(),
            result_limit=1,
        )

        self.assertIn('"serialization":"json"', context.calls[0]["prompt"])


class UnverifiedSuggestionFormattingTest(unittest.TestCase):
    def test_preserves_notice_and_rule_nodes_before_disclaimer_and_suggestions(self) -> None:
        service = fallback_module()
        suggestions = (
            service.UnverifiedGameSuggestion("Game A", "适合轻松合作。"),
            service.UnverifiedGameSuggestion("Game B", "符合解谜偏好。"),
        )

        messages = format_recommendation_messages(
            GamePreference(parse_warnings=["参考游戏未能可靠解析"]),
            [],
            limit=2,
            run_notices=(
                RunNotice("first", "warning", "第一条运行通知"),
                RunNotice("second", "warning", "第二条运行通知"),
            ),
            unverified_suggestions=suggestions,
        )

        self.assertEqual(messages[0:2], ["第一条运行通知", "第二条运行通知"])
        self.assertIn("暂时没有找到满足当前条件的游戏", messages[2])
        self.assertIn("偏好解析提示", messages[2])
        self.assertIn("参考游戏未能可靠解析", messages[2])
        self.assertEqual(
            messages[3],
            "⚠️ LLM 兜底建议（未经过 Steam 数据验证）",
        )
        self.assertEqual(messages[4], "1. 《Game A》\n模型判断理由：适合轻松合作。")
        self.assertEqual(messages[5], "2. 《Game B》\n模型判断理由：符合解谜偏好。")

    def test_suggestion_nodes_never_use_verified_game_fields(self) -> None:
        service = fallback_module()

        messages = format_recommendation_messages(
            GamePreference(),
            [],
            unverified_suggestions=(
                service.UnverifiedGameSuggestion("Game A", "符合玩法偏好。"),
            ),
        )

        suggestion_message = messages[-1]
        for prohibited in ("/100", "价格", "评测", "AppID", "steam://", "http"):
            with self.subTest(prohibited=prohibited):
                self.assertNotIn(prohibited, suggestion_message)

    def test_verified_results_ignore_unverified_suggestions(self) -> None:
        service = fallback_module()

        messages = format_recommendation_messages(
            GamePreference(result_count=1),
            [RankedGame(appid=1, title="Verified Game", app_type="game", score=80)],
            limit=1,
            unverified_suggestions=(
                service.UnverifiedGameSuggestion("Game A", "模型理由。"),
            ),
        )

        rendered = "\n".join(messages)
        self.assertIn("Verified Game", rendered)
        self.assertNotIn("LLM 兜底建议", rendered)
        self.assertNotIn("Game A", rendered)


class SequencedContext:
    def __init__(self, *responses: object) -> None:
        self.responses = list(responses)
        self.calls: list[dict] = []
        self.current_provider_calls = 0

    async def llm_generate(self, **kwargs):
        self.calls.append(kwargs)
        response = self.responses.pop(0)
        if isinstance(response, BaseException):
            raise response
        return SimpleNamespace(completion_text=response)

    async def get_current_chat_provider_id(self, **_kwargs):
        self.current_provider_calls += 1
        raise AssertionError("fallback must not resolve the current session provider")


class JsonModePreference:
    def model_dump(self, *, mode: str) -> dict[str, str]:
        if mode != "json":
            raise AssertionError("preference payload must use JSON serialization mode")
        return {"serialization": mode}


if __name__ == "__main__":
    unittest.main()
