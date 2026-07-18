from __future__ import annotations

import json
import sys
import types
import unittest
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

api_module = types.ModuleType("astrbot.api")
api_module.logger = types.SimpleNamespace(
    debug=lambda *_args, **_kwargs: None,
    exception=lambda *_args, **_kwargs: None,
    info=lambda *_args, **_kwargs: None,
    warning=lambda *_args, **_kwargs: None,
)
event_module = types.ModuleType("astrbot.api.event")
event_module.AstrMessageEvent = object
star_module = types.ModuleType("astrbot.api.star")
star_module.Context = object


class FakeStar:
    def __init__(self, context) -> None:
        self.context = context


star_module.Star = FakeStar
sys.modules.setdefault("astrbot", types.ModuleType("astrbot"))
sys.modules.setdefault("astrbot.api", api_module)
sys.modules.setdefault("astrbot.api.event", event_module)
sys.modules.setdefault("astrbot.api.star", star_module)

from astrbot_plugin_steam_game_recommender.services import preference_parser as parser_module
from astrbot_plugin_steam_game_recommender.services.preference_parser import (
    PreferencePayloadError,
    PreferenceParser,
    parse_preference_json,
)
from astrbot_plugin_steam_game_recommender.storage.models import GamePreference


class ParseOutcomeTest(unittest.IsolatedAsyncioTestCase):
    async def test_success_returns_outcome_and_ignores_llm_result_count(self) -> None:
        context = SequencedContext(
            {
                "genres_like": ["puzzle"],
                "result_count": 2,
            }
        )
        parser = PreferenceParser(context, provider_id="provider/test")

        outcome = await parser.parse_preference(object(), "推荐解谜游戏")

        self.assertEqual(outcome.path, "llm")
        self.assertEqual(outcome.prelude_messages, ())
        self.assertEqual(outcome.preference.result_count, 10)
        self.assertIn("puzzle", outcome.preference.genres_like)
        self.assertEqual(len(context.calls), 1)

    async def test_provider_error_skips_repair_and_emits_independent_notice(self) -> None:
        context = SequencedContext(RuntimeError("provider unavailable"))
        parser = PreferenceParser(context, provider_id="provider/test")

        outcome = await parser.parse_preference(object(), "推荐三款解谜游戏")

        self.assertEqual(outcome.path, "keyword_fallback")
        self.assertEqual(outcome.preference.result_count, 3)
        self.assertEqual(outcome.preference.parse_warnings, [])
        self.assertEqual(len(outcome.prelude_messages), 1)
        notice = outcome.prelude_messages[0]
        self.assertEqual(notice.code, "preference_parser_unavailable")
        self.assertEqual(notice.severity, "error")
        self.assertIn("关键词", notice.text)
        self.assertNotIn("provider unavailable", notice.text)
        self.assertEqual(len(context.calls), 1)

    async def test_provider_exception_text_is_not_logged(self) -> None:
        secret = "secret /private/provider/path?token=abcdef"
        warnings: list[tuple[object, ...]] = []
        parser = PreferenceParser(
            SequencedContext(RuntimeError(secret)),
            provider_id="provider/test",
        )

        with patch.object(
            parser_module.logger,
            "warning",
            side_effect=lambda *args, **_kwargs: warnings.append(args),
        ):
            outcome = await parser.parse_preference(object(), "推荐解谜游戏")

        output = " ".join(str(value) for call in warnings for value in call)
        self.assertEqual(outcome.path, "keyword_fallback")
        self.assertNotIn(secret, output)
        self.assertNotIn("token=", output)
        self.assertIn("PreferenceProviderError", output)

    async def test_invalid_payload_repairs_exactly_once(self) -> None:
        context = SequencedContext(
            "not json",
            {"genres_like": ["puzzle"], "result_count": 1},
        )
        parser = PreferenceParser(context, provider_id="provider/test")

        outcome = await parser.parse_preference(object(), "推荐五款解谜游戏")

        self.assertEqual(outcome.path, "llm_repair")
        self.assertEqual(outcome.preference.result_count, 5)
        self.assertEqual(len(context.calls), 2)

    async def test_contract_overflow_repairs_once_instead_of_silent_truncation(self) -> None:
        overflowing = {
            "derived_intent_tags": [
                {"tag": f"tag_{index}", "source_span": "玩法"}
                for index in range(4)
            ]
        }
        context = SequencedContext(overflowing, {"genres_like": ["puzzle"]})
        parser = PreferenceParser(context, provider_id="provider/test")

        outcome = await parser.parse_preference(object(), "推荐解谜玩法")

        self.assertEqual(outcome.path, "llm_repair")
        self.assertEqual(len(context.calls), 2)

    async def test_invalid_payload_and_invalid_repair_fall_back_after_two_calls(self) -> None:
        context = SequencedContext("not json", "still not json")
        parser = PreferenceParser(context, provider_id="provider/test")

        outcome = await parser.parse_preference(object(), "推荐解谜游戏")

        self.assertEqual(outcome.path, "keyword_fallback")
        self.assertEqual(len(context.calls), 2)
        self.assertEqual(len(outcome.prelude_messages), 1)

    async def test_merge_programming_error_propagates_without_repair(self) -> None:
        context = SequencedContext({"genres_like": ["puzzle"]})
        parser = PreferenceParser(context, provider_id="provider/test")

        with (
            patch.object(
                parser_module,
                "merge_text_preference",
                side_effect=RuntimeError("merge bug"),
            ),
            self.assertRaisesRegex(RuntimeError, "merge bug"),
        ):
            await parser.parse_preference(object(), "推荐解谜游戏")

        self.assertEqual(len(context.calls), 1)

    async def test_llm_parse_programming_error_is_not_provider_fallback(self) -> None:
        parser = PreferenceParser(SequencedContext(), provider_id="provider/test")

        with (
            patch.object(
                parser,
                "_llm_parse",
                AsyncMock(side_effect=AssertionError("parse implementation bug")),
            ),
            self.assertRaisesRegex(AssertionError, "parse implementation bug"),
        ):
            await parser.parse_preference(object(), "推荐解谜游戏")

    async def test_llm_repair_programming_error_is_not_provider_fallback(self) -> None:
        context = SequencedContext("not json")
        parser = PreferenceParser(context, provider_id="provider/test")

        with (
            patch.object(
                parser,
                "_llm_repair",
                AsyncMock(side_effect=TypeError("repair implementation bug")),
            ),
            self.assertRaisesRegex(TypeError, "repair implementation bug"),
        ):
            await parser.parse_preference(object(), "推荐解谜游戏")

        self.assertEqual(len(context.calls), 1)

    async def test_validator_programming_error_skips_repair_and_propagates(self) -> None:
        context = SequencedContext({"genres_like": ["puzzle"]})
        parser = PreferenceParser(context, provider_id="provider/test")

        with (
            patch.object(
                GamePreference,
                "model_validate",
                side_effect=TypeError("validator implementation bug"),
            ),
            self.assertRaisesRegex(TypeError, "validator implementation bug"),
        ):
            await parser.parse_preference(object(), "推荐解谜游戏")

        self.assertEqual(len(context.calls), 1)


class ParsePreferenceJsonBoundaryTest(unittest.TestCase):
    def test_pydantic_contract_error_is_wrapped_as_payload_error(self) -> None:
        with self.assertRaises(PreferencePayloadError):
            parse_preference_json('{"allow_unreleased":[]}')

    def test_validator_programming_type_error_propagates(self) -> None:
        with (
            patch.object(
                GamePreference,
                "model_validate",
                side_effect=TypeError("validator implementation bug"),
            ),
            self.assertRaisesRegex(TypeError, "validator implementation bug"),
        ):
            parse_preference_json("{}")

    def test_validator_programming_value_error_propagates_without_rewrapping(self) -> None:
        with patch.object(
            GamePreference,
            "model_validate",
            side_effect=ValueError("validator implementation bug"),
        ):
            with self.assertRaises(ValueError) as raised:
                parse_preference_json("{}")

        self.assertIs(type(raised.exception), ValueError)
        self.assertEqual(str(raised.exception), "validator implementation bug")


class SequencedContext:
    def __init__(self, *responses: object) -> None:
        self.responses = list(responses)
        self.calls: list[dict] = []

    async def llm_generate(self, **kwargs):
        self.calls.append(kwargs)
        response = self.responses.pop(0)
        if isinstance(response, BaseException):
            raise response
        if isinstance(response, str):
            text = response
        else:
            text = json.dumps(response, ensure_ascii=False)
        return SimpleNamespace(completion_text=text)


if __name__ == "__main__":
    unittest.main()
