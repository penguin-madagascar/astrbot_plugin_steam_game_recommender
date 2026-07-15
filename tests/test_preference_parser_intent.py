from __future__ import annotations

import sys
import types
import unittest
from types import SimpleNamespace
from unittest.mock import patch


class FakeStar:
    def __init__(self, context) -> None:
        self.context = context


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
star_module.Star = FakeStar
sys.modules.setdefault("astrbot", types.ModuleType("astrbot"))
sys.modules.setdefault("astrbot.api", api_module)
sys.modules.setdefault("astrbot.api.event", event_module)
sys.modules.setdefault("astrbot.api.star", star_module)

from astrbot_plugin_steam_game_recommender.services.preference_parser import (
    PREFERENCE_SCHEMA_HINT,
    PreferenceParser,
    parse_preference_json,
)  # noqa: E402
from astrbot_plugin_steam_game_recommender.services import preference_parser as parser_module  # noqa: E402


class PreferenceParserIntentTest(unittest.TestCase):
    def test_schema_requests_quality_and_release_intents_without_aaa_expansion(self) -> None:
        self.assertIn('"quality_intent": "normal"', PREFERENCE_SCHEMA_HINT)
        self.assertIn('"allow_unreleased": false', PREFERENCE_SCHEMA_HINT)
        self.assertNotIn("genres_like 可包含 action、adventure、rpg", PREFERENCE_SCHEMA_HINT)
        self.assertNotIn("extra_tags 包含 aaa、story rich、open world", PREFERENCE_SCHEMA_HINT)
        self.assertIn('"explicit_tag_evidence": []', PREFERENCE_SCHEMA_HINT)
        self.assertIn("span 必须逐字复制用户原文", PREFERENCE_SCHEMA_HINT)
        self.assertIn("3A/AAA/大作本身都不能作为标签证据", PREFERENCE_SCHEMA_HINT)
        self.assertIn("genres_dislike，并且必须与标签所在字段一致", PREFERENCE_SCHEMA_HINT)

    def test_schema_describes_singleplayer_tags_with_explicit_polarity(self) -> None:
        self.assertNotIn("肉鸽、纯单人、pvp", PREFERENCE_SCHEMA_HINT)
        self.assertIn("没有否定或排除措辞时", PREFERENCE_SCHEMA_HINT)
        self.assertIn("明确说不要单机、不要单人或排除 singleplayer", PREFERENCE_SCHEMA_HINT)

    def test_schema_explains_semantic_and_company_contracts(self) -> None:
        for field in (
            '"derived_intent_tags": []',
            '"soft_features": []',
            '"company_preferences": []',
        ):
            self.assertIn(field, PREFERENCE_SCHEMA_HINT)
        self.assertIn("proxy_tags 只用于召回", PREFERENCE_SCHEMA_HINT)
        self.assertIn("不得因为 3A、AAA、大作", PREFERENCE_SCHEMA_HINT)

    def test_llm_json_fields_are_normalized_by_game_preference(self) -> None:
        preference = parse_preference_json(
            '{"quality_intent":"MAINSTREAM","allow_unreleased":true}'
        )

        self.assertEqual(preference.quality_intent, "mainstream")
        self.assertTrue(preference.allow_unreleased)

    def test_explicit_tag_evidence_is_parse_only_metadata(self) -> None:
        preference = parse_preference_json(
            '{"genres_like":["dynamic_tag"],"explicit_tag_evidence":['
            '{"target":"genres_like","tag":"dynamic_tag","span":"未知玩法"}]}'
        )

        self.assertEqual(preference.explicit_tag_evidence[0].span, "未知玩法")
        payload = (
            preference.model_dump()
            if hasattr(preference, "model_dump")
            else preference.dict()
        )
        self.assertNotIn("explicit_tag_evidence", payload)


class PreferenceParserDiagnosticsTest(unittest.IsolatedAsyncioTestCase):
    async def test_empty_input_logs_the_empty_path(self) -> None:
        parser = PreferenceParser(SimpleNamespace(), provider_id="provider/test")

        with patch.object(parser_module.logger, "debug") as debug:
            outcome = await parser.parse_preference(object(), "  ")

        self.assertIn("需求为空", outcome.preference.parse_warnings[0])
        self.assertEqual(outcome.path, "empty")
        debug.assert_called_once_with(
            "recommendation_parse event=parse_complete path=%s",
            "empty",
        )

    async def test_success_logs_only_the_parse_path(self) -> None:
        context = SimpleNamespace(
            llm_generate=lambda **_kwargs: None,
        )

        async def generate(**_kwargs):
            return SimpleNamespace(completion_text='{"genres_like":["puzzle"]}')

        context.llm_generate = generate
        parser = PreferenceParser(context, provider_id="provider/test")

        with patch.object(parser_module.logger, "debug") as debug:
            outcome = await parser.parse_preference(object(), "想玩解谜游戏")

        self.assertIn("puzzle", outcome.preference.genres_like)
        debug.assert_called_once_with(
            "recommendation_parse event=parse_complete path=%s",
            "llm",
        )

    async def test_repaired_json_logs_the_repair_path(self) -> None:
        responses = iter(
            [
                SimpleNamespace(completion_text="not json"),
                SimpleNamespace(completion_text='{"genres_like":["puzzle"]}'),
            ]
        )

        async def generate(**_kwargs):
            return next(responses)

        parser = PreferenceParser(
            SimpleNamespace(llm_generate=generate),
            provider_id="provider/test",
        )
        with patch.object(parser_module.logger, "debug") as debug:
            outcome = await parser.parse_preference(object(), "想玩解谜游戏")

        self.assertIn("puzzle", outcome.preference.genres_like)
        debug.assert_called_once_with(
            "recommendation_parse event=parse_complete path=%s",
            "llm_repair",
        )

    async def test_invalid_json_logs_the_keyword_fallback_path(self) -> None:
        async def generate(**_kwargs):
            return SimpleNamespace(completion_text="not json")

        parser = PreferenceParser(
            SimpleNamespace(llm_generate=generate),
            provider_id="provider/test",
        )
        with patch.object(parser_module.logger, "debug") as debug:
            outcome = await parser.parse_preference(object(), "想玩解谜游戏")

        self.assertEqual(outcome.preference.parse_warnings, [])
        self.assertEqual(outcome.prelude_messages[0].code, "preference_parser_unavailable")
        debug.assert_called_once_with(
            "recommendation_parse event=parse_complete path=%s",
            "keyword_fallback",
        )


if __name__ == "__main__":
    unittest.main()
