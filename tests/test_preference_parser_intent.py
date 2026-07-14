from __future__ import annotations

import sys
import types
import unittest


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
    parse_preference_json,
)  # noqa: E402


class PreferenceParserIntentTest(unittest.TestCase):
    def test_schema_requests_quality_and_release_intents_without_aaa_expansion(self) -> None:
        self.assertIn('"quality_intent": "normal"', PREFERENCE_SCHEMA_HINT)
        self.assertIn('"allow_unreleased": false', PREFERENCE_SCHEMA_HINT)
        self.assertNotIn("genres_like 可包含 action、adventure、rpg", PREFERENCE_SCHEMA_HINT)
        self.assertNotIn("extra_tags 包含 aaa、story rich、open world", PREFERENCE_SCHEMA_HINT)

    def test_llm_json_fields_are_normalized_by_game_preference(self) -> None:
        preference = parse_preference_json(
            '{"quality_intent":"MAINSTREAM","allow_unreleased":true}'
        )

        self.assertEqual(preference.quality_intent, "mainstream")
        self.assertTrue(preference.allow_unreleased)


if __name__ == "__main__":
    unittest.main()
