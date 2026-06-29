from __future__ import annotations

from pathlib import Path
import sys
import types
import unittest

api_module = types.ModuleType("astrbot.api")
api_module.logger = types.SimpleNamespace(
    debug=lambda *_args, **_kwargs: None,
    warning=lambda *_args, **_kwargs: None,
)
event_module = types.ModuleType("astrbot.api.event")
event_module.AstrMessageEvent = object
star_module = types.ModuleType("astrbot.api.star")
star_module.Context = object
sys.modules.setdefault("astrbot", types.ModuleType("astrbot"))
sys.modules.setdefault("astrbot.api", api_module)
sys.modules.setdefault("astrbot.api.event", event_module)
sys.modules.setdefault("astrbot.api.star", star_module)

from astrbot_plugin_game_recommender.services.preference_parser import parse_preference_json
from astrbot_plugin_game_recommender.services.similarity_ranker import (
    build_profile_from_preference,
)
from astrbot_plugin_game_recommender.services.steam_index import (
    STEAM_ONLY_SCOPE_WARNING,
    steam_only_scope_warning_for,
)
from astrbot_plugin_game_recommender.storage.models import GamePreference


ROOT = Path(__file__).resolve().parents[1]


class SteamPeekOnlyMetadataTest(unittest.TestCase):
    def test_version_is_0_3_2_and_legacy_provider_config_is_removed(self) -> None:
        main_text = (ROOT / "main.py").read_text(encoding="utf-8")
        metadata_text = (ROOT / "metadata.yaml").read_text(encoding="utf-8")
        schema_text = (ROOT / "_conf_schema.json").read_text(encoding="utf-8")
        legacy_key = "ra" + "wg_api_key"

        self.assertIn('PLUGIN_VERSION = "0.3.2"', main_text)
        self.assertIn("version: 0.3.2", metadata_text)
        self.assertNotIn(legacy_key, schema_text)

    def test_legacy_provider_files_are_removed(self) -> None:
        legacy_name = "ra" + "wg.py"

        self.assertFalse((ROOT / "clients" / legacy_name).exists())
        self.assertFalse((ROOT / "services" / "search_plan.py").exists())
        self.assertFalse((ROOT / "services" / "reference_data.py").exists())


class SteamPeekOnlyPreferenceTest(unittest.TestCase):
    def test_llm_json_accepts_extra_tags_and_reference_titles(self) -> None:
        preference = parse_preference_json(
            """
            {
              "platforms": ["steam"],
              "genres_like": ["co-op"],
              "extra_tags": ["轻松", "解谜", "本地合作"],
              "genres_dislike": ["恐怖"],
              "reference_games_like": ["双人成行"],
              "reference_search_terms": ["It Takes Two"],
              "players": 2,
              "result_count": 5
            }
            """
        )

        self.assertEqual(preference.extra_tags, ["轻松", "解谜", "本地合作"])
        self.assertEqual(preference.reference_games_like, ["双人成行"])
        self.assertEqual(preference.reference_search_terms, ["It Takes Two"])

        profile = build_profile_from_preference(preference)

        self.assertIn("co_op", profile.include_tags)
        self.assertIn("local_coop", profile.include_tags)
        self.assertIn("puzzle", profile.include_tags)
        self.assertIn("relaxing", profile.include_tags)
        self.assertIn("horror", profile.exclude_tags)

    def test_non_steam_platforms_are_reported_as_out_of_scope(self) -> None:
        warning = steam_only_scope_warning_for(
            GamePreference(platforms=["nintendo switch", "steam"])
        )

        self.assertEqual(warning, STEAM_ONLY_SCOPE_WARNING)


if __name__ == "__main__":
    unittest.main()
