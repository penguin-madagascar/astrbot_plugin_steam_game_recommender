from __future__ import annotations

import sys
import types
import unittest
from pathlib import Path

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

from astrbot_plugin_steam_game_recommender.services.preference_parser import (  # noqa: E402
    parse_preference_json,
)
from astrbot_plugin_steam_game_recommender.services.similarity_ranker import (  # noqa: E402
    build_profile_from_preference,
)
from astrbot_plugin_steam_game_recommender.services.steam_index import (  # noqa: E402
    STEAM_ONLY_SCOPE_WARNING,
    steam_only_scope_warning_for,
)
from astrbot_plugin_steam_game_recommender.storage.models import GamePreference  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]


class SteamOnlyMetadataTest(unittest.TestCase):
    def test_plugin_id_display_name_and_version_are_0_6_1(self) -> None:
        main_text = (ROOT / "main.py").read_text(encoding="utf-8")
        metadata_text = (ROOT / "metadata.yaml").read_text(encoding="utf-8")

        self.assertIn('PLUGIN_NAME = "astrbot_plugin_steam_game_recommender"', main_text)
        self.assertIn('PLUGIN_VERSION = "0.6.1"', main_text)
        self.assertIn("class SteamGameRecommenderPlugin", main_text)
        self.assertIn("name: astrbot_plugin_steam_game_recommender", metadata_text)
        self.assertIn("display_name: Steam 游戏推荐助手", metadata_text)
        self.assertIn("version: 0.6.1", metadata_text)

    def test_readme_documents_only_current_steam_interfaces(self) -> None:
        readme = (ROOT / "README.md").read_text(encoding="utf-8")

        for command in ("/gamerec", "/gamerec_retry", "/accountbind", "/randomrec"):
            self.assertIn(command, readme)
        self.assertNotIn("/unplayedrec", readme)
        self.assertNotIn("/未玩推荐", readme)
        self.assertIn("-US", readme)
        self.assertIn("推荐分：86%", readme)


class SteamOnlyPreferenceTest(unittest.TestCase):
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
              "library_filter_mode": "only_owned",
              "players": 2,
              "result_count": 5
            }
            """
        )

        self.assertEqual(preference.extra_tags, ["轻松", "解谜", "本地合作"])
        self.assertEqual(preference.reference_games_like, ["双人成行"])
        self.assertEqual(preference.reference_search_terms, ["It Takes Two"])
        self.assertEqual(preference.library_filter_mode, "only_owned")

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
