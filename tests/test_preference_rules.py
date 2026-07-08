from __future__ import annotations

import unittest

from astrbot_plugin_game_recommender.services.preference_rules import (
    extract_reference_games,
    infer_preference_from_text,
    merge_text_preference,
)
from astrbot_plugin_game_recommender.storage.models import GamePreference


class PreferenceRulesTest(unittest.TestCase):
    def test_infers_steampeek_profile_terms_from_user_text(self) -> None:
        preference = infer_preference_from_text(
            "推荐几个适合 Switch 和 Steam 的双人游戏，不要恐怖，"
            "最好支持中文，预算 100 以内，类似双人成行但别太难。"
        )

        self.assertIn("nintendo switch", preference.platforms)
        self.assertIn("steam", preference.platforms)
        self.assertEqual(preference.players, 2)
        self.assertEqual(preference.budget, 100)
        self.assertEqual(preference.language, "中文")
        self.assertEqual(preference.difficulty, "easy")
        self.assertIn("horror", preference.genres_dislike)
        self.assertIn("双人成行", preference.reference_games_like)
        self.assertIn("co-op", preference.genres_like)
        self.assertIn("relaxing", preference.extra_tags)
        self.assertNotIn("platformer", preference.genres_like)

    def test_merges_llm_extra_tags_with_keyword_rules(self) -> None:
        llm_preference = GamePreference(
            platforms=[],
            genres_like=["puzzle"],
            extra_tags=["剧情合作"],
            genres_dislike=[],
            reference_games_like=[],
            players=None,
            budget=None,
            language=None,
            difficulty=None,
            result_count=5,
        )

        merged = merge_text_preference(
            llm_preference,
            "推荐几个适合 Steam 的双人游戏，不要恐怖，"
            "最好支持中文，预算 100 以内，类似双人成行但别太难。",
        )

        self.assertEqual(merged.platforms, ["steam"])
        self.assertEqual(merged.players, 2)
        self.assertEqual(merged.budget, 100)
        self.assertEqual(merged.language, "中文")
        self.assertEqual(merged.difficulty, "easy")
        self.assertIn("horror", merged.genres_dislike)
        self.assertIn("双人成行", merged.reference_games_like)
        self.assertIn("剧情合作", merged.extra_tags)
        self.assertIn("relaxing", merged.extra_tags)

    def test_text_platforms_override_llm_platform_hallucinations(self) -> None:
        llm_preference = GamePreference(
            platforms=["steam", "playstation", "nintendo switch"],
            result_count=5,
        )

        merged = merge_text_preference(
            llm_preference,
            "推荐几个适合 Steam 的双人游戏，类似双人成行。",
        )

        self.assertEqual(merged.platforms, ["steam"])

    def test_pc_and_steam_are_distinct_platform_preferences(self) -> None:
        pc_preference = infer_preference_from_text("我想找 PC 上玩的合作射击游戏")
        steam_preference = infer_preference_from_text("Steam 上有没有双人合作游戏")

        self.assertIn("pc", pc_preference.platforms)
        self.assertNotIn("steam", pc_preference.platforms)
        self.assertIn("steam", steam_preference.platforms)

    def test_explicit_text_count_overrides_llm_default_count(self) -> None:
        llm_preference = GamePreference(result_count=5)

        merged = merge_text_preference(
            llm_preference,
            "想找 3 款 PC/Steam 上可以和朋友线上合作的轻松解谜游戏",
        )

        self.assertEqual(merged.result_count, 3)

    def test_reference_title_extraction_is_generic(self) -> None:
        self.assertEqual(
            extract_reference_games("想找类似星露谷物语的多人种田经营游戏"),
            ["星露谷物语"],
        )
        self.assertEqual(
            extract_reference_games("Steam Deck 上找短局卡牌策略，similar to Slay the Spire"),
            ["Slay the Spire"],
        )

    def test_dark_souls_like_request_extracts_searchable_soulslike_profile(self) -> None:
        preference = infer_preference_from_text("类似黑暗之魂的游戏")

        self.assertEqual(preference.reference_games_like, ["黑暗之魂"])
        self.assertIn("soulslike", preference.extra_tags)
        self.assertIn("action", preference.extra_tags)
        self.assertIn("rpg", preference.extra_tags)

    def test_aaa_request_extracts_broad_blockbuster_profile(self) -> None:
        preference = infer_preference_from_text("推荐一下3a游戏")

        self.assertIn("action", preference.genres_like)
        self.assertIn("adventure", preference.genres_like)
        self.assertIn("rpg", preference.genres_like)
        self.assertIn("aaa", preference.extra_tags)
        self.assertIn("story rich", preference.extra_tags)
        self.assertIn("open world", preference.extra_tags)
        self.assertEqual(preference.result_count, 5)

    def test_infers_library_filter_mode_from_text(self) -> None:
        self.assertEqual(
            infer_preference_from_text("推荐几个合作游戏，排除已有").library_filter_mode,
            "exclude_owned",
        )
        self.assertEqual(
            infer_preference_from_text("recommend co-op games only-owned").library_filter_mode,
            "only_owned",
        )

    def test_merges_llm_library_filter_mode(self) -> None:
        merged = merge_text_preference(
            GamePreference(library_filter_mode="exclude_owned"),
            "推荐几个 Steam 合作游戏",
        )

        self.assertEqual(merged.library_filter_mode, "exclude_owned")


if __name__ == "__main__":
    unittest.main()
