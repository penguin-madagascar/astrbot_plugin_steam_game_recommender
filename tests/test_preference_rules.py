from __future__ import annotations

import unittest

from astrbot_plugin_game_recommender.services.preference_rules import (
    infer_preference_from_text,
    merge_text_preference,
)
from astrbot_plugin_game_recommender.services.reference_resolver import (
    ReferenceGameResolver,
    normalize_reference_title,
)
from astrbot_plugin_game_recommender.storage.models import GameCandidate
from astrbot_plugin_game_recommender.storage.models import GamePreference


class PreferenceRulesTest(unittest.TestCase):
    def test_infers_high_confidence_preferences_from_user_text(self) -> None:
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
        self.assertIn("it takes two", preference.reference_games_like)
        for term in ("co-op", "local co-op", "puzzle", "adventure", "casual", "platformer"):
            self.assertIn(term, preference.genres_like)

    def test_merges_keyword_rules_into_empty_llm_preference(self) -> None:
        llm_preference = GamePreference(
            platforms=[],
            genres_like=[],
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
            "推荐几个适合 Switch 和 Steam 的双人游戏，不要恐怖，"
            "最好支持中文，预算 100 以内，类似双人成行但别太难。",
        )

        self.assertIn("nintendo switch", merged.platforms)
        self.assertIn("steam", merged.platforms)
        self.assertEqual(merged.players, 2)
        self.assertEqual(merged.budget, 100)
        self.assertEqual(merged.language, "中文")
        self.assertEqual(merged.difficulty, "easy")
        self.assertIn("horror", merged.genres_dislike)
        self.assertIn("it takes two", merged.reference_games_like)

    def test_text_platforms_override_llm_platform_hallucinations(self) -> None:
        llm_preference = GamePreference(
            platforms=["steam", "playstation", "nintendo switch"],
            result_count=5,
        )

        merged = merge_text_preference(
            llm_preference,
            "推荐几个适合 Switch 和 Steam 的双人游戏，类似双人成行。",
        )

        self.assertEqual(merged.platforms, ["steam", "nintendo switch"])

    def test_pc_and_steam_are_distinct_platform_preferences(self) -> None:
        pc_preference = infer_preference_from_text("我想找 PC 和 Xbox 都能玩的合作射击游戏")
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


class ReferenceGameResolverTest(unittest.IsolatedAsyncioTestCase):
    async def test_chinese_alias_resolves_to_rawg_entity_without_sentence_special_case(self) -> None:
        resolver = ReferenceGameResolver(FakeGameSource([]))

        resolved = await resolver.resolve_reference_games(
            "像《双人成行》一样的合作游戏",
            GamePreference(reference_games_like=["双人成行"]),
        )

        self.assertEqual(len(resolved), 1)
        self.assertEqual(resolved[0].canonical_title, "It Takes Two")
        self.assertEqual(resolved[0].rawg_slug, "it-takes-two")
        self.assertGreaterEqual(resolved[0].confidence, 0.88)
        self.assertEqual(resolved[0].source, "alias")

    async def test_stardew_chinese_alias_resolves_to_reference_profile(self) -> None:
        resolver = ReferenceGameResolver(FakeGameSource([]))

        resolved = await resolver.resolve_reference_games(
            "类似星露谷物语的多人种田经营游戏",
            GamePreference(reference_games_like=["星露谷物语"]),
        )

        self.assertEqual(len(resolved), 1)
        self.assertEqual(resolved[0].canonical_title, "Stardew Valley")
        self.assertEqual(resolved[0].rawg_slug, "stardew-valley")
        self.assertGreaterEqual(resolved[0].confidence, 0.88)
        self.assertEqual(resolved[0].source, "alias")

    async def test_english_and_edition_suffixes_normalize_to_same_reference(self) -> None:
        resolver = ReferenceGameResolver(FakeGameSource([]))

        resolved = await resolver.resolve_reference_games(
            "接近 It Takes Two - Friend's Pass",
            GamePreference(),
        )

        self.assertEqual(resolved[0].canonical_title, "It Takes Two")
        self.assertEqual(
            normalize_reference_title("《It Takes Two - Friend's Pass》"),
            normalize_reference_title("it takes two"),
        )

    async def test_unknown_reference_uses_rawg_title_search_when_alias_is_missing(self) -> None:
        source = FakeGameSource(
            [
                GameCandidate(
                    title="Mystery Farm",
                    rawg_id=12345,
                    raw_url="https://rawg.io/games/mystery-farm",
                    genres=["Simulation"],
                    tags=["Co-op", "Casual"],
                )
            ]
        )

        resolved = await ReferenceGameResolver(source).resolve_reference_games(
            "类似 Mystery Farm 的轻松合作游戏",
            GamePreference(),
        )

        self.assertEqual(resolved[0].canonical_title, "Mystery Farm")
        self.assertEqual(resolved[0].rawg_slug, "mystery-farm")
        self.assertGreaterEqual(resolved[0].confidence, 0.70)
        self.assertEqual(resolved[0].source, "rawg")
        self.assertEqual(source.calls[0]["search"], "Mystery Farm")


class FakeGameSource:
    def __init__(self, games: list[GameCandidate]) -> None:
        self.games = games
        self.calls: list[dict] = []

    async def search_games(self, **kwargs):
        self.calls.append(kwargs)
        return self.games


if __name__ == "__main__":
    unittest.main()
