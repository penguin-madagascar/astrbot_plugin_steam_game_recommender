from __future__ import annotations

import unittest

try:
    from astrbot_plugin_game_recommender.services.preference_parser import keyword_fallback
    from astrbot_plugin_game_recommender.services.ranker import score_game
    from astrbot_plugin_game_recommender.services.recommender import GameRecommender
    from astrbot_plugin_game_recommender.storage.models import GameCandidate, GamePreference
except ModuleNotFoundError as exc:
    if exc.name in {"pydantic", "astrbot"}:
        raise unittest.SkipTest(f"{exc.name} is not installed in this environment")
    raise


class PreferenceFallbackTest(unittest.TestCase):
    def test_keyword_fallback_extracts_common_preferences(self) -> None:
        preference = keyword_fallback(
            "推荐几个适合 Switch 和 Steam 的双人游戏，不要恐怖，最好支持中文，预算 100 以内"
        )

        self.assertIn("nintendo switch", preference.platforms)
        self.assertIn("steam", preference.platforms)
        self.assertEqual(preference.players, 2)
        self.assertEqual(preference.budget, 100)
        self.assertIn("horror", preference.genres_dislike)
        self.assertEqual(preference.language, "中文")


class RankerTest(unittest.TestCase):
    def test_score_game_returns_explanations(self) -> None:
        preference = GamePreference(
            platforms=["steam"],
            genres_like=["co-op"],
            players=2,
            language="中文",
            budget=100,
        )
        game = GameCandidate(
            title="Example Co-op Game",
            platforms=["PC"],
            genres=["Adventure"],
            tags=["Co-op", "Multiplayer", "Simplified Chinese"],
            rating=4.2,
            metacritic=82,
            stores=["Steam"],
            raw_url="https://rawg.io/games/example",
        )

        score, reasons, warnings = score_game(game, preference)

        self.assertGreater(score, 0)
        self.assertTrue(reasons)
        self.assertTrue(any("预算" in warning for warning in warnings))


class FilteringTest(unittest.TestCase):
    def test_filter_excludes_platform_mismatch_and_disliked_tags(self) -> None:
        recommender = GameRecommender(rawg_client=None, max_results=5)  # type: ignore[arg-type]
        preference = GamePreference(platforms=["steam"], genres_dislike=["horror"])
        games = [
            GameCandidate(title="Switch Only", platforms=["Nintendo Switch"]),
            GameCandidate(title="Scary PC", platforms=["PC"], tags=["Horror"]),
            GameCandidate(title="Safe PC", platforms=["PC"], tags=["Co-op"]),
        ]

        filtered = recommender._filter_candidates(games, preference)

        self.assertEqual([game.title for game in filtered], ["Safe PC"])


if __name__ == "__main__":
    unittest.main()

