from __future__ import annotations

import unittest

from astrbot_plugin_game_recommender.services.explanation_builder import (
    validate_polished_points,
)
from astrbot_plugin_game_recommender.services.formatter import format_recommendation_messages
from astrbot_plugin_game_recommender.services.similarity_ranker import (
    build_profile_from_preference,
    rank_steam_candidates,
)
from astrbot_plugin_game_recommender.storage.models import GameCandidate, GamePreference


class TieredRecommendationTest(unittest.TestCase):
    def test_similarity_tiers_block_bad_high_review_games(self) -> None:
        preference = GamePreference(
            platforms=["steam"],
            genres_like=["co-op", "local co-op", "puzzle"],
            extra_tags=["轻松"],
            genres_dislike=["horror"],
            players=2,
            language="中文",
            result_count=3,
        )
        ranked = rank_steam_candidates(
            [
                steam_game("High Review Generic", ["Co-op", "Multiplayer"], reviews=90000),
                steam_game("Scary Match", ["Co-op", "Local Co-op", "Puzzle", "Horror"]),
                steam_game(
                    "Strong Local Puzzle",
                    ["Co-op", "Local Co-op", "Puzzle", "Casual", "Relaxing", "Simplified Chinese"],
                    reviews=700,
                ),
            ],
            build_profile_from_preference(preference),
        )

        self.assertEqual(
            [game.title for game in ranked],
            ["Strong Local Puzzle", "High Review Generic"],
        )
        self.assertEqual(ranked[0].tier, "strong")
        self.assertNotEqual(ranked[1].tier, "strong")
        self.assertTrue(ranked[0].fit_points)
        self.assertTrue(ranked[0].risk_points)


class TieredFormatterTest(unittest.TestCase):
    def test_recommendation_messages_include_tier_summary_and_game_tier(self) -> None:
        game = steam_game(
            "Unravel Two",
            ["Co-op", "Local Co-op", "Multiplayer", "Puzzle", "Casual"],
        )
        ranked = rank_steam_candidates(
            [game],
            build_profile_from_preference(
                GamePreference(platforms=["steam"], genres_like=["co-op", "puzzle"], players=2)
            ),
        )

        messages = format_recommendation_messages(GamePreference(result_count=1), ranked, limit=1)

        self.assertIn("强烈推荐", messages[0])
        self.assertIn("层级：强烈推荐", messages[1])


class ExplanationValidationTest(unittest.TestCase):
    def test_llm_polish_cannot_delete_required_risk_or_add_untrusted_fact(self) -> None:
        result = validate_polished_points(
            '{"fit_points":["支持中文和史低 10 元"],"risk_points":[]}',
            fallback_fit_points=["Steam 分类确认支持合作", "Steam 页面确认支持简体中文"],
            fallback_risk_points=["未确认支持中文"],
        )

        self.assertEqual(
            result.fit_points,
            ["Steam 分类确认支持合作", "Steam 页面确认支持简体中文"],
        )
        self.assertEqual(result.risk_points, ["未确认支持中文"])


def steam_game(title: str, tags: list[str], reviews: int = 500) -> GameCandidate:
    appid = abs(hash(title)) % 1000000
    return GameCandidate(
        title=title,
        appid=appid,
        platforms=["PC"],
        genres=["Adventure"],
        tags=tags,
        stores=["Steam"],
        raw_url=f"https://store.steampowered.com/app/{appid}/",
        review_total=reviews,
        review_positive_ratio=0.8,
        index_source="steam_index",
    )


if __name__ == "__main__":
    unittest.main()
