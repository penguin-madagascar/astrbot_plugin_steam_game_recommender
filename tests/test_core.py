from __future__ import annotations

import unittest

try:
    from astrbot_plugin_steam_game_recommender.services.preference_parser import keyword_fallback
    from astrbot_plugin_steam_game_recommender.services.similarity_ranker import (
        SteamTagProfile,
        rank_steam_candidates,
    )
    from astrbot_plugin_steam_game_recommender.storage.models import GameCandidate
except ModuleNotFoundError as exc:
    if exc.name in {"pydantic", "astrbot"}:
        raise unittest.SkipTest(f"{exc.name} is not installed in this environment") from exc
    raise


class PreferenceFallbackTest(unittest.TestCase):
    def test_keyword_fallback_extracts_common_preferences(self) -> None:
        preference = keyword_fallback(
            "推荐几个适合 Steam 的双人游戏，不要恐怖，最好支持中文，预算 100 以内"
        )

        self.assertEqual(preference.platforms, ["steam"])
        self.assertEqual(preference.players, 2)
        self.assertEqual(preference.budget, 100)
        self.assertIn("horror", preference.genres_dislike)
        self.assertEqual(preference.preferred_languages, ["schinese"])


class SimilarityRankerCoreTest(unittest.TestCase):
    def test_ranker_filters_disliked_tags_and_orders_by_similarity(self) -> None:
        profile = SteamTagProfile(
            include_tags=["co_op", "local_coop", "puzzle", "relaxing"],
            exclude_tags=["horror"],
            reference_titles=[],
        )
        ranked = rank_steam_candidates(
            [
                steam_game("Scary Puzzle", ["co-op", "puzzle", "horror"]),
                steam_game("Generic Co-op", ["co-op", "multiplayer"], reviews=50000),
                steam_game("Focused Match", ["co-op", "local co-op", "puzzle", "relaxing"]),
            ],
            profile,
        )

        self.assertEqual(
            [game.title for game in ranked],
            ["Focused Match", "Generic Co-op"],
        )
        self.assertGreater(
            ranked[0].score_breakdown.tag_coverage,
            ranked[1].score_breakdown.tag_coverage,
        )


def steam_game(title: str, tags: list[str], reviews: int = 500) -> GameCandidate:
    return GameCandidate(
        title=title,
        appid=abs(hash(title)) % 1000000,
        platforms=["PC"],
        tags=tags,
        stores=["Steam"],
        raw_url=f"https://store.steampowered.com/app/{abs(hash(title)) % 1000000}/",
        review_total=reviews,
        review_positive_ratio=0.8,
    )


if __name__ == "__main__":
    unittest.main()
