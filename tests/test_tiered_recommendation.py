from __future__ import annotations

import unittest
from typing import Any

from astrbot_plugin_game_recommender.services.explanation_builder import (
    validate_polished_points,
)
from astrbot_plugin_game_recommender.services.formatter import format_recommendation_messages
from astrbot_plugin_game_recommender.services.recommender import GameRecommender
from astrbot_plugin_game_recommender.storage.models import GameCandidate, GamePreference


USER_TEXT_PREFERENCE = GamePreference(
    platforms=["nintendo switch", "steam"],
    genres_dislike=["horror"],
    reference_games_like=["双人成行"],
    players=2,
    language="中文",
    difficulty="easy",
    budget=100,
    result_count=5,
)


class TieredRecommendationTest(unittest.IsolatedAsyncioTestCase):
    async def test_current_user_request_uses_tiers_and_blocks_bad_high_score_games(self) -> None:
        rawg = TieredFakeSource(
            generic=[
                rawg_game(
                    "The Witcher 3 Wild Hunt - Complete Edition",
                    platforms=["PC", "Nintendo Switch"],
                    genres=["RPG", "Adventure"],
                    tags=["Fantasy", "War"],
                    rating=4.8,
                ),
                rawg_game(
                    "Batman: Arkham Asylum Game of the Year Edition",
                    platforms=["PC"],
                    genres=["Action"],
                    tags=["Singleplayer", "Horror"],
                    rating=4.4,
                ),
                rawg_game(
                    "Persona 5 Royal",
                    platforms=["PC", "Nintendo Switch"],
                    genres=["RPG"],
                    tags=["Singleplayer", "JRPG"],
                    rating=4.8,
                ),
                rawg_game(
                    "Baldur's Gate III",
                    platforms=["PC", "PlayStation 5"],
                    genres=["RPG"],
                    tags=["Multiplayer", "Co-op", "Turn-Based"],
                    rating=4.4,
                ),
                rawg_game(
                    "Super Smash Bros. Ultimate",
                    platforms=["Nintendo Switch"],
                    genres=["Fighting"],
                    tags=["Multiplayer", "Fighter", "Arena"],
                    rating=4.37,
                ),
            ],
            by_search={
                "Split Fiction": [
                    rawg_game(
                        "Split Fiction",
                        platforms=["PC", "Nintendo Switch 2"],
                        genres=["Adventure", "Puzzle"],
                        tags=["Co-op", "Online Co-Op", "Local Co-op", "Split Screen"],
                        rating=4.7,
                    )
                ],
                "Unravel Two": [seed_game("Unravel Two")],
                "Overcooked! All You Can Eat": [seed_game("Overcooked! All You Can Eat")],
                "Moving Out 2": [seed_game("Moving Out 2")],
                "KeyWe": [seed_game("KeyWe")],
            },
        )
        steam = TieredFakeSource(
            by_search={
                "Split Fiction": [
                    steam_game(
                        "Split Fiction",
                        categories=[
                            "Co-op",
                            "Online Co-op",
                            "Shared/Split Screen Co-op",
                            "Remote Play Together",
                            "Simplified Chinese",
                        ],
                    )
                ],
                "Unravel Two": [
                    steam_game(
                        "Unravel Two",
                        categories=[
                            "Co-op",
                            "Shared/Split Screen Co-op",
                            "Remote Play Together",
                            "Simplified Chinese",
                        ],
                    )
                ],
                "Overcooked! All You Can Eat": [
                    steam_game(
                        "Overcooked! All You Can Eat",
                        categories=["Co-op", "Online Co-op", "Shared/Split Screen Co-op"],
                    )
                ],
                "Moving Out 2": [
                    steam_game(
                        "Moving Out 2",
                        categories=["Co-op", "Online Co-op", "Shared/Split Screen Co-op"],
                    )
                ],
                "KeyWe": [
                    steam_game(
                        "KeyWe",
                        categories=["Co-op", "Online Co-op", "Remote Play Together"],
                    )
                ],
                "Baldur's Gate III": [
                    steam_game("Baldur's Gate III", categories=["Co-op", "Online Co-op"])
                ],
                "Super Smash Bros. Ultimate": [],
            }
        )

        ranked = await GameRecommender(rawg, max_results=5, steam_source=steam).recommend(
            USER_TEXT_PREFERENCE,
            candidate_pool_size=12,
        )

        top = ranked[:5]
        titles = [game.title for game in top]
        self.assertEqual(len(top), 5)
        self.assertIn("Split Fiction", titles)
        self.assertIn("Unravel Two", titles)
        for blocked in (
            "The Witcher 3 Wild Hunt - Complete Edition",
            "Batman: Arkham Asylum Game of the Year Edition",
            "Persona 5 Royal",
        ):
            self.assertNotIn(blocked, titles)

        for game in top:
            self.assertIn(game.tier, {"strong", "recommended", "backup"})
            self.assertTrue(game.fit_points)
            self.assertTrue(game.risk_points)
            self.assertGreater(game.facts.match_score, 0)

        weak_by_title = {game.title: game for game in ranked if game.title in {"Baldur's Gate III", "Super Smash Bros. Ultimate"}}
        for weak in weak_by_title.values():
            self.assertNotEqual(weak.tier, "strong")
            self.assertTrue(any("弱" in point or "未确认" in point for point in weak.risk_points))

    async def test_steam_categories_fill_multiplayer_language_facts_even_when_rawg_is_sparse(self) -> None:
        rawg = TieredFakeSource(
            by_search={
                "Split Fiction": [
                    rawg_game(
                        "Split Fiction",
                        platforms=["PC"],
                        genres=["Adventure"],
                        tags=[],
                        rating=4.7,
                    )
                ],
            }
        )
        steam = TieredFakeSource(
            by_search={
                "Split Fiction": [
                    steam_game(
                        "Split Fiction",
                        categories=[
                            "Co-op",
                            "Online Co-op",
                            "Shared/Split Screen Co-op",
                            "Remote Play Together",
                            "Simplified Chinese",
                        ],
                    )
                ],
            }
        )

        ranked = await GameRecommender(rawg, max_results=1, steam_source=steam).recommend(
            USER_TEXT_PREFERENCE,
            candidate_pool_size=3,
        )

        game = ranked[0]
        self.assertTrue(any("Steam" in point and "合作" in point for point in game.fit_points))
        self.assertTrue(any("中文" in point for point in game.fit_points))
        self.assertGreater(game.facts.confidence, 0)

    async def test_explicit_preferences_never_trigger_popular_or_empty_full_library_queries(self) -> None:
        source = TieredFakeSource()

        await GameRecommender(source, max_results=5).recommend(
            GamePreference(players=2, budget=100, result_count=5)
        )

        self.assertTrue(source.calls)
        for call in source.calls:
            search = call.get("search")
            self.assertNotEqual(search, "popular games")
            self.assertTrue(
                search or call.get("platforms") or call.get("genres") or call.get("tags")
            )


class TieredFormatterTest(unittest.TestCase):
    def test_recommendation_messages_include_tier_summary_and_game_tier(self) -> None:
        game = seed_game("Unravel Two")
        game.tier = "strong"
        game.fit_points = ["双人合作核心玩法"]
        game.risk_points = ["Steam 价格未获取到"]

        messages = format_recommendation_messages(GamePreference(result_count=1), [game], limit=1)

        self.assertIn("强烈推荐", messages[0])
        self.assertIn("层级：强烈推荐", messages[1])


class ExplanationValidationTest(unittest.TestCase):
    def test_llm_polish_cannot_delete_required_risk_or_add_untrusted_fact(self) -> None:
        result = validate_polished_points(
            '{"fit_points":["支持中文和史低 10 元"],"risk_points":[]}',
            fallback_fit_points=["Steam 分类确认支持合作", "Steam 页面确认支持简体中文"],
            fallback_risk_points=["未确认支持原版 Switch"],
        )

        self.assertEqual(
            result.fit_points,
            ["Steam 分类确认支持合作", "Steam 页面确认支持简体中文"],
        )
        self.assertEqual(result.risk_points, ["未确认支持原版 Switch"])


def rawg_game(
    title: str,
    platforms: list[str],
    genres: list[str],
    tags: list[str],
    rating: float = 4.0,
) -> GameCandidate:
    return GameCandidate(
        title=title,
        platforms=platforms,
        genres=genres,
        tags=tags,
        rating=rating,
        stores=["Steam"] if "PC" in platforms else ["Nintendo Store"],
        raw_url=f"https://rawg.io/games/{title.lower().replace(' ', '-')}",
    )


def seed_game(title: str) -> GameCandidate:
    return rawg_game(
        title,
        platforms=["PC", "Nintendo Switch"],
        genres=["Adventure", "Puzzle"],
        tags=["Co-op", "Local Co-op", "Puzzle", "Casual", "Simplified Chinese"],
        rating=4.2,
    )


def steam_game(title: str, categories: list[str]) -> GameCandidate:
    return GameCandidate(
        title=title,
        platforms=["PC"],
        genres=["Adventure"],
        tags=categories,
        stores=["Steam"],
        raw_url=f"https://store.steampowered.com/app/{abs(hash(title)) % 100000}/",
    )


class TieredFakeSource:
    def __init__(
        self,
        generic: list[GameCandidate] | None = None,
        by_search: dict[str, list[GameCandidate]] | None = None,
    ) -> None:
        self.generic = generic or []
        self.by_search = by_search or {}
        self.calls: list[dict[str, Any]] = []

    async def search_games(self, **kwargs: Any) -> list[GameCandidate]:
        self.calls.append(kwargs)
        query = kwargs.get("search")
        if query in self.by_search:
            return self.by_search[query]
        return self.generic


if __name__ == "__main__":
    unittest.main()
