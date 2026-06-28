from __future__ import annotations

import unittest
from typing import Any

from astrbot_plugin_game_recommender.services.recommender import (
    GameRecommender,
    dedupe_candidates,
)
from astrbot_plugin_game_recommender.services.game_facts import build_game_facts
from astrbot_plugin_game_recommender.services.tiered_ranker import build_ranked_game
from astrbot_plugin_game_recommender.storage.models import GameCandidate, GamePreference


class RecommendationQualityTest(unittest.IsolatedAsyncioTestCase):
    async def test_reference_profile_seed_recall_prioritizes_similar_coop_games(self) -> None:
        source = SearchAwareFakeGameSource(
            generic_games=[
                GameCandidate(
                    title="The Witcher 3 Wild Hunt - Complete Edition",
                    platforms=["PC", "Nintendo Switch"],
                    genres=["RPG"],
                    tags=["Fantasy", "Singleplayer"],
                    rating=4.8,
                    stores=["Steam", "Nintendo Store"],
                ),
                GameCandidate(
                    title="Baldur's Gate III",
                    platforms=["PC", "PlayStation 5"],
                    genres=["RPG"],
                    tags=["Choices Matter", "Turn-Based"],
                    rating=4.4,
                    stores=["Steam"],
                ),
                GameCandidate(
                    title="Batman: Arkham City - Game of the Year Edition",
                    platforms=["PC"],
                    genres=["Action"],
                    tags=["Singleplayer"],
                    rating=4.4,
                    stores=["Steam"],
                ),
            ],
            search_games={
                "Split Fiction": [
                    co_op_game(
                        "Split Fiction",
                        rating=4.7,
                        platforms=["PC", "Nintendo Switch 2"],
                        tags=[
                            "Co-op",
                            "Online Co-Op",
                            "Local Co-op",
                            "Split Screen",
                            "Puzzle",
                            "Adventure",
                            "Simplified Chinese",
                        ],
                    )
                ],
                "Unravel Two": [co_op_game("Unravel Two", rating=4.2)],
                "Overcooked! All You Can Eat": [
                    co_op_game("Overcooked! All You Can Eat", rating=4.1, tags=["Co-op", "Party"])
                ],
            },
        )
        preference = GamePreference(
            platforms=["nintendo switch", "steam"],
            genres_dislike=["horror"],
            reference_games_like=["双人成行"],
            players=2,
            language="中文",
            difficulty="easy",
            result_count=5,
        )

        ranked = await GameRecommender(source, max_results=5).recommend(
            preference,
            candidate_pool_size=10,
        )

        titles = [game.title for game in ranked[:5]]
        self.assertIn("Split Fiction", titles[:3])
        self.assertIn("Unravel Two", titles)
        self.assertNotIn("The Witcher 3 Wild Hunt - Complete Edition", titles)
        self.assertNotIn("Baldur's Gate III", titles)
        self.assertNotIn("Batman: Arkham City - Game of the Year Edition", titles)
        split_fiction = next(game for game in ranked if game.title == "Split Fiction")
        self.assertTrue(any("Switch 2" in warning for warning in split_fiction.warnings))
        self.assertTrue(any(call.get("search") == "Split Fiction" for call in source.calls))

    async def test_it_takes_two_like_request_filters_bad_matches(self) -> None:
        source = FakeGameSource(
            [
                GameCandidate(
                    title="The Witcher 3: Wild Hunt - Blood and Wine",
                    platforms=["PC", "PlayStation 4"],
                    genres=["RPG"],
                    tags=["Horror", "Blood"],
                    rating=4.8,
                    stores=["Steam"],
                ),
                GameCandidate(
                    title="Persona 5 Royal",
                    platforms=["PC", "Nintendo Switch"],
                    genres=["RPG"],
                    tags=["Singleplayer", "JRPG"],
                    rating=4.8,
                    stores=["Steam", "Nintendo Store"],
                ),
                GameCandidate(
                    title="Warhammer 40,000: Dawn of War - Definitive Edition",
                    platforms=["PC"],
                    genres=["Strategy"],
                    tags=["Singleplayer", "Multiplayer", "Co-op"],
                    rating=4.8,
                    stores=["Steam"],
                ),
                GameCandidate(
                    title="It Takes Two",
                    platforms=["PC", "Nintendo Switch"],
                    genres=["Adventure", "Platformer"],
                    tags=["Co-op", "Local Co-op", "Split Screen", "Simplified Chinese"],
                    rating=4.5,
                    stores=["Steam", "Nintendo Store"],
                ),
                co_op_game("Unravel Two", rating=4.2),
                co_op_game("Overcooked! All You Can Eat", rating=4.1, tags=["Co-op", "Party"]),
                co_op_game("PHOGS!", rating=3.9, tags=["Co-op", "Puzzle", "Casual"]),
            ]
        )
        preference = GamePreference(
            platforms=["nintendo switch", "steam"],
            genres_like=[
                "co-op",
                "local co-op",
                "puzzle",
                "adventure",
                "casual",
                "platformer",
            ],
            genres_dislike=["horror"],
            reference_games_like=["it takes two"],
            players=2,
            language="中文",
            difficulty="easy",
            result_count=3,
        )

        ranked = await GameRecommender(source, max_results=3).recommend(
            preference,
            candidate_pool_size=8,
        )

        titles = [game.title for game in ranked[:3]]
        self.assertEqual(titles, ["Unravel Two", "Overcooked! All You Can Eat", "PHOGS!"])
        self.assertTrue(all("It Takes Two" not in title for title in titles))
        self.assertTrue(all("Witcher" not in title for title in titles))
        self.assertNotIn("Persona 5 Royal", titles)
        self.assertNotIn("Warhammer 40,000: Dawn of War - Definitive Edition", titles)

    async def test_stardew_like_request_prioritizes_farming_coop_games(self) -> None:
        source = SearchAwareFakeGameSource(
            generic_games=[
                GameCandidate(
                    title="Chess",
                    platforms=["PC"],
                    genres=["Puzzle", "Casual", "Simulation"],
                    tags=["Split Screen", "Remote Play Together", "Simplified Chinese"],
                    rating=4.4,
                    stores=["Steam"],
                ),
                GameCandidate(
                    title="Monster Prom 3: Monster Roadtrip",
                    platforms=["PC"],
                    genres=["Casual", "Strategy", "Simulation"],
                    tags=["Co-op", "Online Co-op"],
                    rating=4.4,
                    stores=["Steam"],
                ),
                GameCandidate(
                    title="Stardew Valley",
                    platforms=["PC"],
                    genres=["Simulation", "RPG"],
                    tags=["Co-op", "Farming", "Crafting", "Simplified Chinese"],
                    rating=4.4,
                    stores=["Steam"],
                ),
                GameCandidate(
                    title="Minecraft",
                    platforms=["PC"],
                    genres=["Simulation"],
                    tags=["Multiplayer", "Crafting", "Building", "Simplified Chinese"],
                    rating=4.4,
                    stores=[],
                ),
                GameCandidate(
                    title="Factorio",
                    platforms=["PC"],
                    genres=["Strategy", "Simulation"],
                    tags=["Co-op", "Automation", "Management", "Simplified Chinese"],
                    rating=4.4,
                    stores=["Steam"],
                ),
                GameCandidate(
                    title="Baldur's Gate III",
                    platforms=["PC"],
                    genres=["RPG", "Strategy"],
                    tags=[
                        "Co-op",
                        "Online Co-op",
                        "Local Co-op",
                        "Split Screen",
                        "Story Rich",
                        "Simplified Chinese",
                    ],
                    rating=5.0,
                    metacritic=96,
                    stores=["Steam"],
                ),
                GameCandidate(
                    title="Divinity: Original Sin 2",
                    platforms=["PC"],
                    genres=["RPG", "Strategy"],
                    tags=[
                        "Co-op",
                        "Online Co-op",
                        "Local Co-op",
                        "Split Screen",
                        "Turn-Based",
                        "Simplified Chinese",
                    ],
                    rating=5.0,
                    metacritic=95,
                    stores=["Steam"],
                ),
            ],
            search_games={
                "Sun Haven": [
                    farming_coop_game(
                        "Sun Haven",
                        tags=["Co-op", "Online Co-op", "Farming", "Crafting", "Fantasy", "Simplified Chinese"],
                        rating=4.1,
                    )
                ],
                "Roots of Pacha": [
                    farming_coop_game(
                        "Roots of Pacha",
                        tags=["Co-op", "Online Co-op", "Farming", "Crafting", "Relaxing", "Simplified Chinese"],
                        rating=4.2,
                    )
                ],
                "Farm Together 2": [
                    farming_coop_game(
                        "Farm Together 2",
                        tags=["Co-op", "Online Co-op", "Farming", "Management", "Casual", "Simplified Chinese"],
                        rating=4.3,
                    )
                ],
                "Dinkum": [
                    farming_coop_game(
                        "Dinkum",
                        tags=["Co-op", "Online Co-op", "Farming", "Crafting", "Building", "Simplified Chinese"],
                        rating=4.0,
                    )
                ],
                "Fae Farm": [
                    farming_coop_game(
                        "Fae Farm",
                        tags=["Co-op", "Online Co-op", "Farming", "Crafting", "Relaxing", "Simplified Chinese"],
                        rating=3.9,
                    )
                ],
            },
        )
        preference = GamePreference(
            platforms=["steam"],
            genres_like=[
                "simulation",
                "casual",
                "rpg",
                "co-op",
                "local co-op",
                "multiplayer",
                "farming",
                "management",
                "crafting",
                "building",
                "relaxing",
            ],
            genres_dislike=["horror"],
            reference_games_like=["星露谷物语"],
            players=2,
            language="中文",
            difficulty="easy",
            result_count=5,
        )

        ranked = await GameRecommender(source, max_results=5).recommend(
            preference,
            candidate_pool_size=10,
        )

        titles = [game.title for game in ranked[:5]]
        self.assertEqual(
            titles,
            ["Farm Together 2", "Roots of Pacha", "Sun Haven", "Dinkum", "Fae Farm"],
        )
        self.assertNotIn("Stardew Valley", titles)
        self.assertNotIn("Chess", titles)
        self.assertNotIn("Monster Prom 3: Monster Roadtrip", titles)
        self.assertNotIn("Minecraft", titles)
        self.assertNotIn("Baldur's Gate III", titles)
        self.assertNotIn("Divinity: Original Sin 2", titles)

    async def test_stardew_profile_does_not_treat_generic_coop_games_as_strong_match(self) -> None:
        preference = GamePreference(
            platforms=["steam"],
            genres_like=[
                "simulation",
                "casual",
                "rpg",
                "co-op",
                "local co-op",
                "multiplayer",
                "farming",
                "management",
                "crafting",
                "building",
                "relaxing",
            ],
            reference_games_like=["Stardew Valley"],
            players=2,
            language="中文",
            difficulty="easy",
        )
        generic_games = [
            GameCandidate(
                title="Baldur's Gate III",
                platforms=["PC"],
                genres=["RPG", "Strategy"],
                tags=[
                    "Co-op",
                    "Online Co-op",
                    "Local Co-op",
                    "Multiplayer",
                    "Split Screen",
                    "Story Rich",
                    "Simplified Chinese",
                ],
                rating=5.0,
                metacritic=96,
                stores=["Steam"],
            ),
            GameCandidate(
                title="Factorio",
                platforms=["PC"],
                genres=["Strategy", "Simulation"],
                tags=[
                    "Co-op",
                    "Online Co-op",
                    "Multiplayer",
                    "Crafting",
                    "Building",
                    "Management",
                    "Simplified Chinese",
                ],
                rating=5.0,
                metacritic=90,
                stores=["Steam"],
            ),
        ]

        for candidate in generic_games:
            with self.subTest(candidate=candidate.title):
                facts = build_game_facts(candidate, preference)
                ranked = build_ranked_game(candidate, preference, facts)

                self.assertIsNotNone(ranked)
                assert ranked is not None
                self.assertLess(ranked.facts.reference_similarity, 0.75)
                self.assertNotEqual(ranked.tier, "strong")

    async def test_explicit_preferences_do_not_fall_back_to_empty_rawg_query(self) -> None:
        source = FakeGameSource([co_op_game("Unravel Two")])
        preference = GamePreference(
            players=2,
            language="中文",
            budget=100,
            result_count=1,
        )

        await GameRecommender(source, max_results=1).recommend(preference)

        self.assertTrue(source.calls)
        self.assertTrue(
            all(
                call.get("search")
                or call.get("platforms")
                or call.get("genres")
                or call.get("tags")
                for call in source.calls
            )
        )


class CandidateDedupeTest(unittest.TestCase):
    def test_dedupe_prefers_complete_game_over_witcher_dlc_entries(self) -> None:
        deduped = dedupe_candidates(
            [
                GameCandidate(
                    title="The Witcher 3: Wild Hunt - Blood and Wine",
                    platforms=["PC"],
                    tags=["DLC"],
                    stores=["Steam"],
                ),
                GameCandidate(
                    title="The Witcher 3 Wild Hunt - Complete Edition",
                    platforms=["PC", "Nintendo Switch"],
                    stores=["Steam", "Nintendo Store"],
                ),
                GameCandidate(
                    title="The Witcher 3: Wild Hunt - Hearts of Stone",
                    platforms=["PC"],
                    tags=["Expansion"],
                    stores=["Steam"],
                ),
            ]
        )

        self.assertEqual(
            [game.title for game in deduped],
            ["The Witcher 3 Wild Hunt - Complete Edition"],
        )


def co_op_game(
    title: str,
    rating: float = 4.0,
    tags: list[str] | None = None,
    platforms: list[str] | None = None,
) -> GameCandidate:
    return GameCandidate(
        title=title,
        platforms=platforms or ["PC", "Nintendo Switch"],
        genres=["Adventure", "Puzzle"],
        tags=tags or ["Co-op", "Local Co-op", "Puzzle", "Casual", "Simplified Chinese"],
        rating=rating,
        stores=["Steam", "Nintendo Store"],
    )


def farming_coop_game(
    title: str,
    tags: list[str],
    rating: float = 4.1,
) -> GameCandidate:
    return GameCandidate(
        title=title,
        platforms=["PC"],
        genres=["Simulation", "Casual", "RPG"],
        tags=tags,
        rating=rating,
        stores=["Steam"],
    )


class FakeGameSource:
    def __init__(self, games: list[GameCandidate]) -> None:
        self.games = games
        self.calls: list[dict[str, Any]] = []

    async def search_games(self, **kwargs: Any) -> list[GameCandidate]:
        self.calls.append(kwargs)
        return self.games


class SearchAwareFakeGameSource:
    def __init__(
        self,
        generic_games: list[GameCandidate],
        search_games: dict[str, list[GameCandidate]],
    ) -> None:
        self.generic_games = generic_games
        self.search_games_by_query = search_games
        self.calls: list[dict[str, Any]] = []

    async def search_games(self, **kwargs: Any) -> list[GameCandidate]:
        self.calls.append(kwargs)
        query = kwargs.get("search")
        if query in self.search_games_by_query:
            return self.search_games_by_query[query]
        return self.generic_games


if __name__ == "__main__":
    unittest.main()
