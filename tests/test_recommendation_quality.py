from __future__ import annotations

import unittest
from typing import Any

from astrbot_plugin_steam_game_recommender.services.similarity_ranker import (
    build_profile_from_preference,
    rank_steam_candidates,
)
from astrbot_plugin_steam_game_recommender.services.steam_index import SteamGameIndexService
from astrbot_plugin_steam_game_recommender.storage.models import GameCandidate, GamePreference


class RecommendationQualityTest(unittest.IsolatedAsyncioTestCase):
    async def test_reference_game_tags_expand_profile_without_seed_titles(self) -> None:
        cache = MemoryCache()
        steam = SearchAwareSteamClient(
            {
                "双人成行": [
                    steam_game(
                        "It Takes Two",
                        ["Co-op", "Local Co-op", "Puzzle", "Adventure", "Split Screen"],
                    )
                ],
                "It Takes Two": [
                    steam_game(
                        "It Takes Two",
                        ["Co-op", "Local Co-op", "Puzzle", "Adventure", "Split Screen"],
                    )
                ],
                "co op": [
                    steam_game(
                        "Focused Co-op Puzzle",
                        ["Co-op", "Local Co-op", "Puzzle", "Adventure", "Casual"],
                    ),
                    steam_game("Generic Multiplayer", ["Multiplayer"]),
                ],
                "local coop": [steam_game("Local Co-op Game", ["Co-op", "Local Co-op"])],
                "puzzle": [steam_game("Puzzle Game", ["Puzzle", "Singleplayer"])],
                "adventure": [steam_game("Adventure Game", ["Adventure", "Singleplayer"])],
            }
        )
        service = SteamGameIndexService(steam, cache, min_review_count=50)

        ranked = await service.recommend(
            GamePreference(
                platforms=["steam"],
                genres_like=["co-op"],
                extra_tags=["轻松"],
                reference_games_like=["双人成行"],
                reference_search_terms=["It Takes Two"],
                players=2,
                result_count=3,
            ),
            limit=3,
        )

        self.assertEqual(ranked[0].title, "Focused Co-op Puzzle")
        self.assertNotIn("It Takes Two", [game.title for game in ranked])
        self.assertTrue(any("puzzle" in item.text for item in ranked[0].recommendation_evidence))
        self.assertTrue(any(call["search"] == "双人成行" for call in steam.calls))

    async def test_reference_search_terms_and_tags_find_dark_souls_like_games(self) -> None:
        cache = MemoryCache()
        steam = SearchAwareSteamClient(
            {
                "黑暗之魂": [],
                "Dark Souls": [
                    steam_game("Dark Souls: Remastered", ["Soulslike", "Action", "RPG"])
                ],
                "soulslike": [
                    steam_game("Mortal Shell", ["Soulslike", "Action", "RPG"]),
                    steam_game("Generic Action", ["Action"]),
                    steam_game("Salt and Sanctuary", ["Soulslike", "Action", "RPG"]),
                ],
                "action": [steam_game("Action Only", ["Action"])],
                "rpg": [steam_game("RPG Only", ["RPG"])],
            }
        )
        service = SteamGameIndexService(steam, cache, min_review_count=50)

        ranked = await service.recommend(
            GamePreference(
                reference_games_like=["黑暗之魂"],
                reference_search_terms=["Dark Souls"],
                extra_tags=["soulslike", "action", "rpg"],
                result_count=3,
            ),
            limit=3,
        )

        titles = [game.title for game in ranked]
        self.assertEqual(titles[0], "Mortal Shell")
        self.assertNotIn("Dark Souls: Remastered", titles)
        self.assertTrue(any(call["search"] == "Dark Souls" for call in steam.calls))
        self.assertTrue(any(call["search"] == "soulslike" for call in steam.calls))
        self.assertFalse(any(call["search"] == "soulslike action rpg" for call in steam.calls))

    async def test_negative_reference_similarity_penalizes_without_excluding_similar_games(
        self,
    ) -> None:
        steam = SearchAwareSteamClient(
            {
                "Overcooked": [steam_game("Overcooked", ["Management", "Casual", "Co-op"])],
                "co op": [
                    steam_game("Kitchen Shift", ["Management", "Casual", "Co-op"]),
                    steam_game("Puzzle Team", ["Puzzle", "Adventure", "Co-op"]),
                ],
            }
        )
        service = SteamGameIndexService(steam, MemoryCache(), min_review_count=50)

        ranked = await service.recommend(
            GamePreference(
                genres_like=["co-op"],
                reference_games_dislike=["Overcooked"],
                result_count=2,
            ),
            limit=2,
        )

        self.assertEqual([game.title for game in ranked], ["Puzzle Team", "Kitchen Shift"])
        self.assertNotIn("Overcooked", [game.title for game in ranked])
        self.assertGreater(
            ranked[1].score_breakdown.negative_reference_penalty,
            ranked[0].score_breakdown.negative_reference_penalty,
        )

    async def test_cached_index_orders_tag_coverage_before_reviews(self) -> None:
        cache = MemoryCache(
            {
                "steam_index:v4": [
                    dump_model(
                        steam_game(
                            "High Review Generic Co-op",
                            ["Co-op", "Multiplayer"],
                            reviews=90000,
                            ratio=0.96,
                        )
                    ),
                    dump_model(
                        steam_game(
                            "Lower Review Better Match",
                            ["Co-op", "Local Co-op", "Puzzle", "Casual", "Relaxing"],
                            reviews=600,
                            ratio=0.78,
                        )
                    ),
                ]
            }
        )
        service = SteamGameIndexService(NoLiveSearchSteamClient(), cache)

        ranked = await service.recommend(
            GamePreference(
                platforms=["steam"],
                genres_like=["co-op", "local co-op", "puzzle"],
                extra_tags=["轻松"],
                players=2,
                result_count=2,
            ),
            limit=2,
        )

        self.assertEqual(
            [game.title for game in ranked],
            ["Lower Review Better Match", "High Review Generic Co-op"],
        )
        self.assertGreater(
            ranked[0].score_breakdown.tag_coverage,
            ranked[1].score_breakdown.tag_coverage,
        )

    async def test_cached_index_defaults_to_strict_primary_ranking(self) -> None:
        cache = MemoryCache(
            {
                "steam_index:v4": [
                    dump_model(
                        steam_game("Farm Co-op A", ["Co-op", "Puzzle", "Farming", "Crafting"])
                    ),
                    dump_model(
                        steam_game("Farm Co-op B", ["Co-op", "Puzzle", "Farming", "Crafting"])
                    ),
                    dump_model(
                        steam_game(
                            "Story Co-op",
                            ["Co-op", "Puzzle", "Story Rich", "Choices Matter"],
                        )
                    ),
                    dump_model(
                        steam_game("Lower Match Builder", ["Co-op", "Building", "Automation"])
                    ),
                ]
            }
        )
        service = SteamGameIndexService(NoLiveSearchSteamClient(), cache)

        ranked = await service.recommend(
            GamePreference(platforms=["steam"], genres_like=["co-op", "puzzle"]),
            limit=4,
        )

        self.assertEqual(
            {game.title for game in ranked[:3]},
            {"Farm Co-op A", "Farm Co-op B", "Story Co-op"},
        )
        self.assertEqual(ranked[-1].title, "Lower Match Builder")

    async def test_cached_index_returns_continuous_score_order(self) -> None:
        cache = MemoryCache(
            {
                "steam_index:v4": [
                    dump_model(
                        steam_game("Farm Co-op A", ["Co-op", "Puzzle", "Farming", "Crafting"])
                    ),
                    dump_model(
                        steam_game("Farm Co-op B", ["Co-op", "Puzzle", "Farming", "Crafting"])
                    ),
                    dump_model(
                        steam_game(
                            "Story Co-op",
                            ["Co-op", "Puzzle", "Story Rich", "Choices Matter"],
                        )
                    ),
                    dump_model(
                        steam_game("Lower Match Builder", ["Co-op", "Building", "Automation"])
                    ),
                ]
            }
        )
        service = SteamGameIndexService(NoLiveSearchSteamClient(), cache)

        ranked = await service.recommend(
            GamePreference(platforms=["steam"], genres_like=["co-op", "puzzle"]),
            limit=4,
        )

        self.assertEqual(
            {game.title for game in ranked[:3]},
            {"Farm Co-op A", "Farm Co-op B", "Story Co-op"},
        )
        self.assertEqual(ranked[-1].title, "Lower Match Builder")

    def test_exclude_tags_filter_horror_and_singleplayer_only(self) -> None:
        ranked = rank_steam_candidates(
            [
                steam_game("Safe Co-op Puzzle", ["Co-op", "Puzzle"]),
                steam_game("Scary Co-op Puzzle", ["Co-op", "Puzzle", "Horror"]),
                steam_game("Solo Puzzle", ["Singleplayer", "Puzzle"]),
            ],
            build_profile_from_preference(
                GamePreference(
                    platforms=["steam"],
                    required_tags=["co-op"],
                    genres_like=["co-op", "puzzle"],
                    genres_dislike=["horror"],
                    players=2,
                )
            ),
        )

        self.assertEqual([game.title for game in ranked], ["Safe Co-op Puzzle"])


def steam_game(
    title: str,
    tags: list[str],
    reviews: int = 500,
    ratio: float = 0.8,
) -> GameCandidate:
    appid = abs(hash(title)) % 1000000
    return GameCandidate(
        title=title,
        appid=appid,
        app_type="game",
        platforms=["PC"],
        genres=[],
        tags=tags,
        stores=["Steam"],
        raw_url=f"https://store.steampowered.com/app/{appid}/",
        review_total=reviews,
        review_positive_ratio=ratio,
        review_recent_ratio=ratio,
        internal_source_markers=["steam_index"],
    )


def dump_model(model: Any) -> dict[str, Any]:
    dumper = getattr(model, "model_dump", None)
    return dumper() if dumper else model.dict()


class MemoryCache:
    def __init__(self, payloads: dict[str, Any] | None = None) -> None:
        self.payloads = payloads or {}
        self.writes: dict[str, Any] = {}

    async def get_json(self, key: str, _ttl_hours: int) -> Any | None:
        payload = self.payloads.get(key)
        if key == "steam_index:v4" and isinstance(payload, list):
            return {
                "version": 4,
                "entries": [{"candidate": entry, "refreshed_at": 1.0} for entry in payload],
                "search_coverage": {},
            }
        return payload

    async def set_json(self, key: str, payload: Any) -> None:
        self.writes[key] = payload
        self.payloads[key] = payload


class SearchAwareSteamClient:
    def __init__(self, by_search: dict[str, list[GameCandidate]]) -> None:
        self.by_search = by_search
        self.calls: list[dict[str, Any]] = []

    async def search_games(self, **kwargs: Any) -> list[GameCandidate]:
        self.calls.append(kwargs)
        return self.by_search.get(kwargs.get("search"), [])

    async def get_review_summary(self, appid: int):
        del appid
        return None


class NoLiveSearchSteamClient:
    async def search_games(self, **_kwargs: Any) -> list[GameCandidate]:
        return []


if __name__ == "__main__":
    unittest.main()
