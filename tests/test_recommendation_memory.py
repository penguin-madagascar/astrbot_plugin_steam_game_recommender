from __future__ import annotations

import unittest
from typing import Any

from astrbot_plugin_game_recommender.services.recommendation_memory import (
    RecommendationMemory,
    append_shown_games,
    build_recommendation_memory,
    load_recommendation_memory,
    recommendation_memory_key,
    save_recommendation_memory,
)
from astrbot_plugin_game_recommender.services.steam_index import SteamGameIndexService
from astrbot_plugin_game_recommender.storage.models import GamePreference, RankedGame


class RecommendationMemoryTest(unittest.IsolatedAsyncioTestCase):
    async def test_save_and_load_recent_memory(self) -> None:
        cache = MemoryCache()
        memory = build_recommendation_memory(
            chat_platform="qq",
            chat_user_id="user-1",
            raw_query="Steam 合作解谜",
            preference=GamePreference(platforms=["steam"], genres_like=["co-op"]),
            diversity_mode="strict",
            result_limit=2,
            games=[
                ranked_game("Game A", 1),
                ranked_game("Game B", 2),
            ],
            now=1000,
        )

        await save_recommendation_memory(cache, memory)
        loaded = await load_recommendation_memory("qq", "user-1", cache, now=1010)

        self.assertIsNotNone(loaded)
        self.assertEqual(loaded.raw_query, "Steam 合作解谜")
        self.assertEqual(loaded.diversity_mode, "strict")
        self.assertEqual(loaded.shown_appids, [1, 2])
        self.assertEqual(loaded.preference.platforms, ["steam"])

    async def test_expired_memory_is_not_loaded(self) -> None:
        cache = MemoryCache()
        await save_recommendation_memory(
            cache,
            RecommendationMemory(
                chat_platform="qq",
                chat_user_id="user-1",
                raw_query="Steam 合作解谜",
                preference=GamePreference(platforms=["steam"]),
                diversity_mode="strict",
                result_limit=2,
                shown_appids=[1],
                shown_titles=["game a"],
                created_at=1000,
            ),
        )

        loaded = await load_recommendation_memory(
            "qq",
            "user-1",
            cache,
            ttl_minutes=30,
            now=1000 + 31 * 60,
        )

        self.assertIsNone(loaded)

    def test_append_shown_games_accumulates_appids_and_title_fallbacks(self) -> None:
        memory = RecommendationMemory(
            chat_platform="qq",
            chat_user_id="user-1",
            raw_query="Steam 合作解谜",
            preference=GamePreference(platforms=["steam"]),
            diversity_mode="balanced",
            result_limit=2,
            shown_appids=[1],
            shown_titles=["game a"],
            created_at=1000,
        )

        updated = append_shown_games(
            memory,
            [
                ranked_game("Game B", 2),
                RankedGame(title="No Appid Game", score=10),
            ],
        )

        self.assertEqual(updated.shown_appids, [1, 2])
        self.assertEqual(updated.shown_titles, ["game a", "game b", "no appid game"])


def ranked_game(title: str, appid: int) -> RankedGame:
    return RankedGame(title=title, appid=appid, score=10)


class MemoryCache:
    def __init__(self) -> None:
        self.payloads: dict[str, Any] = {}

    async def get_json(self, key: str, _ttl_hours: int) -> Any | None:
        return self.payloads.get(key)

    async def set_json(self, key: str, payload: Any) -> None:
        self.payloads[key] = payload


class MemoryKeyTest(unittest.TestCase):
    def test_key_uses_platform_and_user(self) -> None:
        self.assertEqual(
            recommendation_memory_key("qq", "user-1"),
            "recommendation_memory:qq:user-1",
        )


class ExcludedRecommendationTest(unittest.IsolatedAsyncioTestCase):
    async def test_steam_index_excludes_previously_shown_appids_and_titles(self) -> None:
        service = SteamGameIndexService(
            steam_client=NoLiveSearchSteamClient(),
            cache=IndexCache(
                [
                    RankedGame(title="Shown Appid", appid=1, tags=["Co-op", "Puzzle"], score=10),
                    RankedGame(title="Shown Title", tags=["Co-op", "Puzzle"], score=9),
                    RankedGame(title="Fresh Game", appid=3, tags=["Co-op", "Puzzle"], score=8),
                ]
            ),
            min_review_count=0,
        )

        ranked = await service.recommend(
            GamePreference(platforms=["steam"], genres_like=["co-op", "puzzle"]),
            limit=3,
            excluded_appids=[1],
            excluded_titles=["shown title"],
        )

        self.assertEqual([game.title for game in ranked], ["Fresh Game"])


class IndexCache:
    def __init__(self, entries: list[RankedGame]) -> None:
        self.entries = entries

    async def get_json(self, _key: str, _ttl_hours: int) -> Any:
        return {
            "version": 2,
            "entries": [
                {"candidate": entry.model_dump(), "refreshed_at": 1.0} for entry in self.entries
            ],
            "search_coverage": {},
        }

    async def set_json(self, _key: str, _payload: Any) -> None:
        return None


class NoLiveSearchSteamClient:
    async def search_games(self, **_kwargs: Any):
        return []


if __name__ == "__main__":
    unittest.main()
