from __future__ import annotations

import unittest
from typing import Any

from astrbot_plugin_steam_game_recommender.services.recommendation_memory import (
    PreferencePatch,
    RecommendationMemory,
    append_feedback,
    append_shown_games,
    build_recommendation_memory,
    load_recommendation_memory,
    recommendation_memory_key,
    save_recommendation_memory,
)
from astrbot_plugin_steam_game_recommender.services.steam_index import SteamGameIndexService
from astrbot_plugin_steam_game_recommender.storage.models import GamePreference, RankedGame


class RecommendationMemoryTest(unittest.IsolatedAsyncioTestCase):
    async def test_save_and_load_recent_memory(self) -> None:
        cache = MemoryCache()
        memory = build_recommendation_memory(
            chat_platform="qq",
            chat_user_id="user-1",
            raw_query="Steam 合作解谜",
            preference=GamePreference(
                platforms=["steam"],
                genres_like=["co-op"],
                budget=100,
                budget_is_required=True,
            ),
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
        self.assertEqual(loaded.shown_appids, [1, 2])
        self.assertEqual(loaded.preference.platforms, ["steam"])
        self.assertTrue(loaded.preference.budget_is_required)
        self.assertEqual([item.title for item in loaded.last_results], ["Game A", "Game B"])

    async def test_legacy_memory_defaults_budget_requirement_to_false(self) -> None:
        cache = MemoryCache()
        memory = build_recommendation_memory(
            chat_platform="qq",
            chat_user_id="user-1",
            raw_query="预算 100 元",
            preference=GamePreference(budget=100),
            result_limit=1,
            games=[ranked_game("Game A", 1)],
            now=1000,
        )
        await save_recommendation_memory(cache, memory)
        cache.payloads[recommendation_memory_key("qq", "user-1")]["preference"].pop(
            "budget_is_required"
        )

        loaded = await load_recommendation_memory("qq", "user-1", cache, now=1010)

        self.assertIsNotNone(loaded)
        self.assertFalse(loaded.preference.budget_is_required)

    async def test_expired_memory_is_not_loaded(self) -> None:
        cache = MemoryCache()
        await save_recommendation_memory(
            cache,
            RecommendationMemory(
                chat_platform="qq",
                chat_user_id="user-1",
                raw_query="Steam 合作解谜",
                preference=GamePreference(platforms=["steam"]),
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

    async def test_feedback_is_capped_and_individually_expires_after_thirty_minutes(self) -> None:
        cache = MemoryCache()
        memory = build_recommendation_memory(
            chat_platform="qq",
            chat_user_id="user-1",
            raw_query="query",
            preference=GamePreference(),
            result_limit=1,
            games=[ranked_game("Game A", 1)],
            now=2_000,
        )
        for index in range(12):
            memory = append_feedback(
                memory,
                PreferencePatch(add_tags=[f"tag-{index}"]),
                now=1_000 + index * 100,
            )
        await save_recommendation_memory(cache, memory)

        loaded = await load_recommendation_memory("qq", "user-1", cache, now=3_100)

        self.assertIsNotNone(loaded)
        assert loaded is not None
        self.assertLess(len(loaded.feedback), 10)
        self.assertTrue(all(item.created_at >= 1_300 for item in loaded.feedback))


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
                    RankedGame(
                        title="Shown Appid",
                        appid=1,
                        app_type="game",
                        tags=["Co-op", "Puzzle"],
                        score=10,
                    ),
                    RankedGame(
                        title="Shown Title",
                        app_type="game",
                        tags=["Co-op", "Puzzle"],
                        score=9,
                    ),
                    RankedGame(
                        title="Fresh Game",
                        appid=3,
                        app_type="game",
                        tags=["Co-op", "Puzzle"],
                        score=8,
                    ),
                ]
            ),
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
            "version": 4,
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
