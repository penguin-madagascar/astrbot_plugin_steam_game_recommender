from __future__ import annotations

import unittest
from typing import Any

from astrbot_plugin_steam_game_recommender.services.recommendation_memory import (
    PreferencePatch,
    RecommendationMemory,
    append_feedback,
    append_shown_games,
    build_recommendation_memory,
    dump_memory,
    load_recommendation_memory,
    recommendation_memory_key,
    recommendation_owner_scope,
    save_recommendation_memory,
)
from astrbot_plugin_steam_game_recommender.services.steam_index import (
    STEAM_INDEX_SCHEMA_VERSION,
    SteamGameIndexService,
)
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

    async def test_session_and_sender_isolate_retry_memory(self) -> None:
        cache = MemoryCache()
        group_memory = build_recommendation_memory(
            chat_platform="onebot-instance",
            chat_user_id="user-1",
            conversation_scope="onebot-instance:GroupMessage:group-1",
            raw_query="群聊需求",
            preference=GamePreference(),
            result_limit=1,
            games=[ranked_game("Game A", 1)],
            now=1000,
        )
        await save_recommendation_memory(cache, group_memory)

        same_user_other_group = await load_recommendation_memory(
            "onebot-instance:GroupMessage:group-2",
            "user-1",
            cache,
            now=1010,
        )
        other_user_same_group = await load_recommendation_memory(
            "onebot-instance:GroupMessage:group-1",
            "user-2",
            cache,
            now=1010,
        )

        self.assertIsNone(same_user_other_group)
        self.assertIsNone(other_user_same_group)

    async def test_does_not_fall_back_to_legacy_unscoped_memory(self) -> None:
        cache = MemoryCache()
        memory = build_recommendation_memory(
            chat_platform="qq",
            chat_user_id="user-1",
            raw_query="private query",
            preference=GamePreference(),
            result_limit=1,
            games=[ranked_game("Game A", 1)],
            now=1000,
        )
        legacy_key = "recommendation_memory:qq:user-1"
        cache.payloads[legacy_key] = dump_memory(memory)

        loaded = await load_recommendation_memory(
            "onebot-instance:GroupMessage:group-1",
            "user-1",
            cache,
            now=1010,
        )

        self.assertIsNone(loaded)
        self.assertNotIn(legacy_key, cache.read_keys)

    async def test_cache_entry_has_explicit_ttl_and_personal_owner_scope(self) -> None:
        cache = BoundedCache()
        memory = build_recommendation_memory(
            chat_platform="onebot-instance",
            chat_user_id="user-1",
            conversation_scope="onebot-instance:FriendMessage:user-1",
            raw_query="query",
            preference=GamePreference(),
            result_limit=1,
            games=[ranked_game("Game A", 1)],
            now=1000,
        )

        await save_recommendation_memory(cache, memory)
        loaded = await load_recommendation_memory(
            memory.conversation_scope,
            memory.chat_user_id,
            cache,
            now=1010,
        )

        self.assertIsNotNone(loaded)
        self.assertEqual(cache.ttl_seconds, 30 * 60)
        self.assertEqual(
            cache.owner_scope,
            recommendation_owner_scope("onebot-instance", "user-1"),
        )
        self.assertEqual(cache.allow_stale_seconds, 0)

    async def test_can_delete_all_retry_memory_owned_by_the_bound_account(self) -> None:
        from astrbot_plugin_steam_game_recommender.services.recommendation_memory import (
            delete_recommendation_memories,
        )

        cache = BoundedCache()

        await delete_recommendation_memories(
            cache,
            "onebot-instance",
            "user-1",
        )

        self.assertEqual(
            cache.deleted_owner_scopes,
            [recommendation_owner_scope("onebot-instance", "user-1")],
        )

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

    async def test_malformed_memory_is_ignored_instead_of_breaking_retry(self) -> None:
        cache = MemoryCache()
        memory = build_recommendation_memory(
            chat_platform="qq",
            chat_user_id="user-1",
            raw_query="query",
            preference=GamePreference(),
            result_limit=1,
            games=[ranked_game("Game A", 1)],
            now=1000,
        )
        payload = dump_memory(memory)
        payload["result_limit"] = "not-a-number"
        cache.payloads[recommendation_memory_key("qq", "user-1")] = payload

        loaded = await load_recommendation_memory(
            "qq",
            "user-1",
            cache,
            now=1010,
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
        self.read_keys: list[str] = []

    async def get_json(self, key: str, _ttl_hours: int) -> Any | None:
        self.read_keys.append(key)
        return self.payloads.get(key)

    async def set_json(self, key: str, payload: Any) -> None:
        self.payloads[key] = payload


class BoundedCache:
    def __init__(self) -> None:
        self.payloads: dict[str, Any] = {}
        self.ttl_seconds: int | None = None
        self.owner_scope = ""
        self.allow_stale_seconds: int | None = None
        self.deleted_owner_scopes: list[str] = []

    async def get_json(
        self,
        key: str,
        _ttl_hours: int = 24,
        *,
        allow_stale_seconds: int = 0,
    ) -> Any | None:
        self.allow_stale_seconds = allow_stale_seconds
        return self.payloads.get(key)

    async def set_json(
        self,
        key: str,
        payload: Any,
        _ttl_hours: int = 24,
        *,
        ttl_seconds: int | None = None,
        owner_scope: str = "",
    ) -> None:
        self.payloads[key] = payload
        self.ttl_seconds = ttl_seconds
        self.owner_scope = owner_scope

    async def delete_owner_scope(self, owner_scope: str) -> None:
        self.deleted_owner_scopes.append(owner_scope)


class MemoryKeyTest(unittest.TestCase):
    def test_key_is_versioned_hashed_and_session_scoped(self) -> None:
        first = recommendation_memory_key(
            "onebot-instance:GroupMessage:group-1",
            "user-1",
        )
        same = recommendation_memory_key(
            "onebot-instance:GroupMessage:group-1",
            "user-1",
        )
        other_session = recommendation_memory_key(
            "onebot-instance:GroupMessage:group-2",
            "user-1",
        )
        other_user = recommendation_memory_key(
            "onebot-instance:GroupMessage:group-1",
            "user-2",
        )

        self.assertEqual(first, same)
        self.assertTrue(first.startswith("recommendation_memory:v2:"))
        self.assertNotIn("group-1", first)
        self.assertNotIn("user-1", first)
        self.assertNotEqual(first, other_session)
        self.assertNotEqual(first, other_user)


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
            clock=lambda: 1.0,
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
            "schema_version": STEAM_INDEX_SCHEMA_VERSION,
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
