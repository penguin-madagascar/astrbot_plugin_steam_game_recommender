from __future__ import annotations

import asyncio
import unittest
from types import SimpleNamespace
from typing import Any
from unittest.mock import patch

from astrbot_plugin_steam_game_recommender.clients.steam import SteamStorefrontPage
from astrbot_plugin_steam_game_recommender.services.steam_index import (
    STEAM_INDEX_CACHE_KEY,
    STEAM_TAG_RECALL_DEGRADED_WARNING,
    SteamGameIndexService,
)
from astrbot_plugin_steam_game_recommender.storage.models import (
    GameCandidate,
    GamePreference,
    SteamSearchHit,
)


class SteamTagRecallIntegrationTest(unittest.IsolatedAsyncioTestCase):
    async def test_three_anchor_tags_use_independent_storefront_queries(self) -> None:
        client = RecallSteamClient(
            tag_ids={"Action": 1, "RPG": 2, "Puzzle": 3},
            tag_results={
                1: [SteamSearchHit(appid=101, title="Action Match")],
                2: [SteamSearchHit(appid=102, title="RPG Match")],
                3: [SteamSearchHit(appid=103, title="Puzzle Match")],
            },
        )
        service = SteamGameIndexService(client, MemoryCache())

        await service.recommend(
            GamePreference(genres_like=["action", "rpg", "puzzle"]),
            limit=3,
        )

        self.assertEqual(client.tag_calls, [(1, 20), (2, 20), (3, 20)])
        self.assertEqual(client.text_calls, [])
        self.assertEqual(client.top_seller_calls, [])

    async def test_mainstream_uses_top_sellers_but_normal_query_does_not(self) -> None:
        mainstream_client = RecallSteamClient(
            top_results=[SteamSearchHit(appid=201, title="Top Seller")]
        )
        mainstream = SteamGameIndexService(mainstream_client, MemoryCache())

        await mainstream.recommend(
            GamePreference(quality_intent="mainstream"),
            limit=1,
        )

        self.assertEqual(mainstream_client.top_seller_calls, [60])
        self.assertEqual(mainstream_client.text_calls, [])

        normal_client = RecallSteamClient()
        normal = SteamGameIndexService(normal_client, MemoryCache())
        await normal.recommend(GamePreference(), limit=1)
        self.assertEqual(normal_client.top_seller_calls, [])

    async def test_missing_tag_id_falls_back_to_that_tag_alone(self) -> None:
        client = RecallSteamClient(
            text_results={
                "soulslike": [SteamSearchHit(appid=301, title="Text Match")]
            }
        )
        service = SteamGameIndexService(client, MemoryCache())
        preference = GamePreference(genres_like=["soulslike"])

        with self.assertLogs(
            "astrbot_plugin_steam_game_recommender.services.steam_index",
            level="DEBUG",
        ) as logs:
            with patch(
                "astrbot_plugin_steam_game_recommender.services.steam_index.steam_tag_id_for",
                return_value=None,
            ):
                await service.recommend(preference, limit=1)

        self.assertEqual(client.text_calls, [("soulslike", 20)])
        self.assertEqual(client.tag_calls, [])
        self.assertEqual(
            preference.parse_warnings.count(STEAM_TAG_RECALL_DEGRADED_WARNING),
            1,
        )
        self.assertTrue(any("degraded=True" in line for line in logs.output))

    async def test_missing_storefront_method_uses_text_and_marks_degraded(self) -> None:
        client = RecallSteamClient(
            tag_ids={"Action": 11},
            text_results={
                "action": [SteamSearchHit(appid=311, title="Fallback Match")]
            },
        )
        client.search_storefront_tag = None  # type: ignore[method-assign]
        preference = GamePreference(genres_like=["action"])
        service = SteamGameIndexService(client, MemoryCache())

        await service.recommend(preference, limit=1)

        self.assertEqual(client.text_calls, [("action", 20)])
        self.assertEqual(
            preference.parse_warnings.count(STEAM_TAG_RECALL_DEGRADED_WARNING),
            1,
        )

    async def test_one_failed_tag_keeps_other_sources_and_warns_once(self) -> None:
        client = RecallSteamClient(
            tag_ids={"Action": 1, "Puzzle": 2},
            tag_results={
                1: RuntimeError("storefront unavailable"),
                2: [SteamSearchHit(appid=402, title="Successful Source")],
            },
            text_results={
                "action": [SteamSearchHit(appid=401, title="Fallback Source")]
            },
        )
        preference = GamePreference(genres_like=["action", "puzzle"])
        service = SteamGameIndexService(client, MemoryCache())

        with self.assertLogs(
            "astrbot_plugin_steam_game_recommender.services.steam_index",
            level="DEBUG",
        ) as logs:
            ranked = await service.recommend(preference, limit=2)

        self.assertEqual({game.appid for game in ranked}, {401, 402})
        self.assertEqual(
            preference.parse_warnings.count(STEAM_TAG_RECALL_DEGRADED_WARNING),
            1,
        )
        self.assertTrue(any("degraded=True" in line for line in logs.output))

    async def test_specific_intent_cannot_use_cached_quality_early_return(self) -> None:
        cached = [candidate(index, f"Cached {index}", ["Action"]) for index in range(1, 13)]
        cache = MemoryCache(snapshot(cached))
        client = RecallSteamClient(
            tag_ids={"Puzzle": 3},
            tag_results={3: [SteamSearchHit(appid=500, title="Anchor Match")]},
        )
        service = SteamGameIndexService(client, cache)

        await service.recommend(GamePreference(genres_like=["puzzle"]), limit=3)

        self.assertEqual(client.tag_calls, [(3, 20)])

        broad_client = RecallSteamClient()
        broad = SteamGameIndexService(broad_client, MemoryCache(snapshot(cached)))
        await broad.recommend(GamePreference(), limit=3)
        self.assertEqual(broad_client.tag_calls, [])
        self.assertEqual(broad_client.top_seller_calls, [])
        self.assertEqual(broad_client.text_calls, [])

    async def test_validation_pool_is_capped_and_dlc_is_discarded(self) -> None:
        tag_results = {
            tag_id: [
                SteamSearchHit(
                    appid=tag_id * 1_000 + offset,
                    title=f"Tag {tag_id}-{offset}",
                )
                for offset in range(1, 31)
            ]
            for tag_id in (1, 2, 3)
        }
        top_results = [
            SteamSearchHit(appid=10_000 + offset, title=f"Top {offset}")
            for offset in range(80)
        ]
        client = RecallSteamClient(
            tag_ids={"Action": 1, "RPG": 2, "Puzzle": 3},
            tag_results=tag_results,
            top_results=top_results,
            dlc_appids={1_001},
        )
        service = SteamGameIndexService(client, MemoryCache())

        ranked = await service.recommend(
            GamePreference(
                genres_like=["action", "rpg", "puzzle"],
                quality_intent="mainstream",
            ),
            limit=100,
        )

        self.assertLessEqual(len(client.detail_calls), 100)
        self.assertNotIn(1_001, {game.appid for game in ranked})

    async def test_query_relevant_old_cache_entry_survives_hundred_candidate_cap(
        self,
    ) -> None:
        cached = [
            candidate(index, f"Recent Generic {index}", ["Action"])
            for index in range(1, 121)
        ]
        relevant = candidate(
            9_999,
            "Old Strong Puzzle",
            ["Puzzle", "Adventure"],
        )
        cache = MemoryCache(snapshot_with_refresh_order([*cached, relevant]))
        client = RecallSteamClient(
            tag_ids={"Puzzle": 13},
            tag_results={13: []},
        )
        service = SteamGameIndexService(client, cache)
        loaded = await service.load_entries()
        self.assertGreater(
            next(index for index, item in enumerate(loaded) if item.appid == 9_999),
            100,
        )

        ranked = await service.recommend(
            GamePreference(genres_like=["puzzle"]),
            limit=5,
        )

        self.assertIn(9_999, {game.appid for game in ranked})
        self.assertEqual(client.detail_calls, [])

    async def test_concurrent_requests_share_alias_load_and_keep_both_snapshots(self) -> None:
        client = ConcurrentFallbackClient()
        cache = YieldingMemoryCache()
        service = SteamGameIndexService(client, cache)

        with patch(
            "astrbot_plugin_steam_game_recommender.services.steam_index.steam_tag_id_for",
            return_value=None,
        ):
            first, second = await asyncio.gather(
                service.recommend(GamePreference(genres_like=["action"]), limit=1),
                service.recommend(GamePreference(genres_like=["puzzle"]), limit=1),
            )

        self.assertEqual(client.popular_tag_calls, 1)
        self.assertEqual({first[0].appid, second[0].appid}, {701, 702})
        stored = cache.payloads[STEAM_INDEX_CACHE_KEY]
        stored_appids = {
            item["candidate"]["appid"]
            for item in stored["entries"]
        }
        self.assertTrue({701, 702} <= stored_appids)
        self.assertNotIn("source_kind", repr(stored))
        self.assertNotIn("source_rank", repr(stored))
        markers = [
            marker
            for item in stored["entries"]
            for marker in item["candidate"].get("internal_source_markers", [])
        ]
        self.assertFalse(any("retrieval" in marker for marker in markers))

    async def test_failed_alias_initialization_is_retryable(self) -> None:
        client = RetryableAliasClient()
        service = SteamGameIndexService(client, MemoryCache())

        self.assertFalse(await service.ensure_steam_tag_aliases())
        self.assertTrue(await service.ensure_steam_tag_aliases())
        self.assertEqual(client.popular_tag_calls, 2)


class RecallSteamClient:
    language = "english"

    def __init__(
        self,
        *,
        tag_ids: dict[str, int] | None = None,
        tag_results: dict[int, list[SteamSearchHit] | Exception] | None = None,
        top_results: list[SteamSearchHit] | None = None,
        text_results: dict[str, list[SteamSearchHit]] | None = None,
        dlc_appids: set[int] | None = None,
    ) -> None:
        self.tag_ids = tag_ids or {}
        self.tag_results = tag_results or {}
        self.top_results = top_results or []
        self.text_results = text_results or {}
        self.dlc_appids = dlc_appids or set()
        self.tag_calls: list[tuple[int, int]] = []
        self.top_seller_calls: list[int] = []
        self.text_calls: list[tuple[str, int]] = []
        self.detail_calls: list[int] = []

    async def get_popular_tags(self) -> list[dict[str, Any]]:
        return [
            {"tagid": tag_id, "name": name}
            for name, tag_id in self.tag_ids.items()
        ]

    async def search_storefront_tag(
        self,
        tag_id: int,
        page_size: int = 20,
    ) -> SteamStorefrontPage:
        self.tag_calls.append((tag_id, page_size))
        result = self.tag_results.get(tag_id, [])
        if isinstance(result, Exception):
            raise result
        return SteamStorefrontPage(tuple(result), len(result), 0)

    async def browse_top_sellers(self, page_size: int = 60) -> SteamStorefrontPage:
        self.top_seller_calls.append(page_size)
        return SteamStorefrontPage(tuple(self.top_results), len(self.top_results), 0)

    async def search_game_refs(
        self,
        search: str,
        page_size: int,
        **_kwargs: Any,
    ) -> list[SteamSearchHit]:
        self.text_calls.append((search, page_size))
        return list(self.text_results.get(search, []))

    async def get_game_detail(self, appid: int) -> GameCandidate:
        self.detail_calls.append(appid)
        return candidate(
            appid,
            f"Game {appid}",
            ["Action", "RPG", "Puzzle", "Souls-like"],
            app_type="dlc" if appid in self.dlc_appids else "game",
        )

    async def get_store_page_tags(self, _appid: int) -> list[str]:
        return ["Action", "RPG", "Puzzle", "Souls-like"]

    async def get_review_summary(self, _appid: int) -> SimpleNamespace:
        return SimpleNamespace(
            total_reviews=1_000,
            positive_ratio=0.9,
            recent_positive_ratio=0.9,
        )


class ConcurrentFallbackClient:
    language = "english"

    def __init__(self) -> None:
        self.popular_tag_calls = 0

    async def get_popular_tags(self) -> list[dict[str, Any]]:
        self.popular_tag_calls += 1
        await asyncio.sleep(0.01)
        return [{"tagid": 99, "name": "Strategy"}]

    async def search_games(
        self,
        search: str,
        page_size: int,
        **_kwargs: Any,
    ) -> list[GameCandidate]:
        del page_size
        await asyncio.sleep(0)
        appid = 701 if search == "action" else 702
        return [candidate(appid, f"Prefetched {search}", [search])]

    async def get_store_page_tags(self, appid: int) -> list[str]:
        return ["Action"] if appid == 701 else ["Puzzle"]

    async def get_review_summary(self, _appid: int) -> SimpleNamespace:
        return SimpleNamespace(
            total_reviews=500,
            positive_ratio=0.8,
            recent_positive_ratio=0.8,
        )


class RetryableAliasClient:
    def __init__(self) -> None:
        self.popular_tag_calls = 0

    async def get_popular_tags(self) -> list[dict[str, Any]]:
        self.popular_tag_calls += 1
        if self.popular_tag_calls == 1:
            raise RuntimeError("temporary")
        return [{"tagid": 1, "name": "Action"}]


class MemoryCache:
    def __init__(self, payload: dict[str, Any] | None = None) -> None:
        self.payloads = {STEAM_INDEX_CACHE_KEY: payload} if payload else {}

    async def get_json(self, key: str, _ttl_hours: int) -> Any | None:
        return self.payloads.get(key)

    async def set_json(self, key: str, payload: Any) -> None:
        self.payloads[key] = payload


class YieldingMemoryCache(MemoryCache):
    async def get_json(self, key: str, ttl_hours: int) -> Any | None:
        await asyncio.sleep(0)
        return await super().get_json(key, ttl_hours)

    async def set_json(self, key: str, payload: Any) -> None:
        await asyncio.sleep(0)
        await super().set_json(key, payload)


def candidate(
    appid: int,
    title: str,
    tags: list[str],
    *,
    app_type: str = "game",
) -> GameCandidate:
    return GameCandidate(
        appid=appid,
        title=title,
        app_type=app_type,
        tags=tags,
        platforms=["PC"],
        stores=["Steam"],
        review_total=1_000,
        review_positive_ratio=0.9,
        review_recent_ratio=0.9,
    )


def snapshot(candidates: list[GameCandidate]) -> dict[str, Any]:
    return {
        "version": 4,
        "entries": [
            {
                "candidate": (
                    item.model_dump() if hasattr(item, "model_dump") else item.dict()
                ),
                "refreshed_at": 1.0,
            }
            for item in candidates
        ],
        "search_coverage": {},
    }


def snapshot_with_refresh_order(candidates: list[GameCandidate]) -> dict[str, Any]:
    payload = snapshot(candidates)
    for position, item in enumerate(payload["entries"]):
        item["refreshed_at"] = float(len(candidates) - position)
    return payload


if __name__ == "__main__":
    unittest.main()
