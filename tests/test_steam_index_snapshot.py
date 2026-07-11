from __future__ import annotations

import asyncio
import unittest
from types import SimpleNamespace
from typing import Any

from astrbot_plugin_game_recommender.services.steam_index import (
    STEAM_INDEX_CACHE_KEY,
    SteamGameIndexService,
    SteamIndexEntry,
    SteamIndexSnapshot,
    prune_snapshot,
    reference_candidates,
)
from astrbot_plugin_game_recommender.storage.models import (
    GameCandidate,
    GamePreference,
    SteamSearchHit,
)


class SteamIndexSnapshotTest(unittest.IsolatedAsyncioTestCase):
    def test_snapshot_evicts_oldest_entries_and_search_terms(self) -> None:
        snapshot = SteamIndexSnapshot(
            entries=[
                SteamIndexEntry(
                    candidate=game(index + 1, f"Game {index}", ["Co-op"]),
                    refreshed_at=float(index),
                )
                for index in range(3_005)
            ],
            search_coverage={f"query {index}": float(index) for index in range(260)},
        )

        pruned = prune_snapshot(snapshot)

        self.assertEqual(len(pruned.entries), 3_000)
        self.assertEqual(len(pruned.search_coverage), 256)
        self.assertEqual(pruned.entries[0].candidate.appid, 3_005)
        self.assertNotIn("query 0", pruned.search_coverage)
        self.assertIn("query 259", pruned.search_coverage)

    async def test_v2_cache_key_ignores_legacy_index_payload(self) -> None:
        cache = MemoryCache(
            {
                "steam_index:entries": [dump_model(game(1, "Legacy", ["Co-op"]))],
            }
        )
        service = SteamGameIndexService(NoopSteamClient(), cache, clock=lambda: 1_000.0)

        entries = await service.load_entries()

        self.assertEqual(STEAM_INDEX_CACHE_KEY, "steam_index:v2")
        self.assertEqual(entries, [])
        self.assertEqual(cache.read_keys, ["steam_index:v2"])

    async def test_weak_cached_pool_triggers_deduplicated_query_recall(self) -> None:
        cache = MemoryCache(
            {
                "steam_index:v2": snapshot_payload(
                    [game(1, "Cached Generic", ["Multiplayer"])],
                    refreshed_at=900.0,
                )
            }
        )
        client = QueryAwareSteamClient()
        service = SteamGameIndexService(client, cache, clock=lambda: 1_000.0)

        ranked = await service.recommend(
            GamePreference(
                genres_like=["co-op", "puzzle", "farming", "crafting"],
                result_count=3,
            ),
            limit=3,
        )

        self.assertEqual(len(ranked), 3)
        self.assertTrue(client.search_queries)
        self.assertLessEqual(len(client.search_queries), 8)
        self.assertTrue(all(page_size == 10 for page_size in client.search_page_sizes))
        self.assertEqual(len(client.detail_appids), len(set(client.detail_appids)))
        self.assertLessEqual(len(client.detail_appids), 60)
        self.assertLessEqual(client.max_active_searches, 6)
        self.assertLessEqual(client.max_active_enrichments, 6)
        self.assertEqual(client.popular_tag_calls, 1)

        written = cache.payloads["steam_index:v2"]
        self.assertEqual(written["version"], 2)
        self.assertTrue(written["entries"])
        self.assertTrue(
            all("refreshed_at" in record and "candidate" in record for record in written["entries"])
        )
        self.assertTrue(written["search_coverage"])

    async def test_limits_each_round_to_sixty_new_appids(self) -> None:
        cache = MemoryCache({})
        client = UniqueHitSteamClient()
        service = SteamGameIndexService(client, cache, clock=lambda: 1_000.0)
        preference = GamePreference(
            reference_games_like=["Query 1", "Query 2", "Query 3"],
            reference_search_terms=["Query 4", "Query 5", "Query 6"],
            genres_like=["co-op", "puzzle", "farming", "crafting"],
            result_count=10,
        )

        entries = await service.refresh_entries(preference, [], target_pool=60)

        self.assertEqual(len(client.detail_appids), 60)
        self.assertEqual(len(client.detail_appids), len(set(client.detail_appids)))
        self.assertEqual(len(entries), 60)

    async def test_reference_resolution_requires_title_confidence_and_keeps_polarity(self) -> None:
        cache = MemoryCache({})
        client = ReferenceSteamClient()
        service = SteamGameIndexService(client, cache, clock=lambda: 1_000.0)
        preference = GamePreference(
            reference_games_like=["Dark Souls", "Slay the Spire"],
            reference_games_dislike=["Overcooked"],
        )

        entries = await service.refresh_entries(preference, [], target_pool=30)

        resolved = {item.raw_text: item for item in preference.resolved_reference_games}
        self.assertGreaterEqual(resolved["Dark Souls"].confidence, 0.75)
        self.assertEqual(resolved["Dark Souls"].polarity, "like")
        self.assertGreaterEqual(resolved["Overcooked"].confidence, 0.75)
        self.assertEqual(resolved["Overcooked"].polarity, "dislike")
        self.assertLess(resolved["Slay the Spire"].confidence, 0.75)
        self.assertTrue(any("Slay the Spire" in warning for warning in preference.parse_warnings))

        markers = [reason for entry in entries for reason in entry.source_reasons]
        self.assertTrue(any(marker.startswith("reference_query:like:") for marker in markers))
        self.assertTrue(any(marker.startswith("reference_query:dislike:") for marker in markers))
        self.assertFalse(any("Slay the Spire" in marker for marker in markers))
        self.assertTrue(reference_candidates(preference, entries))
        self.assertTrue(
            all(
                any(reason.startswith("reference_query:like:") for reason in item.source_reasons)
                for item in reference_candidates(preference, entries)
            )
        )


class MemoryCache:
    def __init__(self, payloads: dict[str, Any]) -> None:
        self.payloads = payloads
        self.read_keys: list[str] = []

    async def get_json(self, key: str, _ttl_hours: int) -> Any | None:
        self.read_keys.append(key)
        return self.payloads.get(key)

    async def set_json(self, key: str, payload: Any) -> None:
        self.payloads[key] = payload


class NoopSteamClient:
    pass


class QueryAwareSteamClient:
    def __init__(self) -> None:
        self.search_queries: list[str] = []
        self.search_page_sizes: list[int] = []
        self.detail_appids: list[int] = []
        self.popular_tag_calls = 0
        self.active_searches = 0
        self.max_active_searches = 0
        self.active_enrichments = 0
        self.max_active_enrichments = 0

    async def get_popular_tags(self) -> list[dict[str, Any]]:
        self.popular_tag_calls += 1
        return [{"tagid": 1685, "name": "Co-op"}]

    async def search_game_refs(self, search: str, page_size: int, **_kwargs: Any):
        self.search_queries.append(search)
        self.search_page_sizes.append(page_size)
        self.active_searches += 1
        self.max_active_searches = max(self.max_active_searches, self.active_searches)
        await asyncio.sleep(0)
        self.active_searches -= 1
        return [SteamSearchHit(appid=100 + index, title=f"Match {index}") for index in range(10)]

    async def get_game_detail(self, appid: int) -> GameCandidate:
        self.detail_appids.append(appid)
        self.active_enrichments += 1
        self.max_active_enrichments = max(self.max_active_enrichments, self.active_enrichments)
        await asyncio.sleep(0)
        self.active_enrichments -= 1
        return game(appid, f"Match {appid}", ["Co-op", "Puzzle", "Farming", "Crafting"])

    async def get_store_page_tags(self, _appid: int) -> list[str]:
        return ["Co-op", "Puzzle", "Farming", "Crafting"]

    async def get_review_summary(self, _appid: int):
        return SimpleNamespace(
            total_reviews=500,
            positive_ratio=0.8,
            recent_positive_ratio=0.8,
        )


class UniqueHitSteamClient(QueryAwareSteamClient):
    async def search_game_refs(self, search: str, page_size: int, **kwargs: Any):
        query_index = len(self.search_queries)
        await super().search_game_refs(search, page_size, **kwargs)
        return [
            SteamSearchHit(
                appid=1_000 + query_index * 10 + index,
                title=f"Unique {query_index}-{index}",
            )
            for index in range(10)
        ]


class ReferenceSteamClient(QueryAwareSteamClient):
    async def search_game_refs(self, search: str, page_size: int, **_kwargs: Any):
        del page_size
        self.search_queries.append(search)
        mapping = {
            "Dark Souls": SteamSearchHit(appid=10, title="Dark Souls Remastered"),
            "Slay the Spire": SteamSearchHit(appid=20, title="Unrelated Adventure"),
            "Overcooked": SteamSearchHit(appid=30, title="Overcooked! 2"),
        }
        hit = mapping.get(search)
        return [hit] if hit else []


def snapshot_payload(
    candidates: list[GameCandidate],
    refreshed_at: float,
) -> dict[str, Any]:
    return {
        "version": 2,
        "entries": [
            {"candidate": dump_model(candidate), "refreshed_at": refreshed_at}
            for candidate in candidates
        ],
        "search_coverage": {},
    }


def game(
    appid: int,
    title: str,
    tags: list[str],
    description: str | None = None,
) -> GameCandidate:
    return GameCandidate(
        appid=appid,
        title=title,
        platforms=["PC"],
        genres=[],
        tags=tags,
        stores=["Steam"],
        review_total=500,
        review_positive_ratio=0.8,
        review_recent_ratio=0.8,
        description=description,
    )


def dump_model(model: Any) -> dict[str, Any]:
    dumper = getattr(model, "model_dump", None)
    return dumper() if dumper else model.dict()


if __name__ == "__main__":
    unittest.main()
