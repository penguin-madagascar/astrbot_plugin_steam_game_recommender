from __future__ import annotations

import asyncio
import unittest
from types import SimpleNamespace
from typing import Any
from unittest.mock import patch

from astrbot_plugin_steam_game_recommender.clients.steam import SteamStorefrontPage
from astrbot_plugin_steam_game_recommender.services import tag_normalizer
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
    async def test_first_reference_query_counts_top_five_before_selecting_anchors(
        self,
    ) -> None:
        reference = GameCandidate(
            appid=900,
            title="Anchor Seed",
            app_type="game",
            ordered_tags=["Action", "RPG", "Souls-like", "Dark Fantasy", "Difficult"],
            platforms=["PC"],
            stores=["Steam"],
        )
        tag_ids = {
            "Action": 1,
            "RPG": 2,
            "Souls-like": 3,
            "Dark Fantasy": 4,
            "Difficult": 5,
        }
        tag_results = {
            tag_id: [
                SteamSearchHit(
                    appid=tag_id * 1_000 + index,
                    title=f"Tag {tag_id}-{index}",
                )
                for index in range(1, count + 1)
            ]
            for tag_id, count in {1: 10, 2: 8, 3: 2, 4: 3, 5: 6}.items()
        }
        client = RecallSteamClient(tag_ids=tag_ids, tag_results=tag_results)
        service = SteamGameIndexService(client, MemoryCache(snapshot([reference])))
        preference = GamePreference(reference_games_like=["Anchor Seed"])

        with (
            patch.dict(tag_normalizer.STEAM_TAG_IDS, {}, clear=True),
        ):
            with self.assertLogs(
                "astrbot_plugin_steam_game_recommender.services.steam_index",
                level="DEBUG",
            ) as logs:
                ranked = await service.recommend(preference, limit=5)

        rank_logs = [
            message
            for message in logs.output
            if "recommendation_rank event=rank_complete" in message
        ]
        self.assertIn("anchors=['soulslike', 'dark_fantasy']", rank_logs[-1])
        self.assertEqual(
            client.tag_calls,
            [(1, 20), (2, 20), (3, 20), (4, 20), (5, 20)],
        )
        self.assertTrue({game.appid for game in ranked} <= {3_001, 3_002, 4_001, 4_002, 4_003})
        self.assertFalse({1_001, 2_001, 5_001} & set(client.detail_calls))

    async def test_failed_count_prefetch_falls_back_without_retrying_storefront(
        self,
    ) -> None:
        reference = GameCandidate(
            appid=910,
            title="Fallback Seed",
            app_type="game",
            ordered_tags=["Action", "RPG", "Puzzle", "Strategy", "Simulation"],
            platforms=["PC"],
            stores=["Steam"],
        )
        client = RecallSteamClient(
            tag_ids={"Action": 11, "RPG": 12, "Puzzle": 13, "Strategy": 14, "Simulation": 15},
            tag_results={
                11: RuntimeError("count endpoint unavailable"),
                12: [SteamSearchHit(appid=12_001, title="RPG Match")],
                13: [SteamSearchHit(appid=13_001, title="Puzzle Match")],
                14: [SteamSearchHit(appid=14_001, title="Strategy Match")],
                15: [SteamSearchHit(appid=15_001, title="Simulation Match")],
            },
            text_results={
                "action": [SteamSearchHit(appid=11_001, title="Action Text Match")]
            },
        )
        service = SteamGameIndexService(client, MemoryCache(snapshot([reference])))
        preference = GamePreference(reference_games_like=["Fallback Seed"])

        with (
            patch.dict(tag_normalizer.STEAM_TAG_IDS, {}, clear=True),
        ):
            ranked = await service.recommend(preference, limit=5)

        self.assertEqual(client.tag_calls.count((11, 20)), 1)
        self.assertEqual(client.text_calls, [("action", 20)])
        self.assertEqual({game.appid for game in ranked}, {11_001, 12_001})
        self.assertEqual(
            preference.parse_warnings.count(STEAM_TAG_RECALL_DEGRADED_WARNING),
            1,
        )

    async def test_missing_vocabulary_preserves_canonical_required_tag_safely(
        self,
    ) -> None:
        client = RecallSteamClient(
            text_results={"precision duel": []},
        )
        service = SteamGameIndexService(
            client,
            MemoryCache(
                snapshot([candidate(30_001, "Unrelated Action", ["Action"])])
            ),
        )
        preference = GamePreference(required_tags=["precision_duel"])

        with (
            patch.object(tag_normalizer, "STEAM_TAG_ALIASES", {}),
            patch.object(tag_normalizer, "STEAM_CANONICAL_TAGS", set()),
            patch.object(tag_normalizer, "STEAM_TAG_IDS", {}),
        ):
            ranked = await service.recommend(preference, limit=5)

        self.assertEqual(ranked, [])
        self.assertEqual(client.text_calls, [("precision duel", 20)])
        self.assertIn(STEAM_TAG_RECALL_DEGRADED_WARNING, preference.parse_warnings)

    async def test_missing_vocabulary_preserves_canonical_exclusion_tag(self) -> None:
        forbidden = candidate(31_001, "A Forbidden", ["Precision Duel"])
        allowed = [
            candidate(31_100 + index, f"B Allowed {index}", ["Action"])
            for index in range(10)
        ]
        client = RecallSteamClient()
        service = SteamGameIndexService(
            client,
            MemoryCache(snapshot([forbidden, *allowed])),
        )
        preference = GamePreference(genres_dislike=["precision_duel"])

        with (
            patch.object(tag_normalizer, "STEAM_TAG_ALIASES", {}),
            patch.object(tag_normalizer, "STEAM_CANONICAL_TAGS", set()),
            patch.object(tag_normalizer, "STEAM_TAG_IDS", {}),
        ):
            ranked = await service.recommend(preference, limit=5)

        self.assertNotIn(forbidden.appid, [game.appid for game in ranked])
        self.assertIn(STEAM_TAG_RECALL_DEGRADED_WARNING, preference.parse_warnings)

    async def test_request_retrieval_rank_breaks_equal_final_scores(self) -> None:
        client = RecallSteamClient(
            tag_ids={"Action": 1, "Puzzle": 2},
            tag_results={
                1: [
                    SteamSearchHit(appid=601, title="Action First"),
                    SteamSearchHit(appid=603, title="Action Second"),
                ],
                2: [SteamSearchHit(appid=602, title="Puzzle First")],
            },
        )
        service = SteamGameIndexService(client, MemoryCache())

        ranked = await service.recommend(
            GamePreference(genres_like=["action", "puzzle"]),
            limit=3,
        )

        self.assertEqual([game.appid for game in ranked], [601, 602, 603])
        self.assertEqual(
            [game.score_breakdown.retrieval_rank for game in ranked],
            [1, 2, 3],
        )

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
            await service.recommend(preference, limit=1)

        self.assertEqual(client.text_calls, [("soulslike", 20)])
        self.assertEqual(client.tag_calls, [])
        self.assertEqual(
            preference.parse_warnings.count(STEAM_TAG_RECALL_DEGRADED_WARNING),
            1,
        )
        self.assertTrue(
            any(
                "recommendation_recall event=recall_complete" in line
                for line in logs.output
            )
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

    async def test_loaded_tag_vocabulary_refreshes_after_twenty_four_hours(self) -> None:
        client = RefreshingAliasClient()
        now = [1_000.0]
        service = SteamGameIndexService(
            client,
            MemoryCache(),
            clock=lambda: now[0],
        )

        self.assertTrue(await service.ensure_steam_tag_aliases())
        now[0] += 24 * 60 * 60 - 1
        self.assertTrue(await service.ensure_steam_tag_aliases())
        self.assertEqual(client.popular_tag_calls, 1)

        now[0] += 1
        self.assertTrue(await service.ensure_steam_tag_aliases())
        self.assertEqual(client.popular_tag_calls, 2)

    async def test_expired_tag_vocabulary_cannot_reuse_old_dynamic_tag_id(self) -> None:
        now = [1_000_000.0]
        client = ExpiringVocabularyClient(now)
        service = SteamGameIndexService(
            client,
            MemoryCache(),
            clock=lambda: now[0],
        )
        preference = GamePreference(genres_like=["expiry probe"])

        with (
            patch.object(tag_normalizer, "STEAM_TAG_ALIASES", {}),
            patch.object(tag_normalizer, "STEAM_CANONICAL_TAGS", set()),
            patch.object(tag_normalizer, "STEAM_TAG_IDS", {}),
        ):
            first = await service.recommend(preference, limit=1)
            now[0] += 8 * 24 * 60 * 60
            client.fail_vocabulary_refresh = True
            second_preference = GamePreference(genres_like=["expiry probe"])
            second = await service.recommend(second_preference, limit=1)

        self.assertEqual([game.appid for game in first], [801])
        self.assertEqual([game.appid for game in second], [802])
        self.assertEqual(client.tag_calls, [(99_999, 20)])
        self.assertEqual(client.text_calls, [("expiry probe", 20)])
        self.assertIn(STEAM_TAG_RECALL_DEGRADED_WARNING, second_preference.parse_warnings)

    async def test_service_instances_keep_independent_dynamic_tag_ids(self) -> None:
        alpha_client = RecallSteamClient(
            tag_ids={"Alpha Dynamic": 71},
            tag_results={71: [SteamSearchHit(appid=871, title="Alpha Match")]},
        )
        beta_client = RecallSteamClient(
            tag_ids={"Beta Dynamic": 72},
            tag_results={72: [SteamSearchHit(appid=872, title="Beta Match")]},
        )
        alpha = SteamGameIndexService(alpha_client, MemoryCache())
        beta = SteamGameIndexService(beta_client, MemoryCache())

        with (
            patch.object(tag_normalizer, "STEAM_TAG_ALIASES", {}),
            patch.object(tag_normalizer, "STEAM_CANONICAL_TAGS", set()),
            patch.object(tag_normalizer, "STEAM_TAG_IDS", {}),
        ):
            await alpha.recommend(
                GamePreference(genres_like=["alpha dynamic"]),
                limit=1,
            )
            await beta.recommend(
                GamePreference(genres_like=["beta dynamic"]),
                limit=1,
            )
            await alpha.recommend(
                GamePreference(genres_like=["alpha dynamic"]),
                limit=1,
            )
            cross_preference = GamePreference(genres_like=["beta dynamic"])
            await alpha.recommend(cross_preference, limit=1)

        self.assertEqual(alpha_client.tag_calls, [(71, 20), (71, 20)])
        self.assertEqual(beta_client.tag_calls, [(72, 20)])
        self.assertIn(STEAM_TAG_RECALL_DEGRADED_WARNING, cross_preference.parse_warnings)

    async def test_cancelling_one_alias_waiter_keeps_shared_load_alive(self) -> None:
        client = CancellationSteamClient()
        service = SteamGameIndexService(client, MemoryCache())
        first = asyncio.create_task(service.ensure_steam_tag_aliases())
        second = asyncio.create_task(service.ensure_steam_tag_aliases())
        await client.alias_started.wait()

        first.cancel()
        with self.assertRaises(asyncio.CancelledError):
            await first
        client.alias_release.set()

        self.assertTrue(await second)
        await asyncio.sleep(0)
        self.assertEqual(client.popular_tag_calls, 1)
        self.assertIsNone(service._tag_aliases_task)

    async def test_cancelling_one_storefront_waiter_keeps_shared_page_alive(self) -> None:
        client = CancellationSteamClient()
        service = SteamGameIndexService(client, MemoryCache())
        first = asyncio.create_task(
            service._shared_storefront_tag_page(81, client.search_storefront_tag)
        )
        second = asyncio.create_task(
            service._shared_storefront_tag_page(81, client.search_storefront_tag)
        )
        await client.tag_started.wait()

        first.cancel()
        with self.assertRaises(asyncio.CancelledError):
            await first
        client.tag_release.set()

        page = await second
        await asyncio.sleep(0)
        self.assertEqual(page.total_count, 1)
        self.assertEqual(client.tag_calls, [(81, 20)])
        self.assertEqual(service._storefront_tag_tasks, {})

    async def test_concurrent_reference_requests_share_count_prefetches(self) -> None:
        reference = GameCandidate(
            appid=920,
            title="Concurrent Seed",
            app_type="game",
            ordered_tags=["Action", "RPG", "Puzzle", "Strategy", "Simulation"],
            platforms=["PC"],
            stores=["Steam"],
        )
        client = SlowRecallSteamClient(
            tag_ids={"Action": 21, "RPG": 22, "Puzzle": 23, "Strategy": 24, "Simulation": 25},
            tag_results={
                tag_id: [
                    SteamSearchHit(appid=tag_id * 1_000, title=f"Tag {tag_id}")
                ]
                for tag_id in range(21, 26)
            },
        )
        service = SteamGameIndexService(
            client,
            YieldingMemoryCache(snapshot([reference])),
        )

        with (
            patch.dict(tag_normalizer.STEAM_TAG_IDS, {}, clear=True),
        ):
            await asyncio.gather(
                service.recommend(
                    GamePreference(reference_games_like=["Concurrent Seed"]),
                    limit=2,
                ),
                service.recommend(
                    GamePreference(reference_games_like=["Concurrent Seed"]),
                    limit=2,
                ),
            )

        self.assertEqual(
            sorted(client.tag_calls),
            [(tag_id, 20) for tag_id in range(21, 26)],
        )


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


class SlowRecallSteamClient(RecallSteamClient):
    async def search_storefront_tag(
        self,
        tag_id: int,
        page_size: int = 20,
    ) -> SteamStorefrontPage:
        await asyncio.sleep(0.01)
        return await super().search_storefront_tag(tag_id, page_size)


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


class RefreshingAliasClient:
    def __init__(self) -> None:
        self.popular_tag_calls = 0

    async def get_popular_tags(self) -> list[dict[str, Any]]:
        self.popular_tag_calls += 1
        return [{"tagid": 1, "name": "Action"}]


class ExpiringVocabularyClient(RecallSteamClient):
    def __init__(self, now: list[float]) -> None:
        super().__init__(
            tag_results={
                99_999: [SteamSearchHit(appid=801, title="Dynamic Tag Match")]
            },
            text_results={
                "expiry probe": [SteamSearchHit(appid=802, title="Text Fallback")]
            },
        )
        self.now = now
        self.initial_fetch_time = now[0]
        self.fail_vocabulary_refresh = False

    async def get_popular_tags_snapshot(self) -> SimpleNamespace:
        if self.fail_vocabulary_refresh:
            raise RuntimeError("vocabulary refresh unavailable")
        return SimpleNamespace(
            tags=({"tagid": 99_999, "name": "Expiry Probe"},),
            fetched_at=self.initial_fetch_time,
        )


class CancellationSteamClient(RecallSteamClient):
    def __init__(self) -> None:
        super().__init__()
        self.popular_tag_calls = 0
        self.alias_started = asyncio.Event()
        self.alias_release = asyncio.Event()
        self.tag_started = asyncio.Event()
        self.tag_release = asyncio.Event()

    async def get_popular_tags(self) -> list[dict[str, Any]]:
        self.popular_tag_calls += 1
        self.alias_started.set()
        await self.alias_release.wait()
        return [{"tagid": 81, "name": "Cancellation Probe"}]

    async def search_storefront_tag(
        self,
        tag_id: int,
        page_size: int = 20,
    ) -> SteamStorefrontPage:
        self.tag_calls.append((tag_id, page_size))
        self.tag_started.set()
        await self.tag_release.wait()
        return SteamStorefrontPage(
            (SteamSearchHit(appid=881, title="Cancellation Match"),),
            1,
            0,
        )


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
        "schema_version": 1,
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
