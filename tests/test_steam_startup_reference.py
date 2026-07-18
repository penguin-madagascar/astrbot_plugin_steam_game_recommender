from __future__ import annotations

import asyncio
import unittest
from datetime import datetime, timedelta, timezone
from email.utils import format_datetime
from types import SimpleNamespace
from typing import Any

import httpx

from astrbot_plugin_steam_game_recommender.clients.steam import (
    SteamApiError,
    SteamClient,
    SteamStorefrontPage,
    SteamTransientError,
    parse_storefront_results_html,
)
from astrbot_plugin_steam_game_recommender.services import steam_index as steam_index_module
from astrbot_plugin_steam_game_recommender.services.recommendation_intent import (
    ReferencePolarity,
    ReferenceQuery,
    build_recommendation_intent,
)
from astrbot_plugin_steam_game_recommender.services.reference_matching import (
    match_reference_query,
)
from astrbot_plugin_steam_game_recommender.services.tag_normalizer import normalize_tag
from astrbot_plugin_steam_game_recommender.services.steam_index import (
    STEAM_INDEX_CACHE_KEY,
    STEAM_INDEX_SCHEMA_VERSION,
    SteamGameIndexService,
)
from astrbot_plugin_steam_game_recommender.services.steam_recall import (
    RecallUnavailableError,
)
from astrbot_plugin_steam_game_recommender.storage.models import (
    GameCandidate,
    GamePreference,
    SteamSearchHit,
)


VALID_STORE_HTML = """
<html><body><div class="game_page_background">
Popular user-defined tags for this product:
<a class="app_tag">RPG</a><a class="app_tag">Action</a>
</div></body></html>
"""


class TitleAndAliasRegressionTest(unittest.TestCase):
    def test_trademark_is_removed_before_edition_normalization(self) -> None:
        reference = ReferenceQuery(
            "Dark Souls",
            ("Dark Souls",),
            ReferencePolarity.POSITIVE,
        )

        match = match_reference_query(
            reference,
            [SteamSearchHit(appid=10, title="DARK SOULS™: REMASTERED")],
        )

        self.assertIsNotNone(match)
        self.assertEqual(match.hit.appid if match else None, 10)

    def test_trademark_normalization_does_not_drop_sequel_numbers(self) -> None:
        base_reference = ReferenceQuery(
            "Dark Souls",
            ("Dark Souls",),
            ReferencePolarity.POSITIVE,
        )
        sequel_reference = ReferenceQuery(
            "Dark Souls II",
            ("Dark Souls II",),
            ReferencePolarity.POSITIVE,
        )
        sequel_hit = SteamSearchHit(
            appid=20,
            title="DARK SOULS™ II: REMASTERED",
        )

        self.assertIsNone(match_reference_query(base_reference, [sequel_hit]))
        self.assertIsNotNone(match_reference_query(sequel_reference, [sequel_hit]))

    def test_one_original_mention_collects_localized_and_english_aliases(self) -> None:
        intent = build_recommendation_intent(
            GamePreference(
                reference_games_like=["黑暗之魂"],
                reference_search_terms=["黑暗之魂", "Dark Souls"],
            )
        )

        self.assertEqual(len(intent.references), 1)
        self.assertEqual(intent.references[0].display_title, "黑暗之魂")
        self.assertEqual(intent.references[0].aliases, ("黑暗之魂", "Dark Souls"))

    def test_two_original_mentions_remain_two_reference_entities(self) -> None:
        intent = build_recommendation_intent(
            GamePreference(
                reference_games_like=["Portal", "Half-Life"],
                reference_search_terms=["Portal", "Half-Life"],
            )
        )

        self.assertEqual(
            [(item.display_title, item.aliases) for item in intent.references],
            [("Portal", ("Portal",)), ("Half-Life", ("Half-Life",))],
        )


class StorefrontContractRegressionTest(unittest.IsolatedAsyncioTestCase):
    def test_storefront_rows_preserve_ordered_tag_ids(self) -> None:
        hits = parse_storefront_results_html(
            '<a class="search_result_row" data-ds-appid="10" '
            'data-ds-tagids="[29482, 19, 122]">'
            '<span class="title">Dark Souls</span></a>'
        )

        self.assertTrue(hasattr(hits[0], "tag_ids"))
        self.assertEqual(getattr(hits[0], "tag_ids", None), [29482, 19, 122])

    async def test_storefront_term_search_uses_term_and_language(self) -> None:
        http_client = QueueHttpClient([json_response(storefront_payload())])
        client = SteamClient(http_client, MemoryCache(), default_country="CN")
        search = getattr(client, "search_storefront_term", None)

        self.assertTrue(callable(search))
        page = await search(
            " Dark Souls ",
            page_size=100,
            start=-2,
            language="english",
        )

        self.assertEqual([hit.appid for hit in page.hits], [10])
        self.assertEqual(
            http_client.calls[0]["params"],
            {
                "ignore_preferences": 1,
                "term": "Dark Souls",
                "ndl": 1,
                "l": "english",
                "cc": "CN",
                "start": 0,
                "count": 60,
                "infinite": 1,
            },
        )

    async def test_store_page_uses_age_cookies_and_caches_only_validated_tags(self) -> None:
        cache = MemoryCache()
        http_client = QueueHttpClient(
            [text_response(VALID_STORE_HTML, url="https://store.steampowered.com/app/10/")]
        )
        client = SteamClient(http_client, cache)

        tags = await client.get_store_page_tags(10)
        cached = await client.get_store_page_tags(10)

        self.assertEqual(tags, ["RPG", "Action"])
        self.assertEqual(cached, tags)
        self.assertEqual(len(http_client.calls), 1)
        cookies = http_client.calls[0]["kwargs"].get("cookies")
        self.assertIsInstance(cookies, dict)
        self.assertIn("birthtime", cookies)
        compact_keys = [
            key for key in cache.payloads if key.startswith("steam:store-page-tags:")
        ]
        self.assertEqual(len(compact_keys), 1)
        self.assertEqual(cache.payloads[compact_keys[0]], ["RPG", "Action"])
        self.assertFalse(any("<html" in str(value).lower() for value in cache.payloads.values()))

    async def test_age_gate_descriptor_tags_are_rejected_and_not_cached(self) -> None:
        age_gate = """
        <html><body id="agecheck"><form action="/agecheck/app/10/">
        <div class="content_descriptor"><a class="app_tag">Violent</a>
        <a class="app_tag">Gore</a></div></form></body></html>
        """
        cache = MemoryCache()
        client = SteamClient(
            QueueHttpClient(
                [text_response(age_gate, url="https://store.steampowered.com/agecheck/app/10/")]
            ),
            cache,
        )

        with self.assertRaises(SteamApiError):
            await client.get_store_page_tags(10)

        self.assertEqual(cache.payloads, {})

    async def test_redirected_or_missing_main_content_pages_are_rejected(self) -> None:
        redirected = text_response(
            VALID_STORE_HTML,
            url="https://store.steampowered.com/app/10/",
            redirected=True,
        )
        missing_main = text_response(
            '<html><body><a class="app_tag">RPG</a></body></html>',
            url="https://store.steampowered.com/app/10/",
        )

        for response, expected_calls in (
            (redirected, 1),
            (missing_main, 2),
        ):
            with self.subTest(response=response):
                cache = MemoryCache()
                http_client = QueueHttpClient([response] * expected_calls)
                client = SteamClient(http_client, cache)
                with self.assertRaises(SteamApiError):
                    await client.get_store_page_tags(10)
                self.assertEqual(len(http_client.calls), expected_calls)
                self.assertEqual(cache.payloads, {})


class SteamReadRetryRegressionTest(unittest.IsolatedAsyncioTestCase):
    async def test_connection_failure_retries_once_and_returns_second_response(self) -> None:
        request = httpx.Request("GET", "https://store.steampowered.com/api/storesearch/")
        http_client = QueueHttpClient(
            [
                httpx.ConnectError("offline", request=request),
                json_response({"items": []}),
            ]
        )
        client = SteamClient(http_client, MemoryCache())

        try:
            hits = await client.search_game_refs(search="Portal")
        except SteamApiError as exc:
            self.fail(f"retryable read did not retry: {exc}")

        self.assertEqual(hits, [])
        self.assertEqual(len(http_client.calls), 2)

    async def test_retry_after_is_honored_before_the_only_retry(self) -> None:
        sleeps: list[float] = []

        async def sleeper(delay: float) -> None:
            sleeps.append(delay)

        http_client = QueueHttpClient(
            [
                json_response({}, status_code=429, headers={"Retry-After": "2"}),
                json_response({"items": []}),
            ]
        )
        client = SteamClient(http_client, MemoryCache())
        client._sleeper = sleeper

        try:
            await client.search_game_refs(search="Portal")
        except SteamApiError as exc:
            self.fail(f"429 read did not retry: {exc}")

        self.assertEqual(len(http_client.calls), 2)
        self.assertEqual(sleeps, [2.0])

    async def test_retry_after_is_clamped_for_large_seconds_and_future_dates(self) -> None:
        future = format_datetime(datetime.now(timezone.utc) + timedelta(days=365))
        for header in ("999999999", future):
            with self.subTest(header=header):
                sleeps: list[float] = []

                async def sleeper(delay: float) -> None:
                    sleeps.append(delay)

                http_client = QueueHttpClient(
                    [
                        json_response({}, status_code=429, headers={"Retry-After": header}),
                        json_response({"items": []}),
                    ]
                )
                client = SteamClient(http_client, MemoryCache(), sleeper=sleeper)

                await client.search_game_refs(search="Portal")

                self.assertEqual(sleeps, [5.0])

    async def test_non_finite_retry_after_never_reaches_the_sleeper(self) -> None:
        for header in ("nan", "inf", "-inf"):
            with self.subTest(header=header):
                sleeps: list[float] = []

                async def sleeper(delay: float) -> None:
                    sleeps.append(delay)

                http_client = QueueHttpClient(
                    [
                        json_response({}, status_code=429, headers={"Retry-After": header}),
                        json_response({"items": []}),
                    ]
                )
                client = SteamClient(http_client, MemoryCache(), sleeper=sleeper)

                await client.search_game_refs(search="Portal")

                self.assertEqual(sleeps, [])

    async def test_non_retryable_404_attempts_once(self) -> None:
        not_found_http = QueueHttpClient([json_response({}, status_code=404)])
        with self.assertRaises(SteamApiError):
            await SteamClient(not_found_http, MemoryCache()).search_game_refs(
                search="Portal"
            )
        self.assertEqual(len(not_found_http.calls), 1)

    async def test_temporary_storefront_contract_failure_retries_once(self) -> None:
        contract_http = QueueHttpClient(
            [
                json_response({"success": 1}),
                json_response(storefront_payload()),
            ]
        )

        page = await SteamClient(
            contract_http,
            MemoryCache(),
        ).search_storefront_tag(19)

        self.assertEqual([hit.appid for hit in page.hits], [10])
        self.assertEqual(len(contract_http.calls), 2)

    async def test_storesearch_contract_failure_retries_once_and_is_not_cached(
        self,
    ) -> None:
        cache = MemoryCache()
        http_client = QueueHttpClient([json_response({}), json_response({})])
        client = SteamClient(http_client, cache)

        with self.assertRaises(SteamApiError):
            await client.search_game_refs(search="Portal")

        self.assertEqual(len(http_client.calls), 2)
        self.assertEqual(cache.payloads, {})


class ReferenceOutcomeRegressionTest(unittest.IsolatedAsyncioTestCase):
    async def test_outcome_retains_attempt_success_and_failure_evidence(self) -> None:
        outcome_type = getattr(steam_index_module, "ReferenceSearchOutcome", None)
        self.assertIsNotNone(outcome_type)
        service = SteamGameIndexService(
            TransportFailingReferenceClient(),
            MemoryCache(),
        )
        reference = ReferenceQuery(
            "Portal",
            ("Portal",),
            ReferencePolarity.POSITIVE,
        )

        outcome = await service._search_reference_group(reference, {})

        self.assertIsInstance(outcome, outcome_type)
        self.assertEqual(outcome.attempted, 2)
        self.assertEqual(outcome.succeeded, 0)
        self.assertEqual(len(outcome.failures), 2)

    async def test_all_transport_failures_differ_from_a_successful_empty_search(self) -> None:
        failed_preference = GamePreference(reference_games_like=["Portal"])
        empty_preference = GamePreference(reference_games_like=["Portal"])
        failed_service = SteamGameIndexService(
            TransportFailingReferenceClient(),
            MemoryCache(),
        )
        empty_service = SteamGameIndexService(
            EmptyReferenceClient(),
            MemoryCache(),
        )

        with self.assertLogs(steam_index_module.__name__, level="DEBUG") as logs:
            await failed_service.refresh_entries(failed_preference, [], target_pool=0)
            await empty_service.refresh_entries(empty_preference, [], target_pool=0)

        self.assertEqual(len(failed_preference.parse_warnings), 1)
        self.assertEqual(len(empty_preference.parse_warnings), 1)
        self.assertNotEqual(
            failed_preference.parse_warnings[0],
            empty_preference.parse_warnings[0],
        )
        self.assertTrue(any("status=transient_failure" in line for line in logs.output))
        self.assertTrue(any("status=no_hit" in line for line in logs.output))

    async def test_reference_only_transport_failure_is_unavailable_but_no_hit_is_empty(
        self,
    ) -> None:
        with self.assertRaises(RecallUnavailableError):
            await SteamGameIndexService(
                TransportFailingReferenceClient(),
                MemoryCache(),
            ).recommend(
                GamePreference(reference_games_like=["Portal"]),
                limit=3,
            )

        result = await SteamGameIndexService(
            EmptyReferenceClient(),
            MemoryCache(),
        ).recommend(
            GamePreference(reference_games_like=["Portal"]),
            limit=3,
        )

        self.assertEqual(result, [])

    async def test_reference_programming_error_is_not_reported_as_no_hit(self) -> None:
        with self.assertRaisesRegex(RuntimeError, "reference parser bug"):
            await SteamGameIndexService(
                ProgrammingFailingReferenceClient(),
                MemoryCache(),
            ).recommend(
                GamePreference(reference_games_like=["Portal"]),
                limit=3,
            )

    async def test_reference_prefers_storefront_and_maps_tag_ids_in_order(self) -> None:
        client = StorefrontReferenceClient()
        service = SteamGameIndexService(client, MemoryCache(), clock=lambda: 1_000.0)
        preference = GamePreference(
            reference_games_like=["黑暗之魂"],
            reference_search_terms=["Dark Souls"],
        )

        entries = await service.refresh_entries(preference, [], target_pool=1)

        self.assertEqual(preference.resolved_reference_games[0].appid, 10)
        self.assertEqual(entries[0].ordered_tags, ["action", "rpg"])
        self.assertEqual(client.storesearch_calls, 0)
        self.assertEqual(sorted(client.vocabulary_languages), ["english", "schinese"])

    async def test_irrelevant_storefront_hits_fall_back_to_storesearch(self) -> None:
        client = StorefrontFallbackClient()
        service = SteamGameIndexService(client, MemoryCache(), clock=lambda: 1_000.0)
        preference = GamePreference(reference_games_like=["Portal"])

        await service.refresh_entries(preference, [], target_pool=1)

        self.assertEqual(preference.resolved_reference_games[0].appid, 20)
        self.assertGreater(client.storesearch_calls, 0)

    async def test_failed_storefront_request_falls_back_to_storesearch(self) -> None:
        client = FailingStorefrontFallbackClient()
        service = SteamGameIndexService(client, MemoryCache(), clock=lambda: 1_000.0)
        preference = GamePreference(reference_games_like=["Portal"])

        await service.refresh_entries(preference, [], target_pool=1)

        self.assertEqual(preference.resolved_reference_games[0].appid, 20)

    async def test_cached_candidate_absorbs_ordered_storefront_tag_ids(self) -> None:
        client = StorefrontReferenceClient()
        cached = game(10, "DARK SOULS™: REMASTERED", ["RPG"])
        cache = MemoryCache({STEAM_INDEX_CACHE_KEY: current_snapshot_payload(cached)})
        service = SteamGameIndexService(client, cache, clock=lambda: 1_000.0)
        preference = GamePreference(reference_games_like=["Dark Souls"])

        entries = await service.refresh_entries(preference, [], target_pool=1)

        self.assertEqual(entries[0].ordered_tags, ["action", "rpg"])

    async def test_instance_tag_map_retains_every_id_for_same_canonical_tag(self) -> None:
        client = DuplicateCanonicalReferenceClient()
        service = SteamGameIndexService(client, MemoryCache(), clock=lambda: 1_000.0)
        preference = GamePreference(reference_games_like=["Content Test"])

        entries = await service.refresh_entries(preference, [], target_pool=1)

        self.assertEqual(entries[0].ordered_tags, ["violent"])

    async def test_same_appid_uses_richest_storefront_tag_evidence(self) -> None:
        client = DuplicateEvidenceReferenceClient()
        service = SteamGameIndexService(client, MemoryCache(), clock=lambda: 1_000.0)
        preference = GamePreference(reference_games_like=["Dark Souls"])

        entries = await service.refresh_entries(preference, [], target_pool=1)

        self.assertEqual(entries[0].ordered_tags, ["action", "rpg"])

    async def test_transient_detail_failure_is_not_reported_as_ambiguous(self) -> None:
        service = SteamGameIndexService(
            TransientDetailReferenceClient(),
            MemoryCache(),
            clock=lambda: 1_000.0,
        )
        preference = GamePreference(reference_games_like=["Portal"])

        with self.assertLogs(steam_index_module.__name__, level="DEBUG") as logs:
            await service.refresh_entries(preference, [], target_pool=1)

        self.assertEqual(len(preference.parse_warnings), 1)
        self.assertIn("暂时无法搜索", preference.parse_warnings[0])
        self.assertTrue(any("status=transient_failure" in line for line in logs.output))


class BootstrapAndMigrationRegressionTest(unittest.IsolatedAsyncioTestCase):
    async def test_bootstrap_is_single_flight_and_waiter_cancellation_is_shielded(self) -> None:
        cache = MemoryCache(
            {
                STEAM_INDEX_CACHE_KEY: {
                    "schema_version": 1,
                    "entries": [],
                    "search_coverage": {},
                }
            }
        )
        client = GatedVocabularyClient()
        service = SteamGameIndexService(client, cache, clock=lambda: 1_000.0)
        bootstrap = getattr(service, "bootstrap", None)
        self.assertTrue(callable(bootstrap))

        cancelled_waiter = asyncio.create_task(bootstrap())
        await asyncio.wait_for(client.started.wait(), timeout=1.0)
        surviving_waiter = asyncio.create_task(bootstrap())
        await asyncio.sleep(0)
        cancelled_waiter.cancel()
        with self.assertRaises(asyncio.CancelledError):
            await cancelled_waiter
        client.release.set()
        await surviving_waiter
        await bootstrap()

        self.assertEqual(sorted(client.languages), ["english", "schinese"])

    async def test_refresh_entries_refreshes_vocabulary_after_bootstrap_becomes_stale(self) -> None:
        now = [1_000.0]
        client = RefreshingVocabularyClient(lambda: now[0])
        cache = MemoryCache(
            {
                STEAM_INDEX_CACHE_KEY: {
                    "schema_version": 1,
                    "entries": [],
                    "search_coverage": {},
                }
            }
        )
        service = SteamGameIndexService(client, cache, clock=lambda: now[0])

        await service.bootstrap()
        now[0] += 24 * 60 * 60 + 1
        await service.refresh_entries(GamePreference(), [], target_pool=0)

        self.assertEqual(client.languages.count("english"), 2)
        self.assertEqual(client.languages.count("schinese"), 2)

    async def test_english_failure_does_not_promote_chinese_name_to_canonical(self) -> None:
        client = PartiallyFailingVocabularyClient(
            failed_language="english",
            tag_id=91,
            english_name="English Alpha Probe",
            chinese_name="仅中文甲探针",
        )
        service = SteamGameIndexService(client, MemoryCache(), clock=lambda: 1_000.0)

        self.assertFalse(await service.ensure_steam_tag_aliases())
        self.assertNotIn(91, service._canonical_tag_by_id)
        self.assertTrue(await service.ensure_steam_tag_aliases())

        self.assertEqual(client.languages, ["english", "schinese", "english"])
        self.assertEqual(service._canonical_tag_by_id[91], "english_alpha_probe")
        self.assertEqual(normalize_tag("仅中文甲探针"), "english_alpha_probe")

    async def test_chinese_failure_keeps_english_mapping_and_retries_only_chinese(self) -> None:
        client = PartiallyFailingVocabularyClient(
            failed_language="schinese",
            tag_id=92,
            english_name="English Beta Probe",
            chinese_name="仅中文乙探针",
        )
        service = SteamGameIndexService(client, MemoryCache(), clock=lambda: 1_000.0)

        self.assertTrue(await service.ensure_steam_tag_aliases())
        self.assertEqual(service._canonical_tag_by_id[92], "english_beta_probe")
        self.assertFalse(service._steam_tag_aliases_are_fresh())
        self.assertTrue(await service.ensure_steam_tag_aliases())

        self.assertEqual(client.languages, ["english", "schinese", "schinese"])
        self.assertEqual(service._canonical_tag_by_id[92], "english_beta_probe")
        self.assertEqual(normalize_tag("仅中文乙探针"), "english_beta_probe")

    async def test_v3_and_v4_snapshots_migrate_to_stable_schema_without_deletion(self) -> None:
        self.assertNotRegex(STEAM_INDEX_CACHE_KEY, r":v\d+$")
        for legacy_version in (3, 4):
            with self.subTest(legacy_version=legacy_version):
                legacy_key = f"steam_index:v{legacy_version}"
                legacy_payload = legacy_snapshot_payload(
                    legacy_version,
                    game(legacy_version, f"Legacy {legacy_version}", ["RPG"]),
                )
                cache = MemoryCache({legacy_key: legacy_payload})
                service = SteamGameIndexService(
                    VocabularyClient(),
                    cache,
                    clock=lambda: 1_000.0,
                )
                bootstrap = getattr(service, "bootstrap", None)
                self.assertTrue(callable(bootstrap))

                await bootstrap()
                snapshot = await service.load_snapshot()

                self.assertEqual([entry.candidate.appid for entry in snapshot.entries], [legacy_version])
                self.assertTrue(snapshot.entries[0].needs_revalidation)
                self.assertEqual(
                    cache.payloads[STEAM_INDEX_CACHE_KEY]["schema_version"],
                    STEAM_INDEX_SCHEMA_VERSION,
                )
                self.assertIs(cache.payloads[legacy_key], legacy_payload)

    async def test_v1_snapshot_migrates_to_v2_and_revalidates_preserved_candidate(
        self,
    ) -> None:
        previous = game(98, "Previous Stable RPG", ["RPG"])
        cache = MemoryCache(
            {
                STEAM_INDEX_CACHE_KEY: {
                    "schema_version": 1,
                    "entries": [
                        {
                            "candidate": dump_model(previous),
                            "refreshed_at": 900.0,
                            "needs_revalidation": False,
                        }
                    ],
                    "search_coverage": {"rpg": 900.0},
                }
            }
        )
        client = PreviousSnapshotValidationClient()
        service = SteamGameIndexService(client, cache, clock=lambda: 1_000.0)

        migrated = await service.load_snapshot()

        self.assertEqual([entry.candidate.appid for entry in migrated.entries], [98])
        self.assertTrue(migrated.entries[0].needs_revalidation)
        self.assertEqual(migrated.search_coverage, {})
        self.assertEqual(
            cache.payloads[STEAM_INDEX_CACHE_KEY]["schema_version"],
            STEAM_INDEX_SCHEMA_VERSION,
        )

        entries = await service.refresh_entries(
            GamePreference(genres_like=["RPG"]),
            [],
            target_pool=1,
            snapshot=migrated,
        )
        persisted = await service.load_snapshot()

        self.assertEqual([candidate.appid for candidate in entries], [98])
        self.assertEqual(client.detail_appids, [98])
        self.assertEqual(client.detail_bypass_cache, [True])
        self.assertFalse(persisted.entries[0].needs_revalidation)

    async def test_migrated_candidate_is_revalidated_before_reference_resolution(self) -> None:
        legacy = game(99, "Legacy Seed", ["RPG"])
        cache = MemoryCache(
            {"steam_index:v4": legacy_snapshot_payload(4, legacy)}
        )
        client = LegacyReferenceClient()
        service = SteamGameIndexService(client, cache, clock=lambda: 1_000.0)
        preference = GamePreference(reference_games_like=["Legacy Seed"])

        await service.refresh_entries(preference, [], target_pool=1)

        self.assertEqual(client.detail_appids, [99])
        self.assertEqual(client.detail_bypass_cache, [True])
        self.assertEqual(preference.resolved_reference_games[0].appid, 99)

    async def test_non_reference_refresh_bypasses_detail_cache_for_legacy_hit(self) -> None:
        legacy = game(99, "Legacy Seed", ["RPG"])
        cache = MemoryCache(
            {"steam_index:v4": legacy_snapshot_payload(4, legacy)}
        )
        client = LegacyReferenceClient()
        service = SteamGameIndexService(client, cache, clock=lambda: 1_000.0)

        entries = await service.refresh_entries(
            GamePreference(genres_like=["RPG"]),
            [],
            target_pool=1,
        )
        persisted = await service.load_snapshot()

        self.assertEqual([candidate.appid for candidate in entries], [99])
        self.assertEqual(client.detail_bypass_cache, [True])
        self.assertFalse(persisted.entries[0].needs_revalidation)

    async def test_revalidated_record_replaces_legacy_with_future_timestamp(self) -> None:
        legacy = game(99, "Legacy Seed", ["RPG"])
        cache = MemoryCache(
            {
                "steam_index:v4": legacy_snapshot_payload(
                    4,
                    legacy,
                    refreshed_at=2_000.0,
                )
            }
        )
        client = LegacyReferenceClient()
        service = SteamGameIndexService(client, cache, clock=lambda: 1_000.0)

        await service.refresh_entries(
            GamePreference(reference_games_like=["Legacy Seed"]),
            [],
            target_pool=1,
        )
        persisted = await service.load_snapshot()

        self.assertFalse(persisted.entries[0].needs_revalidation)

    def test_boolean_schema_version_is_not_a_current_snapshot(self) -> None:
        payload = current_snapshot_payload(game(1, "Bool Schema", ["RPG"]))
        payload["schema_version"] = True

        self.assertEqual(steam_index_module.parse_snapshot(payload).entries, [])

    async def test_migrated_candidate_cannot_be_final_output_when_revalidation_fails(self) -> None:
        legacy = game(99, "Legacy RPG", ["RPG"])
        cache = MemoryCache(
            {"steam_index:v4": legacy_snapshot_payload(4, legacy)}
        )
        client = FailingLegacyValidationClient()
        service = SteamGameIndexService(client, cache, clock=lambda: 1_000.0)

        with self.assertRaises(RecallUnavailableError):
            await service.recommend(
                GamePreference(genres_like=["RPG"]),
                limit=3,
            )

        self.assertEqual(client.detail_appids, [99])

    async def test_old_current_upcoming_candidate_is_revalidated_before_output(
        self,
    ) -> None:
        upcoming = game(77, "Transition RPG", ["RPG"])
        upcoming.coming_soon = True
        cache = MemoryCache(
            {STEAM_INDEX_CACHE_KEY: current_snapshot_payload(upcoming)}
        )
        client = ReleaseTransitionClient()
        service = SteamGameIndexService(client, cache, clock=lambda: 10_000.0)

        ranked = await service.recommend(
            GamePreference(genres_like=["RPG"]),
            limit=1,
        )

        self.assertEqual(client.detail_calls, [(77, True)])
        self.assertEqual([item.appid for item in ranked], [77])
        self.assertFalse(ranked[0].coming_soon)

    async def test_old_current_upcoming_reference_is_revalidated_before_resolution(
        self,
    ) -> None:
        upcoming = game(77, "Transition RPG", ["RPG"])
        upcoming.coming_soon = True
        cache = MemoryCache(
            {STEAM_INDEX_CACHE_KEY: current_snapshot_payload(upcoming)}
        )
        client = ReleaseTransitionClient()
        service = SteamGameIndexService(client, cache, clock=lambda: 10_000.0)
        preference = GamePreference(reference_games_like=["Transition RPG"])

        await service.refresh_entries(preference, [], target_pool=1)

        self.assertEqual(client.detail_calls, [(77, True)])
        self.assertEqual(preference.resolved_reference_games[0].appid, 77)


class MemoryCache:
    def __init__(self, payloads: dict[str, Any] | None = None) -> None:
        self.payloads = dict(payloads or {})
        self.read_keys: list[str] = []

    async def get_json(self, key: str, _ttl_hours: int) -> Any | None:
        self.read_keys.append(key)
        return self.payloads.get(key)

    async def set_json(self, key: str, payload: Any) -> None:
        self.payloads[key] = payload


class QueueHttpClient:
    def __init__(self, outcomes: list[Any]) -> None:
        self.outcomes = list(outcomes)
        self.calls: list[dict[str, Any]] = []

    async def get(self, url: str, params: dict[str, Any], **kwargs: Any) -> Any:
        self.calls.append({"url": url, "params": dict(params), "kwargs": dict(kwargs)})
        outcome = self.outcomes.pop(0)
        if isinstance(outcome, BaseException):
            raise outcome
        return outcome


def json_response(
    payload: Any,
    *,
    status_code: int = 200,
    headers: dict[str, str] | None = None,
) -> httpx.Response:
    request = httpx.Request("GET", "https://store.steampowered.com/api/storesearch/")
    return httpx.Response(
        status_code,
        request=request,
        json=payload,
        headers=headers,
    )


def text_response(
    text: str,
    *,
    url: str,
    redirected: bool = False,
) -> httpx.Response:
    request = httpx.Request("GET", url)
    history = (
        [httpx.Response(302, request=request, headers={"Location": url})]
        if redirected
        else None
    )
    return httpx.Response(200, request=request, text=text, history=history)


def storefront_payload() -> dict[str, Any]:
    return {
        "success": 1,
        "results_html": (
            '<a class="search_result_row" data-ds-appid="10" '
            'data-ds-tagids="[2, 1]">'
            '<span class="title">DARK SOULS™: REMASTERED</span></a>'
        ),
        "total_count": 1,
        "start": 0,
    }


def game(appid: int, title: str, tags: list[str]) -> GameCandidate:
    return GameCandidate(
        appid=appid,
        title=title,
        app_type="game",
        platforms=["PC"],
        tags=tags,
        stores=["Steam"],
        raw_url=f"https://store.steampowered.com/app/{appid}/",
    )


def dump_model(model: Any) -> dict[str, Any]:
    dumper = getattr(model, "model_dump", None)
    return dumper() if dumper else model.dict()


def legacy_snapshot_payload(
    version: int,
    candidate: GameCandidate,
    *,
    refreshed_at: float = 900.0,
) -> dict[str, Any]:
    return {
        "version": version,
        "entries": [{"candidate": dump_model(candidate), "refreshed_at": refreshed_at}],
        "search_coverage": {"rpg": 900.0},
    }


def current_snapshot_payload(candidate: GameCandidate) -> dict[str, Any]:
    return {
        "schema_version": STEAM_INDEX_SCHEMA_VERSION,
        "entries": [
            {
                "candidate": dump_model(candidate),
                "refreshed_at": 900.0,
                "needs_revalidation": False,
            }
        ],
        "search_coverage": {},
    }


class TransportFailingReferenceClient:
    language = "schinese"

    async def search_game_refs(self, **_kwargs: Any) -> list[SteamSearchHit]:
        request = httpx.Request("GET", "https://store.steampowered.com/api/storesearch/")
        raise httpx.ConnectError("offline", request=request)


class EmptyReferenceClient:
    language = "schinese"

    async def search_game_refs(self, **_kwargs: Any) -> list[SteamSearchHit]:
        return []


class ProgrammingFailingReferenceClient:
    language = "schinese"

    async def search_game_refs(self, **_kwargs: Any) -> list[SteamSearchHit]:
        raise RuntimeError("reference parser bug")


class VocabularyClient:
    async def get_popular_tags_snapshot(self, language: str = "english") -> Any:
        names = {
            "english": [(1, "RPG"), (2, "Action")],
            "schinese": [(1, "角色扮演"), (2, "动作")],
        }[language]
        return SimpleNamespace(
            tags=tuple({"tagid": tag_id, "name": name} for tag_id, name in names),
            fetched_at=1_000.0,
        )


class StorefrontReferenceClient(VocabularyClient):
    language = "schinese"

    def __init__(self) -> None:
        self.vocabulary_languages: list[str] = []
        self.storesearch_calls = 0

    async def get_popular_tags_snapshot(self, language: str = "english") -> Any:
        self.vocabulary_languages.append(language)
        return await super().get_popular_tags_snapshot(language)

    async def search_storefront_term(self, _term: str, **_kwargs: Any) -> SteamStorefrontPage:
        return SteamStorefrontPage(
            hits=(
                SteamSearchHit(
                    appid=10,
                    title="DARK SOULS™: REMASTERED",
                    tag_ids=[2, 1],
                ),
            ),
            total_count=1,
            start=0,
        )

    async def search_game_refs(self, **_kwargs: Any) -> list[SteamSearchHit]:
        self.storesearch_calls += 1
        return []

    async def get_game_detail(self, appid: int) -> GameCandidate:
        return game(appid, "DARK SOULS™: REMASTERED", ["RPG"])

    async def get_store_page_tags(self, _appid: int) -> list[str]:
        return []


class DuplicateCanonicalReferenceClient(StorefrontReferenceClient):
    language = "english"

    async def get_popular_tags_snapshot(self, language: str = "english") -> Any:
        self.vocabulary_languages.append(language)
        return SimpleNamespace(
            tags=(
                {"tagid": 1, "name": "Violent"},
                {"tagid": 2, "name": "Gore"},
            ),
            fetched_at=1_000.0,
        )

    async def search_storefront_term(self, _term: str, **_kwargs: Any) -> SteamStorefrontPage:
        return SteamStorefrontPage(
            hits=(
                SteamSearchHit(
                    appid=30,
                    title="Content Test",
                    tag_ids=[1],
                ),
            ),
            total_count=1,
            start=0,
        )

    async def get_game_detail(self, appid: int) -> GameCandidate:
        return game(appid, "Content Test", ["Action"])


class DuplicateEvidenceReferenceClient(StorefrontReferenceClient):
    async def search_storefront_term(self, _term: str, **_kwargs: Any) -> SteamStorefrontPage:
        return SteamStorefrontPage(
            hits=(
                SteamSearchHit(appid=10, title="DARK SOULS™: REMASTERED"),
                SteamSearchHit(
                    appid=10,
                    title="DARK SOULS™: REMASTERED",
                    tag_ids=[2, 1],
                ),
            ),
            total_count=2,
            start=0,
        )


class StorefrontFallbackClient(VocabularyClient):
    language = "english"

    def __init__(self) -> None:
        self.storesearch_calls = 0

    async def search_storefront_term(self, _term: str, **_kwargs: Any) -> SteamStorefrontPage:
        return SteamStorefrontPage(
            hits=(SteamSearchHit(appid=21, title="Unrelated Game"),),
            total_count=1,
            start=0,
        )

    async def search_game_refs(self, **_kwargs: Any) -> list[SteamSearchHit]:
        self.storesearch_calls += 1
        return [SteamSearchHit(appid=20, title="Portal")]

    async def get_game_detail(self, appid: int) -> GameCandidate:
        return game(appid, "Portal", ["Puzzle"])

    async def get_store_page_tags(self, _appid: int) -> list[str]:
        return []


class TransientDetailReferenceClient(StorefrontFallbackClient):
    async def get_game_detail(self, _appid: int) -> GameCandidate:
        raise SteamTransientError("detail temporarily unavailable")


class FailingStorefrontFallbackClient(StorefrontFallbackClient):
    async def search_storefront_term(self, _term: str, **_kwargs: Any) -> SteamStorefrontPage:
        request = httpx.Request("GET", "https://store.steampowered.com/search/results")
        raise httpx.ConnectError("offline", request=request)


class GatedVocabularyClient(VocabularyClient):
    def __init__(self) -> None:
        self.languages: list[str] = []
        self.started = asyncio.Event()
        self.release = asyncio.Event()

    async def get_popular_tags_snapshot(self, language: str = "english") -> Any:
        self.languages.append(language)
        self.started.set()
        await self.release.wait()
        return await super().get_popular_tags_snapshot(language)


class RefreshingVocabularyClient(VocabularyClient):
    def __init__(self, clock: Any) -> None:
        self.clock = clock
        self.languages: list[str] = []

    async def get_popular_tags_snapshot(self, language: str = "english") -> Any:
        self.languages.append(language)
        payload = await super().get_popular_tags_snapshot(language)
        return SimpleNamespace(tags=payload.tags, fetched_at=self.clock())

    async def search_game_refs(self, **_kwargs: Any) -> list[SteamSearchHit]:
        return []


class PartiallyFailingVocabularyClient:
    def __init__(
        self,
        *,
        failed_language: str,
        tag_id: int,
        english_name: str,
        chinese_name: str,
    ) -> None:
        self.failed_language = failed_language
        self.tag_id = tag_id
        self.names = {
            "english": english_name,
            "schinese": chinese_name,
        }
        self.languages: list[str] = []

    async def get_popular_tags_snapshot(self, language: str = "english") -> Any:
        self.languages.append(language)
        if language == self.failed_language and self.languages.count(language) == 1:
            raise RuntimeError(f"{language} vocabulary temporarily unavailable")
        return SimpleNamespace(
            tags=({"tagid": self.tag_id, "name": self.names[language]},),
            fetched_at=1_000.0,
        )


class LegacyReferenceClient(VocabularyClient):
    language = "english"

    def __init__(self) -> None:
        self.detail_appids: list[int] = []
        self.detail_bypass_cache: list[bool] = []

    async def search_game_refs(self, **_kwargs: Any) -> list[SteamSearchHit]:
        return [SteamSearchHit(appid=99, title="Legacy Seed")]

    async def get_game_detail(
        self,
        appid: int,
        bypass_cache: bool = False,
    ) -> GameCandidate:
        self.detail_appids.append(appid)
        self.detail_bypass_cache.append(bypass_cache)
        return game(appid, "Legacy Seed", ["RPG"])

    async def get_store_page_tags(self, _appid: int) -> list[str]:
        return []


class PreviousSnapshotValidationClient(LegacyReferenceClient):
    async def search_game_refs(self, **_kwargs: Any) -> list[SteamSearchHit]:
        return [SteamSearchHit(appid=98, title="Previous Stable RPG")]


class FailingLegacyValidationClient(VocabularyClient):
    language = "english"

    def __init__(self) -> None:
        self.detail_appids: list[int] = []

    async def search_game_refs(self, **_kwargs: Any) -> list[SteamSearchHit]:
        return []

    async def get_game_detail(self, appid: int) -> GameCandidate:
        self.detail_appids.append(appid)
        raise SteamApiError("detail unavailable")


class ReleaseTransitionClient(VocabularyClient):
    def __init__(self) -> None:
        self.detail_calls: list[tuple[int, bool]] = []

    async def search_storefront_tag(
        self,
        tag_id: int,
        page_size: int = 20,
    ) -> SteamStorefrontPage:
        del tag_id, page_size
        return SteamStorefrontPage(
            hits=(SteamSearchHit(appid=77, title="Transition RPG", tag_ids=[1]),),
            total_count=1,
            start=0,
        )

    async def get_game_detail(
        self,
        appid: int,
        bypass_cache: bool = False,
    ) -> GameCandidate:
        self.detail_calls.append((appid, bypass_cache))
        return game(appid, "Transition RPG", ["RPG"])

    async def get_store_page_tags(self, _appid: int) -> list[str]:
        return ["RPG"]

    async def get_review_summary(self, _appid: int) -> Any:
        return SimpleNamespace(
            total_reviews=500,
            positive_ratio=0.9,
            recent_positive_ratio=0.9,
        )


if __name__ == "__main__":
    unittest.main()
