from __future__ import annotations

import unittest

from astrbot_plugin_steam_game_recommender.clients.steam import (
    SteamStorefrontPage,
    SteamTransientError,
)
from astrbot_plugin_steam_game_recommender.services.steam_index import (
    STEAM_INDEX_SCHEMA_VERSION,
    RecallSourceFetch,
    SteamGameIndexService,
    SteamIndexEntry,
    SteamIndexSnapshot,
    successful_source_fetch,
)
from astrbot_plugin_steam_game_recommender.services.steam_recall import (
    RRF_K,
    CandidateHit,
    CandidateSourceHit,
    RecallSource,
    RecallSourceStatus,
    merge_candidate_sources,
)
from astrbot_plugin_steam_game_recommender.storage.models import (
    CompanyPreference,
    GameCandidate,
    GamePreference,
    SteamSearchHit,
)


def preference() -> CompanyPreference:
    return CompanyPreference(
        display_name="Acme Games",
        aliases=["Acme Games", "Acme Interactive"],
        role="either",
        strength="preferred",
        source_span="Acme Games",
    )


def storefront_page(*appids: int) -> SteamStorefrontPage:
    return SteamStorefrontPage(
        hits=tuple(
            SteamSearchHit(appid=appid, title=f"Game {appid}") for appid in appids
        ),
        total_count=len(appids),
        start=0,
    )


class MemoryCache:
    def __init__(self, payload=None) -> None:
        self.payload = payload
        self.saved = []

    async def get_json(self, _key: str, _ttl_hours: int):
        return self.payload

    async def set_json(self, _key: str, payload) -> None:
        self.payload = payload
        self.saved.append(payload)


class CompanySteamClient:
    def __init__(self) -> None:
        self.company_calls: list[tuple[str, str, int]] = []
        self.detail_calls: list[tuple[int, bool]] = []

    async def search_storefront_company(
        self,
        alias: str,
        role: str,
        page_size: int = 20,
    ) -> SteamStorefrontPage:
        self.company_calls.append((alias, role, page_size))
        pages = {
            ("Acme Games", "developer"): storefront_page(1, 2, 4),
            ("Acme Games", "publisher"): storefront_page(3, 2),
            ("Acme Interactive", "developer"): storefront_page(3, 4),
            ("Acme Interactive", "publisher"): storefront_page(4),
        }
        return pages[(alias, role)]

    async def get_game_detail(self, appid: int, bypass_cache: bool = False) -> GameCandidate:
        self.detail_calls.append((appid, bypass_cache))
        matching = appid in {1, 3, 4}
        return GameCandidate(
            appid=appid,
            title=f"Game {appid}",
            app_type="game",
            developers=["Acme Games Ltd."] if matching else ["Different Studio"],
            publishers=["Acme Interactive LLC"] if matching else ["Other Publisher"],
            developer_data_available=True,
            publisher_data_available=True,
            company_data_available=True,
            internal_source_markers=["steam_appdetails"],
        )


class PartiallyFailingCompanySteamClient(CompanySteamClient):
    async def search_storefront_company(
        self,
        alias: str,
        role: str,
        page_size: int = 20,
    ) -> SteamStorefrontPage:
        if (alias, role) == ("Acme Interactive", "publisher"):
            self.company_calls.append((alias, role, page_size))
            raise SteamTransientError("one company alias failed")
        return await super().search_storefront_company(alias, role, page_size)


class SparseCompanySteamClient(CompanySteamClient):
    def __init__(self) -> None:
        super().__init__()
        self.top_calls: list[int] = []

    async def search_storefront_company(
        self,
        alias: str,
        role: str,
        page_size: int = 20,
    ) -> SteamStorefrontPage:
        self.company_calls.append((alias, role, page_size))
        return storefront_page(1)

    async def browse_top_sellers(self, page_size: int = 60) -> SteamStorefrontPage:
        self.top_calls.append(page_size)
        return storefront_page(2, 5)

    async def search_game_refs(self, **_kwargs) -> list[SteamSearchHit]:
        return []


class ProgressiveCompanyRankClient:
    def __init__(self, match_from: int = 61) -> None:
        self.match_from = int(match_from)

    async def get_game_detail(
        self,
        appid: int,
        bypass_cache: bool = False,
    ) -> GameCandidate:
        matches = appid >= self.match_from
        return GameCandidate(
            appid=appid,
            title=f"Game {appid}",
            app_type="game",
            developers=["Acme Games Ltd."] if matches else ["Different Studio"],
            publishers=["Other Publisher"],
            developer_data_available=True,
            publisher_data_available=True,
            company_data_available=True,
            internal_source_markers=["steam_appdetails"],
        )


class ProgressiveCompanyRankService(SteamGameIndexService):
    def __init__(self, match_from: int = 61) -> None:
        super().__init__(ProgressiveCompanyRankClient(match_from), MemoryCache())
        self.validation_batches: list[tuple[int, ...]] = []

    def _source_fetch(
        self,
        source_id: str,
        source_kind: str,
        weight: float,
    ) -> RecallSourceFetch:
        candidates = tuple(
            GameCandidate(appid=appid, title=f"Game {appid}")
            for appid in range(1, 101)
        )
        return successful_source_fetch(
            RecallSource(
                source_id=source_id,
                source_kind=source_kind,
                source_tag=None,
                candidates=candidates,
                weight=weight,
                candidate_ranks=tuple(range(1, 101)),
            ),
            total_count=100,
        )

    async def _fetch_company_source(
        self,
        _preference: CompanyPreference,
        *,
        source_index: int,
    ) -> RecallSourceFetch:
        return self._source_fetch(f"company:{source_index}", "company", 1.0)

    async def _fetch_top_sellers(self) -> RecallSourceFetch:
        return self._source_fetch("top_seller", "top_seller", 0.5)

    async def _validate_recall_hits(self, hits, records, prefetched, **kwargs):
        self.validation_batches.append(
            tuple(int(hit.candidate.appid or 0) for hit in hits)
        )
        return await super()._validate_recall_hits(
            hits,
            records,
            prefetched,
            **kwargs,
        )


class CompanyRrfTest(unittest.TestCase):
    def test_explicit_source_ranks_preserve_multiple_alias_rank_ones(self) -> None:
        source = RecallSource(
            source_id="company:0",
            source_kind="company",
            source_tag=None,
            candidates=(
                GameCandidate(appid=1, title="One"),
                GameCandidate(appid=2, title="Two"),
                GameCandidate(appid=3, title="Three"),
            ),
            weight=1.0,
            candidate_ranks=(1, 1, 5),
        )

        result = merge_candidate_sources([source])
        by_appid = {hit.candidate.appid: hit for hit in result.hits}

        self.assertEqual(by_appid[1].source_rank, 1)
        self.assertEqual(by_appid[2].source_rank, 1)
        self.assertEqual(by_appid[3].source_rank, 5)
        self.assertEqual(by_appid[1].rrf_score, 1.0 / (RRF_K + 1))
        self.assertEqual(by_appid[2].rrf_score, 1.0 / (RRF_K + 1))


class CompanyRecallServiceTest(unittest.IsolatedAsyncioTestCase):
    async def test_alias_and_role_queries_form_one_best_rank_source(self) -> None:
        client = CompanySteamClient()
        service = SteamGameIndexService(client, MemoryCache())

        fetch = await service._fetch_company_source(preference(), source_index=0)

        self.assertEqual(
            client.company_calls,
            [
                ("Acme Games", "developer", 20),
                ("Acme Games", "publisher", 20),
                ("Acme Interactive", "developer", 20),
                ("Acme Interactive", "publisher", 20),
            ],
        )
        self.assertEqual(fetch.source.source_id, "company:0")
        self.assertEqual(fetch.source.weight, 1.0)
        self.assertEqual(
            list(
                zip(
                    [item.appid for item in fetch.source.candidates],
                    fetch.source.candidate_ranks,
                    strict=True,
                )
            ),
            [(1, 1), (3, 1), (4, 1), (2, 2)],
        )

    async def test_partial_company_source_failure_keeps_hits_and_marks_health_stale(self) -> None:
        client = PartiallyFailingCompanySteamClient()
        service = SteamGameIndexService(client, MemoryCache())

        fetch = await service._fetch_company_source(preference(), source_index=0)

        self.assertEqual(fetch.health.status, RecallSourceStatus.STALE)
        self.assertTrue(fetch.health.critical)
        self.assertEqual([item.appid for item in fetch.source.candidates], [1, 3, 2, 4])
        self.assertEqual(fetch.source.candidate_ranks, (1, 1, 2, 2))

    async def test_company_aliases_are_capped_before_role_queries(self) -> None:
        client = SparseCompanySteamClient()
        service = SteamGameIndexService(client, MemoryCache())
        company = CompanyPreference(
            display_name="Acme Games",
            aliases=[f"Alias {index}" for index in range(200)],
            role="either",
            source_span="Acme Games",
        )

        await service._fetch_company_source(company, source_index=0)

        self.assertLessEqual(len(company.aliases), 5)
        self.assertLessEqual(len(client.company_calls), 10)

    async def test_progressive_company_validation_globally_reranks_after_provenance_removal(
        self,
    ) -> None:
        service = ProgressiveCompanyRankService()
        game_preference = GamePreference(
            company_preferences=[preference()],
            quality_intent="mainstream",
        )

        recall, _references, _intent = await service._recall_specific_candidates(
            game_preference,
            SteamIndexSnapshot(),
            requested_limit=5,
        )

        self.assertEqual(
            service.validation_batches,
            [tuple(range(1, 61)), tuple(range(61, 101))],
        )
        self.assertEqual(
            [hit.candidate.appid for hit in recall.hits],
            [*range(61, 101), *range(1, 61)],
        )
        self.assertEqual(
            [hit.retrieval_rank for hit in recall.hits],
            list(range(1, 101)),
        )
        by_appid = {int(hit.candidate.appid or 0): hit for hit in recall.hits}
        self.assertEqual(
            [source.source_id for source in by_appid[1].source_hits],
            ["top_seller"],
        )
        self.assertEqual(
            [source.source_id for source in by_appid[61].source_hits],
            ["company:0", "top_seller"],
        )
        self.assertAlmostEqual(by_appid[1].rrf_score, 0.5 / (RRF_K + 1))
        self.assertAlmostEqual(by_appid[61].rrf_score, 1.5 / (RRF_K + 61))

    async def test_progressive_company_validation_stops_when_first_batch_has_enough_matches(
        self,
    ) -> None:
        service = ProgressiveCompanyRankService(match_from=1)
        game_preference = GamePreference(
            company_preferences=[preference()],
            quality_intent="mainstream",
        )

        recall, _references, _intent = await service._recall_specific_candidates(
            game_preference,
            SteamIndexSnapshot(),
            requested_limit=5,
        )

        self.assertEqual(service.validation_batches, [tuple(range(1, 61))])
        self.assertEqual(len(recall.hits), 60)

    async def test_appdetails_removes_false_positive_company_provenance_and_refreshes_old_metadata(
        self,
    ) -> None:
        client = CompanySteamClient()
        service = SteamGameIndexService(client, MemoryCache())
        company_hit = CandidateSourceHit("company:0", "company", None, 1, 1.0)
        index_hit = CandidateSourceHit("index", "index", None, 1, 0.35)
        hits = (
            CandidateHit(GameCandidate(appid=1, title="One"), (company_hit,), 1 / 61, 1),
            CandidateHit(
                GameCandidate(appid=2, title="Two"),
                (company_hit, index_hit),
                1 / 61 + 0.35 / 61,
                2,
            ),
        )
        old = GameCandidate(
            appid=1,
            title="Old One",
            app_type="game",
            internal_source_markers=["steam_appdetails"],
        )
        records = {"appid:1": SteamIndexEntry(old, refreshed_at=1.0)}

        validated = await service._validate_recall_hits(
            hits,
            records,
            {},
            company_preferences_by_source={"company:0": preference()},
        )

        self.assertEqual([hit.candidate.appid for hit in validated.hits], [1, 2])
        refreshed_match = validated.hits[0]
        false_positive = validated.hits[1]
        self.assertEqual(refreshed_match.source_hits, (company_hit,))
        self.assertEqual(false_positive.source_hits, (index_hit,))
        self.assertAlmostEqual(false_positive.rrf_score, 0.35 / 61)
        self.assertIn((1, True), client.detail_calls)

    async def test_cached_company_metadata_is_refreshed_for_the_requested_role(self) -> None:
        client = CompanySteamClient()
        service = SteamGameIndexService(client, MemoryCache(), clock=lambda: 200.0)
        publisher_preference = CompanyPreference(
            display_name="Acme Interactive",
            role="publisher",
            source_span="Acme Interactive",
        )
        either_preference = preference()
        publisher_hit = CandidateSourceHit(
            "company:publisher", "company", None, 1, 1.0
        )
        either_hit = CandidateSourceHit("company:either", "company", None, 1, 1.0)
        hits = tuple(
            CandidateHit(
                GameCandidate(appid=appid, title=f"Game {appid}"),
                (source,),
                1 / (RRF_K + 1),
                appid,
            )
            for appid, source in (
                (1, publisher_hit),
                (3, either_hit),
                (4, either_hit),
            )
        )
        records = {
            "appid:1": SteamIndexEntry(
                GameCandidate(
                    appid=1,
                    title="Cached Publisher Candidate",
                    app_type="game",
                    developers=["Different Studio"],
                    developer_data_available=True,
                    publisher_data_available=False,
                    company_data_available=True,
                    internal_source_markers=["steam_appdetails"],
                ),
                refreshed_at=100.0,
            ),
            "appid:3": SteamIndexEntry(
                GameCandidate(
                    appid=3,
                    title="Cached Either Candidate",
                    app_type="game",
                    developers=["Different Studio"],
                    developer_data_available=True,
                    publisher_data_available=False,
                    company_data_available=True,
                    internal_source_markers=["steam_appdetails"],
                ),
                refreshed_at=100.0,
            ),
            "appid:4": SteamIndexEntry(
                GameCandidate(
                    appid=4,
                    title="Cached Matching Either Candidate",
                    app_type="game",
                    developers=["Acme Games Ltd."],
                    developer_data_available=True,
                    publisher_data_available=False,
                    company_data_available=True,
                    internal_source_markers=["steam_appdetails"],
                ),
                refreshed_at=100.0,
            ),
        }

        validated = await service._validate_recall_hits(
            hits,
            records,
            {},
            company_preferences_by_source={
                "company:publisher": publisher_preference,
                "company:either": either_preference,
            },
        )

        self.assertEqual(
            [hit.candidate.appid for hit in validated.hits],
            [1, 3, 4],
        )
        self.assertEqual(client.detail_calls, [(1, True), (3, True)])

    async def test_company_only_cold_start_uses_verified_company_source(self) -> None:
        client = SparseCompanySteamClient()
        cache = MemoryCache()
        service = SteamGameIndexService(client, cache)
        game_preference = GamePreference(company_preferences=[preference()])

        recall, _references, _intent = await service._recall_specific_candidates(
            game_preference,
            SteamIndexSnapshot(),
            requested_limit=2,
        )

        self.assertEqual([hit.candidate.appid for hit in recall.hits], [1, 2, 5])
        self.assertEqual(client.top_calls, [60])

    async def test_unresolved_reference_does_not_skip_company_recall(self) -> None:
        client = SparseCompanySteamClient()
        service = SteamGameIndexService(client, MemoryCache())
        game_preference = GamePreference(
            reference_games_like=["Missing Reference"],
            company_preferences=[preference()],
        )

        recall, _references, _intent = await service._recall_specific_candidates(
            game_preference,
            SteamIndexSnapshot(),
            requested_limit=2,
        )

        self.assertTrue(client.company_calls)
        self.assertEqual(client.top_calls, [60])
        self.assertEqual([hit.candidate.appid for hit in recall.hits], [1, 2, 5])

    async def test_v2_snapshot_forces_bypass_cache_before_company_metadata_is_trusted(self) -> None:
        legacy = GameCandidate(
            appid=1,
            title="Legacy One",
            app_type="game",
            internal_source_markers=["steam_appdetails"],
        )
        dump = getattr(legacy, "model_dump", None)
        cache = MemoryCache(
            {
                "schema_version": 2,
                "entries": [
                    {
                        "candidate": dump() if dump else legacy.dict(),
                        "refreshed_at": 1.0,
                    }
                ],
                "search_coverage": {"legacy": 1.0},
            }
        )
        client = CompanySteamClient()
        service = SteamGameIndexService(client, cache)

        snapshot = await service.load_snapshot()
        record = snapshot.entries[0]
        index_hit = CandidateSourceHit("index", "index", None, 1, 0.35)
        validated = await service._validate_recall_hits(
            (
                CandidateHit(
                    record.candidate,
                    (index_hit,),
                    0.35 / (RRF_K + 1),
                    1,
                ),
            ),
            {"appid:1": record},
            {},
        )

        self.assertTrue(record.needs_revalidation)
        self.assertEqual(snapshot.search_coverage, {})
        self.assertEqual(cache.saved[-1]["schema_version"], STEAM_INDEX_SCHEMA_VERSION)
        self.assertTrue(cache.saved[-1]["entries"][0]["needs_revalidation"])
        self.assertEqual(client.detail_calls, [(1, True)])
        self.assertEqual([hit.candidate.appid for hit in validated.hits], [1])


if __name__ == "__main__":
    unittest.main()
