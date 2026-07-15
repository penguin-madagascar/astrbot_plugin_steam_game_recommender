from __future__ import annotations

import unittest
from types import SimpleNamespace
from typing import Any

from astrbot_plugin_steam_game_recommender.clients.steam import (
    SteamMoreLikeResult,
    SteamStorefrontPage,
    SteamTransientError,
)
from astrbot_plugin_steam_game_recommender.services.recommendation_intent import (
    ReferencePolarity,
    ReferenceQuery,
)
from astrbot_plugin_steam_game_recommender.services.steam_index import (
    STEAM_INDEX_CACHE_KEY,
    STEAM_INDEX_SCHEMA_VERSION,
    SteamGameIndexService,
    callable_accepts_keyword,
)
from astrbot_plugin_steam_game_recommender.storage.models import (
    CompanyPreference,
    GameCandidate,
    GamePreference,
    SteamSearchHit,
)


class SteamIndexDiscoveryPolicyTest(unittest.IsolatedAsyncioTestCase):
    async def test_false_policy_reaches_every_discovery_source_on_sequential_calls(
        self,
    ) -> None:
        client = RecordingDiscoveryClient()
        service = SteamGameIndexService(client, PolicyCache(), reuse_cache=False)
        reference = ReferenceQuery(
            display_title="Portal",
            aliases=("Portal",),
            polarity=ReferencePolarity.POSITIVE,
        )
        company = CompanyPreference(
            display_name="Valve",
            aliases=[],
            role="developer",
            strength="preferred",
            source_span="Valve",
        )

        for _ in range(2):
            await service._shared_storefront_tag_page(
                1,
                client.search_storefront_tag,
            )
            await service._shared_intersection_page(
                (1, 2),
                client.search_storefront_tags,
                page_size=40,
            )
            await service._shared_more_like_page(
                10,
                False,
                client.get_more_like,
            )
            await service._fetch_company_source(company, source_index=0)
            await service._fetch_top_sellers()
            await service._search_reference_group(reference, {})

        expected = [False, False]
        self.assertEqual(client.tag_policies, expected)
        self.assertEqual(client.intersection_policies, expected)
        self.assertEqual(client.more_like_policies, expected)
        self.assertEqual(client.company_policies, expected)
        self.assertEqual(client.top_seller_policies, expected)
        self.assertEqual(client.term_policies, expected)
        self.assertEqual(client.storesearch_policies, expected)
        self.assertEqual(service._storefront_tag_tasks, {})
        self.assertEqual(service._intersection_tasks, {})
        self.assertEqual(service._more_like_tasks, {})

    async def test_search_games_fallback_receives_false_policy_when_supported(
        self,
    ) -> None:
        client = SearchGamesOnlyClient()
        service = SteamGameIndexService(client, PolicyCache(), reuse_cache=False)

        hits = await service._search_refs("portal")

        self.assertEqual(hits, [])
        self.assertEqual(client.policies, [False])

    async def test_positional_only_reuse_parameter_is_not_passed_as_keyword(self) -> None:
        client = PositionalOnlyDiscoveryClient()
        service = SteamGameIndexService(client, PolicyCache(), reuse_cache=False)

        page = await service._shared_storefront_tag_page(
            1,
            client.search_storefront_tag,
        )

        self.assertEqual(page.hits, ())
        self.assertEqual(client.calls, [(1, True, 20)])
        self.assertFalse(
            callable_accepts_keyword(client.search_storefront_tag, "reuse_cache")
        )


class SteamIndexCoveragePolicyTest(unittest.IsolatedAsyncioTestCase):
    async def test_preloaded_coverage_is_reused_only_when_policy_is_enabled(self) -> None:
        preference = GamePreference(genres_like=["RPG"])
        enabled_client = CoverageClient()
        enabled_cache = PolicyCache(covered_snapshot())
        enabled = SteamGameIndexService(
            enabled_client,
            enabled_cache,
            clock=lambda: 1_000.0,
            reuse_cache=True,
        )
        disabled_client = CoverageClient()
        disabled_cache = PolicyCache(covered_snapshot())
        disabled = SteamGameIndexService(
            disabled_client,
            disabled_cache,
            clock=lambda: 1_000.0,
            reuse_cache=False,
        )

        await enabled.refresh_entries(preference, [], target_pool=1)
        await disabled.refresh_entries(
            GamePreference(genres_like=["RPG"]),
            [],
            target_pool=1,
        )

        self.assertEqual(enabled_client.searches, [])
        self.assertEqual(disabled_client.searches, [("rpg", False)])
        self.assertEqual(
            disabled_cache.payloads[STEAM_INDEX_CACHE_KEY]["search_coverage"]["rpg"],
            1_000.0,
        )


class SteamIndexReferencePolicyTest(unittest.IsolatedAsyncioTestCase):
    async def test_local_exact_reference_still_queries_remote_without_refetching_detail(
        self,
    ) -> None:
        client = LocalReferenceClient()
        cache = PolicyCache(reference_snapshot(game(10, "Portal")))
        service = SteamGameIndexService(
            client,
            cache,
            clock=lambda: 1_000.0,
            reuse_cache=False,
        )

        for _ in range(2):
            preference = GamePreference(reference_games_like=["Portal"])
            await service.refresh_entries(preference, [], target_pool=0)
            self.assertEqual(preference.resolved_reference_games[0].appid, 10)

        self.assertEqual(client.term_policies, [False, False])
        self.assertEqual(client.storesearch_policies, [False, False])
        self.assertEqual(client.detail_calls, [])

    async def test_remote_failure_keeps_local_exact_reference(self) -> None:
        client = LocalReferenceClient(fail=True)
        service = SteamGameIndexService(
            client,
            PolicyCache(reference_snapshot(game(10, "Portal"))),
            clock=lambda: 1_000.0,
            reuse_cache=False,
        )
        preference = GamePreference(reference_games_like=["Portal"])

        await service.refresh_entries(preference, [], target_pool=0)

        self.assertEqual(preference.resolved_reference_games[0].appid, 10)
        self.assertEqual(client.detail_calls, [])

    async def test_enabled_policy_reuses_local_exact_reference_without_remote_search(
        self,
    ) -> None:
        client = LocalReferenceClient()
        service = SteamGameIndexService(
            client,
            PolicyCache(reference_snapshot(game(10, "Portal"))),
            clock=lambda: 1_000.0,
            reuse_cache=True,
        )

        await service.refresh_entries(
            GamePreference(reference_games_like=["Portal"]),
            [],
            target_pool=0,
        )

        self.assertEqual(client.term_policies, [])
        self.assertEqual(client.storesearch_policies, [])
        self.assertEqual(client.detail_calls, [])


class RecordingDiscoveryClient:
    language = "english"

    def __init__(self) -> None:
        self.tag_policies: list[bool] = []
        self.intersection_policies: list[bool] = []
        self.more_like_policies: list[bool] = []
        self.company_policies: list[bool] = []
        self.top_seller_policies: list[bool] = []
        self.term_policies: list[bool] = []
        self.storesearch_policies: list[bool] = []

    async def search_storefront_tag(
        self,
        _tag_id: int,
        page_size: int = 20,
        reuse_cache: bool = True,
    ) -> SteamStorefrontPage:
        del page_size
        self.tag_policies.append(reuse_cache)
        return empty_page()

    async def search_storefront_tags(
        self,
        _tag_ids: list[int],
        page_size: int = 40,
        reuse_cache: bool = True,
    ) -> SteamStorefrontPage:
        del page_size
        self.intersection_policies.append(reuse_cache)
        return empty_page()

    async def get_more_like(
        self,
        _appid: int,
        *,
        allow_unreleased: bool = False,
        reuse_cache: bool = True,
    ) -> SteamMoreLikeResult:
        del allow_unreleased
        self.more_like_policies.append(reuse_cache)
        return SteamMoreLikeResult(hits=())

    async def search_storefront_company(
        self,
        _term: str,
        _role: str,
        page_size: int = 20,
        reuse_cache: bool = True,
    ) -> SteamStorefrontPage:
        del page_size
        self.company_policies.append(reuse_cache)
        return empty_page()

    async def browse_top_sellers(
        self,
        page_size: int = 60,
        reuse_cache: bool = True,
    ) -> SteamStorefrontPage:
        del page_size
        self.top_seller_policies.append(reuse_cache)
        return empty_page()

    async def search_storefront_term(
        self,
        _term: str,
        *,
        page_size: int,
        start: int,
        language: str,
        reuse_cache: bool = True,
    ) -> SteamStorefrontPage:
        del page_size, start, language
        self.term_policies.append(reuse_cache)
        return empty_page()

    async def search_game_refs(
        self,
        *,
        search: str,
        page_size: int,
        ordering: str,
        language: str,
        reuse_cache: bool = True,
    ) -> list[SteamSearchHit]:
        del search, page_size, ordering, language
        self.storesearch_policies.append(reuse_cache)
        return []


class SearchGamesOnlyClient:
    def __init__(self) -> None:
        self.policies: list[bool] = []

    async def search_games(
        self,
        *,
        search: str,
        page_size: int,
        ordering: str,
        language: str | None,
        reuse_cache: bool = True,
    ) -> list[GameCandidate]:
        del search, page_size, ordering, language
        self.policies.append(reuse_cache)
        return []


class PositionalOnlyDiscoveryClient:
    def __init__(self) -> None:
        self.calls: list[tuple[int, bool, int]] = []

    async def search_storefront_tag(
        self,
        tag_id: int,
        reuse_cache: bool = True,
        /,
        *,
        page_size: int = 20,
    ) -> SteamStorefrontPage:
        self.calls.append((tag_id, reuse_cache, page_size))
        return empty_page()


class CoverageClient:
    language = "english"

    def __init__(self) -> None:
        self.searches: list[tuple[str, bool]] = []

    async def get_popular_tags_snapshot(
        self,
        language: str = "english",
    ) -> SimpleNamespace:
        del language
        return SimpleNamespace(tags=(), fetched_at=1_000.0)

    async def search_game_refs(
        self,
        *,
        search: str,
        page_size: int,
        ordering: str,
        reuse_cache: bool = True,
    ) -> list[SteamSearchHit]:
        del page_size, ordering
        self.searches.append((search, reuse_cache))
        return []


class LocalReferenceClient(CoverageClient):
    def __init__(self, *, fail: bool = False) -> None:
        super().__init__()
        self.fail = fail
        self.term_policies: list[bool] = []
        self.storesearch_policies: list[bool] = []
        self.detail_calls: list[int] = []

    async def search_storefront_term(
        self,
        _term: str,
        *,
        page_size: int,
        start: int,
        language: str,
        reuse_cache: bool = True,
    ) -> SteamStorefrontPage:
        del page_size, start, language
        self.term_policies.append(reuse_cache)
        if self.fail:
            raise SteamTransientError("offline")
        return empty_page()

    async def search_game_refs(
        self,
        *,
        search: str,
        page_size: int,
        ordering: str,
        language: str,
        reuse_cache: bool = True,
    ) -> list[SteamSearchHit]:
        del search, page_size, ordering, language
        self.storesearch_policies.append(reuse_cache)
        if self.fail:
            raise SteamTransientError("offline")
        return []

    async def get_game_detail(self, appid: int) -> GameCandidate:
        self.detail_calls.append(appid)
        raise AssertionError("current local reference must reuse AppDetails")


class PolicyCache:
    def __init__(self, snapshot: dict[str, Any] | None = None) -> None:
        self.payloads = (
            {STEAM_INDEX_CACHE_KEY: snapshot}
            if snapshot is not None
            else {}
        )

    async def get_json(self, key: str, _ttl_hours: int) -> Any | None:
        return self.payloads.get(key)

    async def set_json(self, key: str, payload: Any) -> None:
        self.payloads[key] = payload


def empty_page() -> SteamStorefrontPage:
    return SteamStorefrontPage(hits=(), total_count=0, start=0)


def game(appid: int, title: str) -> GameCandidate:
    return GameCandidate(
        appid=appid,
        title=title,
        app_type="game",
        platforms=["PC"],
        tags=["Puzzle"],
        stores=["Steam"],
        raw_url=f"https://store.steampowered.com/app/{appid}/",
    )


def dump_model(model: Any) -> dict[str, Any]:
    dumper = getattr(model, "model_dump", None)
    return dumper() if dumper else model.dict()


def covered_snapshot() -> dict[str, Any]:
    return {
        "schema_version": STEAM_INDEX_SCHEMA_VERSION,
        "entries": [],
        "search_coverage": {"rpg": 1_000.0},
    }


def reference_snapshot(candidate: GameCandidate) -> dict[str, Any]:
    return {
        "schema_version": STEAM_INDEX_SCHEMA_VERSION,
        "entries": [
            {
                "candidate": dump_model(candidate),
                "refreshed_at": 1_000.0,
                "needs_revalidation": False,
            }
        ],
        "search_coverage": {},
    }


if __name__ == "__main__":
    unittest.main()
