from __future__ import annotations

import unittest
from typing import Any
from unittest.mock import AsyncMock

import httpx
from astrbot_plugin_steam_game_recommender.clients.steam import (
    STEAM_POPULAR_TAGS_URL,
    SteamApiError,
    SteamClient,
    parse_storefront_results_html,
)
from astrbot_plugin_steam_game_recommender.services.tag_normalizer import (
    register_steam_tag_aliases,
)
from astrbot_plugin_steam_game_recommender.storage.models import (
    GameCandidate,
    SteamSearchHit,
)

RESULTS_HTML = """
<a class="search_result_row ds_collapse_flag" data-ds-appid="10">
  <span class="title">Alpha &amp; Omega</span>
</a>
<a class="search_result_row" data-ds-appid="11">
  <span class="title">Second Game</span>
</a>
<a class="search_result_row" data-ds-appid="10">
  <span class="title">Duplicate</span>
</a>
<a class="search_result_row" data-ds-appid="bad">
  <span class="title">Malformed</span>
</a>
<a class="search_result_row" data-ds-appid="0">
  <span class="title">Zero</span>
</a>
<a class="search_result_row" data-ds-appid="-12">
  <span class="title">Negative</span>
</a>
<a class="search_result_row" data-ds-appid="12"></a>
"""


def storefront_payload() -> dict[str, Any]:
    return {
        "success": 1,
        "results_html": RESULTS_HTML,
        "total_count": "321",
        "start": "0",
    }


class StorefrontSearchTest(unittest.IsolatedAsyncioTestCase):
    async def test_storesearch_reuse_false_overwrites_the_single_snapshot(
        self,
    ) -> None:
        cache = MemoryCache()
        http_client = FakeHttpClient(
            {"items": [{"type": "app", "id": 10, "name": "Old Result"}]}
        )
        client = SteamClient(http_client, cache)

        first = await client.search_game_refs(search="portal")
        http_client.payload = {
            "items": [{"type": "app", "id": 11, "name": "Live Result"}]
        }
        second = await client.search_game_refs(search="portal", reuse_cache=False)

        self.assertEqual([hit.appid for hit in first], [10])
        self.assertEqual([hit.appid for hit in second], [11])
        self.assertEqual(http_client.call_count, 2)
        self.assertEqual(len(cache.payloads), 1)
        self.assertEqual(next(iter(cache.payloads.values()))["items"][0]["id"], 11)

    async def test_storesearch_default_policy_reuses_fresh_snapshot(self) -> None:
        cache = MemoryCache()
        http_client = FakeHttpClient(
            {"items": [{"type": "app", "id": 10, "name": "Cached Result"}]}
        )
        client = SteamClient(http_client, cache)
        first = await client.search_game_refs(search="portal")
        http_client.payload = {
            "items": [{"type": "app", "id": 11, "name": "Unexpected Live Result"}]
        }

        second = await client.search_game_refs(search="portal")

        self.assertEqual(second, first)
        self.assertEqual(http_client.call_count, 1)

    async def test_storesearch_live_empty_is_success_and_does_not_restore_old_stale(
        self,
    ) -> None:
        cache = MemoryCache()
        http_client = FakeHttpClient(
            {"items": [{"type": "app", "id": 10, "name": "Old Result"}]}
        )
        client = SteamClient(http_client, cache)
        await client.search_game_refs(search="portal")
        http_client.payload = {"items": []}

        hits = await client.search_game_refs(search="portal", reuse_cache=False)

        self.assertEqual(hits, [])
        self.assertEqual(http_client.call_count, 2)
        self.assertEqual(len(cache.payloads), 1)
        self.assertEqual(next(iter(cache.payloads.values())), {"items": []})

    async def test_storesearch_retries_invalid_json_once(self) -> None:
        calls = 0

        def handler(request: httpx.Request) -> httpx.Response:
            nonlocal calls
            calls += 1
            if calls == 1:
                return httpx.Response(200, text="not-json", request=request)
            return httpx.Response(
                200,
                json={"items": [{"type": "app", "id": 10, "name": "Portal"}]},
                request=request,
            )

        async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as http:
            hits = await SteamClient(http, MemoryCache()).search_game_refs(
                search="portal"
            )

        self.assertEqual([hit.appid for hit in hits], [10])
        self.assertEqual(calls, 2)

    async def test_storesearch_retries_temporary_contract_failure_once(self) -> None:
        calls = 0

        def handler(request: httpx.Request) -> httpx.Response:
            nonlocal calls
            calls += 1
            payload = (
                {"unexpected": []}
                if calls == 1
                else {"items": [{"type": "app", "id": 10, "name": "Portal"}]}
            )
            return httpx.Response(200, json=payload, request=request)

        async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as http:
            hits = await SteamClient(http, MemoryCache()).search_game_refs(
                search="portal"
            )

        self.assertEqual([hit.appid for hit in hits], [10])
        self.assertEqual(calls, 2)

    async def test_storesearch_reuse_false_falls_back_to_seven_day_stale(self) -> None:
        cache = MemoryCache()
        online = SteamClient(
            FakeHttpClient(
                {"items": [{"type": "app", "id": 10, "name": "Old Result"}]}
            ),
            cache,
        )
        await online.search_game_refs(search="portal")
        offline = SteamClient(FailingHttpClient(), cache)

        hits = await offline.search_game_refs(search="portal", reuse_cache=False)

        self.assertEqual([hit.appid for hit in hits], [10])
        self.assertIn(168, cache.requested_ttls)

    async def test_storesearch_reuse_false_raises_when_seven_day_stale_is_absent(
        self,
    ) -> None:
        cache = MemoryCache()
        client = SteamClient(FailingHttpClient(), cache)

        with self.assertRaises(SteamApiError):
            await client.search_game_refs(search="portal", reuse_cache=False)

        self.assertEqual(cache.requested_ttls, [168])

    async def test_storesearch_stale_older_than_seven_days_is_rejected(self) -> None:
        now = [1_000.0]
        cache = ExpiringMemoryCache(lambda: now[0])
        online = SteamClient(
            FakeHttpClient(
                {"items": [{"type": "app", "id": 10, "name": "Old Result"}]}
            ),
            cache,
        )
        await online.search_game_refs(search="portal")
        now[0] += 7 * 24 * 60 * 60 + 1
        offline = SteamClient(FailingHttpClient(), cache)

        with self.assertRaises(SteamApiError):
            await offline.search_game_refs(search="portal", reuse_cache=False)

        self.assertEqual(cache.requested_ttls[-1], 168)

    async def test_storesearch_permanent_http_error_never_uses_stale(self) -> None:
        cache = MemoryCache()
        await SteamClient(
            FakeHttpClient(
                {"items": [{"type": "app", "id": 10, "name": "Cached Result"}]}
            ),
            cache,
        ).search_game_refs(search="portal")
        calls = 0

        def handler(request: httpx.Request) -> httpx.Response:
            nonlocal calls
            calls += 1
            return httpx.Response(404, request=request)

        async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as http:
            with self.assertRaises(SteamApiError):
                await SteamClient(http, cache).search_game_refs(
                    search="portal",
                    reuse_cache=False,
                )

        self.assertEqual(calls, 1)

    async def test_search_games_only_forwards_reuse_policy_to_search_refs(self) -> None:
        client = SteamClient(FakeHttpClient({"items": []}), MemoryCache())
        candidate = GameCandidate(appid=10, title="Portal", app_type="game")
        client.search_game_refs = AsyncMock(
            return_value=[SteamSearchHit(appid=10, title="Portal")]
        )
        client.get_game_detail = AsyncMock(return_value=candidate)

        games = await client.search_games(search="portal", reuse_cache=False)

        self.assertEqual(games, [candidate])
        self.assertIs(client.search_game_refs.await_args.kwargs["reuse_cache"], False)
        self.assertEqual(client.get_game_detail.await_args.args, (10,))
        self.assertEqual(client.get_game_detail.await_args.kwargs, {})

    async def test_popular_tags_returns_valid_fresh_cache_without_network(self) -> None:
        now = 1_000.0
        cache = MemoryCache()
        http_client = FakeHttpClient([{"tagid": 999, "name": "unexpected"}])
        client = SteamClient(http_client, cache, clock=lambda: now)
        cache_key = f"{client._cache_key(STEAM_POPULAR_TAGS_URL, {})}:v2"
        cache.payloads[cache_key] = {
            "fetched_at": now,
            "tags": [{"tagid": 29482, "name": "Souls-like"}],
        }

        tags = await client.get_popular_tags()

        self.assertEqual(tags, [{"tagid": 29482, "name": "Souls-like"}])
        self.assertEqual(http_client.call_count, 0)
        self.assertEqual(cache.requested_ttls, [24])

    async def test_popular_tags_network_success_writes_one_snapshot(self) -> None:
        now = 1_234.0
        payload = [{"tagid": 29482, "name": "Souls-like", "count": 123}]
        cache = MemoryCache()
        http_client = FakeHttpClient(payload)
        client = SteamClient(http_client, cache, clock=lambda: now)
        base_key = f"{client._cache_key(STEAM_POPULAR_TAGS_URL, {})}:v2"

        tags = await client.get_popular_tags()

        expected = [{"tagid": 29482, "name": "Souls-like", "count": 123}]
        expected_snapshot = {"fetched_at": now, "tags": expected}
        self.assertEqual(tags, expected)
        self.assertEqual(http_client.call_count, 1)
        self.assertEqual(cache.payloads, {base_key: expected_snapshot})

    async def test_popular_tag_optional_counts_are_preserved_by_the_contract(self) -> None:
        payload = [
            {"tagid": 91_001, "name": "Broad Fixture Tag", "total_count": 50_000},
            {"tagid": 91_002, "name": "Narrow Fixture Tag", "count": 120},
        ]
        client = SteamClient(FakeHttpClient(payload), MemoryCache())

        tags = await client.get_popular_tags()

        self.assertEqual(tags[0]["count"], 50_000)
        self.assertEqual(tags[1]["count"], 120)

    async def test_popular_tag_contract_rejects_invalid_optional_count(self) -> None:
        for value in (True, -1, 1.5, float("inf"), "1.5"):
            with self.subTest(value=value):
                client = SteamClient(
                    FakeHttpClient(
                        [{"tagid": 91_003, "name": "Fixture Tag", "count": value}]
                    ),
                    MemoryCache(),
                )

                with self.assertRaises(SteamApiError):
                    await client.get_popular_tags()

    async def test_popular_tags_network_failure_uses_seven_day_stale_cache(self) -> None:
        now = 1_000_000.0
        cache = MemoryCache()
        client = SteamClient(FailingHttpClient(), cache, clock=lambda: now)
        cache_key = f"{client._cache_key(STEAM_POPULAR_TAGS_URL, {})}:v2"
        cache.payloads[cache_key] = {
            "fetched_at": now - 6 * 24 * 60 * 60,
            "tags": [{"tagid": 29482, "name": "Souls-like"}],
        }

        tags = await client.get_popular_tags()

        self.assertEqual(tags, [{"tagid": 29482, "name": "Souls-like"}])
        self.assertIn(168, cache.requested_ttls)

    async def test_popular_tags_contract_failure_uses_seven_day_stale_cache(self) -> None:
        now = 1_000_000.0
        cache = MemoryCache()
        http_client = FakeHttpClient([{"tagid": 0, "name": "invalid"}])
        client = SteamClient(http_client, cache, clock=lambda: now)
        cache_key = f"{client._cache_key(STEAM_POPULAR_TAGS_URL, {})}:v2"
        cache.payloads[cache_key] = {
            "fetched_at": now - 6 * 24 * 60 * 60,
            "tags": [{"tagid": 29482, "name": "Souls-like"}],
        }

        tags = await client.get_popular_tags()

        self.assertEqual(tags, [{"tagid": 29482, "name": "Souls-like"}])
        self.assertEqual(http_client.call_count, 2)
        self.assertIn(168, cache.requested_ttls)

    async def test_popular_tags_retries_temporary_contract_failure_once(self) -> None:
        calls = 0

        def handler(request: httpx.Request) -> httpx.Response:
            nonlocal calls
            calls += 1
            payload = (
                [{"tagid": 0, "name": "invalid"}]
                if calls == 1
                else [{"tagid": 29482, "name": "Souls-like"}]
            )
            return httpx.Response(200, json=payload, request=request)

        async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as http:
            tags = await SteamClient(http, MemoryCache()).get_popular_tags()

        self.assertEqual(tags, [{"tagid": 29482, "name": "Souls-like"}])
        self.assertEqual(calls, 2)

    async def test_popular_tags_permanent_http_error_never_uses_stale(self) -> None:
        now = 1_000_000.0
        cache = MemoryCache()
        seeded = SteamClient(
            FakeHttpClient([{"tagid": 29482, "name": "Souls-like"}]),
            cache,
            clock=lambda: now,
        )
        await seeded.get_popular_tags()
        calls = 0

        def handler(request: httpx.Request) -> httpx.Response:
            nonlocal calls
            calls += 1
            return httpx.Response(403, request=request)

        async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as http:
            with self.assertRaises(SteamApiError):
                await SteamClient(
                    http,
                    cache,
                    clock=lambda: now + 24 * 60 * 60 + 1,
                ).get_popular_tags()

        self.assertEqual(calls, 1)

    async def test_popular_tags_invalid_stale_cache_raises(self) -> None:
        now = 1_000_000.0
        cache = MemoryCache()
        client = SteamClient(FailingHttpClient(), cache, clock=lambda: now)
        cache_key = f"{client._cache_key(STEAM_POPULAR_TAGS_URL, {})}:v2"
        cache.payloads[cache_key] = {
            "fetched_at": now,
            "tags": [{"tagid": True, "name": "invalid"}],
        }

        with self.assertRaises(SteamApiError):
            await client.get_popular_tags()

    async def test_stale_popular_tags_never_extend_the_original_fetch_time(self) -> None:
        now = [1_000_000.0]
        cache = MemoryCache()
        first = SteamClient(
            FakeHttpClient([{"tagid": 29482, "name": "Souls-like"}]),
            cache,
            clock=lambda: now[0],
        )
        original = await first.get_popular_tags_snapshot()

        now[0] += 6 * 24 * 60 * 60
        offline = SteamClient(FailingHttpClient(), cache, clock=lambda: now[0])
        stale = await offline.get_popular_tags_snapshot()

        self.assertEqual(stale.fetched_at, original.fetched_at)
        now[0] += 2 * 24 * 60 * 60
        with self.assertRaises(SteamApiError):
            await offline.get_popular_tags_snapshot()

    async def test_tag_search_uses_default_relevance_and_parses_contract(self) -> None:
        register_steam_tag_aliases([{"tagid": 29482, "name": "Souls-like"}])
        cache = MemoryCache()
        http_client = FakeHttpClient(storefront_payload())
        client = SteamClient(http_client, cache, default_country="CN")

        page = await client.search_storefront_tag(29482, page_size=100, start=-3)

        self.assertEqual([hit.appid for hit in page.hits], [10, 11])
        self.assertEqual([hit.title for hit in page.hits], ["Alpha & Omega", "Second Game"])
        self.assertEqual(page.total_count, 321)
        self.assertEqual(page.start, 0)
        self.assertEqual(
            http_client.last_params,
            {
                "ignore_preferences": 1,
                "tags": 29482,
                "ndl": 1,
                "l": "english",
                "cc": "CN",
                "start": 0,
                "count": 60,
                "infinite": 1,
            },
        )
        self.assertNotIn("sort_by", http_client.last_params)
        cached = await client.search_storefront_tag(29482, page_size=60)

        self.assertEqual(cached, page)
        self.assertEqual(http_client.call_count, 1)

    async def test_every_storefront_discovery_method_supports_network_first(self) -> None:
        calls = (
            ("search_storefront_tag", (29482,), {}),
            ("search_storefront_tags", ([29482, 19],), {}),
            ("search_storefront_term", ("portal",), {}),
            ("search_storefront_company", ("Valve", "developer"), {}),
            ("browse_top_sellers", (), {}),
        )
        for method_name, args, kwargs in calls:
            with self.subTest(method=method_name):
                cache = MemoryCache()
                http_client = FakeHttpClient(storefront_payload())
                client = SteamClient(http_client, cache)
                method = getattr(client, method_name)

                await method(*args, **kwargs)
                await method(*args, **kwargs, reuse_cache=False)

                self.assertEqual(http_client.call_count, 2)

    async def test_storefront_live_empty_overwrites_old_stale_without_fallback(self) -> None:
        cache = MemoryCache()
        http_client = FakeHttpClient(storefront_payload())
        client = SteamClient(http_client, cache)
        await client.search_storefront_tag(29482)
        http_client.payload = {
            "success": 1,
            "results_html": "",
            "total_count": 0,
            "start": 0,
        }

        page = await client.search_storefront_tag(29482, reuse_cache=False)

        self.assertEqual(page.hits, ())
        self.assertFalse(page.stale)
        self.assertEqual(http_client.call_count, 2)
        self.assertEqual(len(cache.payloads), 1)
        self.assertEqual(next(iter(cache.payloads.values()))["results_html"], "")

    async def test_top_sellers_use_browse_filter_without_tag_or_sort(self) -> None:
        http_client = FakeHttpClient(storefront_payload())
        client = SteamClient(http_client, MemoryCache(), default_country="US")

        await client.browse_top_sellers(page_size=80)

        self.assertEqual(http_client.last_params["filter"], "topsellers")
        self.assertEqual(http_client.last_params["count"], 60)
        self.assertEqual(http_client.last_params["cc"], "US")
        self.assertNotIn("tags", http_client.last_params)
        self.assertNotIn("sort_by", http_client.last_params)

    async def test_company_search_uses_exact_role_filter_and_caps_each_alias_at_twenty(
        self,
    ) -> None:
        for role in ("developer", "publisher"):
            with self.subTest(role=role):
                http_client = FakeHttpClient(storefront_payload())
                client = SteamClient(
                    http_client,
                    MemoryCache(),
                    default_country="JP",
                )

                await client.search_storefront_company(
                    "Acme Games",
                    role,
                    page_size=99,
                    start=-5,
                )

                self.assertEqual(http_client.last_params[role], "Acme Games")
                self.assertNotIn(
                    "publisher" if role == "developer" else "developer",
                    http_client.last_params,
                )
                self.assertEqual(http_client.last_params["count"], 20)
                self.assertEqual(http_client.last_params["start"], 0)
                self.assertEqual(http_client.last_params["cc"], "JP")
                self.assertNotIn("term", http_client.last_params)
                self.assertNotIn("sort_by", http_client.last_params)

    async def test_invalid_contract_raises_without_stale_cache(self) -> None:
        client = SteamClient(FakeHttpClient({"success": 1}), MemoryCache())

        with self.assertRaises(SteamApiError):
            await client.search_storefront_tag(29482)

    async def test_storefront_retries_invalid_json_once(self) -> None:
        calls = 0

        def handler(request: httpx.Request) -> httpx.Response:
            nonlocal calls
            calls += 1
            if calls == 1:
                return httpx.Response(200, text="not-json", request=request)
            return httpx.Response(200, json=storefront_payload(), request=request)

        async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as http:
            page = await SteamClient(http, MemoryCache()).search_storefront_tag(29482)

        self.assertEqual([hit.appid for hit in page.hits], [10, 11])
        self.assertEqual(calls, 2)

    async def test_storefront_retries_temporary_contract_failure_once(self) -> None:
        calls = 0

        def handler(request: httpx.Request) -> httpx.Response:
            nonlocal calls
            calls += 1
            payload = {"success": 1} if calls == 1 else storefront_payload()
            return httpx.Response(200, json=payload, request=request)

        async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as http:
            page = await SteamClient(http, MemoryCache()).search_storefront_tag(29482)

        self.assertEqual([hit.appid for hit in page.hits], [10, 11])
        self.assertEqual(calls, 2)

    async def test_contract_rejects_boolean_fractional_and_non_finite_integers(self) -> None:
        invalid_values = (True, 1.5, float("inf"), "1.5")
        for value in invalid_values:
            with self.subTest(value=value):
                payload = storefront_payload()
                payload["total_count"] = value
                client = SteamClient(FakeHttpClient(payload), MemoryCache())

                with self.assertRaises(SteamApiError):
                    await client.search_storefront_tag(29482)

    async def test_live_failure_uses_single_seven_day_snapshot_as_stale(self) -> None:
        now = [1_000.0]
        cache = ExpiringMemoryCache(lambda: now[0])
        first = SteamClient(FakeHttpClient(storefront_payload()), cache)
        expected = await first.search_storefront_tag(29482)
        now[0] += 24 * 60 * 60 + 1
        failing = SteamClient(FailingHttpClient(), cache)

        actual = await failing.search_storefront_tag(29482)

        self.assertEqual(actual.hits, expected.hits)
        self.assertEqual(actual.total_count, expected.total_count)
        self.assertTrue(actual.stale)
        self.assertIn(168, cache.requested_ttls)

    async def test_network_first_live_failure_skips_fresh_then_uses_stale(self) -> None:
        cache = MemoryCache()
        first = SteamClient(FakeHttpClient(storefront_payload()), cache)
        expected = await first.search_storefront_tag(29482)
        failing = SteamClient(FailingHttpClient(), cache)

        actual = await failing.search_storefront_tag(29482, reuse_cache=False)

        self.assertEqual(actual.hits, expected.hits)
        self.assertTrue(actual.stale)
        self.assertEqual(cache.requested_ttls[-1], 168)

    async def test_storefront_permanent_http_error_never_uses_stale(self) -> None:
        cache = MemoryCache()
        await SteamClient(FakeHttpClient(storefront_payload()), cache).search_storefront_tag(
            29482
        )
        calls = 0

        def handler(request: httpx.Request) -> httpx.Response:
            nonlocal calls
            calls += 1
            return httpx.Response(404, request=request)

        async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as http:
            with self.assertRaises(SteamApiError):
                await SteamClient(http, cache).search_storefront_tag(
                    29482,
                    reuse_cache=False,
                )

        self.assertEqual(calls, 1)

    async def test_live_failure_without_stale_cache_raises(self) -> None:
        cache = MemoryCache()
        client = SteamClient(FailingHttpClient(), cache)

        with self.assertRaises(SteamApiError):
            await client.search_storefront_tag(29482, reuse_cache=False)

        self.assertEqual(cache.requested_ttls, [168])

    async def test_storefront_stale_older_than_seven_days_is_rejected(self) -> None:
        now = [1_000.0]
        cache = ExpiringMemoryCache(lambda: now[0])
        online = SteamClient(FakeHttpClient(storefront_payload()), cache)
        await online.search_storefront_tag(29482)
        now[0] += 7 * 24 * 60 * 60 + 1
        offline = SteamClient(FailingHttpClient(), cache)

        with self.assertRaises(SteamApiError):
            await offline.search_storefront_tag(29482, reuse_cache=False)

        self.assertEqual(cache.requested_ttls[-1], 168)

    async def test_title_search_accepts_language_override(self) -> None:
        payload = {"items": [{"type": "app", "id": 10, "name": "English Title"}]}
        http_client = FakeHttpClient(payload)
        client = SteamClient(http_client, MemoryCache(), language="schinese")

        await client.search_game_refs(search="title", language="english")

        self.assertEqual(http_client.last_params["l"], "english")

    def test_html_entities_are_decoded_exactly_once(self) -> None:
        hits = parse_storefront_results_html(
            '<a class="search_result_row" data-ds-appid="9">'
            '<span class="title">Rock &amp;amp; Roll</span></a>'
        )

        self.assertEqual(hits[0].title, "Rock &amp; Roll")


class FakeResponse:
    def __init__(self, payload: Any) -> None:
        self.payload = payload

    def raise_for_status(self) -> None:
        return None

    def json(self) -> Any:
        return self.payload


class FakeHttpClient:
    def __init__(self, payload: Any) -> None:
        self.payload = payload
        self.call_count = 0
        self.last_params: dict[str, Any] = {}

    async def get(self, _url: str, params: dict[str, Any]) -> FakeResponse:
        self.call_count += 1
        self.last_params = dict(params)
        return FakeResponse(self.payload)


class FailingHttpClient:
    async def get(self, _url: str, params: dict[str, Any]) -> FakeResponse:
        request = httpx.Request(
            "GET",
            "https://store.steampowered.com/search/results",
            params=params,
        )
        raise httpx.ConnectError("offline", request=request)


class MemoryCache:
    def __init__(self) -> None:
        self.payloads: dict[str, Any] = {}
        self.requested_ttls: list[int] = []

    async def get_json(self, key: str, ttl_hours: int) -> Any | None:
        self.requested_ttls.append(ttl_hours)
        return self.payloads.get(key)

    async def set_json(self, key: str, payload: Any) -> None:
        self.payloads[key] = payload


class ExpiringMemoryCache(MemoryCache):
    def __init__(self, clock) -> None:
        super().__init__()
        self.clock = clock
        self.written_at: dict[str, float] = {}

    async def get_json(self, key: str, ttl_hours: int) -> Any | None:
        self.requested_ttls.append(ttl_hours)
        if key not in self.payloads:
            return None
        if self.clock() - self.written_at[key] > ttl_hours * 60 * 60:
            return None
        return self.payloads[key]

    async def set_json(self, key: str, payload: Any) -> None:
        self.payloads[key] = payload
        self.written_at[key] = self.clock()


if __name__ == "__main__":
    unittest.main()
