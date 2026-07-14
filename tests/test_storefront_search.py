from __future__ import annotations

import unittest
from typing import Any

import httpx

from astrbot_plugin_steam_game_recommender.clients.steam import (
    SteamApiError,
    SteamClient,
)
from astrbot_plugin_steam_game_recommender.services.tag_normalizer import (
    register_steam_tag_aliases,
    steam_tag_result_count_for,
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
        self.assertEqual(steam_tag_result_count_for("soulslike"), 321)

        cached = await client.search_storefront_tag(29482, page_size=60)

        self.assertEqual(cached, page)
        self.assertEqual(http_client.call_count, 1)

    async def test_top_sellers_use_browse_filter_without_tag_or_sort(self) -> None:
        http_client = FakeHttpClient(storefront_payload())
        client = SteamClient(http_client, MemoryCache(), default_country="US")

        await client.browse_top_sellers(page_size=80)

        self.assertEqual(http_client.last_params["filter"], "topsellers")
        self.assertEqual(http_client.last_params["count"], 60)
        self.assertEqual(http_client.last_params["cc"], "US")
        self.assertNotIn("tags", http_client.last_params)
        self.assertNotIn("sort_by", http_client.last_params)

    async def test_invalid_contract_raises_without_stale_cache(self) -> None:
        client = SteamClient(FakeHttpClient({"success": 1}), MemoryCache())

        with self.assertRaises(SteamApiError):
            await client.search_storefront_tag(29482)

    async def test_live_failure_uses_separate_seven_day_stale_cache(self) -> None:
        cache = MemoryCache()
        first = SteamClient(FakeHttpClient(storefront_payload()), cache)
        expected = await first.search_storefront_tag(29482)
        fresh_key = next(key for key in cache.payloads if key.endswith(":fresh"))
        cache.payloads.pop(fresh_key)
        failing = SteamClient(FailingHttpClient(), cache)

        actual = await failing.search_storefront_tag(29482)

        self.assertEqual(actual, expected)
        self.assertIn(168, cache.requested_ttls)

    async def test_live_failure_without_stale_cache_raises(self) -> None:
        client = SteamClient(FailingHttpClient(), MemoryCache())

        with self.assertRaises(SteamApiError):
            await client.search_storefront_tag(29482)

    async def test_title_search_accepts_language_override(self) -> None:
        payload = {"items": [{"type": "app", "id": 10, "name": "English Title"}]}
        http_client = FakeHttpClient(payload)
        client = SteamClient(http_client, MemoryCache(), language="schinese")

        await client.search_game_refs(search="title", language="english")

        self.assertEqual(http_client.last_params["l"], "english")


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
        request = httpx.Request("GET", "https://store.steampowered.com/search/results", params=params)
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


if __name__ == "__main__":
    unittest.main()
