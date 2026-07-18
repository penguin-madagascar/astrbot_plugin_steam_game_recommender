from __future__ import annotations

import sqlite3
import tempfile
import time
import unittest
from pathlib import Path
from typing import Any
from unittest.mock import patch

import httpx
from astrbot_plugin_steam_game_recommender.clients.steam import (
    SteamApiError,
    SteamClient,
    optional_int,
    parse_steam_game,
    parse_storefront_tag_ids,
)
from astrbot_plugin_steam_game_recommender.storage import repository as repository_module
from astrbot_plugin_steam_game_recommender.storage.models import (
    GameCandidate,
    SteamSearchHit,
)
from astrbot_plugin_steam_game_recommender.storage.repository import SQLiteCacheRepository


class SteamClientTest(unittest.IsolatedAsyncioTestCase):
    def test_optional_int_only_accepts_exact_finite_integers(self) -> None:
        for value in (True, False, 1.5, -2.5, float("nan"), float("inf"), "1.0"):
            with self.subTest(value=value):
                self.assertIsNone(optional_int(value))

        for value, expected in ((12, 12), (12.0, 12), ("+12", 12), ("-12", -12)):
            with self.subTest(value=value):
                self.assertEqual(optional_int(value), expected)

    def test_tag_and_genre_ids_only_keep_positive_exact_integers(self) -> None:
        self.assertEqual(
            parse_storefront_tag_ids('[true, 1.5, -2, 0, 3, 4.0, "+5", "6.0"]'),
            [3, 4, 5],
        )
        payload = steam_detail_payload()
        payload["genres"] = [
            {"id": True, "description": "Boolean"},
            {"id": 1.5, "description": "Fractional"},
            {"id": -2, "description": "Negative"},
            {"id": 3, "description": "Valid"},
            {"id": "4", "description": "String Valid"},
        ]

        candidate = parse_steam_game(1, payload)

        self.assertEqual(candidate.genre_ids, [3, 4])

    def test_metacritic_only_accepts_scores_between_zero_and_one_hundred(self) -> None:
        for value, expected in (
            (0, 0),
            (100, 100),
            (88.0, 88),
            (True, None),
            (88.5, None),
            (-1, None),
            (101, None),
            (float("nan"), None),
            (float("inf"), None),
        ):
            with self.subTest(value=value):
                payload = steam_detail_payload()
                payload["metacritic"] = {"score": value}
                self.assertEqual(parse_steam_game(1, payload).metacritic, expected)

    async def test_public_appid_and_tag_queries_reject_non_positive_exact_ids(self) -> None:
        client = SteamClient(FakeHttpClient({}), MemoryCache(), cache_ttl_hours=24)
        for invalid in (True, 1.5, float("nan"), float("inf"), 0, -1, "1.0"):
            for operation in (
                lambda value=invalid: client.get_game_detail(value),
                lambda value=invalid: client.get_review_summary(value),
                lambda value=invalid: client.get_more_like(value),
                lambda value=invalid: client.get_store_page_tags(value),
                lambda value=invalid: client.search_storefront_tag(value),
                lambda value=invalid: client.search_storefront_tags((19, value)),
            ):
                with self.subTest(invalid=invalid, operation=operation):
                    with self.assertRaises(ValueError):
                        await operation()

    def test_game_candidate_defaults_to_released(self) -> None:
        candidate = GameCandidate(title="Released Game")

        self.assertFalse(candidate.coming_soon)

    def test_steam_detail_preserves_release_availability(self) -> None:
        released_payload = steam_detail_payload()
        coming_soon_payload = steam_detail_payload()
        coming_soon_payload["release_date"] = {
            "date": "即将推出",
            "coming_soon": True,
        }

        released = parse_steam_game(1, released_payload)
        coming_soon = parse_steam_game(2, coming_soon_payload)

        self.assertFalse(released.coming_soon)
        self.assertTrue(coming_soon.coming_soon)

    def test_steam_detail_does_not_coerce_non_boolean_release_status(self) -> None:
        payload = steam_detail_payload()
        payload["release_date"] = {
            "date": "即将推出",
            "coming_soon": "true",
        }

        candidate = parse_steam_game(1, payload)

        self.assertFalse(candidate.coming_soon)

    async def test_search_game_refs_returns_lightweight_hits_without_detail_calls(self) -> None:
        cache = MemoryCache()
        http_client = FakeHttpClient(
            {
                "https://store.steampowered.com/api/storesearch/": {
                    "items": [
                        {"type": "app", "id": 123, "name": "Co-op Test Game"},
                        {"type": "app", "appid": 456, "name": "Other Game"},
                    ]
                }
            }
        )
        client = SteamClient(http_client, cache, cache_ttl_hours=24)

        hits = await client.search_game_refs(search="co-op", page_size=2)

        self.assertTrue(all(isinstance(hit, SteamSearchHit) for hit in hits))
        self.assertEqual([hit.appid for hit in hits], [123, 456])
        self.assertEqual([hit.title for hit in hits], ["Co-op Test Game", "Other Game"])
        self.assertEqual(hits[0].store_url, "https://store.steampowered.com/app/123/")
        self.assertEqual(http_client.call_count, 1)

    async def test_search_game_refs_ignores_packages_bundles_and_unknown_items(self) -> None:
        http_client = FakeHttpClient(
            {
                "https://store.steampowered.com/api/storesearch/": {
                    "items": [
                        {"type": "app", "id": 123, "name": "Base Game"},
                        {"type": "sub", "id": 456, "name": "Complete Package"},
                        {"type": "bundle", "id": 789, "name": "Franchise Bundle"},
                        {"id": 999, "name": "Unknown Store Item"},
                    ]
                }
            }
        )
        client = SteamClient(http_client, MemoryCache(), cache_ttl_hours=24)

        hits = await client.search_game_refs(search="base game", page_size=10)

        self.assertEqual([hit.appid for hit in hits], [123])

    async def test_invalid_search_entities_do_not_consume_page_size(self) -> None:
        http_client = FakeHttpClient(
            {
                "https://store.steampowered.com/api/storesearch/": {
                    "items": [
                        {"type": "sub", "id": 10, "name": "Package"},
                        {"type": "bundle", "id": 20, "name": "Bundle"},
                        {"type": "app", "id": 30, "name": "First Game"},
                        {"type": "app", "id": 40, "name": "Second Game"},
                        {"type": "app", "id": 50, "name": "Third Game"},
                    ]
                }
            }
        )
        client = SteamClient(http_client, MemoryCache(), cache_ttl_hours=24)

        hits = await client.search_game_refs(search="game", page_size=2)

        self.assertEqual([hit.appid for hit in hits], [30, 40])

    async def test_search_games_parses_steam_details_and_uses_cache(self) -> None:
        cache = MemoryCache()
        http_client = FakeHttpClient(
            {
                "https://store.steampowered.com/api/storesearch/": {
                    "items": [
                        {"type": "app", "id": 123, "name": "Co-op Test Game"},
                        {"type": "app", "appid": 456, "name": "Other Game"},
                    ]
                },
                "https://store.steampowered.com/api/appdetails": {
                    "123": {"success": True, "data": steam_detail_payload()},
                    "456": {
                        "success": True,
                        "data": {**steam_detail_payload(), "name": "Other Game"},
                    },
                },
            }
        )
        client = SteamClient(http_client, cache, cache_ttl_hours=24)

        games = await client.search_games(
            search="co-op",
            platforms=["steam"],
            genres=["action"],
            tags=["co-op"],
            page_size=2,
        )

        self.assertEqual([game.title for game in games], ["Co-op Test Game", "Other Game"])
        first = games[0]
        self.assertEqual(first.appid, 123)
        self.assertEqual(first.app_type, "game")
        self.assertEqual(first.platforms, ["pc", "macos", "linux"])
        self.assertIn("action", first.genres)
        self.assertIn("co-op", first.tags)
        self.assertNotIn("simplified chinese", first.tags)
        self.assertEqual(first.supported_languages, ["english", "schinese", "japanese"])
        self.assertTrue(first.language_data_available)
        self.assertIn("steam_appdetails", first.internal_source_markers)
        self.assertEqual(first.metacritic, 88)
        self.assertEqual(first.released, "2026 年 1 月 1 日")
        self.assertEqual(first.release_date, "2026 年 1 月 1 日")
        self.assertEqual(first.raw_url, "https://store.steampowered.com/app/123/")
        self.assertIn("Steam description", first.description or "")
        self.assertTrue(cache.keys)
        self.assertTrue(all(key.startswith("steam:") for key in cache.keys))

        await client.search_games(search="co-op", page_size=2)

        self.assertEqual(http_client.call_count, 3)

    async def test_game_detail_release_status_refreshes_after_one_hour(self) -> None:
        now = [1.0]
        cache = TimedMemoryCache(lambda: now[0])
        upcoming = steam_detail_payload()
        upcoming["release_date"] = {
            "date": "Coming soon",
            "coming_soon": True,
        }
        http_client = MutableDetailHttpClient(upcoming)
        client = SteamClient(
            http_client,
            cache,
            cache_ttl_hours=24,
            clock=lambda: now[0],
        )

        first = await client.get_game_detail(123)
        now[0] += 60 * 60 - 1
        still_fresh = await client.get_game_detail(123)
        http_client.payload = steam_detail_payload()
        now[0] += 2
        refreshed = await client.get_game_detail(123)

        self.assertTrue(first.coming_soon)
        self.assertTrue(still_fresh.coming_soon)
        self.assertFalse(refreshed.coming_soon)
        self.assertEqual(http_client.call_count, 2)

    async def test_game_detail_uses_stale_release_status_for_at_most_six_hours(
        self,
    ) -> None:
        now = [1.0]
        cache = TimedMemoryCache(lambda: now[0])
        upcoming = steam_detail_payload()
        upcoming["release_date"] = {
            "date": "Coming soon",
            "coming_soon": True,
        }
        http_client = MutableDetailHttpClient(upcoming)
        client = SteamClient(
            http_client,
            cache,
            cache_ttl_hours=24,
            clock=lambda: now[0],
        )

        await client.get_game_detail(123)
        http_client.failure = httpx.ConnectError("offline")
        now[0] += 60 * 60 + 1
        stale = await client.get_game_detail(123)

        self.assertTrue(stale.coming_soon)
        self.assertEqual(http_client.call_count, 3)

        now[0] += 5 * 60 * 60 + 1
        with self.assertRaises(SteamApiError):
            await client.get_game_detail(123)
        self.assertEqual(http_client.call_count, 5)

    async def test_sqlite_game_detail_refreshes_after_one_hour(self) -> None:
        now = 1_700_000_000.0
        upcoming = steam_detail_payload()
        upcoming["release_date"] = {
            "date": "Coming soon",
            "coming_soon": True,
        }
        released = steam_detail_payload()

        with (
            tempfile.TemporaryDirectory() as temp_dir,
            patch.object(repository_module.time, "time", return_value=now),
        ):
            cache = SQLiteCacheRepository(Path(temp_dir) / "cache.sqlite3")
            http_client = MutableDetailHttpClient(upcoming)
            client = SteamClient(
                http_client,
                cache,
                cache_ttl_hours=24,
                clock=lambda: now,
            )

            first = await client.get_game_detail(123)
            age_sqlite_cache_rows(
                cache,
                "appdetails:v1",
                60 * 60 - 1,
                now=now,
                retention_seconds=6 * 60 * 60,
            )
            still_fresh = await client.get_game_detail(123)
            http_client.payload = released
            age_sqlite_cache_rows(
                cache,
                "appdetails:v1",
                60 * 60 + 1,
                now=now,
                retention_seconds=6 * 60 * 60,
            )
            refreshed = await client.get_game_detail(123)

        self.assertTrue(first.coming_soon)
        self.assertTrue(still_fresh.coming_soon)
        self.assertFalse(refreshed.coming_soon)
        self.assertEqual(http_client.call_count, 2)

    async def test_sqlite_game_detail_stale_fallback_expires_after_six_hours(
        self,
    ) -> None:
        now = 1_700_000_000.0
        upcoming = steam_detail_payload()
        upcoming["release_date"] = {
            "date": "Coming soon",
            "coming_soon": True,
        }

        with (
            tempfile.TemporaryDirectory() as temp_dir,
            patch.object(repository_module.time, "time", return_value=now),
        ):
            cache = SQLiteCacheRepository(Path(temp_dir) / "cache.sqlite3")
            http_client = MutableDetailHttpClient(upcoming)
            client = SteamClient(
                http_client,
                cache,
                cache_ttl_hours=24,
                clock=lambda: now,
            )

            await client.get_game_detail(123)
            http_client.failure = httpx.ConnectError("offline")
            age_sqlite_cache_rows(
                cache,
                "appdetails:v1",
                60 * 60 + 1,
                now=now,
                retention_seconds=6 * 60 * 60,
            )
            stale = await client.get_game_detail(123)

            self.assertTrue(stale.coming_soon)
            self.assertEqual(http_client.call_count, 3)

            age_sqlite_cache_rows(
                cache,
                "appdetails:v1",
                6 * 60 * 60 + 1,
                now=now,
                retention_seconds=6 * 60 * 60,
            )
            with self.assertRaises(SteamApiError):
                await client.get_game_detail(123)
            self.assertEqual(http_client.call_count, 5)
            self.assertEqual(sqlite_cache_row_count(cache, "appdetails:v1"), 0)

    async def test_search_games_skips_non_games_and_failed_details(self) -> None:
        http_client = FakeHttpClient(
            {
                "https://store.steampowered.com/api/storesearch/": {
                    "items": [
                        {"type": "app", "id": 123, "name": "Base Game"},
                        {"type": "app", "id": 456, "name": "Expansion"},
                        {"type": "app", "id": 789, "name": "Missing Detail"},
                    ]
                },
                "https://store.steampowered.com/api/appdetails": {
                    "123": {"success": True, "data": steam_detail_payload()},
                    "456": {
                        "success": True,
                        "data": {
                            **steam_detail_payload(),
                            "name": "Expansion",
                            "type": "dlc",
                        },
                    },
                    "789": {"success": False},
                },
            }
        )
        client = SteamClient(http_client, MemoryCache(), cache_ttl_hours=24)

        games = await client.search_games(search="base game", page_size=3)

        self.assertEqual([game.appid for game in games], [123])

    async def test_game_detail_preserves_all_reported_app_types(self) -> None:
        for app_type in ("game", "dlc", "demo", "music", "tool"):
            with self.subTest(app_type=app_type):
                payload = {**steam_detail_payload(), "type": app_type}
                client = SteamClient(
                    FakeHttpClient(
                        {
                            "https://store.steampowered.com/api/appdetails": {
                                "123": {"success": True, "data": payload}
                            }
                        }
                    ),
                    MemoryCache(),
                    cache_ttl_hours=24,
                )

                game = await client.get_game_detail(123)

                self.assertEqual(game.app_type, app_type)

        payload = steam_detail_payload()
        payload.pop("type")
        client = SteamClient(
            FakeHttpClient(
                {
                    "https://store.steampowered.com/api/appdetails": {
                        "123": {"success": True, "data": payload}
                    }
                }
            ),
            MemoryCache(),
            cache_ttl_hours=24,
        )

        game = await client.get_game_detail(123)

        self.assertIsNone(game.app_type)

    async def test_missing_supported_languages_is_explicitly_unknown(self) -> None:
        payload = steam_detail_payload()
        payload.pop("supported_languages")
        http_client = FakeHttpClient(
            {
                "https://store.steampowered.com/api/appdetails": {
                    "123": {"success": True, "data": payload}
                }
            }
        )
        client = SteamClient(http_client, MemoryCache(), cache_ttl_hours=24)

        game = await client.get_game_detail(123)

        self.assertEqual(game.supported_languages, [])
        self.assertFalse(game.language_data_available)

    async def test_review_summary_parses_total_and_recent_positive_ratio(self) -> None:
        cache = MemoryCache()
        http_client = FakeHttpClient(
            {
                "https://store.steampowered.com/appreviews/123": {
                    "success": 1,
                    "query_summary": {
                        "total_reviews": 100,
                        "total_positive": 78,
                    },
                },
            }
        )
        client = SteamClient(http_client, cache, cache_ttl_hours=24)

        summary = await client.get_review_summary(123)

        self.assertEqual(summary.total_reviews, 100)
        self.assertEqual(summary.positive_ratio, 0.78)
        self.assertEqual(summary.recent_positive_ratio, 0.78)
        self.assertTrue(any(key.startswith("steam:") for key in cache.keys))

    async def test_review_summary_rejects_invalid_counts_and_impossible_ratio(self) -> None:
        invalid_summaries = (
            {"total_reviews": True, "total_positive": 1},
            {"total_reviews": 10.5, "total_positive": 8},
            {"total_reviews": float("nan"), "total_positive": 0},
            {"total_reviews": -1, "total_positive": 0},
            {"total_reviews": 10, "total_positive": -1},
            {"total_reviews": 10, "total_positive": 11},
            {"total_reviews": 10},
        )
        for summary in invalid_summaries:
            with self.subTest(summary=summary):
                client = SteamClient(
                    FakeHttpClient(
                        {
                            "https://store.steampowered.com/appreviews/123": {
                                "success": 1,
                                "query_summary": summary,
                            }
                        }
                    ),
                    MemoryCache(),
                    cache_ttl_hours=24,
                )
                with self.assertRaises(SteamApiError):
                    await client.get_review_summary(123)

    async def test_review_summary_retries_temporary_contract_failure_once(self) -> None:
        calls = 0

        def handler(request: httpx.Request) -> httpx.Response:
            nonlocal calls
            calls += 1
            payload = (
                {"success": 1, "query_summary": {"total_reviews": "invalid"}}
                if calls == 1
                else {
                    "success": 1,
                    "query_summary": {
                        "total_reviews": 100,
                        "total_positive": 80,
                    },
                }
            )
            return httpx.Response(200, json=payload, request=request)

        async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as http:
            summary = await SteamClient(http, MemoryCache()).get_review_summary(123)

        self.assertEqual(summary.positive_ratio, 0.8)
        self.assertEqual(calls, 2)

    async def test_review_summary_explicit_failure_is_permanent(self) -> None:
        calls = 0

        def handler(request: httpx.Request) -> httpx.Response:
            nonlocal calls
            calls += 1
            return httpx.Response(200, json={"success": 0}, request=request)

        async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as http:
            with self.assertRaises(SteamApiError):
                await SteamClient(http, MemoryCache()).get_review_summary(123)

        self.assertEqual(calls, 1)

    async def test_owned_games_use_web_api_key_and_cache_playtime(self) -> None:
        cache = MemoryCache()
        http_client = FakeHttpClient(
            {
                "https://api.steampowered.com/IPlayerService/GetOwnedGames/v0001/": {
                    "response": {
                        "game_count": 2,
                        "games": [
                            {"appid": 123, "name": "Played Game", "playtime_forever": 90},
                            {"appid": 456, "name": "Unplayed Game", "playtime_forever": 0},
                        ],
                    }
                },
            }
        )
        client = SteamClient(
            http_client,
            cache,
            cache_ttl_hours=24,
            steam_api_key="STEAM_KEY",
        )

        games = await client.get_owned_games("76561198000000000")

        self.assertEqual([game.appid for game in games], [123, 456])
        self.assertEqual(games[0].name, "Played Game")
        self.assertEqual(games[0].playtime_forever, 90)
        self.assertNotIn("key", http_client.last_params)
        self.assertEqual(http_client.last_headers["x-webapi-key"], "STEAM_KEY")
        self.assertIs(http_client.last_follow_redirects, False)
        self.assertEqual(http_client.last_params["steamid"], "76561198000000000")
        self.assertEqual(http_client.last_params["include_appinfo"], 1)
        self.assertEqual(http_client.last_params["include_played_free_games"], 1)

        await client.get_owned_games("76561198000000000")

        self.assertEqual(http_client.call_count, 1)

    async def test_owned_games_cache_varies_by_api_key_fingerprint(self) -> None:
        cache = MemoryCache()
        response = {
            "https://api.steampowered.com/IPlayerService/GetOwnedGames/v0001/": {
                "response": {"games": []}
            }
        }
        first_http = FakeHttpClient(response)
        second_http = FakeHttpClient(response)

        await SteamClient(
            first_http,
            cache,
            steam_api_key="FIRST_PRIVATE_KEY",
        ).get_owned_games("76561198000000000")
        await SteamClient(
            second_http,
            cache,
            steam_api_key="SECOND_PRIVATE_KEY",
        ).get_owned_games("76561198000000000")

        self.assertEqual(first_http.call_count, 1)
        self.assertEqual(second_http.call_count, 1)
        self.assertEqual(len(cache.payloads), 2)
        self.assertTrue(
            all(
                secret not in key
                for key in cache.payloads
                for secret in ("FIRST_PRIVATE_KEY", "SECOND_PRIVATE_KEY")
            )
        )

    async def test_owned_games_rejects_redirect_without_disclosing_key(self) -> None:
        secret = "VERY_PRIVATE_STEAM_KEY"
        requests: list[httpx.Request] = []

        def redirect_handler(request: httpx.Request) -> httpx.Response:
            requests.append(request)
            return httpx.Response(
                302,
                headers={"Location": "https://attacker.invalid/collect"},
                request=request,
            )

        async with httpx.AsyncClient(
            transport=httpx.MockTransport(redirect_handler)
        ) as http_client:
            client = SteamClient(
                http_client,
                MemoryCache(),
                steam_api_key=secret,
            )
            with self.assertRaises(SteamApiError) as raised:
                await client.get_owned_games("76561198000000000")

        self.assertEqual(len(requests), 1)
        self.assertEqual(requests[0].url.host, "api.steampowered.com")
        self.assertNotIn(secret, str(requests[0].url))
        self.assertNotIn(secret, str(raised.exception))

    async def test_owned_games_http_error_never_contains_key_or_keyed_url(self) -> None:
        secret = "VERY_PRIVATE_STEAM_KEY"
        requests: list[httpx.Request] = []

        def forbidden_handler(request: httpx.Request) -> httpx.Response:
            requests.append(request)
            return httpx.Response(403, request=request)

        async with httpx.AsyncClient(
            transport=httpx.MockTransport(forbidden_handler)
        ) as http_client:
            client = SteamClient(
                http_client,
                MemoryCache(),
                steam_api_key=secret,
            )
            with self.assertRaises(SteamApiError) as raised:
                await client.get_owned_games("76561198000000000")

        self.assertEqual(len(requests), 1)
        self.assertEqual(requests[0].headers["x-webapi-key"], secret)
        self.assertNotIn(secret, str(requests[0].url))
        self.assertNotIn(secret, str(raised.exception))

    async def test_owned_games_retries_invalid_json_once(self) -> None:
        calls = 0

        def handler(request: httpx.Request) -> httpx.Response:
            nonlocal calls
            calls += 1
            if calls == 1:
                return httpx.Response(200, text="not-json", request=request)
            return httpx.Response(
                200,
                json={
                    "response": {
                        "games": [
                            {
                                "appid": 123,
                                "name": "Recovered Game",
                                "playtime_forever": 0,
                            }
                        ]
                    }
                },
                request=request,
            )

        async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as http:
            games = await SteamClient(
                http,
                MemoryCache(),
                steam_api_key="PRIVATE_KEY",
            ).get_owned_games("76561198000000000")

        self.assertEqual([game.appid for game in games], [123])
        self.assertEqual(calls, 2)

    async def test_owned_games_retries_temporary_contract_failure_once(self) -> None:
        calls = 0

        def handler(request: httpx.Request) -> httpx.Response:
            nonlocal calls
            calls += 1
            payload = (
                {"response": {"games": "temporarily-invalid"}}
                if calls == 1
                else {
                    "response": {
                        "games": [
                            {
                                "appid": 123,
                                "name": "Recovered Game",
                                "playtime_forever": 0,
                            }
                        ]
                    }
                }
            )
            return httpx.Response(200, json=payload, request=request)

        async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as http:
            games = await SteamClient(
                http,
                MemoryCache(),
                steam_api_key="PRIVATE_KEY",
            ).get_owned_games("76561198000000000")

        self.assertEqual([game.appid for game in games], [123])
        self.assertEqual(calls, 2)

    async def test_owned_games_uses_stale_after_two_transient_failures(self) -> None:
        now = [1.0]
        calls = 0

        def handler(request: httpx.Request) -> httpx.Response:
            nonlocal calls
            calls += 1
            if calls == 1:
                return httpx.Response(
                    200,
                    json={
                        "response": {
                            "games": [
                                {
                                    "appid": 123,
                                    "name": "Cached Game",
                                    "playtime_forever": 0,
                                }
                            ]
                        }
                    },
                    request=request,
                )
            return httpx.Response(503, request=request)

        cache = TimedMemoryCache(lambda: now[0])
        async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as http:
            client = SteamClient(
                http,
                cache,
                cache_ttl_hours=24,
                steam_api_key="PRIVATE_KEY",
            )
            await client.get_owned_games("76561198000000000")
            now[0] += 24 * 60 * 60 + 1
            games = await client.get_owned_games("76561198000000000")

        self.assertEqual([game.appid for game in games], [123])
        self.assertEqual(calls, 3)

    async def test_owned_games_permanent_error_never_uses_stale(self) -> None:
        now = [1.0]
        calls = 0

        def handler(request: httpx.Request) -> httpx.Response:
            nonlocal calls
            calls += 1
            if calls == 1:
                return httpx.Response(
                    200,
                    json={"response": {"games": []}},
                    request=request,
                )
            return httpx.Response(403, request=request)

        cache = TimedMemoryCache(lambda: now[0])
        async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as http:
            client = SteamClient(
                http,
                cache,
                cache_ttl_hours=24,
                steam_api_key="PRIVATE_KEY",
            )
            await client.get_owned_games("76561198000000000")
            now[0] += 24 * 60 * 60 + 1
            with self.assertRaises(SteamApiError) as raised:
                await client.get_owned_games("76561198000000000")

        self.assertEqual(raised.exception.status_code, 403)
        self.assertEqual(calls, 2)

    async def test_appdetails_retries_invalid_json_once(self) -> None:
        calls = 0

        def handler(request: httpx.Request) -> httpx.Response:
            nonlocal calls
            calls += 1
            if calls == 1:
                return httpx.Response(200, text="not-json", request=request)
            return httpx.Response(
                200,
                json={"123": {"success": True, "data": steam_detail_payload()}},
                request=request,
            )

        async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as http:
            game = await SteamClient(http, MemoryCache()).get_game_detail(123)

        self.assertEqual(game.appid, 123)
        self.assertEqual(calls, 2)

    async def test_appdetails_retries_transient_contract_failure_once(self) -> None:
        calls = 0

        def handler(request: httpx.Request) -> httpx.Response:
            nonlocal calls
            calls += 1
            payload = (
                {}
                if calls == 1
                else {"123": {"success": True, "data": steam_detail_payload()}}
            )
            return httpx.Response(200, json=payload, request=request)

        async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as http:
            game = await SteamClient(http, MemoryCache()).get_game_detail(123)

        self.assertEqual(game.appid, 123)
        self.assertEqual(calls, 2)

    async def test_appdetails_uses_one_stale_payload_after_invalid_json_retries(
        self,
    ) -> None:
        now = [1.0]
        calls = 0

        def handler(request: httpx.Request) -> httpx.Response:
            nonlocal calls
            calls += 1
            if calls == 1:
                return httpx.Response(
                    200,
                    json={"123": {"success": True, "data": steam_detail_payload()}},
                    request=request,
                )
            return httpx.Response(200, text="not-json", request=request)

        cache = TimedMemoryCache(lambda: now[0])
        async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as http:
            client = SteamClient(http, cache, clock=lambda: now[0])
            first = await client.get_game_detail(123)
            now[0] += 60 * 60 + 1
            stale = await client.get_game_detail(123)

        self.assertEqual(first.title, stale.title)
        self.assertEqual(calls, 3)
        self.assertEqual(len(cache.payloads), 1)

    async def test_appdetails_success_false_is_permanent_and_does_not_use_stale(
        self,
    ) -> None:
        now = [1.0]
        calls = 0

        def handler(request: httpx.Request) -> httpx.Response:
            nonlocal calls
            calls += 1
            payload = (
                {"123": {"success": True, "data": steam_detail_payload()}}
                if calls == 1
                else {"123": {"success": False}}
            )
            return httpx.Response(200, json=payload, request=request)

        cache = TimedMemoryCache(lambda: now[0])
        async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as http:
            client = SteamClient(http, cache, clock=lambda: now[0])
            await client.get_game_detail(123)
            now[0] += 60 * 60 + 1
            with self.assertRaises(SteamApiError):
                await client.get_game_detail(123)

        self.assertEqual(calls, 2)

    async def test_appdetails_404_is_permanent_and_does_not_use_stale(self) -> None:
        now = [1.0]
        calls = 0

        def handler(request: httpx.Request) -> httpx.Response:
            nonlocal calls
            calls += 1
            if calls == 1:
                return httpx.Response(
                    200,
                    json={"123": {"success": True, "data": steam_detail_payload()}},
                    request=request,
                )
            return httpx.Response(404, request=request)

        cache = TimedMemoryCache(lambda: now[0])
        async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as http:
            client = SteamClient(http, cache, clock=lambda: now[0])
            await client.get_game_detail(123)
            now[0] += 60 * 60 + 1
            with self.assertRaises(SteamApiError) as raised:
                await client.get_game_detail(123)

        self.assertEqual(calls, 2)
        self.assertEqual(raised.exception.code, "steam_request_rejected")

    async def test_more_like_retries_temporary_contract_failure_once(self) -> None:
        calls = 0

        def handler(request: httpx.Request) -> httpx.Response:
            nonlocal calls
            calls += 1
            content = (
                "<html><body>temporary response</body></html>"
                if calls == 1
                else '<section id="released"></section>'
            )
            return httpx.Response(200, text=content, request=request)

        async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as http:
            result = await SteamClient(http, MemoryCache()).get_more_like(123)

        self.assertEqual(result.hits, ())
        self.assertEqual(calls, 2)

    async def test_more_like_permanent_error_never_uses_stale(self) -> None:
        cache = MemoryCache()

        def success_handler(request: httpx.Request) -> httpx.Response:
            return httpx.Response(
                200,
                text='<section id="released"></section>',
                request=request,
            )

        async with httpx.AsyncClient(
            transport=httpx.MockTransport(success_handler)
        ) as http:
            await SteamClient(http, cache).get_more_like(123)

        calls = 0

        def missing_handler(request: httpx.Request) -> httpx.Response:
            nonlocal calls
            calls += 1
            return httpx.Response(404, request=request)

        async with httpx.AsyncClient(
            transport=httpx.MockTransport(missing_handler)
        ) as http:
            with self.assertRaises(SteamApiError):
                await SteamClient(http, cache).get_more_like(123, reuse_cache=False)

        self.assertEqual(calls, 1)

    async def test_owned_games_skip_invalid_appids_and_playtime(self) -> None:
        invalid_values = [True, 1.5, float("nan"), float("inf"), -1, "1.0"]
        raw_games = [
            {"appid": value, "name": "Invalid App", "playtime_forever": 0}
            for value in invalid_values
        ]
        raw_games.extend(
            {
                "appid": 100 + index,
                "name": "Invalid Playtime",
                "playtime_forever": value,
            }
            for index, value in enumerate(invalid_values)
        )
        raw_games.extend(
            [
                {"appid": 200, "name": "Valid Zero", "playtime_forever": 0},
                {"appid": "201", "name": "Valid Played", "playtime_forever": "90"},
            ]
        )
        client = SteamClient(
            FakeHttpClient(
                {
                    "https://api.steampowered.com/IPlayerService/GetOwnedGames/v0001/": {
                        "response": {"games": raw_games}
                    }
                }
            ),
            MemoryCache(),
            cache_ttl_hours=24,
            steam_api_key="STEAM_KEY",
        )

        games = await client.get_owned_games("76561198000000000")

        self.assertEqual(
            [(game.appid, game.playtime_forever) for game in games],
            [(200, 0), (201, 90)],
        )

    async def test_popular_tags_fetches_english_tag_vocabulary_and_uses_cache(self) -> None:
        cache = MemoryCache()
        http_client = FakeHttpClient(
            {
                "https://store.steampowered.com/tagdata/populartags/english": [
                    {"tagid": 87918, "name": "Farming Sim"},
                    {"tagid": 10235, "name": "Life Sim"},
                    {"tagid": 3964, "name": "Pixel Graphics"},
                ],
            }
        )
        client = SteamClient(http_client, cache, cache_ttl_hours=24)

        tags = await client.get_popular_tags()

        self.assertEqual(
            tags,
            [
                {"tagid": 87918, "name": "Farming Sim"},
                {"tagid": 10235, "name": "Life Sim"},
                {"tagid": 3964, "name": "Pixel Graphics"},
            ],
        )
        self.assertEqual(http_client.last_params, {})

        cached = await client.get_popular_tags()

        self.assertEqual(cached, tags)
        self.assertEqual(http_client.call_count, 1)

    async def test_store_page_tags_fetches_english_user_tags_and_uses_cache(self) -> None:
        cache = MemoryCache()
        http_client = FakeHttpClient(
            {
                "https://store.steampowered.com/app/413150/": (
                    "<html><body>"
                    "Popular user-defined tags for this product:"
                    '<a class="app_tag" href="/tags/en/Farming%20Sim/">Farming Sim</a>'
                    '<a class="app_tag" href="/tags/en/Pixel%20Graphics/">Pixel Graphics</a>'
                    '<a class="app_tag" href="/tags/en/Multiplayer/">Multiplayer</a>'
                    '<a class="app_tag" href="/tags/en/Life%20Sim/">Life Sim</a>'
                    '<a class="app_tag" href="/tags/en/RPG/">RPG</a>'
                    '<a class="app_tag" href="/tags/en/Relaxing/">Relaxing</a>'
                    "</body></html>"
                ),
            }
        )
        client = SteamClient(http_client, cache, cache_ttl_hours=24)

        tags = await client.get_store_page_tags(413150)

        self.assertEqual(
            tags,
            ["Farming Sim", "Pixel Graphics", "Multiplayer", "Life Sim", "RPG", "Relaxing"],
        )
        self.assertEqual(http_client.last_params, {"l": "english"})

        cached = await client.get_store_page_tags(413150)

        self.assertEqual(cached, tags)
        self.assertEqual(http_client.call_count, 1)

    async def test_store_page_tags_retries_temporary_page_contract_failure(self) -> None:
        calls = 0

        def handler(request: httpx.Request) -> httpx.Response:
            nonlocal calls
            calls += 1
            text = (
                "<html><body>temporary response</body></html>"
                if calls == 1
                else (
                    "<html><body>Popular user-defined tags for this product:"
                    '<a class="app_tag">Puzzle</a></body></html>'
                )
            )
            return httpx.Response(200, text=text, request=request)

        async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as http:
            tags = await SteamClient(http, MemoryCache()).get_store_page_tags(123)

        self.assertEqual(tags, ["Puzzle"])
        self.assertEqual(calls, 2)

    async def test_owned_games_requires_steam_web_api_key(self) -> None:
        client = SteamClient(FakeHttpClient({}), MemoryCache(), cache_ttl_hours=24)

        with self.assertRaises(SteamApiError):
            await client.get_owned_games("76561198000000000")


def steam_detail_payload() -> dict[str, Any]:
    return {
        "name": "Co-op Test Game",
        "short_description": "Steam description",
        "type": "game",
        "platforms": {"windows": True, "mac": True, "linux": True},
        "genres": [{"description": "Action"}, {"description": "Adventure"}],
        "categories": [{"description": "Co-op"}, {"description": "Online Co-op"}],
        "supported_languages": "English, Simplified Chinese, Japanese",
        "metacritic": {"score": 88},
        "release_date": {"date": "2026 年 1 月 1 日", "coming_soon": False},
    }


class FakeResponse:
    def __init__(self, payload: Any) -> None:
        self.payload = payload
        self.text = str(payload)

    def raise_for_status(self) -> None:
        return None

    def json(self) -> Any:
        return self.payload


class FakeHttpClient:
    def __init__(self, responses: dict[str, Any]) -> None:
        self.responses = responses
        self.call_count = 0
        self.last_params: dict[str, Any] = {}
        self.last_headers: dict[str, Any] = {}
        self.last_follow_redirects: bool | None = None

    async def get(
        self,
        url: str,
        params: dict[str, Any],
        **kwargs: Any,
    ) -> FakeResponse:
        self.call_count += 1
        self.last_params = dict(params)
        self.last_headers = dict(kwargs.get("headers") or {})
        self.last_follow_redirects = kwargs.get("follow_redirects")
        return FakeResponse(self.responses[url])


class MemoryCache:
    def __init__(self) -> None:
        self.payloads: dict[str, Any] = {}
        self.keys: list[str] = []

    async def get_json(self, key: str, _ttl_hours: int) -> Any | None:
        return self.payloads.get(key)

    async def set_json(self, key: str, payload: Any) -> None:
        self.keys.append(key)
        self.payloads[key] = payload


class TimedMemoryCache:
    def __init__(self, clock: Any) -> None:
        self.clock = clock
        self.payloads: dict[str, tuple[Any, float]] = {}

    async def get_json(self, key: str, ttl_hours: int) -> Any | None:
        stored = self.payloads.get(key)
        if stored is None:
            return None
        payload, stored_at = stored
        if self.clock() - stored_at > max(int(ttl_hours), 0) * 60 * 60:
            return None
        return payload

    async def set_json(self, key: str, payload: Any) -> None:
        self.payloads[key] = (payload, self.clock())


def age_sqlite_cache_rows(
    repository: SQLiteCacheRepository,
    suffix: str,
    seconds: float,
    *,
    now: float | None = None,
    retention_seconds: float | None = None,
) -> None:
    created_at = (time.time() if now is None else now) - seconds
    with sqlite3.connect(repository.db_path) as connection:
        if retention_seconds is None:
            cursor = connection.execute(
                "UPDATE cache SET created_at = ? WHERE key LIKE ?",
                (created_at, f"%{suffix}"),
            )
        else:
            cursor = connection.execute(
                """
                UPDATE cache
                SET created_at = ?, expires_at = ?
                WHERE key LIKE ?
                """,
                (created_at, created_at + retention_seconds, f"%{suffix}"),
            )
    if cursor.rowcount <= 0:
        raise AssertionError(f"no SQLite cache row matched suffix {suffix}")


def sqlite_cache_row_count(
    repository: SQLiteCacheRepository,
    suffix: str,
) -> int:
    with sqlite3.connect(repository.db_path) as connection:
        row = connection.execute(
            "SELECT COUNT(*) FROM cache WHERE key LIKE ?",
            (f"%{suffix}",),
        ).fetchone()
    return int(row[0]) if row else 0


class MutableDetailHttpClient:
    def __init__(self, payload: dict[str, Any]) -> None:
        self.payload = payload
        self.failure: httpx.HTTPError | None = None
        self.call_count = 0

    async def get(
        self,
        url: str,
        params: dict[str, Any],
        **_kwargs: Any,
    ) -> FakeResponse:
        self.call_count += 1
        if self.failure is not None:
            request = httpx.Request("GET", url, params=params)
            raise type(self.failure)(str(self.failure), request=request)
        return FakeResponse(
            {
                str(params["appids"]): {
                    "success": True,
                    "data": self.payload,
                }
            }
        )


if __name__ == "__main__":
    unittest.main()
