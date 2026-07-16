from __future__ import annotations

import unittest
from types import SimpleNamespace
from typing import Any
from unittest.mock import patch

from astrbot_plugin_steam_game_recommender.clients.steam import SteamApiError
from astrbot_plugin_steam_game_recommender.services.recommendation_intent import (
    build_recommendation_intent,
)
from astrbot_plugin_steam_game_recommender.services.reference_matching import (
    ReferenceMatch,
)
from astrbot_plugin_steam_game_recommender.services.similarity_ranker import (
    build_profile_from_preference,
)
from astrbot_plugin_steam_game_recommender.services.steam_index import (
    MAX_REFERENCE_DETAIL_ATTEMPTS_PER_ENTITY,
    SteamGameIndexService,
    negative_reference_candidates,
    reference_candidates,
    references_are_resolved,
    search_terms_for,
)
from astrbot_plugin_steam_game_recommender.storage.models import (
    GameCandidate,
    GamePreference,
    SteamSearchHit,
)


class ReferenceGroupResolutionTest(unittest.IsolatedAsyncioTestCase):
    async def test_reference_detail_attempts_have_a_per_entity_hard_limit(self) -> None:
        reference = build_recommendation_intent(
            GamePreference(reference_games_like=["Reference"])
        ).references[0]
        hits = [
            SteamSearchHit(appid=index, title=f"Reference Candidate {index}")
            for index in range(1, 11)
        ]
        client = FrozenReferenceSteamClient(search_results={}, details={})
        service = SteamGameIndexService(client, MemoryCache({}), clock=lambda: 1_000.0)

        def select_first(_reference, remaining):
            if not remaining:
                return None
            return ReferenceMatch(
                hit=remaining[0],
                confidence=1.0,
                matched_alias="Reference",
                match_kind="exact",
            )

        with patch(
            "astrbot_plugin_steam_game_recommender.services.steam_index.match_reference_query",
            side_effect=select_first,
        ):
            match, candidate, failures = await service._select_reference_candidate(
                reference,
                hits,
                {},
            )

        self.assertIsNone(match)
        self.assertIsNone(candidate)
        self.assertEqual(
            client.detail_appids,
            list(range(1, MAX_REFERENCE_DETAIL_ATTEMPTS_PER_ENTITY + 1)),
        )
        self.assertEqual(len(failures), MAX_REFERENCE_DETAIL_ATTEMPTS_PER_ENTITY)

    async def test_reference_entity_budget_bounds_actual_search_calls(self) -> None:
        preference = GamePreference(
            reference_entities=[
                {
                    "display_title": f"Reference {index}",
                    "aliases": [
                        f"Reference {index} Alias {alias_index}"
                        for alias_index in range(1, 6)
                    ],
                    "polarity": "positive",
                }
                for index in range(1, 6)
            ]
        )
        client = FrozenReferenceSteamClient(search_results={}, details={})
        service = SteamGameIndexService(client, MemoryCache({}), clock=lambda: 1_000.0)

        await service.refresh_entries(preference, [], target_pool=0)

        # Three entities x three aliases x two locales.  The frozen client only
        # exposes the store-search source, so this is the exact upper bound here.
        self.assertGreater(len(client.search_calls), 0)
        self.assertLessEqual(len(client.search_calls), 18)

    async def test_resolves_localized_aliases_as_one_enriched_group(self) -> None:
        preference = GamePreference(
            reference_games_like=["黑暗之魂"],
            reference_search_terms=["Dark Souls"],
        )
        client = FrozenReferenceSteamClient(
            search_results={
                ("黑暗之魂", "schinese"): [
                    SteamSearchHit(appid=10, title="黑暗之魂：重制版")
                ],
                ("Dark Souls", "english"): [
                    SteamSearchHit(
                        appid=10,
                        title="DARK SOULS: REMASTERED",
                        store_url="https://store.steampowered.com/app/10/",
                    )
                ],
            },
            details={
                10: game(10, "DARK SOULS: REMASTERED", ["Action", "RPG"]),
            },
            store_tags={10: ["Souls-like", "Action", "RPG"]},
            review_totals={10: 12_345},
        )
        service = SteamGameIndexService(client, MemoryCache({}), clock=lambda: 1_000.0)

        with self.assertLogs(
            "astrbot_plugin_steam_game_recommender.services.steam_index",
            level="DEBUG",
        ) as logs:
            entries = await service.refresh_entries(preference, [], target_pool=1)

        self.assertEqual(len(preference.resolved_reference_games), 1)
        resolved = preference.resolved_reference_games[0]
        self.assertEqual(resolved.raw_text, "黑暗之魂")
        self.assertEqual(resolved.canonical_title, "DARK SOULS: REMASTERED")
        self.assertEqual(resolved.appid, 10)
        self.assertEqual(resolved.source, "steam_alias_group")
        self.assertEqual(resolved.polarity, "like")
        self.assertGreaterEqual(resolved.confidence, 0.9)
        self.assertEqual(preference.parse_warnings, [])
        self.assertTrue(references_are_resolved(preference))

        reference = reference_candidates(preference, entries)[0]
        self.assertEqual(reference.appid, 10)
        self.assertEqual(reference.ordered_tags, ["souls-like", "action", "rpg"])
        self.assertEqual(reference.review_total, 12_345)
        self.assertIn("reference_query:like:黑暗之魂", reference.internal_source_markers)
        self.assertTrue(
            any(
                "recommendation_reference event=resolution" in message
                and "alias_count=2" in message
                and "appid=10" in message
                for message in logs.output
            )
        )
        self.assertFalse(any("黑暗之魂" in message for message in logs.output))
        self.assertFalse(any("DARK SOULS" in message for message in logs.output))

        expected_calls = {
            (alias, language)
            for alias in ("黑暗之魂", "Dark Souls")
            for language in ("english", "schinese")
        }
        alias_calls = {
            (call["search"], call["language"])
            for call in client.search_calls
            if call["search"] in {"黑暗之魂", "Dark Souls"}
        }
        self.assertEqual(alias_calls, expected_calls)
        self.assertTrue(
            all(
                call["page_size"] == 20 and call["ordering"] == "-relevance"
                for call in client.search_calls
                if call["search"] in {"黑暗之魂", "Dark Souls"}
            )
        )
        self.assertNotIn(
            "黑暗之魂 Dark Souls",
            [call["search"] for call in client.search_calls],
        )

    async def test_failed_alias_in_every_locale_does_not_warn_when_group_resolves(
        self,
    ) -> None:
        preference = GamePreference(
            reference_games_like=["本地标题"],
            reference_search_terms=["English Alias"],
        )
        client = FrozenReferenceSteamClient(
            search_results={
                ("English Alias", "schinese"): [
                    SteamSearchHit(appid=11, title="English Alias")
                ]
            },
            details={11: game(11, "English Alias", ["Adventure"])},
        )
        service = SteamGameIndexService(client, MemoryCache({}), clock=lambda: 1_000.0)

        await service.refresh_entries(preference, [], target_pool=1)

        self.assertEqual(preference.parse_warnings, [])
        self.assertEqual(len(preference.resolved_reference_games), 1)
        self.assertEqual(preference.resolved_reference_games[0].appid, 11)
        self.assertEqual(
            {
                (call["search"], call["language"])
                for call in client.search_calls
            },
            {
                ("本地标题", "english"),
                ("本地标题", "schinese"),
                ("English Alias", "english"),
                ("English Alias", "schinese"),
            },
        )

    async def test_base_edition_beats_sequel(self) -> None:
        preference = GamePreference(reference_games_like=["Portal"])
        client = FrozenReferenceSteamClient(
            search_results={
                ("Portal", "english"): [
                    SteamSearchHit(appid=20, title="Portal 2"),
                    SteamSearchHit(appid=21, title="Portal Remastered"),
                ]
            },
            details={21: game(21, "Portal Remastered", ["Puzzle"])},
        )
        service = SteamGameIndexService(client, MemoryCache({}), clock=lambda: 1_000.0)

        await service.refresh_entries(preference, [], target_pool=1)

        self.assertEqual(preference.resolved_reference_games[0].appid, 21)
        self.assertEqual(client.detail_appids, [21])

    async def test_exact_title_dlc_is_rejected_then_base_match_is_retried(self) -> None:
        preference = GamePreference(reference_games_like=["Example Quest"])
        client = FrozenReferenceSteamClient(
            search_results={
                ("Example Quest", "english"): [
                    SteamSearchHit(appid=30, title="Example Quest"),
                    SteamSearchHit(appid=31, title="Example Quest Complete Edition"),
                ]
            },
            details={
                30: game(30, "Example Quest", ["RPG"], app_type="dlc"),
                31: game(31, "Example Quest Complete Edition", ["RPG"]),
            },
        )
        service = SteamGameIndexService(client, MemoryCache({}), clock=lambda: 1_000.0)

        entries = await service.refresh_entries(preference, [], target_pool=1)

        self.assertEqual(preference.resolved_reference_games[0].appid, 31)
        self.assertEqual(client.detail_appids, [30, 31])
        self.assertEqual([item.appid for item in reference_candidates(preference, entries)], [31])

    async def test_failed_exact_detail_is_removed_before_base_retry(self) -> None:
        preference = GamePreference(reference_games_like=["Detail Retry"])
        client = FrozenReferenceSteamClient(
            search_results={
                ("Detail Retry", "english"): [
                    SteamSearchHit(appid=32, title="Detail Retry"),
                    SteamSearchHit(appid=33, title="Detail Retry Definitive Edition"),
                ]
            },
            details={
                33: game(33, "Detail Retry Definitive Edition", ["Adventure"]),
            },
        )
        service = SteamGameIndexService(client, MemoryCache({}), clock=lambda: 1_000.0)

        await service.refresh_entries(preference, [], target_pool=1)

        self.assertEqual(client.detail_appids, [32, 33])
        self.assertEqual(preference.resolved_reference_games[0].appid, 33)

    async def test_ambiguous_group_warns_once_across_repeated_refreshes(self) -> None:
        preference = GamePreference(reference_games_like=["Twin Game"])
        client = FrozenReferenceSteamClient(
            search_results={
                ("Twin Game", "english"): [
                    SteamSearchHit(appid=40, title="Twin Game"),
                    SteamSearchHit(appid=41, title="Twin Game"),
                ]
            },
            details={},
        )
        service = SteamGameIndexService(client, MemoryCache({}), clock=lambda: 1_000.0)

        with self.assertLogs(
            "astrbot_plugin_steam_game_recommender.services.steam_index",
            level="DEBUG",
        ) as logs:
            await service.refresh_entries(preference, [], target_pool=0)
            await service.refresh_entries(preference, [], target_pool=0)

        self.assertEqual(len(preference.resolved_reference_games), 1)
        unresolved = preference.resolved_reference_games[0]
        self.assertEqual(unresolved.raw_text, "Twin Game")
        self.assertIsNone(unresolved.appid)
        self.assertEqual(unresolved.source, "steam_alias_group")
        self.assertEqual(
            preference.parse_warnings,
            ["参考游戏“Twin Game”未能可靠解析，未扩展其标签。"],
        )
        self.assertFalse(references_are_resolved(preference))
        self.assertTrue(any("unresolved" in message for message in logs.output))

    async def test_repeated_success_uses_cached_index_without_duplicates(self) -> None:
        preference = GamePreference(reference_games_like=["Cached Seed"])
        client = FrozenReferenceSteamClient(
            search_results={
                ("Cached Seed", "english"): [
                    SteamSearchHit(appid=50, title="Cached Seed")
                ]
            },
            details={50: game(50, "Cached Seed", ["Farming"])},
        )
        cache = MemoryCache({})
        service = SteamGameIndexService(client, cache, clock=lambda: 1_000.0)

        await service.refresh_entries(preference, [], target_pool=1)
        first_alias_call_count = sum(
            call["search"] == "Cached Seed" for call in client.search_calls
        )
        entries = await service.refresh_entries(preference, [], target_pool=1)

        self.assertEqual(len(preference.resolved_reference_games), 1)
        self.assertEqual(preference.resolved_reference_games[0].appid, 50)
        self.assertEqual(len([entry for entry in entries if entry.appid == 50]), 1)
        self.assertEqual(
            sum(call["search"] == "Cached Seed" for call in client.search_calls),
            first_alias_call_count,
        )

    async def test_positive_and_negative_groups_keep_polarity_and_appids(self) -> None:
        preference = GamePreference(
            reference_games_like=["Liked Seed"],
            reference_games_dislike=["Avoided Seed"],
        )
        client = FrozenReferenceSteamClient(
            search_results={
                ("Liked Seed", "english"): [
                    SteamSearchHit(appid=60, title="Liked Seed")
                ],
                ("Avoided Seed", "english"): [
                    SteamSearchHit(appid=61, title="Avoided Seed")
                ],
            },
            details={
                60: game(60, "Liked Seed", ["Farming"]),
                61: game(61, "Avoided Seed", ["Horror"]),
            },
        )
        service = SteamGameIndexService(client, MemoryCache({}), clock=lambda: 1_000.0)

        entries = await service.refresh_entries(preference, [], target_pool=2)

        resolved = {
            (item.raw_text, item.polarity): item.appid
            for item in preference.resolved_reference_games
        }
        self.assertEqual(
            resolved,
            {("Liked Seed", "like"): 60, ("Avoided Seed", "dislike"): 61},
        )
        self.assertEqual([item.appid for item in reference_candidates(preference, entries)], [60])
        self.assertEqual(
            [item.appid for item in negative_reference_candidates(preference, entries)],
            [61],
        )
        self.assertTrue(references_are_resolved(preference))

    async def test_new_reference_resolution_respects_remaining_appid_budget(self) -> None:
        preference = GamePreference(
            reference_games_like=["First Seed", "Second Seed"],
        )
        client = FrozenReferenceSteamClient(
            search_results={
                ("First Seed", "english"): [
                    SteamSearchHit(appid=62, title="First Seed")
                ],
                ("Second Seed", "english"): [
                    SteamSearchHit(appid=63, title="Second Seed")
                ],
            },
            details={
                62: game(62, "First Seed", ["Farming"]),
                63: game(63, "Second Seed", ["Strategy"]),
            },
        )
        service = SteamGameIndexService(client, MemoryCache({}), clock=lambda: 1_000.0)

        entries = await service.refresh_entries(preference, [], target_pool=1)

        self.assertEqual([entry.appid for entry in entries], [62])
        self.assertEqual(client.detail_appids, [62])
        self.assertEqual(
            [(item.raw_text, item.appid) for item in preference.resolved_reference_games],
            [("First Seed", 62)],
        )
        self.assertEqual(preference.parse_warnings, [])
        self.assertIn(
            "Second Seed",
            [call["search"] for call in client.search_calls],
        )

    async def test_no_hit_group_records_unresolved_with_zero_budget(self) -> None:
        preference = GamePreference(reference_games_like=["No Hit Seed"])
        client = FrozenReferenceSteamClient(search_results={}, details={})
        service = SteamGameIndexService(client, MemoryCache({}), clock=lambda: 1_000.0)

        with self.assertLogs(
            "astrbot_plugin_steam_game_recommender.services.steam_index",
            level="DEBUG",
        ) as logs:
            await service.refresh_entries(preference, [], target_pool=0)

        self.assertEqual(len(preference.resolved_reference_games), 1)
        self.assertIsNone(preference.resolved_reference_games[0].appid)
        self.assertEqual(
            preference.parse_warnings,
            ["参考游戏“No Hit Seed”未能可靠解析，未扩展其标签。"],
        )
        self.assertTrue(any("unresolved" in message for message in logs.output))

    async def test_zero_budget_drops_active_resolution_missing_from_records(self) -> None:
        preference = GamePreference(reference_games_like=["Missing Active Seed"])
        preference.resolved_reference_games = [
            resolved_reference("Missing Active Seed", "Missing Active Seed", 64),
        ]
        client = FrozenReferenceSteamClient(
            search_results={
                ("Missing Active Seed", "english"): [
                    SteamSearchHit(appid=64, title="Missing Active Seed")
                ]
            },
            details={64: game(64, "Missing Active Seed", ["Adventure"])},
        )
        service = SteamGameIndexService(client, MemoryCache({}), clock=lambda: 1_000.0)

        await service.refresh_entries(preference, [], target_pool=0)

        self.assertEqual(client.detail_appids, [])
        self.assertEqual(preference.resolved_reference_games, [])
        self.assertEqual(preference.parse_warnings, [])
        self.assertFalse(references_are_resolved(preference))
        self.assertIn(
            "Missing Active Seed",
            [call["search"] for call in client.search_calls],
        )

    async def test_existing_confirmed_entry_resolves_without_search(self) -> None:
        preference = GamePreference(
            reference_games_like=["本地标题"],
            reference_search_terms=["Existing Seed"],
        )
        client = FrozenReferenceSteamClient(search_results={}, details={})
        service = SteamGameIndexService(client, MemoryCache({}), clock=lambda: 1_000.0)

        entries = await service.refresh_entries(
            preference,
            [game(70, "Existing Seed", ["Strategy"])],
            target_pool=0,
        )

        self.assertEqual(preference.resolved_reference_games[0].appid, 70)
        self.assertEqual(preference.resolved_reference_games[0].raw_text, "本地标题")
        self.assertEqual(client.search_calls, [])
        self.assertEqual([item.appid for item in reference_candidates(preference, entries)], [70])

    async def test_cached_sequel_does_not_preempt_remote_base_edition(self) -> None:
        preference = GamePreference(reference_games_like=["Portal"])
        client = FrozenReferenceSteamClient(
            search_results={
                ("Portal", "english"): [
                    SteamSearchHit(appid=75, title="Portal Remastered")
                ]
            },
            details={75: game(75, "Portal Remastered", ["Puzzle"])},
        )
        service = SteamGameIndexService(client, MemoryCache({}), clock=lambda: 1_000.0)

        entries = await service.refresh_entries(
            preference,
            [game(74, "Portal 2", ["Puzzle"])],
            target_pool=1,
        )

        self.assertEqual(preference.resolved_reference_games[0].appid, 75)
        self.assertIn("Portal", [call["search"] for call in client.search_calls])
        self.assertEqual([item.appid for item in reference_candidates(preference, entries)], [75])

    async def test_cached_edition_does_not_preempt_remote_exact_title(self) -> None:
        preference = GamePreference(reference_games_like=["Portal"])
        client = FrozenReferenceSteamClient(
            search_results={
                ("Portal", "english"): [SteamSearchHit(appid=77, title="Portal")]
            },
            details={77: game(77, "Portal", ["Puzzle"])},
        )
        service = SteamGameIndexService(client, MemoryCache({}), clock=lambda: 1_000.0)

        entries = await service.refresh_entries(
            preference,
            [game(76, "Portal Remastered", ["Puzzle"])],
            target_pool=1,
        )

        self.assertEqual(preference.resolved_reference_games[0].appid, 77)
        self.assertIn("Portal", [call["search"] for call in client.search_calls])
        self.assertEqual([item.appid for item in reference_candidates(preference, entries)], [77])

    async def test_invalid_remote_exact_falls_back_to_cached_edition(self) -> None:
        preference = GamePreference(reference_games_like=["Portal"])
        client = FrozenReferenceSteamClient(
            search_results={
                ("Portal", "english"): [SteamSearchHit(appid=79, title="Portal")]
            },
            details={},
        )
        service = SteamGameIndexService(client, MemoryCache({}), clock=lambda: 1_000.0)

        entries = await service.refresh_entries(
            preference,
            [game(78, "Portal Remastered", ["Puzzle"])],
            target_pool=1,
        )

        self.assertEqual(preference.resolved_reference_games[0].appid, 78)
        self.assertEqual(preference.parse_warnings, [])
        self.assertEqual(client.detail_appids, [79])
        self.assertEqual([item.appid for item in reference_candidates(preference, entries)], [78])

    async def test_remote_fuzzy_cannot_replace_cached_base_after_exact_fails(
        self,
    ) -> None:
        preference = GamePreference(reference_games_like=["Portal"])
        client = FrozenReferenceSteamClient(
            search_results={
                ("Portal", "english"): [
                    SteamSearchHit(appid=80, title="Portal"),
                    SteamSearchHit(appid=81, title="Portals"),
                ]
            },
            details={81: game(81, "Portals", ["Puzzle"])},
        )
        service = SteamGameIndexService(client, MemoryCache({}), clock=lambda: 1_000.0)

        entries = await service.refresh_entries(
            preference,
            [game(78, "Portal Remastered", ["Puzzle"])],
            target_pool=1,
        )

        self.assertEqual(preference.resolved_reference_games[0].appid, 78)
        self.assertEqual(client.detail_appids, [80])
        self.assertEqual([item.appid for item in reference_candidates(preference, entries)], [78])

    async def test_refresh_prunes_removed_references_and_their_warnings(self) -> None:
        preference = GamePreference()
        preference.resolved_reference_games = [
            resolved_reference("Removed Seed", "Removed Seed", 72),
        ]
        preference.parse_warnings = [
            "参考游戏“Removed Seed”未能可靠解析，未扩展其标签。",
            "保留这个其他解析提醒。",
        ]
        service = SteamGameIndexService(
            FrozenReferenceSteamClient(search_results={}, details={}),
            MemoryCache({}),
            clock=lambda: 1_000.0,
        )

        await service.refresh_entries(preference, [], target_pool=0)

        self.assertEqual(preference.resolved_reference_games, [])
        self.assertEqual(preference.parse_warnings, ["保留这个其他解析提醒。"])

    async def test_polarity_flip_replaces_stale_resolution(self) -> None:
        preference = GamePreference(reference_games_dislike=["Flip Seed"])
        preference.resolved_reference_games = [
            resolved_reference("Flip Seed", "Flip Seed", 73, polarity="like"),
        ]
        service = SteamGameIndexService(
            FrozenReferenceSteamClient(search_results={}, details={}),
            MemoryCache({}),
            clock=lambda: 1_000.0,
        )
        existing = [game(73, "Flip Seed", ["Horror"])]

        entries = await service.refresh_entries(
            preference,
            existing,
            target_pool=0,
        )

        self.assertEqual(
            [(item.raw_text, item.polarity) for item in preference.resolved_reference_games],
            [("Flip Seed", "dislike")],
        )
        self.assertEqual(reference_candidates(preference, entries), [])
        self.assertEqual(
            [item.appid for item in negative_reference_candidates(preference, entries)],
            [73],
        )

    async def test_configured_english_locale_deduplicates_search_calls(self) -> None:
        preference = GamePreference(reference_games_like=["One Locale"])
        client = FrozenReferenceSteamClient(
            search_results={
                ("One Locale", "english"): [
                    SteamSearchHit(appid=71, title="One Locale")
                ]
            },
            details={71: game(71, "One Locale", ["Strategy"])},
        )
        client.language = "english"
        service = SteamGameIndexService(client, MemoryCache({}), clock=lambda: 1_000.0)

        await service.refresh_entries(preference, [], target_pool=1)

        calls = [call for call in client.search_calls if call["search"] == "One Locale"]
        self.assertEqual(len(calls), 1)
        self.assertEqual(calls[0]["language"], "english")

    def test_group_resolution_and_generic_terms_use_grouped_semantics(self) -> None:
        preference = GamePreference(
            reference_games_like=["本地标题"],
            reference_search_terms=["English Alias"],
        )
        preference.resolved_reference_games = [
            resolved_reference(
                raw_text="本地标题",
                canonical_title="English Alias",
                appid=80,
            )
        ]

        self.assertTrue(references_are_resolved(preference))
        self.assertEqual(
            search_terms_for(preference, build_profile_from_preference(preference)),
            ["popular co-op"],
        )
        self.assertEqual(len(build_recommendation_intent(preference).references), 1)

    def test_reference_candidates_ignore_stale_titles_with_matching_polarity(self) -> None:
        preference = GamePreference(reference_games_like=["Current Seed"])
        preference.resolved_reference_games = [
            resolved_reference("Current Seed", "Current Seed", 80),
            resolved_reference("Removed Seed", "Removed Seed", 81),
        ]
        entries = [
            game(80, "Current Seed", ["Farming"]),
            game(81, "Removed Seed", ["Strategy"]),
        ]

        self.assertEqual(
            [item.appid for item in reference_candidates(preference, entries)],
            [80],
        )


class MemoryCache:
    def __init__(self, payloads: dict[str, Any]) -> None:
        self.payloads = payloads

    async def get_json(self, key: str, _ttl_hours: int) -> Any | None:
        return self.payloads.get(key)

    async def set_json(self, key: str, payload: Any) -> None:
        self.payloads[key] = payload


class FrozenReferenceSteamClient:
    language = "schinese"

    def __init__(
        self,
        search_results: dict[tuple[str, str], list[SteamSearchHit]],
        details: dict[int, GameCandidate],
        store_tags: dict[int, list[str]] | None = None,
        review_totals: dict[int, int] | None = None,
    ) -> None:
        self.search_results = search_results
        self.details = details
        self.store_tags = store_tags or {}
        self.review_totals = review_totals or {}
        self.search_calls: list[dict[str, Any]] = []
        self.detail_appids: list[int] = []

    async def get_popular_tags(self) -> list[dict[str, Any]]:
        return []

    async def search_game_refs(
        self,
        search: str,
        page_size: int,
        ordering: str = "-relevance",
        language: str | None = None,
        **_kwargs: Any,
    ) -> list[SteamSearchHit]:
        resolved_language = language or self.language
        self.search_calls.append(
            {
                "search": search,
                "page_size": page_size,
                "ordering": ordering,
                "language": resolved_language,
            }
        )
        return list(self.search_results.get((search, resolved_language), []))

    async def get_game_detail(self, appid: int) -> GameCandidate:
        self.detail_appids.append(appid)
        try:
            return self.details[appid]
        except KeyError as exc:
            raise SteamApiError("detail unavailable") from exc

    async def get_store_page_tags(self, appid: int) -> list[str]:
        return list(self.store_tags.get(appid, []))

    async def get_review_summary(self, appid: int) -> SimpleNamespace:
        return SimpleNamespace(
            total_reviews=self.review_totals.get(appid, 500),
            positive_ratio=0.8,
            recent_positive_ratio=0.75,
        )


def game(
    appid: int,
    title: str,
    tags: list[str],
    app_type: str = "game",
) -> GameCandidate:
    return GameCandidate(
        appid=appid,
        title=title,
        app_type=app_type,
        platforms=["PC"],
        genres=[],
        tags=tags,
        stores=["Steam"],
        raw_url=f"https://store.steampowered.com/app/{appid}/",
        review_total=100,
        review_positive_ratio=0.7,
        review_recent_ratio=0.7,
    )


def resolved_reference(
    raw_text: str,
    canonical_title: str,
    appid: int,
    polarity: str = "like",
):
    from astrbot_plugin_steam_game_recommender.storage.models import ResolvedReferenceGame

    return ResolvedReferenceGame(
        raw_text=raw_text,
        normalized_title=raw_text,
        canonical_title=canonical_title,
        appid=appid,
        confidence=1.0,
        source="steam_alias_group",
        polarity=polarity,
    )


if __name__ == "__main__":
    unittest.main()
