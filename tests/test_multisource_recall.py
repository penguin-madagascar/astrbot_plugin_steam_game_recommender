from __future__ import annotations

import unittest
from types import SimpleNamespace
from typing import Any

from astrbot_plugin_steam_game_recommender.clients import steam
from astrbot_plugin_steam_game_recommender.clients.steam import (
    SteamApiError,
    SteamClient,
    SteamStorefrontPage,
    SteamTransientError,
)
from astrbot_plugin_steam_game_recommender.services import ranking_precedence, steam_recall
from astrbot_plugin_steam_game_recommender.services.game_identity import (
    is_confirmed_base_game,
)
from astrbot_plugin_steam_game_recommender.services.recommendation_intent import (
    IntentTagRole,
    IntentTagSource,
    QualityIntent,
    RecommendationIntent,
    WeightedIntentTag,
)
from astrbot_plugin_steam_game_recommender.services.recommendation_scoring import (
    layer_score,
)
from astrbot_plugin_steam_game_recommender.services.similarity_ranker import (
    rank_steam_candidates,
    ranked_game_sort_key,
)
from astrbot_plugin_steam_game_recommender.services.steam_index import (
    SteamGameIndexService,
    merge_current_recall_candidate_evidence,
)
from astrbot_plugin_steam_game_recommender.services.steam_price_bridge import (
    attach_price_summary,
)
from astrbot_plugin_steam_game_recommender.storage.models import (
    GameCandidate,
    GamePreference,
    GamePriceSummary,
    SteamSearchHit,
)

MORE_LIKE_HTML = """
<main>
  <section id="released">
    <a class="similar_grid_item" data-ds-appid="900"
       data-ds-tagids="[10,20]"><span class="title">Exact Seed</span></a>
    <a class="similar_grid_item" data-ds-appid="901"
       data-ds-tagids="[20,10,30]"><span class="title">Related Sequel</span></a>
  </section>
  <section id="upcoming">
    <a class="similar_grid_item" data-ds-appid="902"
       data-ds-tagids="[30,20]"><span class="title">Future Related</span></a>
  </section>
</main>
"""


def _candidate(
    appid: int,
    tags: list[str] | None = None,
    *,
    title: str | None = None,
    genre_ids: list[int] | None = None,
    app_type: str = "game",
) -> GameCandidate:
    return GameCandidate(
        appid=appid,
        title=title or f"Game {appid}",
        app_type=app_type,
        ordered_tags=tags or [],
        genre_ids=genre_ids or [],
        stores=["Steam"],
    )


def _intent(
    *tags: WeightedIntentTag,
    quality: QualityIntent = QualityIntent.NORMAL,
) -> RecommendationIntent:
    return RecommendationIntent(
        tags=tags,
        references=(),
        quality_intent=quality,
        allow_unreleased=False,
    )


def _tag(name: str, weight: float = 1.0) -> WeightedIntentTag:
    return WeightedIntentTag(
        name,
        IntentTagRole.ANCHOR,
        IntentTagSource.EXPLICIT,
        weight,
    )


class MoreLikeContractTest(unittest.IsolatedAsyncioTestCase):
    async def test_more_like_reuse_false_skips_fresh_and_overwrites_with_live_empty(
        self,
    ) -> None:
        cache = ClientMemoryCache()
        online_http = HtmlHttpClient(MORE_LIKE_HTML)
        online = SteamClient(online_http, cache)
        await online.get_more_like(900)
        empty_http = HtmlHttpClient('<section id="released"></section>')
        network_first = SteamClient(empty_http, cache)

        result = await network_first.get_more_like(900, reuse_cache=False)

        self.assertEqual(result.hits, ())
        self.assertFalse(result.stale)
        self.assertEqual(empty_http.call_count, 1)
        self.assertEqual(len(cache.payloads), 1)
        self.assertEqual(
            next(iter(cache.payloads.values()))["html"],
            '<section id="released"></section>',
        )

    async def test_more_like_network_first_failure_uses_seven_day_stale(self) -> None:
        cache = ClientMemoryCache()
        online = SteamClient(HtmlHttpClient(MORE_LIKE_HTML), cache)
        await online.get_more_like(900)
        offline = SteamClient(FailingHttpClient(), cache)

        result = await offline.get_more_like(900, reuse_cache=False)

        self.assertEqual([hit.appid for hit in result.hits], [901])
        self.assertTrue(result.stale)
        self.assertEqual(cache.requested_ttls[-1], 168)

    async def test_more_like_network_first_raises_without_seven_day_stale(self) -> None:
        cache = ClientMemoryCache()
        offline = SteamClient(FailingHttpClient(), cache)

        with self.assertRaises(SteamApiError):
            await offline.get_more_like(900, reuse_cache=False)

        self.assertEqual(cache.requested_ttls, [168])

    async def test_more_like_stale_older_than_seven_days_is_rejected(self) -> None:
        now = [1_000.0]
        cache = ExpiringClientMemoryCache(lambda: now[0])
        online = SteamClient(
            HtmlHttpClient(MORE_LIKE_HTML),
            cache,
            clock=lambda: now[0],
        )
        await online.get_more_like(900)
        now[0] += 7 * 24 * 60 * 60 + 1
        offline = SteamClient(
            FailingHttpClient(),
            cache,
            clock=lambda: now[0],
        )

        with self.assertRaises(SteamApiError):
            await offline.get_more_like(900, reuse_cache=False)

        self.assertEqual(cache.requested_ttls[-1], 168)

    def test_parser_keeps_section_boundaries_titles_and_ordered_tag_ids(self) -> None:
        parser = getattr(steam, "parse_more_like_html", None)
        self.assertIsNotNone(parser, "More Like This parser is missing")

        page = parser(MORE_LIKE_HTML)

        self.assertEqual([hit.appid for hit in page.released], [900, 901])
        self.assertEqual(page.released[1].title, "Related Sequel")
        self.assertEqual(page.released[1].tag_ids, [20, 10, 30])
        self.assertEqual([hit.appid for hit in page.upcoming], [902])

    def test_missing_released_section_is_contract_failure_but_empty_is_valid(self) -> None:
        parser = getattr(steam, "parse_more_like_html", None)
        self.assertIsNotNone(parser, "More Like This parser is missing")

        with self.assertRaises(SteamApiError):
            parser('<section id="upcoming"></section>')
        page = parser('<section id="released"></section>')
        self.assertEqual(page.released, ())

    def test_unreleased_marker_and_nested_classes_cannot_cross_section_boundaries(
        self,
    ) -> None:
        page = steam.parse_more_like_html(
            """
            <section id="unreleased">
              <a data-ds-appid="1"><span class="title">Ignore Me</span></a>
            </section>
            <section id="released">
              <a data-ds-appid="2"><span class="title">Released Game</span></a>
            </section>
            <section id="upcoming">
              <div class="released">
                <a data-ds-appid="3"><span class="title">Upcoming Game</span></a>
              </div>
            </section>
            """
        )

        self.assertEqual([hit.appid for hit in page.released], [2])
        self.assertEqual([hit.appid for hit in page.upcoming], [3])

    async def test_upcoming_is_gated_exact_seed_is_excluded_and_stale_is_typed(self) -> None:
        now = [1_000.0]
        cache = ClientMemoryCache()
        online = SteamClient(
            HtmlHttpClient(MORE_LIKE_HTML),
            cache,
            clock=lambda: now[0],
        )
        getter = getattr(online, "get_more_like", None)
        self.assertIsNotNone(getter, "More Like This client method is missing")

        released = await getter(900, allow_unreleased=False)
        allowed = await getter(900, allow_unreleased=True)

        self.assertEqual([hit.appid for hit in released.hits], [901])
        self.assertEqual([hit.appid for hit in allowed.hits], [901, 902])
        self.assertFalse(released.stale)

        now[0] += 24 * 60 * 60 + 1
        stale_client = SteamClient(
            FailingHttpClient(),
            cache,
            clock=lambda: now[0],
        )
        stale = await stale_client.get_more_like(900, allow_unreleased=False)
        self.assertEqual([hit.appid for hit in stale.hits], [901])
        self.assertTrue(stale.stale)
        self.assertIn(168, cache.requested_ttls)

    async def test_two_tag_intersection_uses_stable_storefront_parameters(self) -> None:
        http = JsonHttpClient(
            {
                "success": 1,
                "results_html": (
                    '<a class="search_result_row" data-ds-appid="1">'
                    '<span class="title">One</span></a>'
                ),
                "total_count": 1,
                "start": 0,
            }
        )
        client = SteamClient(http, ClientMemoryCache(), default_country="US")
        search = getattr(client, "search_storefront_tags", None)
        self.assertIsNotNone(search, "two-tag storefront search is missing")

        await search([22, 11], page_size=100)

        self.assertEqual(http.last_params["tags"], "22,11")
        self.assertEqual(http.last_params["count"], 40)
        self.assertEqual(http.last_params["cc"], "US")
        self.assertNotIn("sort_by", http.last_params)

    def test_each_more_like_section_is_capped_at_twenty(self) -> None:
        released = tuple(
            SteamSearchHit(appid=appid, title=f"Released {appid}")
            for appid in range(1, 26)
        )
        upcoming = tuple(
            SteamSearchHit(appid=appid, title=f"Upcoming {appid}")
            for appid in range(101, 126)
        )
        sections = steam.SteamMoreLikeSections(released, upcoming)

        normal = steam.select_more_like_hits(
            sections,
            reference_appid=999,
            allow_unreleased=False,
        )
        allowed = steam.select_more_like_hits(
            sections,
            reference_appid=999,
            allow_unreleased=True,
        )

        self.assertEqual(len(normal.hits), 20)
        self.assertEqual(len(allowed.hits), 40)
        self.assertEqual(allowed.hits[-1].appid, 120)


class WeightedRrfTest(unittest.TestCase):
    def _source(
        self,
        source_id: str,
        kind: str,
        candidates: list[GameCandidate],
        weight: float,
        *,
        tag: str | None = None,
        component_tags: tuple[str, ...] = (),
    ) -> Any:
        source_type = getattr(steam_recall, "RecallSource", None)
        self.assertIsNotNone(source_type, "weighted recall source is missing")
        return source_type(
            source_id=source_id,
            source_kind=kind,
            source_tag=tag,
            candidates=tuple(candidates),
            weight=weight,
            component_tags=component_tags,
        )

    def test_weighted_formula_keeps_all_source_hits_and_one_hit_per_source(self) -> None:
        duplicate = _candidate(1)
        result = steam_recall.merge_candidate_sources(
            [
                self._source("more:9", "more_like", [duplicate, duplicate], 1.2),
                self._source("local", "index", [_candidate(1)], 0.35),
                self._source("top", "top_seller", [_candidate(2)], 0.5),
            ]
        )

        first = result.hits[0]
        self.assertEqual(first.candidate.appid, 1)
        self.assertAlmostEqual(first.rrf_score, 1.2 / 61 + 0.35 / 61)
        self.assertEqual(
            [(hit.source_id, hit.source_rank) for hit in first.source_hits],
            [("more:9", 1), ("local", 1)],
        )

    def test_duplicate_source_id_keeps_only_its_best_rank_contribution(self) -> None:
        target = _candidate(1)
        result = steam_recall.merge_candidate_sources(
            [
                self._source(
                    "tag:a",
                    "tag",
                    [_candidate(2), target],
                    1.0,
                    tag="a",
                ),
                self._source("tag:a", "tag", [target], 1.0, tag="a"),
            ]
        )

        target_hit = next(hit for hit in result.hits if hit.candidate.appid == 1)
        self.assertAlmostEqual(target_hit.rrf_score, 1.0 / 61.0)
        self.assertEqual(
            [(hit.source_id, hit.source_rank) for hit in target_hit.source_hits],
            [("tag:a", 1)],
        )

    def test_component_singles_are_capped_and_intersection_replaces_them(self) -> None:
        target = _candidate(1)
        third = self._source("tag:c", "tag", [target], 0.8, tag="c")
        empty_intersection = self._source(
            "intersection:a+b",
            "intersection",
            [],
            1.3,
            component_tags=("a", "b"),
        )
        without_hit = steam_recall.merge_candidate_sources(
            [
                self._source("tag:a", "tag", [target], 1.0, tag="a"),
                self._source("tag:b", "tag", [_candidate(9), target], 1.0, tag="b"),
                third,
                empty_intersection,
            ]
        )
        expected_cap = 1.3 / 61
        self.assertAlmostEqual(
            without_hit.hits[0].rrf_score,
            expected_cap + 0.8 / 61,
        )

        with_hit = steam_recall.merge_candidate_sources(
            [
                self._source("tag:a", "tag", [target], 1.0, tag="a"),
                self._source("tag:b", "tag", [target], 1.0, tag="b"),
                self._source(
                    "intersection:a+b",
                    "intersection",
                    [_candidate(8), target],
                    1.3,
                    component_tags=("a", "b"),
                ),
                third,
            ]
        )
        target_hit = next(hit for hit in with_hit.hits if hit.candidate.appid == 1)
        self.assertAlmostEqual(target_hit.rrf_score, 1.3 / 62 + 0.8 / 61)
        self.assertEqual(len(target_hit.source_hits), 4)

    def test_cross_source_consensus_outranks_an_equal_rank_single_source(self) -> None:
        result = steam_recall.merge_candidate_sources(
            [
                self._source(
                    "more:9",
                    "more_like",
                    [_candidate(1), _candidate(2)],
                    1.2,
                ),
                self._source("local", "index", [_candidate(1)], 0.35),
            ]
        )

        self.assertEqual([hit.candidate.appid for hit in result.hits], [1, 2])
        self.assertEqual([hit.retrieval_rank for hit in result.hits], [1, 2])

    def test_third_tag_source_rank_forty_still_contributes_before_top_100(self) -> None:
        target = _candidate(999)
        first = [_candidate(appid) for appid in range(1, 40)] + [target]
        second = [_candidate(appid) for appid in range(101, 141)]
        third = [_candidate(appid) for appid in range(201, 240)] + [target]

        result = steam_recall.merge_candidate_sources(
            [
                self._source("tag:a", "tag", first, 1.0, tag="a"),
                self._source("tag:b", "tag", second, 1.0, tag="b"),
                self._source("tag:c", "tag", third, 1.0, tag="c"),
            ]
        )

        target_hit = next(hit for hit in result.hits if hit.candidate.appid == 999)
        self.assertAlmostEqual(target_hit.rrf_score, 2.0 / 100.0)
        self.assertEqual(
            [(hit.source_id, hit.source_rank) for hit in target_hit.source_hits],
            [("tag:a", 40), ("tag:c", 40)],
        )


class RankingPolicyTest(unittest.TestCase):
    def test_a_b_c_thresholds_use_weighted_coverage_and_strong_direct_anchor(self) -> None:
        query = _intent(_tag("action"), _tag("puzzle"))
        ranked = rank_steam_candidates(
            [
                _candidate(1, ["Action", "Puzzle"], title="A"),
                _candidate(2, ["Action"], title="B"),
                _candidate(
                    3,
                    ["x1", "x2", "x3", "x4", "x5", "x6", "x7", "Puzzle"],
                    title="C",
                ),
            ],
            query,
        )

        by_title = {game.title: game.score_breakdown.relevance_tier for game in ranked}
        self.assertEqual(by_title, {"A": "A", "B": "B", "C": "C"})

    def test_normal_and_mainstream_layer_weights_are_exact(self) -> None:
        self.assertAlmostEqual(layer_score(0.8, 0.4, QualityIntent.NORMAL), 0.72)
        self.assertAlmostEqual(
            layer_score(0.8, 0.4, QualityIntent.MAINSTREAM),
            0.66,
        )

    def test_displayed_score_and_sort_share_unrounded_effective_score(self) -> None:
        score_function = getattr(ranking_precedence, "effective_score", None)
        self.assertIsNotNone(score_function, "unified effective score is missing")
        games = rank_steam_candidates(
            [
                GameCandidate(
                    appid=1,
                    title="Supported",
                    app_type="game",
                    ordered_tags=["Puzzle"],
                    supported_languages=["schinese"],
                    language_data_available=True,
                ),
                GameCandidate(
                    appid=2,
                    title="Unknown",
                    app_type="game",
                    ordered_tags=["Puzzle"],
                ),
            ],
            _intent(_tag("puzzle")),
            language_profile=SimpleNamespace(
                preferred_languages=["schinese"],
                required_languages=[],
                positive_reference_candidates=[],
                negative_reference_candidates=[],
                reference_appids=[],
                reference_appids_dislike=[],
            ),
        )
        self.assertEqual(games[0].title, "Supported")
        for game in games:
            self.assertEqual(game.score, round(score_function(game.score_breakdown)))

        adjusted = attach_price_summary(
            games[1],
            GamePriceSummary(
                region="CN",
                currency="CNY",
                current_price="¥1",
                current_amount=1,
                current_currency="CNY",
                historic_low="¥1",
                historic_low_amount=1,
                historic_low_currency="CNY",
            ),
            GamePreference(budget=10, budget_currency="CNY"),
        )
        self.assertEqual(adjusted.score, round(score_function(adjusted.score_breakdown)))
        self.assertEqual(
            sorted([adjusted, games[0]], key=ranked_game_sort_key)[0].title,
            max(
                [adjusted, games[0]],
                key=lambda game: score_function(game.score_breakdown),
            ).title,
        )

    def test_official_software_genres_are_preserved_and_rejected_without_title_rules(self) -> None:
        payload = steam_detail_payload()
        payload["genres"] = [
            {"id": "25", "description": "Adventure"},
            {"id": "57", "description": "Utilities"},
        ]
        software = steam.parse_steam_game(1, payload)

        self.assertEqual(getattr(software, "genre_ids", None), [25, 57])
        self.assertFalse(is_confirmed_base_game(software))
        self.assertTrue(
            is_confirmed_base_game(
                _candidate(2, ["Puzzle"], title="Utility Quest", genre_ids=[25])
            )
        )
        self.assertFalse(
            is_confirmed_base_game(
                _candidate(3, ["Puzzle"], title="Ordinary Game", genre_ids=[57])
            )
        )


class RecallPipelineTest(unittest.IsolatedAsyncioTestCase):
    async def test_resolved_reference_replaces_stale_tags_with_current_hit_evidence(
        self,
    ) -> None:
        client = PipelineClient(
            tag_ids={"Puzzle": 1, "Adventure": 2},
            text_results={
                "Seed": [SteamSearchHit(appid=900, title="Seed")],
            },
            more_like_results={
                900: steam.SteamMoreLikeResult(
                    hits=(
                        SteamSearchHit(appid=900, title="Seed", tag_ids=[1, 2]),
                        SteamSearchHit(appid=901, title="Related", tag_ids=[1]),
                    )
                ),
            },
            details={
                900: _candidate(900, ["Puzzle", "Adventure"], title="Seed"),
                901: _candidate(901, ["Unrelated"], title="Related"),
            },
        )
        service = SteamGameIndexService(client, ServiceMemoryCache())

        ranked = await service.recommend(
            GamePreference(
                reference_games_like=["Seed"],
                allow_unreleased=True,
            ),
            limit=3,
        )

        self.assertEqual(client.more_like_calls, [(900, True)])
        self.assertEqual([game.appid for game in ranked], [901])
        self.assertEqual(ranked[0].ordered_tags, ["puzzle"])

    def test_multiple_hits_in_one_refresh_merge_their_current_tag_evidence(self) -> None:
        merged = merge_current_recall_candidate_evidence(
            _candidate(1, ["Puzzle"]),
            _candidate(1, ["Adventure"]),
        )

        self.assertEqual(merged.ordered_tags, ["puzzle", "adventure"])

    async def test_sources_use_intersection_and_three_deep_single_anchor_queries(self) -> None:
        client = PipelineClient(
            tag_ids={"Action": 1, "RPG": 2, "Puzzle": 3, "Strategy": 4},
        )
        service = SteamGameIndexService(client, ServiceMemoryCache())

        await service.recommend(
            GamePreference(genres_like=["action", "rpg", "puzzle", "strategy"]),
            limit=2,
        )

        self.assertEqual(client.intersection_calls, [((1, 2), 40)])
        self.assertEqual(client.tag_calls, [(1, 40), (2, 40), (3, 40)])
        self.assertEqual(client.text_calls, [])

    async def test_loaded_vocabulary_skips_unknown_anchors_before_seed_selection(self) -> None:
        client = PipelineClient(tag_ids={"Known One": 11, "Known Two": 12})
        service = SteamGameIndexService(client, ServiceMemoryCache())

        await service.recommend(
            GamePreference(
                genres_like=[
                    "unknown_one",
                    "unknown_two",
                    "unknown_three",
                    "known_one",
                    "known_two",
                ]
            ),
            limit=2,
        )

        self.assertEqual(client.intersection_calls, [((11, 12), 40)])
        self.assertEqual(client.tag_calls, [(11, 40), (12, 40)])
        self.assertEqual(client.text_calls, [])

    async def test_topsellers_are_gated_to_mainstream_or_no_anchor_queries(self) -> None:
        anchored_client = PipelineClient(tag_ids={"Puzzle": 1})
        await SteamGameIndexService(
            anchored_client,
            ServiceMemoryCache(),
        ).recommend(GamePreference(genres_like=["puzzle"]), limit=1)
        self.assertEqual(anchored_client.top_calls, [])

        mainstream_client = PipelineClient(tag_ids={"Puzzle": 1})
        await SteamGameIndexService(
            mainstream_client,
            ServiceMemoryCache(),
        ).recommend(
            GamePreference(genres_like=["puzzle"], quality_intent="mainstream"),
            limit=1,
        )
        self.assertEqual(mainstream_client.top_calls, [60])

        broad_client = PipelineClient()
        await SteamGameIndexService(broad_client, ServiceMemoryCache()).recommend(
            GamePreference(),
            limit=1,
        )
        self.assertEqual(broad_client.top_calls, [60])

    async def test_progressive_validation_stops_at_60_or_expands_to_100(self) -> None:
        enough_client = progressive_client(first_batch_matches=True)
        enough = await SteamGameIndexService(
            enough_client,
            ServiceMemoryCache(),
        ).recommend(
            GamePreference(genres_like=["action", "rpg", "puzzle"]),
            limit=5,
        )
        self.assertEqual(len(enough_client.detail_calls), 60)
        self.assertTrue(enough)

        sparse_client = progressive_client(first_batch_matches=False)
        sparse = await SteamGameIndexService(
            sparse_client,
            ServiceMemoryCache(),
        ).recommend(
            GamePreference(genres_like=["action", "rpg", "puzzle"]),
            limit=5,
        )
        self.assertEqual(len(sparse_client.detail_calls), 100)
        self.assertTrue(sparse)
        self.assertTrue(all(game.score_breakdown.relevance_tier == "A" for game in sparse))

    async def test_default_path_never_returns_c_but_healthy_all_c_is_empty(self) -> None:
        client = PipelineClient(
            tag_ids={"Puzzle": 1},
            tag_results={1: [SteamSearchHit(appid=100, title="Weak")]},
            details={100: _candidate(100, ["Unrelated"])},
        )
        service = SteamGameIndexService(client, ServiceMemoryCache())

        ranked = await service.recommend(
            GamePreference(genres_like=["puzzle"]),
            limit=3,
        )

        self.assertEqual(ranked, [])

    async def test_known_tag_failure_never_falls_back_to_tag_as_title_and_is_unavailable(
        self,
    ) -> None:
        unavailable_type = getattr(steam_recall, "RecallUnavailableError", None)
        self.assertIsNotNone(unavailable_type, "typed unavailable error is missing")
        client = PipelineClient(
            tag_ids={"Puzzle": 1},
            tag_results={1: SteamTransientError("offline")},
            text_results={"puzzle": [SteamSearchHit(appid=9, title="Wrong fallback")]},
        )
        service = SteamGameIndexService(client, ServiceMemoryCache())

        with self.assertRaises(unavailable_type):
            await service.recommend(
                GamePreference(genres_like=["puzzle"]),
                limit=3,
            )

        self.assertEqual(client.text_calls, [])

    async def test_strict_majority_validation_failure_exposes_typed_health(self) -> None:
        client = PipelineClient(
            tag_ids={"Puzzle": 1},
            tag_results={
                1: [
                    SteamSearchHit(appid=appid, title=f"Game {appid}")
                    for appid in range(1, 6)
                ]
            },
            details={
                1: SteamTransientError("offline"),
                2: SteamTransientError("offline"),
                3: SteamTransientError("offline"),
                4: _candidate(4, ["Puzzle"]),
                5: _candidate(5, ["Puzzle"]),
            },
        )

        with self.assertRaises(steam_recall.RecallUnavailableError) as raised:
            await SteamGameIndexService(client, ServiceMemoryCache()).recommend(
                GamePreference(genres_like=["puzzle"]),
                limit=5,
            )

        health = raised.exception.health
        self.assertEqual(health.validation_attempts, 5)
        self.assertEqual(health.validation_transient_failures, 3)
        self.assertEqual(health.verified, 2)
        self.assertEqual(health.eligible, 2)

    async def test_small_validation_samples_that_all_fail_are_unavailable(self) -> None:
        for candidate_count, failure in (
            (1, SteamTransientError("offline")),
            (2, SteamTransientError("offline")),
            (1, SteamApiError("invalid response")),
            (2, SteamApiError("invalid response")),
        ):
            with self.subTest(
                candidate_count=candidate_count,
                failure_type=type(failure).__name__,
            ):
                appids = list(range(1, candidate_count + 1))
                client = PipelineClient(
                    tag_ids={"Puzzle": 1},
                    tag_results={
                        1: [
                            SteamSearchHit(appid=appid, title=f"Game {appid}")
                            for appid in appids
                        ]
                    },
                    details={appid: failure for appid in appids},
                )

                with self.assertRaises(steam_recall.RecallUnavailableError):
                    await SteamGameIndexService(
                        client,
                        ServiceMemoryCache(),
                    ).recommend(
                        GamePreference(genres_like=["puzzle"]),
                        limit=3,
                    )

    async def test_programming_error_during_candidate_validation_is_not_degraded(self) -> None:
        client = PipelineClient(
            tag_ids={"Puzzle": 1},
            tag_results={
                1: [SteamSearchHit(appid=1, title="Game 1")],
            },
            details={1: RuntimeError("candidate decoder bug")},
        )

        with self.assertRaisesRegex(RuntimeError, "candidate decoder bug"):
            await SteamGameIndexService(client, ServiceMemoryCache()).recommend(
                GamePreference(genres_like=["puzzle"]),
                limit=3,
            )


class RecallHealthTest(unittest.TestCase):
    def test_empty_applicable_set_is_not_systemic_and_valid_empty_is_healthy(self) -> None:
        health_type = getattr(steam_recall, "RecallHealth", None)
        source_health_type = getattr(steam_recall, "RecallSourceHealth", None)
        status_type = getattr(steam_recall, "RecallSourceStatus", None)
        self.assertIsNotNone(health_type, "typed recall health is missing")
        self.assertIsNotNone(source_health_type, "typed source health is missing")
        self.assertIsNotNone(status_type, "typed source status is missing")

        self.assertFalse(health_type().systemic_failure)
        healthy_empty = health_type(
            sources=(
                source_health_type(
                    source_id="tag:1",
                    critical=True,
                    status=status_type.EMPTY,
                ),
            ),
            verified=1,
            eligible=0,
        )
        self.assertFalse(healthy_empty.systemic_failure)
        self.assertFalse(healthy_empty.unavailable(limit=3))

    def test_critical_failures_or_majority_validation_failures_are_systemic(self) -> None:
        health_type = getattr(steam_recall, "RecallHealth", None)
        source_health_type = getattr(steam_recall, "RecallSourceHealth", None)
        status_type = getattr(steam_recall, "RecallSourceStatus", None)
        self.assertIsNotNone(health_type, "typed recall health is missing")

        critical = health_type(
            sources=(
                source_health_type(
                    source_id="tag:1",
                    critical=True,
                    status=status_type.TRANSIENT_FAILURE,
                ),
            ),
            verified=0,
            eligible=0,
        )
        self.assertTrue(critical.systemic_failure)
        self.assertTrue(critical.unavailable(limit=3))

        validation = health_type(
            validation_attempts=5,
            validation_transient_failures=2,
            validation_contract_failures=1,
            verified=2,
            eligible=2,
        )
        self.assertTrue(validation.systemic_failure)
        self.assertTrue(validation.unavailable(limit=5))

        for attempts in (1, 2):
            with self.subTest(attempts=attempts):
                small_sample = health_type(
                    validation_attempts=attempts,
                    validation_transient_failures=attempts,
                    verified=0,
                    eligible=0,
                )
                self.assertTrue(small_sample.systemic_failure)
                self.assertTrue(small_sample.unavailable(limit=3))

    def test_stale_source_and_non_strict_validation_half_prevent_systemic_failure(
        self,
    ) -> None:
        health = steam_recall.RecallHealth(
            sources=(
                steam_recall.RecallSourceHealth(
                    source_id="more:1",
                    critical=True,
                    status=steam_recall.RecallSourceStatus.STALE,
                ),
                steam_recall.RecallSourceHealth(
                    source_id="tag:1",
                    critical=True,
                    status=steam_recall.RecallSourceStatus.TRANSIENT_FAILURE,
                ),
            ),
            validation_attempts=4,
            validation_transient_failures=2,
            verified=2,
            eligible=1,
        )

        self.assertFalse(health.systemic_failure)
        self.assertFalse(health.unavailable(limit=3))


class PipelineClient:
    language = "english"

    def __init__(
        self,
        *,
        tag_ids: dict[str, int] | None = None,
        tag_results: dict[int, list[SteamSearchHit] | Exception] | None = None,
        intersection_results: list[SteamSearchHit] | None = None,
        top_results: list[SteamSearchHit] | None = None,
        text_results: dict[str, list[SteamSearchHit]] | None = None,
        details: dict[int, GameCandidate | Exception] | None = None,
        more_like_results: dict[int, steam.SteamMoreLikeResult | Exception] | None = None,
    ) -> None:
        self.tag_ids = tag_ids or {}
        self.tag_results = tag_results or {}
        self.intersection_results = intersection_results or []
        self.top_results = top_results or []
        self.text_results = text_results or {}
        self.details = details or {}
        self.more_like_results = more_like_results or {}
        self.tag_calls: list[tuple[int, int]] = []
        self.intersection_calls: list[tuple[tuple[int, int], int]] = []
        self.top_calls: list[int] = []
        self.text_calls: list[tuple[str, int]] = []
        self.detail_calls: list[int] = []
        self.more_like_calls: list[tuple[int, bool]] = []

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

    async def search_storefront_tags(
        self,
        tag_ids: list[int] | tuple[int, ...],
        page_size: int = 40,
    ) -> SteamStorefrontPage:
        resolved = tuple(int(item) for item in tag_ids)
        self.intersection_calls.append((resolved, page_size))
        return SteamStorefrontPage(
            tuple(self.intersection_results),
            len(self.intersection_results),
            0,
        )

    async def browse_top_sellers(self, page_size: int = 60) -> SteamStorefrontPage:
        self.top_calls.append(page_size)
        return SteamStorefrontPage(tuple(self.top_results), len(self.top_results), 0)

    async def get_more_like(
        self,
        appid: int,
        *,
        allow_unreleased: bool = False,
    ) -> steam.SteamMoreLikeResult:
        self.more_like_calls.append((int(appid), bool(allow_unreleased)))
        result = self.more_like_results.get(
            int(appid),
            steam.SteamMoreLikeResult(hits=()),
        )
        if isinstance(result, Exception):
            raise result
        return result

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
        result = self.details.get(appid, _candidate(appid, ["Unrelated"]))
        if isinstance(result, Exception):
            raise result
        return result


def progressive_client(*, first_batch_matches: bool) -> PipelineClient:
    tag_ids = {"Action": 1, "RPG": 2, "Puzzle": 3}
    tag_results: dict[int, list[SteamSearchHit]] = {}
    details: dict[int, GameCandidate] = {}
    for tag_id in tag_ids.values():
        hits = []
        for rank in range(1, 41):
            appid = tag_id * 1_000 + rank
            hits.append(SteamSearchHit(appid=appid, title=f"Game {appid}"))
            matches = first_batch_matches if rank <= 20 else not first_batch_matches
            details[appid] = _candidate(
                appid,
                ["Action", "RPG", "Puzzle"] if matches else ["Unrelated"],
            )
        tag_results[tag_id] = hits
    return PipelineClient(
        tag_ids=tag_ids,
        tag_results=tag_results,
        details=details,
    )


class ServiceMemoryCache:
    def __init__(self) -> None:
        self.payloads: dict[str, Any] = {}

    async def get_json(self, key: str, _ttl_hours: int) -> Any | None:
        return self.payloads.get(key)

    async def set_json(self, key: str, payload: Any) -> None:
        self.payloads[key] = payload


class ClientMemoryCache:
    def __init__(self) -> None:
        self.payloads: dict[str, Any] = {}
        self.requested_ttls: list[int] = []

    async def get_json(self, key: str, ttl_hours: int) -> Any | None:
        self.requested_ttls.append(ttl_hours)
        return self.payloads.get(key)

    async def set_json(self, key: str, payload: Any) -> None:
        self.payloads[key] = payload

    def drop_fresh(self) -> None:
        for key in list(self.payloads):
            if key.endswith(":fresh"):
                self.payloads.pop(key)


class ExpiringClientMemoryCache(ClientMemoryCache):
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


class HtmlResponse:
    def __init__(self, text: str) -> None:
        self.text = text

    def raise_for_status(self) -> None:
        return None


class HtmlHttpClient:
    def __init__(self, html: str) -> None:
        self.html = html
        self.call_count = 0

    async def get(self, _url: str, **_kwargs: Any) -> HtmlResponse:
        self.call_count += 1
        return HtmlResponse(self.html)


class JsonResponse:
    def __init__(self, payload: Any) -> None:
        self.payload = payload

    def raise_for_status(self) -> None:
        return None

    def json(self) -> Any:
        return self.payload


class JsonHttpClient:
    def __init__(self, payload: Any) -> None:
        self.payload = payload
        self.last_params: dict[str, Any] = {}

    async def get(self, _url: str, params: dict[str, Any], **_kwargs: Any) -> JsonResponse:
        self.last_params = dict(params)
        return JsonResponse(self.payload)


class FailingHttpClient:
    async def get(self, _url: str, **_kwargs: Any) -> Any:
        raise SteamTransientError("offline")


def steam_detail_payload() -> dict[str, Any]:
    return {
        "name": "Software",
        "type": "game",
        "platforms": {"windows": True},
        "genres": [{"id": "25", "description": "Adventure"}],
        "categories": [],
        "release_date": {"coming_soon": False, "date": "1 Jan, 2025"},
    }


if __name__ == "__main__":
    unittest.main()
