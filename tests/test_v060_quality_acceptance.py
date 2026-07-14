from __future__ import annotations

# ruff: noqa: E402, I001

import os
import unittest
from collections import defaultdict
from copy import deepcopy
from statistics import fmean

import httpx

try:
    _astrbot_stubs = __import__("tests.test_prepare_recommendation")
except ModuleNotFoundError:
    _astrbot_stubs = __import__("test_prepare_recommendation")

from astrbot_plugin_steam_game_recommender.clients.steam import SteamClient
from astrbot_plugin_steam_game_recommender.services.recommendation_evaluation import (
    fill_rate,
    hit_at_k,
    ndcg_at_k,
    policy_pairwise_accuracy,
    recall_at_k,
)
from astrbot_plugin_steam_game_recommender.services.tag_normalizer import (
    register_steam_tag_aliases,
    steam_tag_id_for,
)

try:
    from tests.e2e_recommendation_harness import (
        MemoryIndexCache,
        load_e2e_fixture,
        run_e2e_scenario,
    )
except ModuleNotFoundError:
    from e2e_recommendation_harness import (
        MemoryIndexCache,
        load_e2e_fixture,
        run_e2e_scenario,
    )


QUALITY_SLICES = {
    "souls_like",
    "extraction_shooter",
    "metroidvania",
    "deckbuilding",
    "colony_sim",
    "mainstream",
}


def fixture_tag_id(fixture: dict, name: str) -> int:
    expected = name.casefold()
    return next(
        int(item["tagid"])
        for item in fixture["tags"]
        if str(item["name"]).casefold() == expected
    )


class V060EndToEndQualityAcceptanceTest(unittest.IsolatedAsyncioTestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.fixture = load_e2e_fixture()
        cls.scenarios = {
            scenario["id"]: scenario for scenario in cls.fixture["scenarios"]
        }

    def test_frozen_storefront_contract_is_independent_from_judgments(self) -> None:
        self.assertEqual(
            self.fixture["baseline_snapshot"]["source_commit"],
            "202d2125fac730ab94a9fc08277510b40c513a14",
        )
        self.assertTrue(
            all(
                "count" not in tag and "total_count" not in tag
                for tag in self.fixture["tags"]
            )
        )
        self.assertIn("storefront_tag_results", self.fixture)
        self.assertTrue(
            all(
                "recall_appids" not in scenario
                for scenario in self.fixture["scenarios"]
            )
        )

    async def test_frozen_llm_receives_the_original_query(self) -> None:
        scenario = self.scenarios["souls_cn_original"]
        run = await run_e2e_scenario(self.fixture, scenario)

        self.assertEqual(len(run.llm_prompts), 1)
        self.assertIn(scenario["query"], run.llm_prompts[0])

    async def test_mainstream_keeps_verified_cross_language_gameplay_only(self) -> None:
        run = await run_e2e_scenario(
            self.fixture,
            self.scenarios["mainstream_cross_language_gameplay"],
        )

        self.assertIn("extraction_shooter", run.preference.genres_like)
        self.assertFalse(
            {"action", "rpg", "open_world"} & set(run.preference.genres_like)
        )
        self.assertIn(
            fixture_tag_id(self.fixture, "Extraction Shooter"),
            run.client.storefront_tag_calls,
        )
        self.assertTrue({210, 211, 212} <= set(run.ranking[:5]))

    async def test_wrong_tag_cannot_preserve_unrelated_slice_recall(self) -> None:
        for scenario_id in ("souls_cn_original", "extraction_cn_original"):
            with self.subTest(scenario=scenario_id):
                scenario = self.scenarios[scenario_id]
                wrong_fixture = deepcopy(self.fixture)
                reference_appid = int(scenario["expected_reference_appid"])
                reference = next(
                    game
                    for game in wrong_fixture["games"]
                    if int(game["appid"]) == reference_appid
                )
                reference["ordered_tags"] = ["Deckbuilding", "Card Battler"]

                run = await run_e2e_scenario(wrong_fixture, scenario)
                ranking = [str(appid) for appid in run.ranking]
                relevance = {
                    str(appid): int(value)
                    for appid, value in scenario["relevance"].items()
                }

                self.assertEqual(recall_at_k(ranking, relevance, k=50), 0.0)
                self.assertIn(
                    fixture_tag_id(self.fixture, "Deckbuilding"),
                    run.client.storefront_tag_calls,
                )

    async def test_real_pipeline_meets_retrieval_ranking_and_safety_gates(self) -> None:
        evaluated = [
            await run_e2e_scenario(self.fixture, scenario)
            for scenario in self.fixture["scenarios"]
            if scenario["slice"] in QUALITY_SLICES
        ]
        recalls: list[float] = []
        hits: list[float] = []
        pairwise: list[float] = []
        ndcg_deltas: list[float] = []
        slice_recalls: dict[str, list[float]] = defaultdict(list)
        slice_deltas: dict[str, list[float]] = defaultdict(list)
        reference_total = 0
        reference_correct = 0
        reference_errors = 0

        for run in evaluated:
            scenario = run.scenario
            ranking = [str(appid) for appid in run.ranking]
            retrieved = [str(appid) for appid in run.retrieved_appids]
            relevance = {
                str(appid): int(value)
                for appid, value in scenario["relevance"].items()
            }
            relevant_ids = {appid for appid, value in relevance.items() if value > 0}
            strong_ids = {str(appid) for appid in scenario["strong_appids"]}
            baseline = [str(appid) for appid in scenario["baseline_ranking"]]
            pairs = [
                (str(core_appid), str(broad_appid))
                for core_appid, broad_appid in scenario["pairwise"]
            ]
            self.assertTrue(
                all(comparison in retrieved for _preferred, comparison in pairs),
                scenario["id"],
            )
            recall = recall_at_k(retrieved, relevance, k=100)
            current_ndcg = ndcg_at_k(ranking, relevance, k=5)
            baseline_ndcg = ndcg_at_k(baseline, relevance, k=5)
            delta = current_ndcg - baseline_ndcg

            recalls.append(recall)
            hits.append(hit_at_k(ranking, strong_ids, k=20))
            pairwise.append(policy_pairwise_accuracy(ranking, pairs))
            ndcg_deltas.append(delta)
            slice_recalls[scenario["slice"]].append(recall)
            slice_deltas[scenario["slice"]].append(delta)

            self.assertEqual(len(run.ranking), len(set(run.ranking)), scenario["id"])
            self.assertTrue(
                all(game.app_type == "game" for game in run.ranked),
                scenario["id"],
            )
            self.assertTrue(
                all(not game.coming_soon for game in run.ranked),
                scenario["id"],
            )
            self.assertTrue(
                all(game.score_breakdown.relevance_tier != "C" for game in run.ranked),
                scenario["id"],
            )
            selected = ranking[: int(scenario["target_count"])]
            if len(relevant_ids) >= int(scenario["target_count"]):
                self.assertEqual(
                    fill_rate(selected, int(scenario["target_count"])),
                    1.0,
                    scenario["id"],
                )
            for appid in run.ranking:
                self.assertIn(appid, run.client.detail_calls, scenario["id"])
                self.assertIn(appid, run.client.store_tag_calls, scenario["id"])
                self.assertIn(appid, run.client.review_calls, scenario["id"])

            expected_reference = scenario.get("expected_reference_appid")
            if expected_reference is not None:
                reference_total += 1
                resolved = [
                    item.appid
                    for item in run.preference.resolved_reference_games
                    if item.appid is not None
                ]
                if resolved and int(resolved[0]) == int(expected_reference):
                    reference_correct += 1
                elif resolved:
                    reference_errors += 1

        self.assertGreaterEqual(reference_correct / reference_total, 0.95)
        self.assertLessEqual(reference_errors / reference_total, 0.01)
        self.assertGreaterEqual(fmean(recalls), 0.90)
        self.assertTrue(
            all(fmean(values) >= 0.85 for values in slice_recalls.values())
        )
        self.assertGreaterEqual(fmean(hits), 0.95)
        self.assertGreaterEqual(fmean(pairwise), 0.95)
        self.assertGreaterEqual(fmean(ndcg_deltas), -0.01)
        self.assertTrue(
            all(fmean(values) >= -0.02 for values in slice_deltas.values())
        )

    async def test_alias_group_uses_any_reliable_alias_and_preserves_sequel_number(self) -> None:
        run = await run_e2e_scenario(
            self.fixture,
            self.scenarios["souls_cn_original"],
        )

        self.assertEqual(run.preference.resolved_reference_games[0].appid, 100)
        self.assertNotEqual(run.preference.resolved_reference_games[0].appid, 101)
        searched = {query.casefold() for query, _language in run.client.reference_calls}
        self.assertIn("黑暗之魂".casefold(), searched)
        self.assertIn("dark souls", searched)
        self.assertEqual(
            set(run.client.storefront_tag_calls),
            {
                fixture_tag_id(self.fixture, "Souls-like"),
                fixture_tag_id(self.fixture, "Action"),
                fixture_tag_id(self.fixture, "RPG"),
                fixture_tag_id(self.fixture, "Dark Fantasy"),
                fixture_tag_id(self.fixture, "Difficult"),
            },
        )

    async def test_recall_boundary_excludes_reference_resolution_detail(self) -> None:
        run = await run_e2e_scenario(
            self.fixture,
            self.scenarios["souls_cn_original"],
        )

        self.assertIn(100, run.client.detail_calls)
        self.assertNotIn(100, run.retrieved_appids)
        self.assertLessEqual(len(run.retrieved_appids), 100)

    async def test_multisource_hits_reach_actual_validation_boundary(self) -> None:
        run = await run_e2e_scenario(
            self.fixture,
            self.scenarios["souls_cn_original"],
        )

        self.assertEqual(
            run.client.more_like_calls,
            [(100, False)],
        )
        self.assertEqual(
            run.client.storefront_intersection_calls,
            [((103, 104), 40)],
        )
        self.assertIn(118, run.retrieved_appids)
        self.assertIn(119, run.retrieved_appids)

    async def test_all_aliases_fail_once_without_false_resolution(self) -> None:
        run = await run_e2e_scenario(
            self.fixture,
            self.scenarios["all_aliases_fail"],
        )

        self.assertEqual(run.ranking, [])
        self.assertEqual(run.client.top_seller_calls, 0)
        self.assertTrue(run.preference.resolved_reference_games)
        self.assertIsNone(run.preference.resolved_reference_games[0].appid)
        warnings = [
            warning
            for warning in run.preference.parse_warnings
            if "未能可靠解析" in warning
        ]
        self.assertEqual(len(warnings), 1)

    async def test_review_confidence_matrix_never_imputes_missing_quality(self) -> None:
        run = await run_e2e_scenario(
            self.fixture,
            self.scenarios["review_confidence_matrix"],
        )

        positions = {appid: index for index, appid in enumerate(run.ranking)}
        self.assertLess(positions[706], positions[705])
        self.assertLess(positions[705], positions[704])
        self.assertLess(positions[704], positions[703])
        self.assertLess(positions[703], positions[702])
        by_appid = {int(game.appid): game for game in run.ranked if game.appid is not None}
        self.assertEqual(by_appid[701].score_breakdown.quality_score, 0.0)
        self.assertEqual(by_appid[702].score_breakdown.quality_score, 0.0)
        self.assertEqual(
            by_appid[701].score_breakdown.quality_score,
            by_appid[702].score_breakdown.quality_score,
        )

    @unittest.skipUnless(
        os.getenv("STEAM_STOREFRONT_SMOKE") == "1",
        "set STEAM_STOREFRONT_SMOKE=1 for the periodic live Steam contract check",
    )
    async def test_optional_live_storefront_contract_smoke(self) -> None:
        async with httpx.AsyncClient(timeout=10.0) as http_client:
            client = SteamClient(http_client, MemoryIndexCache(), default_country="CN")
            tags = await client.get_popular_tags()
            register_steam_tag_aliases(tags)
            tag_id = steam_tag_id_for("soulslike")
            self.assertIsNotNone(tag_id)
            page = await client.search_storefront_tag(int(tag_id), page_size=5)

        self.assertGreaterEqual(page.total_count, len(page.hits))
        self.assertTrue(all(hit.appid > 0 and hit.title for hit in page.hits))


if __name__ == "__main__":
    unittest.main()
