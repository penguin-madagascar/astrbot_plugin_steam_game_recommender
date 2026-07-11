from __future__ import annotations

import unittest
from collections import Counter
from statistics import fmean
from typing import Any, Callable

from astrbot_plugin_game_recommender.services.recommendation_evaluation import (
    constraint_violation_rate,
    fill_rate,
    intra_list_tag_similarity,
    ndcg_at_k,
    recall_at_k,
)

try:
    from astrbot_plugin_game_recommender.tests.recommendation_scenario_loader import (
        load_recommendation_quality_fixture,
    )
except ModuleNotFoundError as error:
    if not error.name or not error.name.endswith("recommendation_scenario_loader"):
        raise
    load_recommendation_quality_fixture = None


EXPECTED_CATEGORY_COUNTS = {
    "polarity": 4,
    "hard_constraints": 4,
    "positive_negative_reference": 4,
    "aaa": 3,
    "budget_library": 3,
    "diversity": 3,
    "retry_feedback": 3,
}
EXPECTED_SCENARIO_IDS = {
    "polarity-horror-positive",
    "polarity-horror-negative",
    "polarity-souls-positive",
    "polarity-souls-negative",
    "required-chinese",
    "required-local-coop",
    "required-multiplayer",
    "required-relaxing",
    "reference-stardew-positive",
    "reference-dark-souls-negative-tag",
    "reference-slay-spire-negative",
    "reference-positive-and-negative",
    "aaa-open-world",
    "aaa-story-rich",
    "aaa-broad-empty",
    "budget-soft-ranking",
    "library-exclude-owned",
    "library-only-owned",
    "diversity-strict-similar",
    "diversity-balanced",
    "diversity-high",
    "retry-too-hard",
    "retry-like-second",
    "retry-dislike-first",
}
METRIC_NAMES = {
    "ndcg_at_target",
    "recall_at_20",
    "constraint_violation_rate",
    "fill_rate",
    "intra_list_tag_similarity",
}


class RecommendationScenarioReplayTest(unittest.TestCase):
    def _load_fixture(self) -> dict[str, Any]:
        loader: Callable[[], dict[str, Any]] | None = load_recommendation_quality_fixture
        if loader is None:
            self.fail("recommendation scenario fixture loader is missing")
        return loader()

    def test_fixture_has_exact_scenario_and_category_counts(self) -> None:
        fixture = self._load_fixture()
        scenarios = fixture["scenarios"]

        self.assertIs(type(fixture["schema_version"]), int)
        self.assertEqual(fixture["schema_version"], 1)
        self.assertEqual(len(scenarios), 24)
        self.assertEqual({scenario["id"] for scenario in scenarios}, EXPECTED_SCENARIO_IDS)
        self.assertEqual(
            Counter(scenario["category"] for scenario in scenarios),
            EXPECTED_CATEGORY_COUNTS,
        )

    def test_scenarios_have_complete_valid_references(self) -> None:
        fixture = self._load_fixture()
        scenarios = fixture["scenarios"]
        scenario_ids: set[str] = set()
        candidate_ids: set[str] = set()

        for scenario in scenarios:
            with self.subTest(scenario=scenario.get("id")):
                self.assertTrue(
                    {
                        "id",
                        "category",
                        "query",
                        "target_count",
                        "candidates",
                        "legacy_candidate_ranking",
                        "legacy_ranking",
                        "violating_ids",
                    }
                    <= scenario.keys()
                )
                self.assertIsInstance(scenario["id"], str)
                self.assertTrue(scenario["id"])
                self.assertNotIn(scenario["id"], scenario_ids)
                scenario_ids.add(scenario["id"])
                self.assertIn(scenario["category"], EXPECTED_CATEGORY_COUNTS)
                self.assertIsInstance(scenario["query"], str)
                self.assertTrue(scenario["query"].strip())
                self.assertIs(type(scenario["target_count"]), int)
                self.assertGreaterEqual(scenario["target_count"], 3)
                self.assertLessEqual(scenario["target_count"], 6)
                self.assertIsInstance(scenario["candidates"], list)
                self.assertGreaterEqual(len(scenario["candidates"]), 3)

                local_candidate_ids: set[str] = set()
                for candidate in scenario["candidates"]:
                    self.assertTrue({"id", "title", "tags", "relevance"} <= candidate.keys())
                    self.assertIsInstance(candidate["id"], str)
                    self.assertTrue(candidate["id"])
                    self.assertNotIn(candidate["id"], candidate_ids)
                    candidate_ids.add(candidate["id"])
                    local_candidate_ids.add(candidate["id"])
                    self.assertIsInstance(candidate["title"], str)
                    self.assertTrue(candidate["title"].strip())
                    self.assertIsInstance(candidate["tags"], list)
                    self.assertTrue(
                        all(isinstance(tag, str) and tag.strip() for tag in candidate["tags"])
                    )
                    self.assertIs(type(candidate["relevance"]), int)
                    self.assertGreaterEqual(candidate["relevance"], 0)
                    self.assertLessEqual(candidate["relevance"], 3)

                self.assertTrue(
                    any(candidate["relevance"] > 0 for candidate in scenario["candidates"]),
                    "each scenario must include at least one relevant candidate",
                )
                self.assertIsInstance(scenario["legacy_ranking"], list)
                self.assertEqual(
                    len(scenario["legacy_ranking"]),
                    len(set(scenario["legacy_ranking"])),
                    "legacy ranking must not contain duplicate ids",
                )
                self.assertTrue(
                    set(scenario["legacy_ranking"]) <= local_candidate_ids,
                    "legacy ranking must only reference candidates from its scenario",
                )
                self.assertIsInstance(scenario["legacy_candidate_ranking"], list)
                self.assertLessEqual(len(scenario["legacy_candidate_ranking"]), 20)
                self.assertEqual(
                    len(scenario["legacy_candidate_ranking"]),
                    len(set(scenario["legacy_candidate_ranking"])),
                    "legacy candidate ranking must not contain duplicate ids",
                )
                self.assertTrue(
                    set(scenario["legacy_candidate_ranking"]) <= local_candidate_ids,
                    "legacy candidate ranking must only reference candidates from its scenario",
                )
                self.assertIsInstance(scenario["violating_ids"], list)
                self.assertEqual(
                    len(scenario["violating_ids"]), len(set(scenario["violating_ids"]))
                )
                self.assertTrue(
                    set(scenario["violating_ids"]) <= local_candidate_ids,
                    "violations must only reference candidates from their scenario",
                )

    def test_fixture_preserves_known_legacy_quality_gaps(self) -> None:
        fixture = self._load_fixture()
        evaluations = [_evaluate_scenario(scenario) for scenario in fixture["scenarios"]]

        self.assertTrue(any(not scenario["legacy_ranking"] for scenario in fixture["scenarios"]))
        self.assertTrue(
            any(
                0 < len(scenario["legacy_ranking"]) < scenario["target_count"]
                for scenario in fixture["scenarios"]
            )
        )
        self.assertTrue(any(len(scenario["candidates"]) > 20 for scenario in fixture["scenarios"]))
        self.assertTrue(any(result["fill_rate"] < 1.0 for result in evaluations))
        self.assertTrue(any(result["constraint_violation_rate"] > 0.0 for result in evaluations))
        self.assertTrue(any(result["intra_list_tag_similarity"] >= 0.8 for result in evaluations))
        self.assertTrue(any(0.0 < result["ndcg_at_target"] < 1.0 for result in evaluations))

    def test_legacy_baseline_replays_from_explicit_fixed_values(self) -> None:
        fixture = self._load_fixture()
        self.assertEqual(
            fixture["legacy_source"],
            {
                "commit": "c6af3a9ae310e2ac6b0dcca970d96bbae8086ec6",
                "method": "curated offline legacy behavior snapshot",
                "replay_command": (
                    "PYTHONPATH=/Users/jiangxingda/Projects/QQChatbot "
                    ".venv/bin/python -m unittest tests/test_recommendation_scenario_replay.py"
                ),
            },
        )
        per_scenario = {
            scenario["id"]: _evaluate_scenario(scenario) for scenario in fixture["scenarios"]
        }
        actual_baseline = {
            metric: fmean(result[metric] for result in per_scenario.values())
            for metric in METRIC_NAMES
        }
        expected_baseline = fixture["legacy_baseline"]

        self.assertEqual(len(per_scenario), 24)
        self.assertEqual(set(expected_baseline), METRIC_NAMES)
        self.assertLessEqual(expected_baseline["ndcg_at_target"], 0.95)
        for scenario_id, result in per_scenario.items():
            with self.subTest(scenario=scenario_id):
                self.assertEqual(set(result), METRIC_NAMES)
                self.assertTrue(all(0.0 <= value <= 1.0 for value in result.values()))
        for metric, expected in expected_baseline.items():
            with self.subTest(metric=metric):
                self.assertAlmostEqual(
                    actual_baseline[metric],
                    expected,
                    delta=1e-6,
                    msg="update the fixed baseline only after intentional legacy changes",
                )


def _evaluate_scenario(scenario: dict[str, Any]) -> dict[str, float]:
    relevance_by_id = {
        candidate["id"]: candidate["relevance"] for candidate in scenario["candidates"]
    }
    tags_by_id = {candidate["id"]: candidate["tags"] for candidate in scenario["candidates"]}
    ranking = scenario["legacy_ranking"]
    target_count = scenario["target_count"]
    return {
        "ndcg_at_target": ndcg_at_k(ranking, relevance_by_id, k=target_count),
        "recall_at_20": recall_at_k(scenario["legacy_candidate_ranking"], relevance_by_id, k=20),
        "constraint_violation_rate": constraint_violation_rate(ranking, scenario["violating_ids"]),
        "fill_rate": fill_rate(ranking, target_count=target_count),
        "intra_list_tag_similarity": intra_list_tag_similarity(ranking, tags_by_id),
    }


if __name__ == "__main__":
    unittest.main()
