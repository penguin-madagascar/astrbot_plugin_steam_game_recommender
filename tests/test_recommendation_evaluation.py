from __future__ import annotations

import unittest

from astrbot_plugin_steam_game_recommender.services.recommendation_evaluation import (
    constraint_violation_rate,
    fill_rate,
    ndcg_at_k,
    recall_at_k,
)


class RecommendationEvaluationTest(unittest.TestCase):
    def test_empty_ranking_has_zero_quality_metrics(self) -> None:
        relevance = {"best": 3, "good": 2}

        self.assertEqual(ndcg_at_k([], relevance, k=5), 0.0)
        self.assertEqual(recall_at_k([], relevance, k=20), 0.0)
        self.assertEqual(constraint_violation_rate([], {"blocked"}), 0.0)
        self.assertEqual(fill_rate([], target_count=5), 0.0)

    def test_ideal_ranking_has_perfect_ndcg_and_recall(self) -> None:
        relevance = {"best": 3, "good": 2, "fair": 1, "irrelevant": 0}
        ranking = ["best", "good", "fair", "irrelevant"]

        self.assertAlmostEqual(ndcg_at_k(ranking, relevance, k=5), 1.0)
        self.assertAlmostEqual(recall_at_k(ranking, relevance, k=20), 1.0)

    def test_ndcg_ignores_duplicate_ids(self) -> None:
        relevance = {"best": 3, "good": 2, "fair": 1}
        duplicated = ndcg_at_k(["best", "best", "good", "fair"], relevance, k=5)
        deduplicated = ndcg_at_k(["best", "good", "fair"], relevance, k=5)

        self.assertLessEqual(duplicated, 1.0)
        self.assertAlmostEqual(duplicated, deduplicated)

    def test_recall_ignores_duplicate_ids_before_cutoff(self) -> None:
        relevance = {"first": 3, "second": 2}

        self.assertAlmostEqual(
            recall_at_k(["first", "first", "second"], relevance, k=2),
            1.0,
        )

    def test_constraint_violation_rate_counts_violating_results(self) -> None:
        ranking = ["safe", "over-budget", "excluded-tag", "also-safe"]

        rate = constraint_violation_rate(ranking, {"over-budget", "excluded-tag"})

        self.assertAlmostEqual(rate, 0.5)

    def test_constraint_violation_rate_ignores_duplicate_ids(self) -> None:
        rate = constraint_violation_rate(["blocked", "blocked", "safe"], {"blocked"})

        self.assertAlmostEqual(rate, 0.5)

    def test_fill_rate_is_partial_until_target_is_filled(self) -> None:
        self.assertAlmostEqual(fill_rate(["a", "b", "c"], target_count=5), 0.6)
        self.assertAlmostEqual(fill_rate(["a", "b", "c", "d", "e"], target_count=5), 1.0)
        self.assertAlmostEqual(
            fill_rate(["a", "b", "c", "d", "e", "f"], target_count=5),
            1.0,
        )

    def test_fill_rate_is_zero_for_non_positive_target_count(self) -> None:
        for target_count in (0, -1):
            with self.subTest(target_count=target_count):
                self.assertEqual(fill_rate(["a"], target_count=target_count), 0.0)

    def test_fill_rate_ignores_duplicate_ids(self) -> None:
        self.assertAlmostEqual(fill_rate(["a", "a", "b"], target_count=3), 2 / 3)


if __name__ == "__main__":
    unittest.main()
