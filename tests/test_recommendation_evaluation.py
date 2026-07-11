from __future__ import annotations

import unittest

from astrbot_plugin_game_recommender.services.recommendation_evaluation import (
    constraint_violation_rate,
    fill_rate,
    intra_list_tag_similarity,
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
        self.assertEqual(intra_list_tag_similarity([], {}), 0.0)

    def test_ideal_ranking_has_perfect_ndcg_and_recall(self) -> None:
        relevance = {"best": 3, "good": 2, "fair": 1, "irrelevant": 0}
        ranking = ["best", "good", "fair", "irrelevant"]

        self.assertAlmostEqual(ndcg_at_k(ranking, relevance, k=5), 1.0)
        self.assertAlmostEqual(recall_at_k(ranking, relevance, k=20), 1.0)

    def test_constraint_violation_rate_counts_violating_results(self) -> None:
        ranking = ["safe", "over-budget", "excluded-tag", "also-safe"]

        rate = constraint_violation_rate(ranking, {"over-budget", "excluded-tag"})

        self.assertAlmostEqual(rate, 0.5)

    def test_fill_rate_is_partial_until_target_is_filled(self) -> None:
        self.assertAlmostEqual(fill_rate(["a", "b", "c"], target_count=5), 0.6)
        self.assertAlmostEqual(fill_rate(["a", "b", "c", "d", "e"], target_count=5), 1.0)
        self.assertAlmostEqual(
            fill_rate(["a", "b", "c", "d", "e", "f"], target_count=5),
            1.0,
        )

    def test_identical_tag_sets_have_similarity_one(self) -> None:
        ranking = ["a", "b", "c"]
        tags = {
            "a": {"co-op", "puzzle"},
            "b": {"co-op", "puzzle"},
            "c": {"co-op", "puzzle"},
        }

        self.assertAlmostEqual(intra_list_tag_similarity(ranking, tags), 1.0)

    def test_disjoint_tag_sets_have_similarity_zero(self) -> None:
        ranking = ["a", "b", "c"]
        tags = {
            "a": {"co-op"},
            "b": {"strategy"},
            "c": {"racing"},
        }

        self.assertAlmostEqual(intra_list_tag_similarity(ranking, tags), 0.0)


if __name__ == "__main__":
    unittest.main()
