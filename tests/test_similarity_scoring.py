from __future__ import annotations

import math
import unittest

from astrbot_plugin_game_recommender.services.similarity_ranker import (
    SteamTagProfile,
    bayesian_review_score,
    blend_relevance_components,
    candidate_tag_weights,
    compute_tag_idf,
    ordered_tfidf_cosine,
    rank_steam_candidates,
)
from astrbot_plugin_game_recommender.storage.models import GameCandidate


class OrderedTagSimilarityTest(unittest.TestCase):
    def test_idf_uses_smoothed_candidate_document_frequency(self) -> None:
        idf = compute_tag_idf(
            [
                ["soulslike", "action"],
                ["action", "rpg"],
            ]
        )

        self.assertAlmostEqual(idf["action"], math.log(3 / 3) + 1)
        self.assertAlmostEqual(idf["soulslike"], math.log(3 / 2) + 1)

    def test_steam_tag_order_changes_tfidf_cosine(self) -> None:
        idf = {"soulslike": 1.8, "action": 1.2, "rpg": 1.4}
        reference = ["soulslike", "action", "rpg"]

        same_order = ordered_tfidf_cosine(reference, reference, idf)
        reversed_order = ordered_tfidf_cosine(
            ["rpg", "action", "soulslike"],
            reference,
            idf,
        )

        self.assertAlmostEqual(same_order, 1.0)
        self.assertGreater(same_order, reversed_order)

    def test_type_tags_are_neutral_and_description_tags_are_halved(self) -> None:
        weights = candidate_tag_weights(
            candidate(
                "Weighted Sources",
                tags=["Action"],
                genres=["RPG"],
                inferred_tags=["Horror"],
            ),
            idf={"action": 1.0, "rpg": 1.0, "horror": 1.0},
        )

        self.assertAlmostEqual(weights["action"], 1.0)
        self.assertAlmostEqual(weights["rpg"], 1.0)
        self.assertAlmostEqual(weights["horror"], 0.5)


class HybridRelevanceTest(unittest.TestCase):
    def test_bayesian_review_score_uses_pool_prior_and_minimum_strength(self) -> None:
        game = candidate("Small Sample", tags=["Co-op"], reviews=10, ratio=0.9)

        score = bayesian_review_score(game, prior=0.75, prior_strength=50)

        self.assertAlmostEqual(score, (10 * 0.9 + 50 * 0.75) / 60)
        self.assertEqual(
            bayesian_review_score(
                candidate("No Reviews", tags=["Co-op"], reviews=None, ratio=None),
                prior=0.75,
                prior_strength=50,
            ),
            0.75,
        )

    def test_missing_positive_components_are_redistributed_proportionally(self) -> None:
        score = blend_relevance_components(
            tag_coverage=1.0,
            positive_reference=None,
            library_profile=None,
            review_confidence=0.5,
            negative_reference=0.0,
        )

        self.assertAlmostEqual(score, (0.55 * 1.0 + 0.15 * 0.5) / 0.70)

    def test_negative_reference_similarity_penalizes_without_hard_filtering(self) -> None:
        profile = SteamTagProfile(
            include_tags=["co_op"],
            reference_titles_dislike=["Negative Seed"],
            negative_reference_tag_sequences=[["management", "casual"]],
        )
        ranked = rank_steam_candidates(
            [
                candidate("Negative Seed", ["Management", "Co-op", "Casual"]),
                candidate("Similar Management", ["Management", "Co-op", "Casual"]),
                candidate("Different Co-op", ["Co-op", "Puzzle", "Adventure"]),
            ],
            profile,
            min_review_count=50,
        )

        self.assertNotIn("Negative Seed", [game.title for game in ranked])
        self.assertIn("Similar Management", [game.title for game in ranked])
        similar = next(game for game in ranked if game.title == "Similar Management")
        different = next(game for game in ranked if game.title == "Different Co-op")
        self.assertGreater(similar.facts.negative_reference_score, 0.8)
        self.assertLess(similar.facts.base_relevance_score, different.facts.base_relevance_score)

    def test_all_component_scores_are_recorded_and_drive_ranking(self) -> None:
        profile = SteamTagProfile(
            include_tags=["co_op", "puzzle", "farming"],
            positive_reference_tag_sequences=[["farming", "co_op", "puzzle"]],
            negative_reference_tag_sequences=[["horror", "soulslike", "violent"]],
        )
        ranked = rank_steam_candidates(
            [
                candidate("Positive Match", ["Farming", "Co-op", "Puzzle"]),
                candidate(
                    "Negative Match",
                    ["Horror", "Soulslike", "Violent", "Co-op", "Puzzle"],
                ),
            ],
            profile,
            min_review_count=50,
            profile_tag_weights={"farming": 1.0},
        )

        self.assertEqual(ranked[0].title, "Positive Match")
        positive = ranked[0].facts
        self.assertGreater(positive.tag_coverage_score, 0.0)
        self.assertGreater(positive.positive_reference_score, 0.0)
        self.assertEqual(positive.negative_reference_score, 0.0)
        self.assertGreater(positive.library_profile_score, 0.0)
        self.assertGreater(positive.review_confidence_score, 0.0)
        self.assertAlmostEqual(
            positive.base_relevance_score,
            ranked[0].score / 100,
            places=4,
        )


def candidate(
    title: str,
    tags: list[str],
    genres: list[str] | None = None,
    inferred_tags: list[str] | None = None,
    reviews: int | None = 500,
    ratio: float | None = 0.8,
) -> GameCandidate:
    return GameCandidate(
        title=title,
        appid=abs(hash(title)) % 1_000_000,
        platforms=["PC"],
        genres=genres or [],
        tags=tags,
        inferred_tags=inferred_tags or [],
        stores=["Steam"],
        review_total=reviews,
        review_positive_ratio=ratio,
        review_recent_ratio=ratio,
        index_source="steam_index",
    )


if __name__ == "__main__":
    unittest.main()
