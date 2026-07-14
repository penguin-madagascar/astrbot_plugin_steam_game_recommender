from __future__ import annotations

import math
import unittest

from astrbot_plugin_steam_game_recommender.services.candidate_tag_evidence import (
    CandidateTagEvidence,
)
from astrbot_plugin_steam_game_recommender.services.recommendation_intent import (
    IntentTagRole,
    IntentTagSource,
    QualityIntent,
    RecommendationIntent,
    WeightedIntentTag,
)
from astrbot_plugin_steam_game_recommender.services.recommendation_scoring import (
    RelevanceTier,
    anchor_coverage,
    evidence_scaled_similarity,
    layer_score,
    popularity,
    quality_score,
    relevance_tier,
    semantic_score,
    wilson_lower_bound,
)


def make_intent(
    *tags: WeightedIntentTag,
    quality_intent: QualityIntent = QualityIntent.NORMAL,
) -> RecommendationIntent:
    return RecommendationIntent(
        tags=tags,
        references=(),
        quality_intent=quality_intent,
        allow_unreleased=False,
    )


def intent_tag(
    tag: str,
    role: IntentTagRole,
    weight: float = 1.0,
) -> WeightedIntentTag:
    return WeightedIntentTag(tag, role, IntentTagSource.EXPLICIT, weight)


class AnchorCoverageTest(unittest.TestCase):
    def test_uses_only_anchor_weights_and_direct_evidence(self) -> None:
        intent = make_intent(
            intent_tag("souls_like", IntentTagRole.ANCHOR, 1.0),
            intent_tag("action", IntentTagRole.ANCHOR, 0.8),
            intent_tag("rpg", IntentTagRole.SUPPORTING, 1.0),
        )
        evidence = CandidateTagEvidence(
            direct={"souls_like": 1.0, "action": 0.5},
            supporting={"rpg": 1.0},
        )

        self.assertAlmostEqual(anchor_coverage(intent, evidence), 1.4 / 1.8)

    def test_supporting_only_inferred_evidence_does_not_cover_anchor(self) -> None:
        intent = make_intent(intent_tag("souls_like", IntentTagRole.ANCHOR))
        evidence = CandidateTagEvidence(
            direct={},
            supporting={"souls_like": 0.25},
        )

        self.assertEqual(anchor_coverage(intent, evidence), 0.0)
        self.assertEqual(relevance_tier(intent, evidence), RelevanceTier.C)

    def test_clamps_anchor_weights_and_direct_strengths(self) -> None:
        intent = make_intent(
            intent_tag("first", IntentTagRole.ANCHOR, 2.0),
            intent_tag("second", IntentTagRole.ANCHOR, 1.0),
            intent_tag("ignored", IntentTagRole.ANCHOR, -1.0),
        )
        evidence = CandidateTagEvidence(
            direct={"first": 2.0, "second": -0.5, "ignored": 1.0},
            supporting={},
        )

        self.assertEqual(anchor_coverage(intent, evidence), 0.5)

    def test_no_anchor_query_uses_broad_tier(self) -> None:
        intent = make_intent(intent_tag("rpg", IntentTagRole.SUPPORTING))
        evidence = CandidateTagEvidence(direct={"rpg": 1.0}, supporting={})

        self.assertEqual(anchor_coverage(intent, evidence), 0.0)
        self.assertEqual(relevance_tier(intent, evidence), RelevanceTier.BROAD)

    def test_tier_boundaries_are_exact_and_deterministic(self) -> None:
        intent = make_intent(intent_tag("anchor", IntentTagRole.ANCHOR))

        cases = (
            (0.0, RelevanceTier.C),
            (0.599999, RelevanceTier.C),
            (0.60, RelevanceTier.A),
            (0.600001, RelevanceTier.A),
        )
        for strength, expected in cases:
            with self.subTest(strength=strength):
                evidence = CandidateTagEvidence(
                    direct={"anchor": strength},
                    supporting={},
                )
                self.assertEqual(relevance_tier(intent, evidence), expected)


class EvidenceScaledSimilarityTest(unittest.TestCase):
    def test_empty_query_or_evidence_has_zero_similarity(self) -> None:
        self.assertEqual(evidence_scaled_similarity({}, {"action": 1.0}), 0.0)
        self.assertEqual(evidence_scaled_similarity({"action": 1.0}, {}), 0.0)

    def test_single_inferred_match_scores_below_full_direct_match(self) -> None:
        query = {"action": 1.0}

        inferred = evidence_scaled_similarity(query, {"action": 0.25})
        direct = evidence_scaled_similarity(query, {"action": 1.0})

        self.assertEqual(inferred, 0.25)
        self.assertEqual(direct, 1.0)

    def test_multitag_similarity_is_cosine_scaled_by_weighted_coverage(self) -> None:
        query = {"action": 1.0, "rpg": 0.5}
        candidate = {"action": 1.0, "rpg": 0.5, "unrequested": 1.0}
        weighted_candidate = (1.0, 0.25)
        cosine = (1.0 * weighted_candidate[0] + 0.5 * weighted_candidate[1]) / (
            math.sqrt(1.0**2 + 0.5**2)
            * math.sqrt(weighted_candidate[0] ** 2 + weighted_candidate[1] ** 2)
        )
        coverage = (1.0 * 1.0 + 0.5 * 0.5) / (1.0 + 0.5)

        self.assertAlmostEqual(
            evidence_scaled_similarity(query, candidate),
            cosine * coverage,
        )

    def test_similarity_clamps_invalid_ranges_without_returning_nan(self) -> None:
        score = evidence_scaled_similarity(
            {"full": 2.0, "negative": -1.0, "nan": math.nan},
            {"full": math.inf, "negative": 1.0, "nan": 1.0},
        )

        self.assertEqual(score, 1.0)
        self.assertTrue(math.isfinite(score))


class ReviewQualityTest(unittest.TestCase):
    def test_wilson_lower_bound_matches_standard_formula(self) -> None:
        ratio = 0.9
        count = 100
        z = 1.96
        expected = (
            ratio
            + z**2 / (2 * count)
            - z * math.sqrt((ratio * (1 - ratio) + z**2 / (4 * count)) / count)
        ) / (1 + z**2 / count)

        self.assertAlmostEqual(wilson_lower_bound(ratio, count), expected)

    def test_wilson_clamps_ratio_and_nonpositive_count(self) -> None:
        self.assertEqual(wilson_lower_bound(0.8, None), 0.0)
        self.assertEqual(wilson_lower_bound(0.8, 0), 0.0)
        self.assertEqual(wilson_lower_bound(0.8, -5), 0.0)
        self.assertEqual(
            wilson_lower_bound(2.0, 10),
            wilson_lower_bound(1.0, 10),
        )
        self.assertEqual(wilson_lower_bound(-1.0, 10), 0.0)

    def test_wilson_rejects_non_integer_review_counts(self) -> None:
        invalid_counts = (True, 1.5, "50", math.nan, math.inf, -math.inf)
        for count in invalid_counts:
            with self.subTest(count=count):
                self.assertEqual(wilson_lower_bound(0.8, count), 0.0)  # type: ignore[arg-type]

        self.assertEqual(
            wilson_lower_bound(0.8, 50.0),  # type: ignore[arg-type]
            wilson_lower_bound(0.8, 50),
        )

    def test_popularity_uses_log_scale_and_clamps_count(self) -> None:
        self.assertEqual(popularity(None), 0.0)
        self.assertEqual(popularity(-1), 0.0)
        self.assertAlmostEqual(popularity(999), 3 / 5)
        self.assertEqual(popularity(100_000), 1.0)
        self.assertEqual(popularity(1_000_000), 1.0)

    def test_popularity_rejects_non_integer_review_counts(self) -> None:
        invalid_counts = (True, 1.5, "50", math.nan, math.inf, -math.inf)
        for count in invalid_counts:
            with self.subTest(count=count):
                self.assertEqual(popularity(count), 0.0)  # type: ignore[arg-type]

        self.assertEqual(
            popularity(999.0),  # type: ignore[arg-type]
            popularity(999),
        )

    def test_missing_ratio_or_reviews_make_the_entire_quality_score_zero(self) -> None:
        self.assertEqual(quality_score(None, 100_000), 0.0)
        self.assertEqual(quality_score(0.95, None), 0.0)
        self.assertEqual(quality_score(0.95, 0), 0.0)
        self.assertEqual(quality_score(math.nan, 100_000), 0.0)

    def test_quality_rejects_non_integer_review_counts(self) -> None:
        invalid_counts = (True, 1.5, "50", math.nan, math.inf, -math.inf)
        for count in invalid_counts:
            with self.subTest(count=count):
                self.assertEqual(quality_score(0.8, count), 0.0)  # type: ignore[arg-type]

        self.assertEqual(
            quality_score(0.8, 50.0),  # type: ignore[arg-type]
            quality_score(0.8, 50),
        )

    def test_review_count_matrix_rewards_growing_confidence(self) -> None:
        counts = (None, 0, 1, 50, 1_000, 100_000)
        scores = [quality_score(0.8, count) for count in counts]

        self.assertEqual(scores[:2], [0.0, 0.0])
        self.assertTrue(all(left < right for left, right in zip(scores[2:], scores[3:])))
        for count, score in zip(counts[2:], scores[2:]):
            with self.subTest(count=count):
                expected = 0.60 * wilson_lower_bound(0.8, count) + 0.40 * popularity(
                    count
                )
                self.assertAlmostEqual(score, expected)


class CompositeScoringTest(unittest.TestCase):
    def test_anchor_semantic_formula_and_negative_coefficient(self) -> None:
        intent = make_intent(intent_tag("anchor", IntentTagRole.ANCHOR))

        score = semantic_score(
            intent,
            anchor_coverage_value=0.8,
            supporting_similarity=0.4,
            negative_similarity=0.2,
        )

        self.assertAlmostEqual(score, 0.70 * 0.8 + 0.30 * 0.4 - 0.25 * 0.2)

    def test_no_anchor_semantic_formula_ignores_anchor_value(self) -> None:
        intent = make_intent(intent_tag("support", IntentTagRole.SUPPORTING))

        score = semantic_score(
            intent,
            anchor_coverage_value=1.0,
            supporting_similarity=0.6,
            negative_similarity=0.2,
        )

        self.assertAlmostEqual(score, 0.6 - 0.25 * 0.2)

    def test_negative_reference_penalty_is_exactly_one_quarter(self) -> None:
        intent = make_intent()

        self.assertEqual(
            semantic_score(
                intent,
                anchor_coverage_value=0.0,
                supporting_similarity=1.0,
                negative_similarity=1.0,
            ),
            0.75,
        )

    def test_semantic_score_clamps_every_component(self) -> None:
        intent = make_intent(intent_tag("anchor", IntentTagRole.ANCHOR))

        self.assertEqual(
            semantic_score(
                intent,
                anchor_coverage_value=2.0,
                supporting_similarity=-1.0,
                negative_similarity=-2.0,
            ),
            0.70,
        )
        self.assertEqual(
            semantic_score(
                intent,
                anchor_coverage_value=math.nan,
                supporting_similarity=math.nan,
                negative_similarity=math.nan,
            ),
            0.0,
        )

    def test_normal_and_mainstream_layer_formulas(self) -> None:
        semantic = 0.6
        quality = 0.8

        self.assertAlmostEqual(
            layer_score(semantic, quality, QualityIntent.NORMAL),
            0.80 * semantic + 0.20 * quality,
        )
        self.assertAlmostEqual(
            layer_score(semantic, quality, QualityIntent.MAINSTREAM),
            0.65 * semantic + 0.35 * quality,
        )

    def test_layer_score_clamps_components(self) -> None:
        self.assertEqual(layer_score(2.0, -1.0, "normal"), 0.80)
        self.assertEqual(layer_score(math.nan, math.inf, "mainstream"), 0.35)


if __name__ == "__main__":
    unittest.main()
