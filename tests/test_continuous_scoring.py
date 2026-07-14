from __future__ import annotations

import inspect
import unittest

from astrbot_plugin_steam_game_recommender.services import similarity_ranker
from astrbot_plugin_steam_game_recommender.services.steam_index import (
    SteamGameIndexService,
    rank_entries,
)
from astrbot_plugin_steam_game_recommender.services.similarity_ranker import (
    SteamTagProfile,
    rank_steam_candidates,
    ranked_game_sort_key,
)
from astrbot_plugin_steam_game_recommender.services.recommendation_scoring import (
    popularity,
    quality_score,
)
from astrbot_plugin_steam_game_recommender.storage.models import (
    GameCandidate,
    RankedGame,
    RecommendationEvidence,
    ScoreBreakdown,
)


class ContinuousScoringTest(unittest.TestCase):
    def test_legacy_scoring_weight_api_is_removed(self) -> None:
        for name in (
            "POSITIVE_COMPONENT_WEIGHTS",
            "resolve_positive_component_weights",
            "weighted_positive_score",
        ):
            with self.subTest(name=name):
                self.assertFalse(hasattr(similarity_ranker, name))

        self.assertNotIn(
            "positive_component_weights",
            inspect.signature(rank_steam_candidates).parameters,
        )
        self.assertNotIn(
            "min_review_count",
            inspect.signature(rank_steam_candidates).parameters,
        )
        self.assertNotIn(
            "min_positive_ratio",
            inspect.signature(rank_steam_candidates).parameters,
        )
        service_parameters = inspect.signature(SteamGameIndexService).parameters
        self.assertNotIn("positive_component_weights", service_parameters)
        self.assertNotIn("min_review_count", service_parameters)
        self.assertNotIn("min_positive_ratio", service_parameters)
        rank_entry_parameters = inspect.signature(rank_entries).parameters
        self.assertNotIn("positive_component_weights", rank_entry_parameters)
        self.assertNotIn("min_review_count", rank_entry_parameters)
        self.assertNotIn("min_positive_ratio", rank_entry_parameters)

    def test_popularity_uses_logarithmic_review_count(self) -> None:
        self.assertEqual(popularity(None), 0.0)
        self.assertAlmostEqual(popularity(99_999), 1.0)
        self.assertEqual(popularity(1_000_000), 1.0)

    def test_ranker_uses_fixed_anchor_semantic_and_quality_policy(self) -> None:
        ranked = rank_steam_candidates(
            [candidate("Complete Match", ["Co-op", "Puzzle"], reviews=999)],
            SteamTagProfile(include_tags=["co_op", "puzzle"]),
        )

        game = ranked[0]
        expected_layer = 0.80 * (0.70 * 0.65) + 0.20 * quality_score(0.8, 999)

        self.assertEqual(game.score, round(expected_layer * 100))
        self.assertIsNone(game.score_breakdown.positive_reference)
        self.assertIsNone(game.score_breakdown.library_profile)
        self.assertAlmostEqual(game.score_breakdown.anchor_coverage, 0.65)
        self.assertAlmostEqual(game.score_breakdown.layer_score, expected_layer)

    def test_low_match_popularity_cannot_beat_high_match_candidate(self) -> None:
        ranked = rank_steam_candidates(
            [
                candidate("Huge but Wrong", ["Action"], reviews=2_000_000, ratio=0.95),
                candidate("Small Exact Match", ["Co-op", "Puzzle"], reviews=100, ratio=0.75),
            ],
            SteamTagProfile(include_tags=["co_op", "puzzle"]),
        )

        self.assertEqual(ranked[0].title, "Small Exact Match")
        self.assertGreater(
            ranked[0].score_breakdown.tag_coverage,
            ranked[1].score_breakdown.tag_coverage,
        )

    def test_negative_reference_penalty_is_capped_at_twenty_points(self) -> None:
        seed = candidate("Negative Seed", ["Management", "Casual"])
        ranked = rank_steam_candidates(
            [
                seed,
                candidate("Near Negative", ["Management", "Casual", "Co-op"]),
                candidate("Different", ["Puzzle", "Co-op"]),
            ],
            SteamTagProfile(
                include_tags=["co_op"],
                reference_titles_dislike=["Negative Seed"],
                negative_reference_candidates=[seed],
            ),
        )

        near = next(game for game in ranked if game.title == "Near Negative")
        self.assertLessEqual(near.score_breakdown.negative_reference_penalty, 20)
        self.assertGreater(near.score_breakdown.negative_reference_penalty, 15)
        self.assertTrue(
            any(
                evidence.sentiment == "negative" and evidence.important
                for evidence in near.recommendation_evidence
            )
        )

    def test_unknown_hard_constraints_are_filtered_by_direct_evidence_gate(self) -> None:
        ranked = rank_steam_candidates(
            [
                candidate("Confirmed", ["Online Co-op", "Puzzle"]),
                candidate("Unknown", ["Multiplayer", "Puzzle"]),
                candidate("Violated", ["Single-player", "Puzzle"]),
            ],
            SteamTagProfile(include_tags=["puzzle"], required_tags=["online_coop"]),
        )

        self.assertEqual([game.title for game in ranked], ["Confirmed"])

    def test_ranked_model_uses_only_new_scoring_and_evidence_fields(self) -> None:
        breakdown_fields = (
            getattr(ScoreBreakdown, "model_fields", None) or ScoreBreakdown.__fields__
        )
        evidence_fields = (
            getattr(RecommendationEvidence, "model_fields", None)
            or RecommendationEvidence.__fields__
        )
        ranked = rank_steam_candidates(
            [candidate("Model", ["Puzzle"])],
            SteamTagProfile(include_tags=["puzzle"]),
        )[0]
        ranked_fields = getattr(type(ranked), "model_fields", None) or type(ranked).__fields__

        self.assertIn("popularity", breakdown_fields)
        self.assertIn("evidence_id", evidence_fields)
        self.assertIn("score_breakdown", ranked_fields)
        self.assertIn("recommendation_evidence", ranked_fields)
        self.assertIn("recommendation_reason", ranked_fields)

    def test_stable_sort_uses_tier_raw_layer_then_retrieval_rank(self) -> None:
        games = [
            RankedGame(
                title="Tier B",
                score=99,
                score_breakdown=ScoreBreakdown(
                    relevance_tier="B", layer_score=0.99, retrieval_rank=1
                ),
            ),
            RankedGame(
                title="Tier A",
                score=10,
                score_breakdown=ScoreBreakdown(
                    relevance_tier="A",
                    layer_score=0.10,
                    retrieval_rank=1,
                    language_adjustment=-10,
                ),
            ),
            RankedGame(
                title="Higher Layer",
                score=90,
                score_breakdown=ScoreBreakdown(
                    relevance_tier="A", layer_score=0.90, retrieval_rank=20
                ),
            ),
            RankedGame(
                title="Earlier Retrieval",
                score=90,
                score_breakdown=ScoreBreakdown(
                    relevance_tier="A", layer_score=0.90, retrieval_rank=2
                ),
            ),
        ]

        self.assertEqual(
            [game.title for game in sorted(games, key=ranked_game_sort_key)],
            [
                "Earlier Retrieval",
                "Higher Layer",
                "Tier A",
                "Tier B",
            ],
        )


class LanguageScoringTest(unittest.TestCase):
    def test_language_adjustment_applies_to_zero_layer_ranker_results(self) -> None:
        ranked = rank_steam_candidates(
            [
                candidate(
                    "A Unsupported Zero",
                    [],
                    reviews=None,
                    ratio=None,
                    supported_languages=["tchinese"],
                    language_data_available=True,
                ),
                candidate(
                    "Z Supported Zero",
                    [],
                    reviews=None,
                    ratio=None,
                    supported_languages=["schinese"],
                    language_data_available=True,
                ),
            ],
            SteamTagProfile(required_languages=["schinese"]),
        )

        self.assertEqual(
            [game.title for game in ranked],
            ["A Unsupported Zero", "Z Supported Zero"],
        )
        self.assertTrue(all(game.score == 0 for game in ranked))
        self.assertTrue(
            all(game.score_breakdown.retrieval_rank > 0 for game in ranked)
        )

    def test_language_adjustment_changes_order_within_the_same_tier(self) -> None:
        cases = (
            (
                "unknown",
                SteamTagProfile(
                    include_tags=["puzzle"],
                    preferred_languages=["schinese"],
                ),
                candidate("A Unknown", ["Puzzle"]),
                -2,
            ),
            (
                "preferred",
                SteamTagProfile(
                    include_tags=["puzzle"],
                    preferred_languages=["schinese"],
                ),
                candidate(
                    "A Preferred Mismatch",
                    ["Puzzle"],
                    supported_languages=["tchinese"],
                    language_data_available=True,
                ),
                -5,
            ),
            (
                "required",
                SteamTagProfile(
                    include_tags=["puzzle"],
                    required_languages=["schinese"],
                ),
                candidate(
                    "A Required Mismatch",
                    ["Puzzle"],
                    supported_languages=["tchinese"],
                    language_data_available=True,
                ),
                -10,
            ),
        )
        for name, profile, mismatch, expected_adjustment in cases:
            with self.subTest(name=name):
                supported = candidate(
                    "Z Supported",
                    ["Puzzle"],
                    supported_languages=["schinese"],
                    language_data_available=True,
                )
                ranked = rank_steam_candidates([mismatch, supported], profile)

                self.assertEqual(
                    [game.title for game in ranked],
                    ["Z Supported", mismatch.title],
                )
                self.assertEqual(
                    ranked[1].score_breakdown.language_adjustment,
                    expected_adjustment,
                )

    def test_language_adjustment_cannot_cross_anchor_tiers(self) -> None:
        ranked = rank_steam_candidates(
            [
                candidate(
                    "A Lower Tier Supported",
                    ["Puzzle"],
                    supported_languages=["schinese"],
                    language_data_available=True,
                ),
                candidate(
                    "Z Higher Tier Unsupported",
                    ["Puzzle", "Strategy"],
                    supported_languages=["tchinese"],
                    language_data_available=True,
                ),
            ],
            SteamTagProfile(
                include_tags=["puzzle", "strategy"],
                required_languages=["schinese"],
            ),
        )

        self.assertEqual(
            [game.title for game in ranked],
            ["Z Higher Tier Unsupported", "A Lower Tier Supported"],
        )
        self.assertEqual(ranked[0].score_breakdown.relevance_tier, "A")
        self.assertEqual(ranked[1].score_breakdown.relevance_tier, "B")

    def test_required_language_uses_penalty_without_filtering(self) -> None:
        ranked = rank_steam_candidates(
            [
                candidate(
                    "Simplified",
                    ["Puzzle"],
                    supported_languages=["schinese"],
                    language_data_available=True,
                ),
                candidate(
                    "Traditional Only",
                    ["Puzzle"],
                    supported_languages=["tchinese"],
                    language_data_available=True,
                ),
                candidate("Unknown", ["Puzzle"]),
            ],
            SteamTagProfile(
                include_tags=["puzzle"],
                preferred_languages=["schinese"],
                required_languages=["schinese"],
            ),
        )

        self.assertEqual(
            [game.title for game in ranked],
            ["Simplified", "Unknown", "Traditional Only"],
        )
        by_title = {game.title: game for game in ranked}
        self.assertEqual(by_title["Simplified"].score_breakdown.language_adjustment, 0)
        self.assertEqual(by_title["Unknown"].score_breakdown.language_adjustment, -2)
        self.assertEqual(by_title["Traditional Only"].score_breakdown.language_adjustment, -10)
        self.assertEqual(by_title["Unknown"].score_breakdown.unknown_constraints_penalty, 0)
        self.assertEqual(by_title["Simplified"].score_breakdown.tag_coverage, 0.65)
        self.assertTrue(
            any("简体中文" in item.text for item in by_title["Unknown"].recommendation_evidence)
        )

    def test_soft_language_mismatch_deducts_five_points(self) -> None:
        game = rank_steam_candidates(
            [
                candidate(
                    "Traditional Only",
                    ["Puzzle"],
                    supported_languages=["tchinese"],
                    language_data_available=True,
                )
            ],
            SteamTagProfile(
                include_tags=["puzzle"],
                preferred_languages=["schinese"],
            ),
        )[0]

        self.assertEqual(game.score_breakdown.language_adjustment, -5)
        self.assertEqual(game.score_breakdown.tag_coverage, 0.65)

    def test_multiple_languages_apply_only_the_most_severe_penalty_once(self) -> None:
        game = rank_steam_candidates(
            [
                candidate(
                    "English Only",
                    ["Puzzle"],
                    supported_languages=["english"],
                    language_data_available=True,
                )
            ],
            SteamTagProfile(
                include_tags=["puzzle"],
                preferred_languages=["japanese"],
                required_languages=["schinese", "tchinese"],
            ),
        )[0]

        self.assertEqual(game.score_breakdown.language_adjustment, -10)

    def test_language_evidence_is_absent_when_user_did_not_request_language(self) -> None:
        game = rank_steam_candidates(
            [
                candidate(
                    "Multilingual",
                    ["Puzzle"],
                    supported_languages=["schinese", "english"],
                    language_data_available=True,
                )
            ],
            SteamTagProfile(include_tags=["puzzle"]),
        )[0]

        self.assertFalse(any(item.category == "language" for item in game.recommendation_evidence))

    def test_legacy_score_only_records_do_not_double_apply_adjustments(self) -> None:
        games = [
            RankedGame(
                title="Higher Legacy Score",
                score=80,
                score_breakdown=ScoreBreakdown(language_adjustment=-10),
            ),
            RankedGame(
                title="Lower Legacy Score",
                score=79,
                score_breakdown=ScoreBreakdown(),
            ),
        ]

        self.assertEqual(
            [game.title for game in sorted(games, key=ranked_game_sort_key)],
            ["Higher Legacy Score", "Lower Legacy Score"],
        )


def candidate(
    title: str,
    tags: list[str],
    reviews: int | None = 500,
    ratio: float | None = 0.8,
    supported_languages: list[str] | None = None,
    language_data_available: bool = False,
) -> GameCandidate:
    return GameCandidate(
        title=title,
        appid=abs(hash(title)) % 1_000_000,
        platforms=["PC"],
        tags=tags,
        review_total=reviews,
        review_positive_ratio=ratio,
        review_recent_ratio=ratio,
        release_date="2025",
        supported_languages=supported_languages or [],
        language_data_available=language_data_available,
        internal_source_markers=["steam_index"],
    )


if __name__ == "__main__":
    unittest.main()
