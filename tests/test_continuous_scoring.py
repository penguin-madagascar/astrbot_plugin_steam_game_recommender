from __future__ import annotations

import math
import unittest

from astrbot_plugin_steam_game_recommender.services.similarity_ranker import (
    POSITIVE_COMPONENT_WEIGHTS,
    SteamTagProfile,
    popularity_score,
    rank_steam_candidates,
    ranked_game_sort_key,
    resolve_positive_component_weights,
    weighted_positive_score,
)
from astrbot_plugin_steam_game_recommender.storage.models import (
    GameCandidate,
    RankedGame,
    RecommendationEvidence,
    ScoreBreakdown,
)


class ContinuousScoringTest(unittest.TestCase):
    def test_positive_component_weights_match_rebalanced_scoring(self) -> None:
        self.assertEqual(
            POSITIVE_COMPONENT_WEIGHTS,
            {
                "tag_coverage": 35.0,
                "positive_reference": 25.0,
                "library_profile": 5.0,
                "review_reputation": 20.0,
                "popularity": 15.0,
            },
        )

    def test_optional_components_are_renormalized_for_all_availability_states(self) -> None:
        components = {
            "tag_coverage": 0.8,
            "positive_reference": 0.6,
            "library_profile": 0.4,
            "review_reputation": 0.9,
            "popularity": 0.7,
        }

        self.assertAlmostEqual(weighted_positive_score(**components), 73.5)
        self.assertAlmostEqual(
            weighted_positive_score(**{**components, "library_profile": None}),
            (35 * 0.8 + 25 * 0.6 + 20 * 0.9 + 15 * 0.7) / 95 * 100,
        )
        self.assertAlmostEqual(
            weighted_positive_score(**{**components, "positive_reference": None}),
            (35 * 0.8 + 5 * 0.4 + 20 * 0.9 + 15 * 0.7) / 75 * 100,
        )
        self.assertAlmostEqual(
            weighted_positive_score(
                tag_coverage=0.8,
                positive_reference=None,
                library_profile=None,
                review_reputation=0.9,
                popularity=0.7,
            ),
            (35 * 0.8 + 20 * 0.9 + 15 * 0.7) / 70 * 100,
        )

    def test_positive_weight_config_is_sanitized_and_all_zero_falls_back(self) -> None:
        self.assertEqual(
            resolve_positive_component_weights(
                {
                    "tag_coverage": 120,
                    "positive_reference": -3,
                    "library_profile": "invalid",
                    "review_reputation": float("inf"),
                    "popularity": "10",
                }
            ),
            {
                "tag_coverage": 100.0,
                "positive_reference": 0.0,
                "library_profile": 5.0,
                "review_reputation": 20.0,
                "popularity": 10.0,
            },
        )
        self.assertEqual(
            resolve_positive_component_weights(dict.fromkeys(POSITIVE_COMPONENT_WEIGHTS, 0)),
            POSITIVE_COMPONENT_WEIGHTS,
        )

    def test_custom_positive_weights_are_used_with_dynamic_normalization(self) -> None:
        weights = {
            "tag_coverage": 60,
            "positive_reference": 20,
            "library_profile": 0,
            "review_reputation": 10,
            "popularity": 10,
        }
        components = {
            "tag_coverage": 0.8,
            "positive_reference": 0.6,
            "library_profile": 0.4,
            "review_reputation": 0.9,
            "popularity": 0.7,
        }

        self.assertAlmostEqual(
            weighted_positive_score(
                positive_component_weights=weights,
                **components,
            ),
            76.0,
        )
        self.assertAlmostEqual(
            weighted_positive_score(
                positive_component_weights=weights,
                **{**components, "positive_reference": None},
            ),
            80.0,
        )

    def test_popularity_uses_logarithmic_review_count(self) -> None:
        self.assertEqual(popularity_score(None), 0.0)
        self.assertAlmostEqual(popularity_score(99_999), 1.0)
        self.assertEqual(popularity_score(1_000_000), 1.0)

    def test_missing_reference_and_library_weights_are_renormalized(self) -> None:
        ranked = rank_steam_candidates(
            [candidate("Complete Match", ["Co-op", "Puzzle"], reviews=999)],
            SteamTagProfile(include_tags=["co_op", "puzzle"]),
            min_review_count=50,
        )

        game = ranked[0]
        popularity = min(math.log10(1_000) / 5, 1)
        expected = round((35 * 1.0 + 20 * 0.8 + 15 * popularity) / 70 * 100)

        self.assertEqual(game.score, expected)
        self.assertIsNone(game.score_breakdown.positive_reference)
        self.assertIsNone(game.score_breakdown.library_profile)
        self.assertAlmostEqual(game.score_breakdown.positive_score, expected, delta=0.5)

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

    def test_ranker_uses_instance_weights_without_mutating_defaults(self) -> None:
        candidates = [
            candidate("Exact Small Match", ["Puzzle"], reviews=1, ratio=0.8),
            candidate("Huge Wrong Match", ["Action"], reviews=1_000_000, ratio=0.8),
        ]
        profile = SteamTagProfile(include_tags=["puzzle"])

        default_ranked = rank_steam_candidates(candidates, profile)
        popularity_ranked = rank_steam_candidates(
            candidates,
            profile,
            positive_component_weights={
                "tag_coverage": 0,
                "positive_reference": 0,
                "library_profile": 0,
                "review_reputation": 0,
                "popularity": 100,
            },
        )

        self.assertEqual(default_ranked[0].title, "Exact Small Match")
        self.assertEqual(popularity_ranked[0].title, "Huge Wrong Match")
        self.assertEqual(POSITIVE_COMPONENT_WEIGHTS["tag_coverage"], 35.0)
        self.assertEqual(POSITIVE_COMPONENT_WEIGHTS["popularity"], 15.0)

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

    def test_unknown_hard_constraints_deduct_at_most_fifteen_and_violations_filter(self) -> None:
        ranked = rank_steam_candidates(
            [
                candidate("Confirmed", ["Online Co-op", "Puzzle"]),
                candidate("Unknown", ["Multiplayer", "Puzzle"]),
                candidate("Violated", ["Single-player", "Puzzle"]),
            ],
            SteamTagProfile(include_tags=["puzzle"], required_tags=["online_coop"]),
        )

        self.assertEqual([game.title for game in ranked], ["Confirmed", "Unknown"])
        unknown = ranked[1]
        self.assertEqual(unknown.score_breakdown.unknown_constraints_penalty, 15)
        self.assertTrue(
            any(
                evidence.sentiment == "uncertain" and evidence.important
                for evidence in unknown.recommendation_evidence
            )
        )

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

    def test_stable_sort_uses_score_coverage_reviews_year_and_title(self) -> None:
        games = [
            RankedGame(
                title="Lower Score",
                score=90,
                review_total=10_000,
                release_date="2026",
                score_breakdown=ScoreBreakdown(tag_coverage=1),
            ),
            RankedGame(
                title="Higher Score",
                score=91,
                review_total=1,
                release_date="2020",
                score_breakdown=ScoreBreakdown(tag_coverage=0),
            ),
            RankedGame(
                title="Higher Coverage",
                score=90,
                review_total=1,
                release_date="2020",
                score_breakdown=ScoreBreakdown(tag_coverage=0.9),
            ),
            RankedGame(
                title="More Reviews",
                score=90,
                review_total=100,
                release_date="2020",
                score_breakdown=ScoreBreakdown(tag_coverage=0.8),
            ),
            RankedGame(
                title="Newer",
                score=90,
                review_total=10,
                release_date="2025",
                score_breakdown=ScoreBreakdown(tag_coverage=0.8),
            ),
            RankedGame(
                title="Alpha",
                score=90,
                review_total=10,
                release_date="2024",
                score_breakdown=ScoreBreakdown(tag_coverage=0.8),
            ),
            RankedGame(
                title="Zulu",
                score=90,
                review_total=10,
                release_date="2024",
                score_breakdown=ScoreBreakdown(tag_coverage=0.8),
            ),
        ]

        self.assertEqual(
            [game.title for game in sorted(games, key=ranked_game_sort_key)],
            [
                "Higher Score",
                "Lower Score",
                "Higher Coverage",
                "More Reviews",
                "Newer",
                "Alpha",
                "Zulu",
            ],
        )


class LanguageScoringTest(unittest.TestCase):
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
        self.assertEqual(by_title["Simplified"].score_breakdown.tag_coverage, 1)
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
        self.assertEqual(game.score_breakdown.tag_coverage, 1)

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
