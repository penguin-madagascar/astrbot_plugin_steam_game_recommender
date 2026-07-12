from __future__ import annotations

import unittest

from astrbot_plugin_steam_game_recommender.services.constraint_evaluator import (
    evaluate_candidate_constraints,
)
from astrbot_plugin_steam_game_recommender.services.similarity_ranker import (
    SteamTagProfile,
    rank_steam_candidates,
)
from astrbot_plugin_steam_game_recommender.storage.models import GameCandidate, ScoreBreakdown


class ConstraintEvaluatorTest(unittest.TestCase):
    def test_ordered_steam_tags_are_direct_constraint_evidence(self) -> None:
        assessment = evaluate_candidate_constraints(
            GameCandidate(
                title="Steam Tagged Horror",
                ordered_tags=["Horror", "Co-op"],
            ),
            required_tags=[],
            exclude_tags=["horror"],
        )

        self.assertEqual(assessment.status, "violated")
        self.assertEqual(assessment.violations, ["horror"])

    def test_reports_satisfied_required_tags_with_multiplayer_implications(self) -> None:
        assessment = evaluate_candidate_constraints(
            candidate("Couch Team", ["Local Co-op"]),
            required_tags=["co_op", "local_coop", "multiplayer"],
            exclude_tags=["horror"],
        )

        self.assertEqual(assessment.status, "satisfied")
        self.assertEqual(
            assessment.hits,
            ["co_op", "local_coop", "multiplayer"],
        )
        self.assertEqual(assessment.violations, [])
        self.assertEqual(assessment.unknowns, [])

    def test_reports_confirmed_contradictions_as_violations(self) -> None:
        multiplayer = evaluate_candidate_constraints(
            candidate("Solo Only", ["Single-player"]),
            required_tags=["multiplayer"],
            exclude_tags=[],
        )
        relaxing = evaluate_candidate_constraints(
            candidate("Punishing", ["Difficult"]),
            required_tags=["relaxing"],
            exclude_tags=[],
        )

        for assessment, tag in (
            (multiplayer, "multiplayer"),
            (relaxing, "relaxing"),
        ):
            with self.subTest(tag=tag):
                self.assertEqual(assessment.status, "violated")
                self.assertEqual(assessment.violations, [tag])

    def test_missing_evidence_is_unknown_instead_of_violation(self) -> None:
        assessment = evaluate_candidate_constraints(
            candidate("Unverified Co-op", ["Multiplayer"]),
            required_tags=["online_coop", "crafting"],
            exclude_tags=[],
        )

        self.assertEqual(assessment.status, "unknown")
        self.assertEqual(assessment.violations, [])
        self.assertEqual(assessment.unknowns, ["online_coop", "crafting"])

    def test_exclusions_only_use_direct_steam_tags(self) -> None:
        direct = evaluate_candidate_constraints(
            candidate("Direct Horror", ["Co-op", "Horror"]),
            required_tags=[],
            exclude_tags=["horror"],
        )
        inferred = evaluate_candidate_constraints(
            candidate(
                "Description Horror",
                ["Co-op"],
                description="A horror-themed cooperative story.",
            ),
            required_tags=[],
            exclude_tags=["horror"],
        )

        self.assertEqual(direct.status, "violated")
        self.assertEqual(direct.violations, ["horror"])
        self.assertEqual(inferred.status, "satisfied")


class ConstraintAwareRankerTest(unittest.TestCase):
    def test_filters_violations_and_penalizes_unknown_language_requirement(self) -> None:
        profile = SteamTagProfile(
            include_tags=["co_op", "puzzle"],
            required_languages=["schinese"],
        )
        ranked = rank_steam_candidates(
            [
                candidate(
                    "Confirmed",
                    ["Co-op", "Puzzle"],
                    supported_languages=["schinese"],
                    language_data_available=True,
                ),
                candidate("Unknown", ["Co-op", "Puzzle"]),
                candidate(
                    "Violated",
                    ["Co-op", "Puzzle"],
                    supported_languages=["english"],
                    language_data_available=True,
                ),
            ],
            profile,
            min_review_count=50,
            min_positive_ratio=0.65,
        )

        self.assertEqual([game.title for game in ranked], ["Confirmed", "Unknown"])
        self.assertEqual(ranked[0].score_breakdown.unknown_constraints_penalty, 0)
        self.assertEqual(ranked[1].score_breakdown.unknown_constraints_penalty, 15)
        self.assertTrue(
            any(
                item.important and "简体中文" in item.text
                for item in ranked[1].recommendation_evidence
            )
        )

    def test_description_only_exclusion_does_not_hard_filter_candidate(self) -> None:
        ranked = rank_steam_candidates(
            [
                candidate(
                    "Description Only",
                    ["Co-op", "Puzzle"],
                    description="A horror story inferred only from the description.",
                )
            ],
            SteamTagProfile(include_tags=["co_op", "puzzle"], exclude_tags=["horror"]),
            min_review_count=50,
            min_positive_ratio=0.65,
        )

        self.assertEqual([game.title for game in ranked], ["Description Only"])

    def test_score_breakdown_declares_continuous_components(self) -> None:
        fields = getattr(ScoreBreakdown, "model_fields", None) or ScoreBreakdown.__fields__

        for field in (
            "tag_coverage",
            "positive_reference",
            "negative_reference_penalty",
            "library_profile",
            "review_reputation",
            "popularity",
            "data_completeness",
            "unknown_constraints_penalty",
        ):
            self.assertIn(field, fields)


def candidate(
    title: str,
    tags: list[str],
    description: str | None = None,
    supported_languages: list[str] | None = None,
    language_data_available: bool = False,
) -> GameCandidate:
    return GameCandidate(
        title=title,
        appid=abs(hash(title)) % 1_000_000,
        platforms=["PC"],
        genres=[],
        tags=tags,
        stores=["Steam"],
        review_total=500,
        review_positive_ratio=0.8,
        supported_languages=supported_languages or [],
        language_data_available=language_data_available,
        description=description,
    )


if __name__ == "__main__":
    unittest.main()
