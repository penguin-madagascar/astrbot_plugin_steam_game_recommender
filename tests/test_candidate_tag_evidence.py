from __future__ import annotations

import unittest
from dataclasses import FrozenInstanceError

from astrbot_plugin_steam_game_recommender.services.candidate_tag_evidence import (
    build_candidate_tag_evidence,
    excluded_tag_is_hit,
    matches_excluded_tags,
    required_tag_is_satisfied,
    satisfies_required_tags,
)
from astrbot_plugin_steam_game_recommender.storage.models import GameCandidate


class CandidateTagEvidenceStrengthTest(unittest.TestCase):
    def test_first_ordered_tag_has_full_direct_and_supporting_strength(self) -> None:
        evidence = build_candidate_tag_evidence(
            GameCandidate(title="Candidate", ordered_tags=["Action"])
        )

        self.assertEqual(evidence.direct["action"], 1.0)
        self.assertEqual(evidence.supporting["action"], 1.0)

    def test_ordered_tag_positions_straddle_required_threshold(self) -> None:
        evidence = build_candidate_tag_evidence(
            GameCandidate(
                title="Candidate",
                ordered_tags=[
                    "unknown zero",
                    "unknown one",
                    "unknown two",
                    "unknown three",
                    "unknown four",
                    "Action",
                    "RPG",
                ],
                genres=["Strategy"],
            )
        )

        self.assertAlmostEqual(evidence.direct["action"], 0.70)
        self.assertAlmostEqual(evidence.direct["rpg"], 0.64)
        self.assertEqual(evidence.direct["strategy"], 0.65)
        self.assertTrue(required_tag_is_satisfied(evidence, "action"))
        self.assertFalse(required_tag_is_satisfied(evidence, "rpg"))
        self.assertTrue(required_tag_is_satisfied(evidence, "strategy"))

    def test_ordered_tag_strength_has_point_four_floor(self) -> None:
        evidence = build_candidate_tag_evidence(
            GameCandidate(
                title="Candidate",
                ordered_tags=[*(f"unknown {index}" for index in range(12)), "Action"],
            )
        )

        self.assertEqual(evidence.direct["action"], 0.4)
        self.assertEqual(evidence.supporting["action"], 0.4)

    def test_unknown_ordered_tags_still_consume_their_original_positions(self) -> None:
        evidence = build_candidate_tag_evidence(
            GameCandidate(
                title="Candidate",
                ordered_tags=["Action", "unregistered tag", "RPG"],
            )
        )

        self.assertAlmostEqual(evidence.direct["rpg"], 0.88)

    def test_highest_strength_wins_across_direct_and_inferred_sources(self) -> None:
        evidence = build_candidate_tag_evidence(
            GameCandidate(
                title="Candidate",
                ordered_tags=["Action"],
                inferred_tags=["Action", "Horror"],
                description="A horror adventure.",
            )
        )

        self.assertEqual(evidence.direct["action"], 1.0)
        self.assertEqual(evidence.supporting["action"], 1.0)
        self.assertEqual(evidence.supporting["horror"], 0.25)
        self.assertNotIn("horror", evidence.direct)


class CandidateTagEvidenceConstraintTest(unittest.TestCase):
    def test_inferred_and_description_tags_cannot_satisfy_required_tags(self) -> None:
        evidence = build_candidate_tag_evidence(
            GameCandidate(
                title="Candidate",
                inferred_tags=["Horror"],
                description="A cooperative action game.",
            )
        )

        self.assertEqual(evidence.supporting["horror"], 0.25)
        self.assertEqual(evidence.supporting["co_op"], 0.25)
        self.assertFalse(satisfies_required_tags(evidence, ["horror"]))
        self.assertFalse(satisfies_required_tags(evidence, ["co_op"]))

    def test_inferred_and_description_tags_cannot_hit_exclusions(self) -> None:
        evidence = build_candidate_tag_evidence(
            GameCandidate(
                title="Candidate",
                inferred_tags=["Horror"],
                description="A violent cooperative game.",
            )
        )

        self.assertFalse(excluded_tag_is_hit(evidence, "horror"))
        self.assertFalse(matches_excluded_tags(evidence, ["horror", "co_op"]))

    def test_direct_exclusion_matches_at_any_positive_strength(self) -> None:
        evidence = build_candidate_tag_evidence(
            GameCandidate(
                title="Candidate",
                ordered_tags=[*(f"unknown {index}" for index in range(12)), "Horror"],
            )
        )

        self.assertEqual(evidence.direct["horror"], 0.4)
        self.assertTrue(excluded_tag_is_hit(evidence, "恐怖"))
        self.assertTrue(matches_excluded_tags(evidence, ["rpg", "horror"]))

    def test_required_predicate_requires_every_tag_and_normalizes_aliases(self) -> None:
        evidence = build_candidate_tag_evidence(
            GameCandidate(title="Candidate", tags=["Co-op", "Puzzle"])
        )

        self.assertTrue(satisfies_required_tags(evidence, ["合作", "解谜"]))
        self.assertFalse(satisfies_required_tags(evidence, ["co_op", "rpg"]))
        self.assertTrue(satisfies_required_tags(evidence, []))

    def test_direct_coop_evidence_preserves_strength_for_implications(self) -> None:
        evidence = build_candidate_tag_evidence(
            GameCandidate(title="Candidate", tags=["Local Co-op"])
        )

        self.assertEqual(evidence.direct["local_coop"], 0.65)
        self.assertEqual(evidence.direct["co_op"], 0.65)
        self.assertEqual(evidence.direct["multiplayer"], 0.65)
        self.assertEqual(evidence.supporting["local_coop"], 0.65)
        self.assertEqual(evidence.supporting["co_op"], 0.65)
        self.assertEqual(evidence.supporting["multiplayer"], 0.65)
        self.assertTrue(
            satisfies_required_tags(evidence, ["local_coop", "co_op", "multiplayer"])
        )

    def test_online_and_generic_coop_both_imply_multiplayer(self) -> None:
        for direct_tag in ("Online Co-op", "Co-op"):
            with self.subTest(direct_tag=direct_tag):
                evidence = build_candidate_tag_evidence(
                    GameCandidate(title="Candidate", ordered_tags=[direct_tag])
                )

                self.assertEqual(evidence.direct["co_op"], 1.0)
                self.assertEqual(evidence.direct["multiplayer"], 1.0)

    def test_inferred_coop_implications_remain_supporting_only(self) -> None:
        evidence = build_candidate_tag_evidence(
            GameCandidate(title="Candidate", inferred_tags=["Local Co-op"])
        )

        self.assertEqual(evidence.supporting["local_coop"], 0.25)
        self.assertEqual(evidence.supporting["co_op"], 0.25)
        self.assertEqual(evidence.supporting["multiplayer"], 0.25)
        self.assertNotIn("local_coop", evidence.direct)
        self.assertNotIn("co_op", evidence.direct)
        self.assertNotIn("multiplayer", evidence.direct)


class CandidateTagEvidenceImmutabilityTest(unittest.TestCase):
    def test_evidence_and_its_mappings_are_immutable(self) -> None:
        evidence = build_candidate_tag_evidence(
            GameCandidate(title="Candidate", ordered_tags=["Action"])
        )

        with self.assertRaises(FrozenInstanceError):
            evidence.direct = {}  # type: ignore[misc]
        with self.assertRaises(TypeError):
            evidence.direct["action"] = 0.0  # type: ignore[index]
        with self.assertRaises(TypeError):
            evidence.supporting["action"] = 0.0  # type: ignore[index]


if __name__ == "__main__":
    unittest.main()
