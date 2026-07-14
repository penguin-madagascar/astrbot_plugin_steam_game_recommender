from __future__ import annotations

import unittest

from astrbot_plugin_steam_game_recommender.services.recommendation_intent import (
    IntentTagRole,
    IntentTagSource,
    QualityIntent,
    RecommendationIntent,
    WeightedIntentTag,
)
from astrbot_plugin_steam_game_recommender.services.similarity_ranker import (
    rank_steam_candidates,
)
from astrbot_plugin_steam_game_recommender.storage.models import GameCandidate


def intent(
    *tags: WeightedIntentTag,
    quality: QualityIntent = QualityIntent.NORMAL,
    allow_unreleased: bool = False,
) -> RecommendationIntent:
    return RecommendationIntent(
        tags=tags,
        references=(),
        quality_intent=quality,
        allow_unreleased=allow_unreleased,
    )


def tag(
    name: str,
    role: IntentTagRole,
    weight: float = 1.0,
    source: IntentTagSource = IntentTagSource.EXPLICIT,
) -> WeightedIntentTag:
    return WeightedIntentTag(name, role, source, weight)


def candidate(
    appid: int,
    title: str,
    *,
    ordered: list[str] | None = None,
    tags: list[str] | None = None,
    inferred: list[str] | None = None,
    reviews: int | None = 500,
    ratio: float | None = 0.8,
    coming_soon: bool = False,
) -> GameCandidate:
    return GameCandidate(
        appid=appid,
        title=title,
        app_type="game",
        ordered_tags=ordered or [],
        tags=tags or [],
        inferred_tags=inferred or [],
        review_total=reviews,
        review_positive_ratio=ratio,
        coming_soon=coming_soon,
    )


class AnchorTierRankerTest(unittest.TestCase):
    def test_core_anchor_beats_many_broad_tags_and_mainstream_quality(self) -> None:
        query = intent(
            tag("soulslike", IntentTagRole.ANCHOR),
            tag("action", IntentTagRole.SUPPORTING, 0.35),
            tag("rpg", IntentTagRole.SUPPORTING, 0.35),
        )

        ranked = rank_steam_candidates(
            [
                candidate(
                    1,
                    "Broad Blockbuster",
                    tags=["Action", "RPG"],
                    reviews=100_000,
                    ratio=0.95,
                ),
                candidate(
                    2,
                    "Core Niche",
                    ordered=["Souls-like"],
                    reviews=None,
                    ratio=None,
                ),
            ],
            query,
        )

        self.assertEqual([game.title for game in ranked], ["Core Niche", "Broad Blockbuster"])
        self.assertEqual(ranked[0].score_breakdown.relevance_tier, "A")
        self.assertEqual(ranked[1].score_breakdown.relevance_tier, "C")

    def test_a_tier_cold_candidate_beats_b_tier_mature_candidate(self) -> None:
        query = intent(
            tag("soulslike", IntentTagRole.ANCHOR, 1.0),
            tag("action", IntentTagRole.ANCHOR, 0.8),
        )
        ranked = rank_steam_candidates(
            [
                candidate(
                    1,
                    "Mature B",
                    ordered=["Souls-like"],
                    reviews=100_000,
                    ratio=0.98,
                ),
                candidate(
                    2,
                    "Cold A",
                    tags=["Souls-like", "Action"],
                    reviews=None,
                    ratio=None,
                ),
            ],
            query,
        )

        self.assertEqual([game.title for game in ranked], ["Cold A", "Mature B"])
        self.assertEqual(ranked[0].score_breakdown.relevance_tier, "A")
        self.assertEqual(ranked[1].score_breakdown.relevance_tier, "B")

    def test_same_tier_prefers_mature_quality(self) -> None:
        query = intent(tag("puzzle", IntentTagRole.ANCHOR))
        ranked = rank_steam_candidates(
            [
                candidate(1, "One Review", tags=["Puzzle"], reviews=1, ratio=1.0),
                candidate(
                    2,
                    "Mature",
                    tags=["Puzzle"],
                    reviews=100_000,
                    ratio=0.9,
                ),
            ],
            query,
        )

        self.assertEqual([game.title for game in ranked], ["Mature", "One Review"])

    def test_required_tag_needs_direct_strength_at_least_point_65(self) -> None:
        query = intent(tag("strategy", IntentTagRole.REQUIRED))
        ranked = rank_steam_candidates(
            [
                candidate(1, "Genre Strength", tags=["Strategy"]),
                candidate(
                    2,
                    "Ordered Below Threshold",
                    ordered=["a", "b", "c", "d", "e", "f", "Strategy"],
                ),
                candidate(3, "Inferred Only", inferred=["Strategy"]),
            ],
            query,
        )

        self.assertEqual([game.title for game in ranked], ["Genre Strength"])

    def test_inferred_exclusion_does_not_filter_but_direct_exclusion_does(self) -> None:
        query = intent(
            tag("puzzle", IntentTagRole.SUPPORTING),
            tag("horror", IntentTagRole.EXCLUDE),
        )
        ranked = rank_steam_candidates(
            [
                candidate(1, "Inferred Horror", tags=["Puzzle"], inferred=["Horror"]),
                candidate(2, "Direct Horror", tags=["Puzzle", "Horror"]),
            ],
            query,
        )

        self.assertEqual([game.title for game in ranked], ["Inferred Horror"])

    def test_missing_and_zero_reviews_have_zero_quality(self) -> None:
        query = intent(tag("puzzle", IntentTagRole.SUPPORTING))
        ranked = rank_steam_candidates(
            [
                candidate(index + 1, str(count), tags=["Puzzle"], reviews=count, ratio=0.9)
                for index, count in enumerate((None, 0, 1, 50, 1000, 100_000))
            ],
            query,
        )
        by_title = {game.title: game for game in ranked}

        self.assertEqual(by_title["None"].score_breakdown.quality_score, 0.0)
        self.assertEqual(by_title["0"].score_breakdown.quality_score, 0.0)
        for count in (1, 50, 1000, 100_000):
            self.assertGreater(by_title[str(count)].score_breakdown.quality_score, 0.0)
        self.assertGreater(
            by_title["100000"].score_breakdown.quality_score,
            by_title["1"].score_breakdown.quality_score,
        )

    def test_negative_reference_similarity_subtracts_exactly_one_quarter(self) -> None:
        query = intent(tag("puzzle", IntentTagRole.SUPPORTING))
        seed = candidate(10, "Negative Seed", tags=["Puzzle"])
        ranked = rank_steam_candidates(
            [candidate(1, "Candidate", tags=["Puzzle"])],
            query,
            negative_reference_candidates=[seed],
        )

        breakdown = ranked[0].score_breakdown
        self.assertEqual(breakdown.supporting_similarity, 0.65)
        self.assertEqual(breakdown.negative_reference_similarity, 1.0)
        self.assertAlmostEqual(breakdown.semantic_score, 0.40)

    def test_positive_reference_is_not_scored_again_after_intent_expansion(self) -> None:
        query = intent(tag("puzzle", IntentTagRole.SUPPORTING))
        game = candidate(1, "Candidate", tags=["Puzzle"])
        seed = candidate(10, "Positive Seed", tags=["Puzzle"])

        without_seed = rank_steam_candidates([game], query)[0]
        with_seed = rank_steam_candidates(
            [game, seed],
            query,
            positive_reference_candidates=[seed],
        )[0]

        self.assertEqual(with_seed.title, "Candidate")
        self.assertEqual(
            with_seed.score_breakdown.layer_score,
            without_seed.score_breakdown.layer_score,
        )

    def test_mainstream_changes_only_the_within_tier_blend(self) -> None:
        tags = (
            tag("puzzle", IntentTagRole.SUPPORTING),
            tag("strategy", IntentTagRole.SUPPORTING),
        )
        games = [
            candidate(1, "Semantic", tags=["Puzzle", "Strategy"], reviews=None, ratio=None),
            candidate(2, "Quality", tags=["Puzzle"], reviews=100_000, ratio=0.95),
        ]

        normal = rank_steam_candidates(games, intent(*tags))
        mainstream = rank_steam_candidates(
            games,
            intent(*tags, quality=QualityIntent.MAINSTREAM),
        )

        self.assertEqual(normal[0].title, "Semantic")
        self.assertEqual(mainstream[0].title, "Quality")
        self.assertTrue(all(game.score_breakdown.relevance_tier == "broad" for game in mainstream))
        self.assertTrue(
            any(item.evidence_id == "mainstream_intent" for item in mainstream[0].recommendation_evidence)
        )

    def test_retrieval_rank_breaks_exact_layer_ties(self) -> None:
        query = intent(tag("puzzle", IntentTagRole.ANCHOR))
        ranked = rank_steam_candidates(
            [
                candidate(1, "First Input", tags=["Puzzle"]),
                candidate(2, "Second Input", tags=["Puzzle"]),
            ],
            query,
            retrieval_ranks={1: 20, 2: 3},
        )

        self.assertEqual([game.appid for game in ranked], [2, 1])
        self.assertEqual(ranked[0].score_breakdown.retrieval_rank, 3)

    def test_coming_soon_is_filtered_unless_explicitly_allowed(self) -> None:
        game = candidate(1, "Upcoming", tags=["Puzzle"], coming_soon=True)
        default = rank_steam_candidates(
            [game], intent(tag("puzzle", IntentTagRole.ANCHOR))
        )
        allowed = rank_steam_candidates(
            [game],
            intent(tag("puzzle", IntentTagRole.ANCHOR), allow_unreleased=True),
        )

        self.assertEqual(default, [])
        self.assertEqual([item.title for item in allowed], ["Upcoming"])

    def test_library_tags_are_supporting_only_and_cannot_cross_tiers(self) -> None:
        query = intent(tag("soulslike", IntentTagRole.ANCHOR))
        ranked = rank_steam_candidates(
            [
                candidate(1, "Library Only", tags=["Farming"], reviews=100_000, ratio=0.98),
                candidate(2, "Core", tags=["Souls-like"], reviews=None, ratio=None),
            ],
            query,
            profile_tag_weights={"farming": 1.0},
        )

        self.assertEqual([game.title for game in ranked], ["Core", "Library Only"])
        self.assertEqual(ranked[1].score_breakdown.anchor_coverage, 0.0)

    def test_b_and_c_evidence_explicitly_marks_relaxed_matching(self) -> None:
        query = intent(
            tag("soulslike", IntentTagRole.ANCHOR, 1.0),
            tag("action", IntentTagRole.ANCHOR, 0.8),
        )
        ranked = rank_steam_candidates(
            [
                candidate(1, "B", ordered=["Souls-like"]),
                candidate(2, "C", tags=["Puzzle"]),
            ],
            query,
        )

        for game in ranked:
            relaxed = [item for item in game.recommendation_evidence if item.evidence_id == "core_missing"]
            self.assertEqual(len(relaxed), 1)
            self.assertTrue(relaxed[0].important)
            self.assertIn("宽松匹配", relaxed[0].text)


if __name__ == "__main__":
    unittest.main()
