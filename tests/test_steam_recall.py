from __future__ import annotations

import dataclasses
import unittest

from astrbot_plugin_steam_game_recommender.services.recommendation_intent import (
    IntentTagRole,
    IntentTagSource,
    QualityIntent,
    RecommendationIntent,
    WeightedIntentTag,
)
from astrbot_plugin_steam_game_recommender.services.steam_recall import (
    CandidateRecallResult,
    merge_candidate_sources,
    select_recall_seeds,
)
from astrbot_plugin_steam_game_recommender.storage.models import GameCandidate


def _tag(
    name: str,
    role: IntentTagRole,
    *,
    source: IntentTagSource = IntentTagSource.EXPLICIT,
) -> WeightedIntentTag:
    return WeightedIntentTag(name, role, source, 1.0)


def _intent(*tags: WeightedIntentTag) -> RecommendationIntent:
    return RecommendationIntent(
        tags=tags,
        references=(),
        quality_intent=QualityIntent.NORMAL,
        allow_unreleased=False,
    )


def _candidate(appid: int, title: str | None = None) -> GameCandidate:
    return GameCandidate(title=title or f"Game {appid}", appid=appid)


class RecallSeedSelectionTest(unittest.TestCase):
    def test_required_tags_precede_anchors_while_preserving_role_order(self) -> None:
        intent = _intent(
            _tag("first_anchor", IntentTagRole.ANCHOR),
            _tag("supporting", IntentTagRole.SUPPORTING),
            _tag("first_required", IntentTagRole.REQUIRED),
            _tag("second_anchor", IntentTagRole.ANCHOR),
            _tag("second_required", IntentTagRole.REQUIRED),
            _tag("excluded", IntentTagRole.EXCLUDE),
        )

        seeds = select_recall_seeds(intent)

        self.assertEqual(
            [(seed.tag, seed.role) for seed in seeds],
            [
                ("first_required", IntentTagRole.REQUIRED),
                ("second_required", IntentTagRole.REQUIRED),
                ("first_anchor", IntentTagRole.ANCHOR),
            ],
        )

    def test_deduplicates_tags_and_is_stable(self) -> None:
        intent = _intent(
            _tag("tactical", IntentTagRole.ANCHOR),
            _tag("tactical", IntentTagRole.REQUIRED),
            _tag("simulation", IntentTagRole.ANCHOR),
            _tag("military", IntentTagRole.ANCHOR),
        )

        first = select_recall_seeds(intent)
        second = select_recall_seeds(intent)

        self.assertEqual(first, second)
        self.assertEqual([seed.tag for seed in first], ["tactical", "simulation", "military"])
        self.assertEqual(first[0].role, IntentTagRole.REQUIRED)
        with self.assertRaises(dataclasses.FrozenInstanceError):
            first[0].tag = "changed"

    def test_non_positive_limit_returns_no_seeds(self) -> None:
        intent = _intent(_tag("action", IntentTagRole.ANCHOR))

        self.assertEqual(select_recall_seeds(intent, limit=0), ())


class CandidateSourceMergeTest(unittest.TestCase):
    def test_weighted_rrf_keeps_all_provenance_for_duplicates(self) -> None:
        result = merge_candidate_sources(
            [
                ("tag", "first_tag", [_candidate(1), _candidate(2), _candidate(3)]),
                ("tag", "second_tag", [_candidate(2), _candidate(4), _candidate(5)]),
                ("index", None, [_candidate(6), _candidate(1), _candidate(7)]),
            ]
        )

        self.assertEqual(
            [hit.candidate.appid for hit in result.hits],
            [2, 1, 4, 3, 5, 6, 7],
        )
        self.assertEqual(
            [hit.retrieval_rank for hit in result.hits],
            list(range(1, 8)),
        )
        app_two = next(hit for hit in result.hits if hit.candidate.appid == 2)
        self.assertEqual(
            [(source.source_tag, source.source_rank) for source in app_two.source_hits],
            [("first_tag", 2), ("second_tag", 1)],
        )
        app_four = next(hit for hit in result.hits if hit.candidate.appid == 4)
        self.assertEqual(app_four.source_rank, 2)

    def test_empty_sources_and_invalid_appids_are_ignored(self) -> None:
        result = merge_candidate_sources(
            [
                ("tag", "empty", []),
                (
                    "index",
                    None,
                    [
                        GameCandidate(title="Missing AppID"),
                        _candidate(0),
                        _candidate(-1),
                    ],
                ),
            ]
        )

        self.assertEqual(result.hits, ())

    def test_caps_each_tag_source_tag_total_and_validation_pool(self) -> None:
        sources: list[tuple[str, str | None, list[GameCandidate]]] = []
        for source_index in range(3):
            base = source_index * 1_000
            sources.append(
                (
                    "tag",
                    f"tag_{source_index}",
                    [_candidate(base + offset + 1) for offset in range(25)],
                )
            )
        sources.append(
            (
                "index",
                None,
                [_candidate(10_000 + offset) for offset in range(80)],
            )
        )

        result = merge_candidate_sources(sources)

        self.assertEqual(len(result.hits), 100)
        tag_hits = [hit for hit in result.hits if hit.source_tag is not None]
        self.assertEqual(len(tag_hits), 75)
        for source_tag in ("tag_0", "tag_1", "tag_2"):
            self.assertEqual(
                sum(hit.source_tag == source_tag for hit in tag_hits),
                25,
            )

    def test_custom_caps_are_applied_without_mutating_inputs(self) -> None:
        candidates = [_candidate(appid) for appid in range(1, 10)]

        result = merge_candidate_sources(
            [("tag", "small", candidates)],
            per_tag_limit=3,
            tag_limit=2,
            total_limit=1,
        )

        self.assertEqual([hit.candidate.appid for hit in result.hits], [1])
        self.assertEqual(len(candidates), 9)

    def test_result_is_immutable_and_repeated_merges_are_stable(self) -> None:
        sources = [("top_seller", None, [_candidate(1), _candidate(2)])]

        first = merge_candidate_sources(sources)
        second = merge_candidate_sources(sources)

        self.assertEqual(first, second)
        self.assertIsInstance(first, CandidateRecallResult)
        with self.assertRaises(dataclasses.FrozenInstanceError):
            first.hits = ()
        with self.assertRaises(dataclasses.FrozenInstanceError):
            first.hits[0].source_rank = 99


if __name__ == "__main__":
    unittest.main()
