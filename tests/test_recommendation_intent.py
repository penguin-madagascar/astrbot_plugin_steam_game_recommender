from __future__ import annotations

import dataclasses
import unittest

from astrbot_plugin_steam_game_recommender.services.preference_rules import (
    merge_text_preference,
)
from astrbot_plugin_steam_game_recommender.services.recommendation_intent import (
    IntentTagRole,
    IntentTagSource,
    QualityIntent,
    ReferencePolarity,
    build_recommendation_intent,
)
from astrbot_plugin_steam_game_recommender.storage.models import GamePreference


class RecommendationIntentBuilderTest(unittest.TestCase):
    def test_builds_weighted_roles_and_deduplicates_canonical_tags(self) -> None:
        intent = build_recommendation_intent(
            GamePreference(
                required_tags=["本地合作"],
                genres_like=["Local Co-op", "解谜"],
                extra_tags=["local_coop", "剧情向"],
                genres_dislike=["恐怖"],
                quality_intent="mainstream",
                allow_unreleased=True,
            )
        )

        self.assertEqual(
            [(tag.tag, tag.role, tag.source, tag.weight) for tag in intent.tags],
            [
                ("local_coop", IntentTagRole.REQUIRED, IntentTagSource.EXPLICIT, 1.0),
                ("puzzle", IntentTagRole.ANCHOR, IntentTagSource.EXPLICIT, 1.0),
                ("story_rich", IntentTagRole.SUPPORTING, IntentTagSource.DERIVED, 0.35),
                ("horror", IntentTagRole.EXCLUDE, IntentTagSource.EXPLICIT, 1.0),
            ],
        )
        self.assertEqual(intent.quality_intent, QualityIntent.MAINSTREAM)
        self.assertTrue(intent.allow_unreleased)

    def test_preserves_parser_final_polarity_without_duplicate_intent_tags(self) -> None:
        preference = merge_text_preference(
            GamePreference(genres_like=["horror"], extra_tags=["恐怖"]),
            "这次不要恐怖游戏",
        )

        intent = build_recommendation_intent(preference)

        horror = [tag for tag in intent.tags if tag.tag == "horror"]
        self.assertEqual(len(horror), 1)
        self.assertEqual(horror[0].role, IntentTagRole.EXCLUDE)

    def test_groups_one_reference_title_with_all_search_aliases(self) -> None:
        intent = build_recommendation_intent(
            GamePreference(
                reference_games_like=["黑暗之魂"],
                reference_search_terms=["Dark Souls", "DARK SOULS Remastered"],
            )
        )

        self.assertEqual(len(intent.references), 1)
        reference = intent.references[0]
        self.assertEqual(reference.display_title, "黑暗之魂")
        self.assertEqual(
            reference.aliases,
            ("黑暗之魂", "Dark Souls", "DARK SOULS Remastered"),
        )
        self.assertEqual(reference.polarity, ReferencePolarity.POSITIVE)

    def test_pairs_equal_reference_titles_and_aliases_positionally(self) -> None:
        intent = build_recommendation_intent(
            GamePreference(
                reference_games_like=["黑暗之魂", "星露谷物语"],
                reference_search_terms=["Dark Souls", "Stardew Valley"],
            )
        )

        self.assertEqual(
            [(reference.display_title, reference.aliases) for reference in intent.references],
            [
                ("黑暗之魂", ("黑暗之魂", "Dark Souls")),
                ("星露谷物语", ("星露谷物语", "Stardew Valley")),
            ],
        )

    def test_does_not_attach_mismatched_or_positive_aliases_to_other_references(self) -> None:
        intent = build_recommendation_intent(
            GamePreference(
                reference_games_like=["黑暗之魂", "星露谷物语"],
                reference_search_terms=["Dark Souls"],
                reference_games_dislike=["杀戮尖塔"],
            )
        )

        self.assertEqual(
            [
                (reference.display_title, reference.aliases, reference.polarity)
                for reference in intent.references
            ],
            [
                ("黑暗之魂", ("黑暗之魂",), ReferencePolarity.POSITIVE),
                ("星露谷物语", ("星露谷物语",), ReferencePolarity.POSITIVE),
                ("杀戮尖塔", ("杀戮尖塔",), ReferencePolarity.NEGATIVE),
            ],
        )

    def test_intent_values_are_immutable(self) -> None:
        intent = build_recommendation_intent(GamePreference(genres_like=["解谜"]))

        with self.assertRaises(dataclasses.FrozenInstanceError):
            intent.allow_unreleased = True
        with self.assertRaises(dataclasses.FrozenInstanceError):
            intent.tags[0].weight = 0.5


class GamePreferenceIntentFieldsTest(unittest.TestCase):
    def test_normalizes_quality_intent_and_defaults_release_policy(self) -> None:
        mainstream = GamePreference(quality_intent=" MAINSTREAM ")
        unknown = GamePreference(quality_intent="premium")

        self.assertEqual(mainstream.quality_intent, "mainstream")
        self.assertEqual(unknown.quality_intent, "normal")
        self.assertFalse(mainstream.allow_unreleased)


if __name__ == "__main__":
    unittest.main()
