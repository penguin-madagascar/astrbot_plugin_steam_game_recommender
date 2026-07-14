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
    expand_intent_with_reference_tags,
)
from astrbot_plugin_steam_game_recommender.services.tag_normalizer import (
    register_steam_tag_aliases,
)
from astrbot_plugin_steam_game_recommender.storage.models import (
    GameCandidate,
    GamePreference,
)


class RecommendationIntentBuilderTest(unittest.TestCase):
    def test_adds_gameplay_constraints_as_derived_supporting_tags(self) -> None:
        intent = build_recommendation_intent(
            GamePreference(players=2, difficulty="easy", mood="轻松")
        )
        by_tag = {tag.tag: tag for tag in intent.tags}

        for canonical in ("co_op", "multiplayer", "casual", "relaxing"):
            self.assertEqual(by_tag[canonical].role, IntentTagRole.SUPPORTING)
            self.assertEqual(by_tag[canonical].source, IntentTagSource.DERIVED)
            self.assertEqual(by_tag[canonical].weight, 0.35)

    def test_builds_weighted_roles_and_ignores_unverified_extra_tags(self) -> None:
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


class ReferenceTagIntentExpansionTest(unittest.TestCase):
    def test_selects_specific_cached_tags_and_preserves_reference_order(self) -> None:
        tag_names = [
            "Reference Broad",
            "Reference Specific",
            "Reference Narrow",
            "Reference Middle",
            "Reference Fifth",
            "Reference Sixth",
        ]
        register_steam_tag_aliases(
            [
                {"tagid": index, "name": name}
                for index, name in enumerate(tag_names, start=90_001)
            ]
        )
        intent = build_recommendation_intent(GamePreference())
        reference = GameCandidate(
            title="Reference",
            ordered_tags=tag_names,
        )

        expanded = expand_intent_with_reference_tags(
            intent,
            [reference],
            tag_result_counts={
                name.lower().replace(" ", "_"): count
                for name, count in zip(tag_names[:5], [5_000, 20, 40, 300, 900])
            },
        )

        self.assertEqual(intent.tags, ())
        self.assertEqual(
            [tag.tag for tag in expanded.tags],
            [name.lower().replace(" ", "_") for name in tag_names],
        )
        by_tag = {tag.tag: tag for tag in expanded.tags}
        self.assertEqual(by_tag["reference_specific"].role, IntentTagRole.ANCHOR)
        self.assertEqual(by_tag["reference_specific"].weight, 1.0)
        self.assertEqual(by_tag["reference_narrow"].role, IntentTagRole.ANCHOR)
        self.assertEqual(by_tag["reference_narrow"].weight, 0.8)
        self.assertEqual(by_tag["reference_broad"].weight, 0.5)
        self.assertAlmostEqual(by_tag["reference_middle"].weight, 0.5 * 0.85**3)
        self.assertAlmostEqual(by_tag["reference_fifth"].weight, 0.5 * 0.85**4)
        self.assertAlmostEqual(by_tag["reference_sixth"].weight, 0.5 * 0.85**5)
        self.assertTrue(
            all(tag.source is IntentTagSource.REFERENCE for tag in expanded.tags)
        )

    def test_falls_back_to_first_direct_tags_when_counts_are_missing(self) -> None:
        intent = build_recommendation_intent(GamePreference())
        reference = GameCandidate(
            title="Reference",
            tags=["Action", "RPG"],
            genres=["Adventure"],
        )

        expanded = expand_intent_with_reference_tags(intent, [reference])

        self.assertEqual(
            [(tag.tag, tag.role, tag.weight) for tag in expanded.tags],
            [
                ("action", IntentTagRole.ANCHOR, 1.0),
                ("rpg", IntentTagRole.ANCHOR, 0.8),
                ("adventure", IntentTagRole.SUPPORTING, 0.5 * 0.85**2),
            ],
        )

    def test_request_local_counts_override_process_global_counts(self) -> None:
        tag_names = [
            "Local Broad",
            "Local Common",
            "Local Specific",
            "Local Narrow",
        ]
        register_steam_tag_aliases(
            [
                {"tagid": index, "name": name}
                for index, name in enumerate(tag_names, start=91_001)
            ]
        )
        reference = GameCandidate(title="Reference", ordered_tags=tag_names)

        expanded = expand_intent_with_reference_tags(
            build_recommendation_intent(GamePreference()),
            [reference],
            tag_result_counts={
                "local_broad": 10_000,
                "local_common": 8_000,
                "local_specific": 10,
                "local_narrow": 20,
            },
        )

        anchors = [
            tag.tag for tag in expanded.tags if tag.role is IntentTagRole.ANCHOR
        ]
        self.assertEqual(anchors, ["local_specific", "local_narrow"])

    def test_canonical_dedupe_never_downgrades_explicit_intent(self) -> None:
        intent = build_recommendation_intent(
            GamePreference(
                required_tags=["Action"],
                genres_dislike=["Horror"],
            )
        )
        reference = GameCandidate(
            title="Reference",
            ordered_tags=["动作", "Action", "Horror", "RPG"],
        )

        expanded = expand_intent_with_reference_tags(intent, [reference])

        self.assertEqual([tag.tag for tag in expanded.tags], ["action", "horror", "rpg"])
        by_tag = {tag.tag: tag for tag in expanded.tags}
        self.assertEqual(
            (by_tag["action"].role, by_tag["action"].source, by_tag["action"].weight),
            (IntentTagRole.REQUIRED, IntentTagSource.EXPLICIT, 1.0),
        )
        self.assertEqual(
            (by_tag["horror"].role, by_tag["horror"].source, by_tag["horror"].weight),
            (IntentTagRole.EXCLUDE, IntentTagSource.EXPLICIT, 1.0),
        )
        self.assertEqual(by_tag["rpg"].role, IntentTagRole.SUPPORTING)

    def test_supporting_limit_counts_canonical_tags_not_raw_values(self) -> None:
        intent = build_recommendation_intent(GamePreference())
        reference = GameCandidate(
            title="Reference",
            ordered_tags=[
                "Action",
                "动作",
                "RPG",
                "Adventure",
                "Puzzle",
                "Strategy",
                "Simulation",
                "Co-op",
                "Multiplayer",
                "Crafting",
                "Building",
                "Management",
            ],
        )

        expanded = expand_intent_with_reference_tags(intent, [reference])

        self.assertEqual(len(expanded.tags), 10)
        self.assertEqual(expanded.tags[-1].tag, "building")

    def test_reference_source_outranks_derived_weight_for_the_same_role(self) -> None:
        intent = build_recommendation_intent(
            GamePreference(extra_tags=["building"])
        )
        reference = GameCandidate(
            title="Reference",
            ordered_tags=[
                "Action",
                "RPG",
                "Adventure",
                "Puzzle",
                "Strategy",
                "Simulation",
                "Crafting",
                "Management",
                "Farming",
                "Building",
            ],
        )

        expanded = expand_intent_with_reference_tags(intent, [reference])
        building = next(tag for tag in expanded.tags if tag.tag == "building")

        self.assertEqual(building.role, IntentTagRole.SUPPORTING)
        self.assertEqual(building.source, IntentTagSource.REFERENCE)
        self.assertAlmostEqual(building.weight, 0.5 * 0.85**9)


if __name__ == "__main__":
    unittest.main()
