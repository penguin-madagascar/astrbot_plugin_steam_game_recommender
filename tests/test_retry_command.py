from __future__ import annotations

import unittest

from astrbot_plugin_steam_game_recommender.services.recommendation_memory import (
    RecommendationResultSummary,
)
from astrbot_plugin_steam_game_recommender.services.retry_command import (
    apply_preference_patch,
    merge_retry_preferences,
    parse_preference_patch,
    parse_retry_request,
)
from astrbot_plugin_steam_game_recommender.storage.models import GamePreference


class RetryCommandTest(unittest.TestCase):
    def test_detects_plain_retry_request(self) -> None:
        parsed = parse_retry_request("重新推荐")

        self.assertTrue(parsed.is_retry)
        self.assertEqual(parsed.supplement, "")

    def test_detects_retry_request_with_supplement(self) -> None:
        parsed = parse_retry_request("换一批 不要恐怖，预算 100 以内")

        self.assertTrue(parsed.is_retry)
        self.assertEqual(parsed.supplement, "不要恐怖，预算 100 以内")

    def test_ignores_normal_recommendation(self) -> None:
        parsed = parse_retry_request("推荐几个 Steam 合作解谜")

        self.assertFalse(parsed.is_retry)
        self.assertEqual(parsed.supplement, "")


class PreferencePatchTest(unittest.TestCase):
    def setUp(self) -> None:
        self.results = [
            RecommendationResultSummary(1, "Game A", ["Puzzle"]),
            RecommendationResultSummary(2, "Game B", ["Farming"]),
        ]

    def test_like_ordinal_becomes_positive_reference(self) -> None:
        parsed = parse_preference_patch("喜欢第 2 款这类，再换一批", len(self.results))
        preference, excluded_appids, _titles = apply_preference_patch(
            GamePreference(),
            parsed.patch,
            self.results,
        )

        self.assertEqual(parsed.residual_text, "")
        self.assertEqual(preference.reference_games_like, ["Game B"])
        self.assertEqual(preference.reference_games_dislike, [])
        self.assertEqual(excluded_appids, [])

    def test_dislike_ordinal_becomes_negative_reference(self) -> None:
        parsed = parse_preference_patch("不喜欢第1款这类，换不同玩法", len(self.results))
        preference, _appids, _titles = apply_preference_patch(
            GamePreference(),
            parsed.patch,
            self.results,
        )

        self.assertEqual(parsed.residual_text, "换不同玩法")
        self.assertEqual(preference.reference_games_dislike, ["Game A"])

    def test_dislike_ordinal_flips_existing_positive_reference(self) -> None:
        parsed = parse_preference_patch("不喜欢第1款这类", len(self.results))
        preference, _appids, _titles = apply_preference_patch(
            GamePreference(
                reference_entities=[
                    {
                        "display_title": "Game A",
                        "aliases": ["Localized Game A"],
                        "polarity": "positive",
                    }
                ]
            ),
            parsed.patch,
            self.results,
        )

        self.assertEqual(preference.reference_games_like, [])
        self.assertEqual(preference.reference_games_dislike, ["Game A"])
        self.assertEqual(
            [
                (entity.display_title, entity.aliases, entity.polarity)
                for entity in preference.reference_entities
            ],
            [("Game A", ["Localized Game A"], "negative")],
        )

    def test_like_ordinal_flips_existing_negative_reference(self) -> None:
        parsed = parse_preference_patch("喜欢第1款这类", len(self.results))
        preference, _appids, _titles = apply_preference_patch(
            GamePreference(
                reference_entities=[
                    {
                        "display_title": "Game A",
                        "aliases": ["Localized Game A"],
                        "polarity": "negative",
                    }
                ]
            ),
            parsed.patch,
            self.results,
        )

        self.assertEqual(preference.reference_games_like, ["Game A"])
        self.assertEqual(preference.reference_games_dislike, [])
        self.assertEqual(
            [
                (entity.display_title, entity.aliases, entity.polarity)
                for entity in preference.reference_entities
            ],
            [("Game A", ["Localized Game A"], "positive")],
        )

    def test_ordinal_feedback_matches_an_existing_localized_alias(self) -> None:
        parsed = parse_preference_patch("不喜欢第1款这类", 1)
        preference, _appids, _titles = apply_preference_patch(
            GamePreference(
                reference_entities=[
                    {
                        "display_title": "本地化标题",
                        "aliases": ["Portal"],
                        "polarity": "positive",
                    }
                ]
            ),
            parsed.patch,
            [RecommendationResultSummary(10, "Ｐｏｒｔａｌ™", ["Puzzle"])],
        )

        self.assertEqual(preference.reference_games_like, [])
        self.assertEqual(preference.reference_games_dislike, ["本地化标题"])
        self.assertEqual(
            [
                (entity.display_title, entity.aliases, entity.polarity)
                for entity in preference.reference_entities
            ],
            [("本地化标题", ["Portal"], "negative")],
        )

    def test_ordinal_feedback_matches_an_existing_resolved_appid(self) -> None:
        parsed = parse_preference_patch("不喜欢第1款这类", 1)
        preference, _appids, _titles = apply_preference_patch(
            GamePreference(
                reference_games_like=["本地化标题"],
                resolved_reference_games=[
                    {
                        "raw_text": "本地化标题",
                        "normalized_title": "localizedtitle",
                        "canonical_title": "Portal",
                        "appid": 10,
                        "confidence": 1.0,
                        "polarity": "like",
                    }
                ],
            ),
            parsed.patch,
            [RecommendationResultSummary(10, "Different Store Title", ["Puzzle"])],
        )

        self.assertEqual(preference.reference_games_like, [])
        self.assertEqual(preference.reference_games_dislike, ["本地化标题"])
        self.assertEqual(
            [entity.polarity for entity in preference.reference_entities],
            ["negative"],
        )
        self.assertEqual(
            [reference.polarity for reference in preference.resolved_reference_games],
            ["dislike"],
        )

    def test_plain_rejection_only_excludes_that_result(self) -> None:
        parsed = parse_preference_patch("不要第2款", len(self.results))
        preference, appids, titles = apply_preference_patch(
            GamePreference(),
            parsed.patch,
            self.results,
        )

        self.assertEqual(preference.reference_games_like, [])
        self.assertEqual(preference.reference_games_dislike, [])
        self.assertEqual(appids, [2])
        self.assertEqual(titles, ["game b"])

    def test_out_of_range_ordinal_is_ignored_with_warning(self) -> None:
        parsed = parse_preference_patch("喜欢第9款这类", len(self.results))
        preference, appids, titles = apply_preference_patch(
            GamePreference(),
            parsed.patch,
            self.results,
            parsed.warnings,
        )

        self.assertEqual(appids, [])
        self.assertEqual(titles, [])
        self.assertEqual(preference.reference_games_like, [])
        self.assertTrue(any("超出" in warning for warning in preference.parse_warnings))

    def test_latest_condition_patch_can_override_and_clear(self) -> None:
        overridden = parse_preference_patch("预算改为 80，改成 3 人", len(self.results))
        preference, _appids, _titles = apply_preference_patch(
            GamePreference(
                budget=100,
                budget_currency="CNY",
                budget_is_required=True,
                players=2,
            ),
            overridden.patch,
            self.results,
        )
        cleared = parse_preference_patch("取消预算限制", len(self.results))
        preference, _appids, _titles = apply_preference_patch(
            preference,
            cleared.patch,
            self.results,
        )

        self.assertIsNone(preference.budget)
        self.assertIsNone(preference.budget_currency)
        self.assertFalse(preference.budget_is_required)
        self.assertEqual(preference.players, 3)

    def test_budget_patch_updates_requirement_level(self) -> None:
        soft = parse_preference_patch("预算改为 80 元", len(self.results))
        required = parse_preference_patch("预算必须低于 60 元", len(self.results))

        self.assertEqual(soft.patch.condition_overrides["budget"], 80)
        self.assertFalse(soft.patch.condition_overrides["budget_is_required"])
        self.assertEqual(required.patch.condition_overrides["budget"], 60)
        self.assertTrue(required.patch.condition_overrides["budget_is_required"])

    def test_retry_merge_carries_required_budget_with_new_amount(self) -> None:
        merged = merge_retry_preferences(
            GamePreference(budget=100, budget_is_required=False),
            GamePreference(budget=60, budget_is_required=True),
        )

        self.assertEqual(merged.budget, 60)
        self.assertTrue(merged.budget_is_required)

    def test_retry_merge_preserves_aliases_for_each_reference_entity(self) -> None:
        merged = merge_retry_preferences(
            GamePreference(
                reference_entities=[
                    {
                        "display_title": "Reference A",
                        "aliases": ["Alias A1", "Alias A2"],
                        "polarity": "positive",
                    }
                ]
            ),
            GamePreference(
                reference_entities=[
                    {
                        "display_title": "Reference B",
                        "aliases": ["Alias B1", "Alias B2"],
                        "polarity": "positive",
                    }
                ]
            ),
        )

        self.assertEqual(
            [
                (entity.display_title, entity.aliases)
                for entity in merged.reference_entities
            ],
            [
                ("Reference A", ["Alias A1", "Alias A2"]),
                ("Reference B", ["Alias B1", "Alias B2"]),
            ],
        )
        self.assertFalse(
            any("无法可靠归属" in warning for warning in merged.parse_warnings)
        )

    def test_retry_supplement_flips_reference_polarity_and_merges_aliases(self) -> None:
        merged = merge_retry_preferences(
            GamePreference(
                reference_entities=[
                    {
                        "display_title": "Game A",
                        "aliases": ["Alias A1"],
                        "polarity": "positive",
                    }
                ]
            ),
            GamePreference(
                reference_entities=[
                    {
                        "display_title": "Game A",
                        "aliases": ["Alias A2"],
                        "polarity": "negative",
                    }
                ]
            ),
        )

        self.assertEqual(merged.reference_games_like, [])
        self.assertEqual(merged.reference_games_dislike, ["Game A"])
        self.assertEqual(
            [
                (entity.display_title, entity.aliases, entity.polarity)
                for entity in merged.reference_entities
            ],
            [("Game A", ["Alias A1", "Alias A2"], "negative")],
        )

    def test_retry_supplement_can_flip_negative_reference_back_to_positive(self) -> None:
        merged = merge_retry_preferences(
            GamePreference(
                reference_entities=[
                    {
                        "display_title": "Game A",
                        "aliases": ["Alias A1"],
                        "polarity": "negative",
                    }
                ]
            ),
            GamePreference(
                reference_entities=[
                    {
                        "display_title": "Game A",
                        "aliases": ["Alias A2"],
                        "polarity": "positive",
                    }
                ]
            ),
        )

        self.assertEqual(merged.reference_games_like, ["Game A"])
        self.assertEqual(merged.reference_games_dislike, [])
        self.assertEqual(
            [
                (entity.display_title, entity.aliases, entity.polarity)
                for entity in merged.reference_entities
            ],
            [("Game A", ["Alias A1", "Alias A2"], "positive")],
        )


if __name__ == "__main__":
    unittest.main()
