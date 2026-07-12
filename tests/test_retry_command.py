from __future__ import annotations

import unittest

from astrbot_plugin_game_recommender.services.recommendation_memory import (
    RecommendationResultSummary,
)
from astrbot_plugin_game_recommender.services.retry_command import (
    apply_preference_patch,
    parse_preference_patch,
    parse_retry_request,
)
from astrbot_plugin_game_recommender.storage.models import GamePreference


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
            GamePreference(budget=100, players=2),
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
        self.assertEqual(preference.players, 3)


if __name__ == "__main__":
    unittest.main()
