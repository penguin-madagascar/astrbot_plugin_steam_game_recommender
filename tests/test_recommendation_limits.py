from __future__ import annotations

import unittest

from astrbot_plugin_steam_game_recommender.services.recommendation_limits import (
    effective_result_limit,
)


class RecommendationLimitTest(unittest.TestCase):
    def test_effective_result_limit_respects_user_count_without_exceeding_config(self) -> None:
        self.assertEqual(effective_result_limit(5, 3), 3)
        self.assertEqual(effective_result_limit(3, 5), 3)
        self.assertEqual(effective_result_limit(5, None), 5)


if __name__ == "__main__":
    unittest.main()
