from __future__ import annotations

import unittest

from astrbot_plugin_game_recommender.services.retry_command import parse_retry_request


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


if __name__ == "__main__":
    unittest.main()
