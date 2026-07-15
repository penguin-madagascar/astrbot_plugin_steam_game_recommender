from __future__ import annotations

import json
import unittest
from pathlib import Path

from astrbot_plugin_steam_game_recommender.config_migration import migrate_config_data
from astrbot_plugin_steam_game_recommender.services.preference_rules import (
    extract_result_count,
    infer_preference_from_text,
)
from astrbot_plugin_steam_game_recommender.services.recommendation_limits import (
    DEFAULT_RECOMMENDATION_COUNT,
    MAX_RECOMMENDATION_COUNT,
    effective_result_limit,
)
from astrbot_plugin_steam_game_recommender.storage.models import GamePreference


class RecommendationLimitsV2Test(unittest.TestCase):
    def test_defaults_and_global_max_are_ten(self) -> None:
        self.assertEqual(DEFAULT_RECOMMENDATION_COUNT, 10)
        self.assertEqual(MAX_RECOMMENDATION_COUNT, 10)
        self.assertEqual(GamePreference().result_count, 10)
        self.assertEqual(infer_preference_from_text("推荐解谜游戏").result_count, 10)

    def test_explicit_arabic_and_chinese_counts_are_deterministic(self) -> None:
        cases = {
            "推荐 3 款游戏": 3,
            "推荐三款游戏": 3,
            "推荐两款游戏": 2,
            "Please recommend 7 games": 7,
            "Please recommend 11 results": 10,
            "推荐 99 款游戏": 10,
            "推荐十一款游戏": None,
            "推荐 0 款游戏": None,
            "推荐零款游戏": None,
            "第 3 个标签很重要": None,
            "推荐适合3个人玩的游戏": None,
            "想找能和4个朋友玩的合作游戏": None,
            "我已经买过3款游戏，想推荐点新的": None,
            "推荐3个": 3,
        }

        for text, expected in cases.items():
            with self.subTest(text=text):
                self.assertEqual(extract_result_count(text), expected)

    def test_invalid_earlier_quantity_does_not_hide_later_request_count(self) -> None:
        self.assertEqual(
            extract_result_count("我已经玩过3款游戏，请推荐5款新的"),
            5,
        )

    def test_existing_config_value_caps_default_and_explicit_requests(self) -> None:
        self.assertEqual(effective_result_limit(5, 10), 5)
        self.assertEqual(effective_result_limit(5, 3), 3)
        self.assertEqual(effective_result_limit(None, None), 10)
        self.assertEqual(effective_result_limit(99, 99), 10)

    def test_new_schema_defaults_to_ten_and_keeps_fallback_opt_in(self) -> None:
        root = Path(__file__).resolve().parents[1]
        schema = json.loads((root / "_conf_schema.json").read_text(encoding="utf-8"))

        self.assertEqual(
            schema["recommendation_and_scoring"]["items"]["max_results"]["default"],
            10,
        )
        fallback = schema["model_and_access"]["items"]["llm_fallback_provider_id"]
        self.assertEqual(fallback["default"], "")
        self.assertEqual(fallback["_special"], "select_provider")

    def test_migration_drops_old_bool_but_keeps_provider_and_saved_limit(self) -> None:
        root = Path(__file__).resolve().parents[1]
        schema = json.loads((root / "_conf_schema.json").read_text(encoding="utf-8"))
        migrated, changed = migrate_config_data(
            {
                "model_and_access": {
                    "llm_provider_id": "provider/current",
                    "llm_fallback_provider_id": "provider/old-fallback",
                    "enable_llm_fallback": True,
                },
                "recommendation_and_scoring": {"max_results": 5},
            },
            schema,
        )

        self.assertTrue(changed)
        self.assertEqual(
            migrated["model_and_access"],
            {
                "llm_provider_id": "provider/current",
                "llm_fallback_provider_id": "provider/old-fallback",
            },
        )
        self.assertEqual(migrated["recommendation_and_scoring"]["max_results"], 5)


if __name__ == "__main__":
    unittest.main()
