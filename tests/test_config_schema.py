from __future__ import annotations

import json
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
GROUP_KEYS = [
    "model_and_access",
    "price_and_region",
    "recommendation_and_scoring",
    "cache_and_network",
]


def load_schema() -> dict:
    return json.loads((ROOT / "_conf_schema.json").read_text(encoding="utf-8"))


def group_items(schema: dict, group: str) -> dict:
    if group not in schema:
        raise AssertionError(f"missing dashboard group: {group}")
    return schema[group]["items"]


class ConfigSchemaTest(unittest.TestCase):
    def test_dashboard_uses_four_ordered_object_groups(self) -> None:
        schema = load_schema()

        self.assertEqual(list(schema), GROUP_KEYS)
        self.assertEqual(
            list(group_items(schema, "model_and_access")),
            ["llm_provider_id", "enable_llm_fallback", "steam_api_key"],
        )
        self.assertEqual(
            list(group_items(schema, "price_and_region")),
            ["steam_price_heybox_notice", "default_region"],
        )
        self.assertEqual(
            list(group_items(schema, "recommendation_and_scoring")),
            [
                "max_results",
                "tag_coverage_weight",
                "positive_reference_weight",
                "library_profile_weight",
                "review_reputation_weight",
                "popularity_weight",
                "steam_index_ttl_hours",
                "steam_min_review_count",
                "steam_min_positive_ratio",
            ],
        )
        self.assertEqual(
            list(group_items(schema, "cache_and_network")),
            ["cache_ttl_hours", "timeout_seconds"],
        )
        for group in GROUP_KEYS:
            with self.subTest(group=group):
                self.assertEqual(schema[group]["type"], "object")
                self.assertTrue(schema[group]["description"].strip())
                self.assertTrue(schema[group]["hint"].strip())

    def test_model_and_access_copy_and_order(self) -> None:
        items = group_items(load_schema(), "model_and_access")

        self.assertEqual(
            items["llm_provider_id"]["description"],
            "用于偏好解析和推荐理由的 LLM 提供商",
        )
        self.assertTrue(items["llm_provider_id"]["hint"].startswith("‼️留空时"))
        self.assertEqual(items["llm_provider_id"]["_special"], "select_provider")
        self.assertEqual(items["enable_llm_fallback"]["type"], "bool")
        self.assertIs(items["enable_llm_fallback"]["default"], False)

        steam_key = items["steam_api_key"]
        self.assertEqual(steam_key["type"], "string")
        self.assertEqual(steam_key["default"], "")
        self.assertIn("GetOwnedGames", steam_key["hint"])
        self.assertIn("/randomrec", steam_key["hint"])
        self.assertIn("https://steamcommunity.com/dev/apikey", steam_key["hint"])
        self.assertNotIn("/unplayedrec", steam_key["hint"])

    def test_price_notice_precedes_default_region(self) -> None:
        items = group_items(load_schema(), "price_and_region")
        notice = items["steam_price_heybox_notice"]

        self.assertEqual(notice["type"], "text")
        self.assertIs(notice["_readonly"], True)
        self.assertIs(notice["obvious_hint"], True)
        self.assertIn("无需配置", notice["default"])
        self.assertIn("自动启用", notice["default"])
        self.assertIn("未安装", notice["default"])
        self.assertEqual(items["default_region"]["default"], "CN")

    def test_positive_weights_are_adjustable_relative_percentages(self) -> None:
        items = group_items(load_schema(), "recommendation_and_scoring")
        expected = {
            "tag_coverage_weight": 35.0,
            "positive_reference_weight": 25.0,
            "library_profile_weight": 5.0,
            "review_reputation_weight": 20.0,
            "popularity_weight": 15.0,
        }

        for key, default in expected.items():
            with self.subTest(key=key):
                setting = items[key]
                self.assertEqual(setting["type"], "float")
                self.assertEqual(setting["default"], default)
                self.assertEqual(setting["min"], 0)
                self.assertEqual(setting["max"], 100)
                self.assertIn("相对权重", setting["hint"])
                self.assertIn("无需", setting["hint"])

    def test_recommendation_thresholds_keep_current_command_copy(self) -> None:
        items = group_items(load_schema(), "recommendation_and_scoring")

        self.assertEqual(items["steam_index_ttl_hours"]["default"], 168)
        self.assertEqual(items["steam_min_review_count"]["default"], 50)
        self.assertEqual(items["steam_min_positive_ratio"]["default"], 0.65)
        review_hint = items["steam_min_review_count"]["hint"]
        ratio_setting = items["steam_min_positive_ratio"]
        self.assertIn("/randomrec", review_hint)
        self.assertIn("随机推荐", ratio_setting["description"])
        self.assertIn("/randomrec", ratio_setting["hint"])
        self.assertNotIn("/unplayedrec", review_hint + ratio_setting["hint"])


if __name__ == "__main__":
    unittest.main()
