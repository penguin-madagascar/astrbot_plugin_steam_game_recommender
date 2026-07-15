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
            [
                "llm_provider_id",
                "llm_fallback_provider_id",
                "semantic_verification_batch_size",
                "steam_api_key",
            ],
        )
        self.assertEqual(
            list(group_items(schema, "price_and_region")),
            ["steam_price_heybox_notice", "default_region"],
        )
        self.assertEqual(
            list(group_items(schema, "recommendation_and_scoring")),
            [
                "max_results",
                "steam_index_ttl_hours",
                "steam_min_review_count",
                "steam_min_positive_ratio",
            ],
        )
        self.assertEqual(
            list(group_items(schema, "cache_and_network")),
            ["cache_ttl_hours", "timeout_seconds", "reuse_identical_query_cache"],
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
        fallback = items["llm_fallback_provider_id"]
        self.assertEqual(fallback["type"], "string")
        self.assertEqual(fallback["_special"], "select_provider")
        self.assertEqual(fallback["default"], "")
        self.assertIn("留空", fallback["hint"])
        self.assertIn("关闭", fallback["hint"])
        self.assertIn("正常", fallback["hint"])
        self.assertIn("零结果", fallback["hint"])
        self.assertIn("未经过 Steam", fallback["hint"])
        self.assertIn("绝不自动使用当前会话模型", fallback["hint"])
        batch_size = items.get("semantic_verification_batch_size")
        self.assertIsNotNone(batch_size)
        self.assertEqual(batch_size["type"], "int")
        self.assertEqual(batch_size["default"], 5)
        self.assertIn("语义特征", batch_size["description"])
        self.assertIn("1", batch_size["hint"])
        self.assertIn("10", batch_size["hint"])
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

    def test_gamerec_uses_fixed_scoring_and_thresholds_only_apply_to_randomrec(
        self,
    ) -> None:
        schema = load_schema()
        group = schema["recommendation_and_scoring"]
        items = group["items"]

        self.assertEqual(items["steam_index_ttl_hours"]["default"], 168)
        self.assertEqual(items["max_results"]["default"], 10)
        self.assertEqual(items["steam_min_review_count"]["default"], 50)
        self.assertEqual(items["steam_min_positive_ratio"]["default"], 0.65)
        self.assertIn("/gamerec", group["hint"])
        self.assertIn("Wilson", group["hint"])
        review_hint = items["steam_min_review_count"]["hint"]
        ratio_setting = items["steam_min_positive_ratio"]
        self.assertIn("/randomrec", review_hint)
        self.assertIn("随机推荐", ratio_setting["description"])
        self.assertIn("/randomrec", ratio_setting["hint"])
        self.assertNotIn("/gamerec", review_hint + ratio_setting["hint"])
        self.assertNotIn("索引推荐", review_hint + ratio_setting["hint"])
        self.assertNotIn("/unplayedrec", review_hint + ratio_setting["hint"])

    def test_identical_query_cache_reuse_is_opt_in_and_preserves_other_caches(self) -> None:
        setting = group_items(load_schema(), "cache_and_network").get(
            "reuse_identical_query_cache"
        )

        self.assertIsNotNone(setting)
        self.assertEqual(setting["type"], "bool")
        self.assertIs(setting["default"], False)
        hint = setting["hint"]
        self.assertIn("Steam 发现召回", hint)
        self.assertIn("语义核验", hint)
        self.assertIn("AppDetails", hint)
        self.assertIn("/gamerec_retry", hint)
        self.assertIn("7 天", hint)


if __name__ == "__main__":
    unittest.main()
