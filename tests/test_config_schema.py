from __future__ import annotations

import json
import re
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


def sentence_count(text: str) -> int:
    stripped = text.strip()
    if not stripped:
        return 0
    return max(len(re.findall(r"[。！？!?]+", stripped)), 1)


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
            "自然语言理解与推荐理由模型",
        )
        self.assertTrue(items["llm_provider_id"]["hint"].startswith("留空时"))
        self.assertEqual(items["llm_provider_id"]["_special"], "select_provider")
        fallback = items["llm_fallback_provider_id"]
        self.assertEqual(fallback["type"], "string")
        self.assertEqual(fallback["_special"], "select_provider")
        self.assertEqual(fallback["default"], "")
        self.assertIn("留空", fallback["hint"])
        self.assertIn("关闭", fallback["hint"])
        self.assertIn("Steam 查询正常", fallback["hint"])
        self.assertIn("未经需求验证", fallback["hint"])
        self.assertIn("不显示价格、推荐分或购买链接", fallback["hint"])
        self.assertIn("不能用于“换一批”", fallback["hint"])
        batch_size = items.get("semantic_verification_batch_size")
        self.assertIsNotNone(batch_size)
        self.assertEqual(batch_size["type"], "int")
        self.assertEqual(batch_size["default"], 5)
        self.assertEqual(batch_size["description"], "具体玩法核对数量")
        self.assertIn("1", batch_size["hint"])
        self.assertIn("10", batch_size["hint"])
        steam_key = items["steam_api_key"]
        self.assertEqual(steam_key["type"], "string")
        self.assertEqual(steam_key["default"], "")
        self.assertTrue(steam_key["hint"].startswith("⚠️"))
        self.assertIn("普通推荐无需填写", steam_key["hint"])
        self.assertIn("/randomrec", steam_key["hint"])
        self.assertIn(
            "[Steam Web API Key 申请页](https://steamcommunity.com/dev/apikey)",
            steam_key["hint"],
        )
        self.assertNotIn("/unplayedrec", steam_key["hint"])

    def test_price_notice_precedes_default_region(self) -> None:
        items = group_items(load_schema(), "price_and_region")
        notice = items["steam_price_heybox_notice"]

        self.assertEqual(notice["type"], "text")
        self.assertIs(notice["_readonly"], True)
        self.assertIs(notice["obvious_hint"], True)
        self.assertIn("无需配置", notice["hint"])
        self.assertIn("Steam 价格查询（小黑盒）", notice["hint"])
        self.assertIn("astrbot_plugin_steam_price_heybox", notice["hint"])
        self.assertIn("自动显示", notice["default"])
        self.assertIn("未安装", notice["default"])
        self.assertEqual(items["default_region"]["default"], "CN")

    def test_recommendation_settings_describe_user_visible_results(
        self,
    ) -> None:
        schema = load_schema()
        group = schema["recommendation_and_scoring"]
        items = group["items"]

        self.assertEqual(items["steam_index_ttl_hours"]["default"], 168)
        self.assertEqual(items["max_results"]["default"], 10)
        self.assertEqual(items["steam_min_review_count"]["default"], 50)
        self.assertEqual(items["steam_min_positive_ratio"]["default"], 0.65)
        self.assertIn("默认返回数量", group["hint"])
        self.assertIn("/randomrec", group["hint"])
        self.assertNotIn("Wilson", group["hint"])
        review_hint = items["steam_min_review_count"]["hint"]
        self.assertEqual(
            items["steam_min_review_count"]["description"],
            "随机推荐最低评测数",
        )
        ratio_setting = items["steam_min_positive_ratio"]
        self.assertIn("/randomrec", review_hint)
        self.assertIn("随机推荐", ratio_setting["description"])
        self.assertIn("/randomrec", ratio_setting["hint"])
        self.assertIn("65%", ratio_setting["hint"])
        self.assertNotIn("/gamerec", review_hint + ratio_setting["hint"])
        self.assertNotIn("/unplayedrec", review_hint + ratio_setting["hint"])

    def test_identical_query_cache_reuse_is_opt_in_and_outcome_focused(self) -> None:
        items = group_items(load_schema(), "cache_and_network")
        setting = items.get("reuse_identical_query_cache")

        self.assertIsNotNone(setting)
        self.assertEqual(setting["type"], "bool")
        self.assertIs(setting["default"], False)
        self.assertIn("完全相同", setting["description"])
        hint = setting["hint"]
        self.assertIn("响应更快", hint)
        self.assertIn("请求更少", hint)
        self.assertIn("重新寻找和核对候选", hint)
        self.assertIn("资料更新更慢", items["cache_ttl_hours"]["hint"])

    def test_dashboard_copy_is_user_facing_and_concise(self) -> None:
        schema = load_schema()
        visible_copy: list[str] = []

        for group in schema.values():
            visible_copy.extend([group["description"], group["hint"]])
            self.assertLessEqual(sentence_count(group["hint"]), 2)
            for setting in group["items"].values():
                copy = [setting["description"], setting["hint"]]
                if setting["type"] == "text" and isinstance(setting["default"], str):
                    copy.append(setting["default"])
                visible_copy.extend(copy)
                self.assertLessEqual(sentence_count("".join(copy[1:])), 2)

        combined = "\n".join(visible_copy)
        for internal_term in (
            "Wilson",
            "AppDetails",
            "GetOwnedGames",
            "发现召回",
            "语义核验",
            "响应契约",
        ):
            with self.subTest(internal_term=internal_term):
                self.assertNotIn(internal_term, combined)


if __name__ == "__main__":
    unittest.main()
