from __future__ import annotations

import sys
import types
import unittest
from pathlib import Path

api_module = types.ModuleType("astrbot.api")
api_module.logger = types.SimpleNamespace(
    debug=lambda *_args, **_kwargs: None,
    warning=lambda *_args, **_kwargs: None,
)
event_module = types.ModuleType("astrbot.api.event")
event_module.AstrMessageEvent = object
star_module = types.ModuleType("astrbot.api.star")
star_module.Context = object
sys.modules.setdefault("astrbot", types.ModuleType("astrbot"))
sys.modules.setdefault("astrbot.api", api_module)
sys.modules.setdefault("astrbot.api.event", event_module)
sys.modules.setdefault("astrbot.api.star", star_module)

from astrbot_plugin_steam_game_recommender.services.preference_parser import (  # noqa: E402
    parse_preference_json,
)
from astrbot_plugin_steam_game_recommender.services.similarity_ranker import (  # noqa: E402
    build_profile_from_preference,
)
from astrbot_plugin_steam_game_recommender.services.steam_index import (  # noqa: E402
    STEAM_ONLY_SCOPE_WARNING,
    steam_only_scope_warning_for,
)
from astrbot_plugin_steam_game_recommender.storage.models import GamePreference  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]


class SteamOnlyMetadataTest(unittest.TestCase):
    def test_plugin_id_display_name_and_version_are_0_7_0(self) -> None:
        main_text = (ROOT / "main.py").read_text(encoding="utf-8")
        metadata_text = (ROOT / "metadata.yaml").read_text(encoding="utf-8")

        self.assertIn('PLUGIN_NAME = "astrbot_plugin_steam_game_recommender"', main_text)
        self.assertIn('PLUGIN_VERSION = "0.7.0"', main_text)
        self.assertIn("class SteamGameRecommenderPlugin", main_text)
        self.assertIn("name: astrbot_plugin_steam_game_recommender", metadata_text)
        self.assertIn("display_name: Steam 游戏推荐助手", metadata_text)
        self.assertIn("version: 0.7.0", metadata_text)

    def test_readme_documents_only_current_steam_interfaces(self) -> None:
        readme = (ROOT / "README.md").read_text(encoding="utf-8")

        for command in (
            "/gamerec",
            "/gamerec_retry",
            "/accountbind",
            "/accountunbind",
            "/randomrec",
        ):
            self.assertIn(command, readme)
        for alias in (
            "/游戏推荐",
            "/重新推荐",
            "/换一批",
            "/账号绑定",
            "/解除绑定",
            "/随机推荐",
        ):
            self.assertIn(alias, readme)
        self.assertNotIn("/unplayedrec", readme)
        self.assertNotIn("/未玩推荐", readme)
        self.assertIn("-US", readme)
        self.assertIn("推荐分：86/100", readme)
        self.assertIn("推荐理由：", readme)
        self.assertIn("不推荐理由：", readme)
        self.assertIn("DLC、试玩版、原声、工具和套餐", readme)
        self.assertIn("默认关闭相同查询结果复用", readme)
        self.assertIn("开启后，相同需求通常会更快返回", readme)
        self.assertIn("无结果灵感推荐默认关闭", readme)
        self.assertIn("不会附带购买链接、价格、评测或推荐分", readme)

    def test_readme_is_a_user_facing_overview_with_combined_examples(self) -> None:
        readme = (ROOT / "README.md").read_text(encoding="utf-8")

        headings = (
            "## 安装与配置",
            "## 开始使用",
            "## 怎样提出需求",
            "## 怎样理解结果",
            "## 账号、游戏库与价格",
            "## 错误、降级与空结果",
            "## 数据边界",
        )
        positions = [readme.index(heading) for heading in headings]
        self.assertEqual(positions, sorted(positions))
        self.assertIn(
            "/gamerec [排除已有|仅查看已有] [区域] <自然语言需求>",
            readme,
        )
        for example in (
            "/gamerec 排除已有 -US 双人合作解谜，预算 30 美元",
            "/gamerec 类似黑暗之魂的游戏",
            "/gamerec 推荐 3 款支持简体中文、预算 100 元以内的合作游戏",
            "/gamerec 模拟类游戏，希望有丰富的长期游玩内容，最好由 ConcernedApe 开发",
            "/gamerec 即将发售的类魂游戏",
        ):
            self.assertIn(example, readme)
        for user_visible_behavior in (
            "新安装默认最多返回 10 款",
            "用户原文中明确写出的数量优先",
            "升级前已经设置为 5 款",
            "参考游戏本身不会作为结果返回",
            "续作和同系列作品仍可参与推荐",
            "### 更细致的玩法要求",
            "有些需求无法直接对应固定的 Steam 分类",
            "Steam 标签、类型和商店介绍",
            "明确不符合或无法确认的游戏会被移除",
            "检查服务暂时异常时，游戏会保留并标注未确认",
            "资料不足的游戏只会在结果不足时补位",
            "即使写了“仅限”或“只要”",
            "不会用完全没有核心玩法直接证据的候选凑数",
            "部分核心特征证据不足的候选仍可能返回",
            "不会把默认排序值显示成玩家评价",
            "普通推荐不需要 Steam Web API Key",
            "[Steam Web API Key 申请页](https://steamcommunity.com/dev/apikey)",
            "Steam 价格查询（小黑盒）",
            "astrbot_plugin_steam_price_heybox",
            "偏好解析模型暂时不可用",
            "具体玩法暂时无法确认",
            "暂时没有找到满足当前条件的游戏",
            "只会在 Steam 查询正常但没有合适结果时出现",
            "显示样式取决于当前聊天平台",
            "OneBot v11 通常显示为合并转发",
            "QQ 官方、Telegram、Discord",
            "随机抽样最多 50 款",
            "约 20 秒",
            "近期推荐记录会保留 30 分钟",
            "解绑时会删除当前平台实例的绑定",
        ):
            self.assertIn(user_visible_behavior, readme)

        self.assertIn(
            "推荐理由：合作解谜玩法符合你的需求，Steam 玩家评价也较稳定。",
            readme,
        )
        self.assertIn(
            "不推荐理由：暂时无法确认是否支持简体中文。",
            readme,
        )
        for removed_example in ("节日活动", "结婚", "昼夜循环"):
            self.assertNotIn(removed_example, readme)

        for implementation_detail in (
            "语义分 =",
            "层内分 =",
            "Wilson 好评率下界",
            "z=1.96",
            "RRF",
            "evidence_ids",
            "schema_version",
            "source_kind",
            "quality_source",
            "company_adjustment",
            "llm_fallback_provider_id",
            "steam_index_ttl_hours",
            "cache_ttl_hours",
            "AppDetails",
            "More Like This",
            "发现召回",
            "语义核验",
            "响应契约",
            "每 20 款",
            "最多核验 60 款",
            "60/100 未发售质量先验",
        ):
            self.assertNotIn(implementation_detail, readme)


class SteamOnlyPreferenceTest(unittest.TestCase):
    def test_llm_json_accepts_extra_tags_and_reference_titles(self) -> None:
        preference = parse_preference_json(
            """
            {
              "platforms": ["steam"],
              "genres_like": ["co-op"],
              "extra_tags": ["轻松", "解谜", "本地合作"],
              "genres_dislike": ["恐怖"],
              "reference_games_like": ["双人成行"],
              "reference_search_terms": ["It Takes Two"],
              "library_filter_mode": "only_owned",
              "players": 2,
              "result_count": 5
            }
            """
        )

        self.assertEqual(preference.extra_tags, ["轻松", "解谜", "本地合作"])
        self.assertEqual(preference.reference_games_like, ["双人成行"])
        self.assertEqual(preference.reference_search_terms, ["It Takes Two"])
        self.assertEqual(preference.library_filter_mode, "only_owned")

        profile = build_profile_from_preference(preference)

        self.assertIn("co_op", profile.include_tags)
        self.assertNotIn("local_coop", profile.include_tags)
        self.assertNotIn("puzzle", profile.include_tags)
        self.assertNotIn("relaxing", profile.include_tags)
        self.assertIn("horror", profile.exclude_tags)

    def test_non_steam_platforms_are_reported_as_out_of_scope(self) -> None:
        warning = steam_only_scope_warning_for(
            GamePreference(platforms=["nintendo switch", "steam"])
        )

        self.assertEqual(warning, STEAM_ONLY_SCOPE_WARNING)


if __name__ == "__main__":
    unittest.main()
