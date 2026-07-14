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

        for command in ("/gamerec", "/gamerec_retry", "/accountbind", "/randomrec"):
            self.assertIn(command, readme)
        self.assertNotIn("/unplayedrec", readme)
        self.assertNotIn("/未玩推荐", readme)
        self.assertIn("-US", readme)
        self.assertIn("推荐分：86%", readme)
        self.assertIn("搜索条目必须是 `app`", readme)
        self.assertIn("详情类型必须是 `game`", readme)
        self.assertIn("DLC、Demo、原声、工具和套餐", readme)
        self.assertIn("同一作品只保留一款", readme)
        self.assertIn("`llm_fallback_provider_id`", readme)
        self.assertIn("⚠️ LLM 兜底建议", readme)

    def test_readme_is_a_user_facing_overview_with_combined_examples(self) -> None:
        readme = (ROOT / "README.md").read_text(encoding="utf-8")

        headings = (
            "## 项目定位",
            "## 核心能力",
            "## 快速开始",
            "## 推荐结果",
            "## 安装与配置",
            "## 使用边界",
        )
        positions = [readme.index(heading) for heading in headings]
        self.assertEqual(positions, sorted(positions))
        self.assertIn(
            "/gamerec [排除已有|仅查看已有] [区域] <自然语言需求>",
            readme,
        )
        for example in (
            "/gamerec 排除已有 -US 双人合作解谜，预算 30 美元",
            "/游戏推荐 仅查看已有 日区 适合周末通关的剧情游戏，预算 3000 日元",
            "/gamerec 排除已有 国区 支持简体中文的轻松经营游戏，预算 100 元",
        ):
            self.assertIn(example, readme)
        self.assertIn("游戏库过滤参数必须位于需求开头", readme)
        for scoring_detail in (
            "标签覆盖 | 35%",
            "正向参考 | 25%",
            "游戏库画像 | 5%",
            "口碑 | 20%",
            "知名度 | 15%",
            "两项都不可用 | 50%",
            "数据完整度不再作为独立评分项",
            "上限为 20 分",
            "上限为 15 分",
            "普通语言偏好不支持时为 -5",
            "强制措辞而不支持时为 -10",
            "当前价不高于预算时为 +5",
            "普通预算为 -5",
            "强制措辞的预算为 -10",
        ):
            self.assertIn(scoring_detail, readme)

        for config_detail in (
            "模型与鉴权",
            "价格与地区",
            "推荐与评分",
            "缓存与网络",
            "https://steamcommunity.com/dev/apikey",
            "五项正向权重",
            "无需手工凑满 100",
            "旧版平铺配置会在首次加载时自动迁移",
        ):
            self.assertIn(config_detail, readme)

        for implementation_detail in (
            "## 评分规则",
            "## LLM 行为",
            "## 开发验证",
            "min(log10(",
            "evidence_ids",
            "最多 5 路并发",
            "steam_index_ttl_hours",
            "cache_ttl_hours",
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
        self.assertIn("local_coop", profile.include_tags)
        self.assertIn("puzzle", profile.include_tags)
        self.assertIn("relaxing", profile.include_tags)
        self.assertIn("horror", profile.exclude_tags)

    def test_non_steam_platforms_are_reported_as_out_of_scope(self) -> None:
        warning = steam_only_scope_warning_for(
            GamePreference(platforms=["nintendo switch", "steam"])
        )

        self.assertEqual(warning, STEAM_ONLY_SCOPE_WARNING)


if __name__ == "__main__":
    unittest.main()
