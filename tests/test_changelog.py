from __future__ import annotations

import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


class ChangelogTest(unittest.TestCase):
    def test_changelog_tracks_current_release_and_version_history(self) -> None:
        changelog = ROOT / "CHANGELOG.md"

        self.assertTrue(changelog.exists())
        text = changelog.read_text(encoding="utf-8")
        self.assertIn("# 更新日志", text)
        self.assertNotIn("## 未发布", text)
        self.assertIn("## 0.7.0 - 2026-07-17", text)
        self.assertIn("## 0.6.1 - 2026-07-13", text)
        self.assertIn("`/randomrec`", text)
        self.assertIn("`/随机推荐`", text)
        self.assertIn("README", text)
        self.assertIn("用户指南", text)
        release = text.split("## 0.7.0 - 2026-07-17", 1)[1].split(
            "## 0.6.1 - 2026-07-13",
            1,
        )[0]
        self.assertLessEqual(release.count("\n- "), 3)
        self.assertIn("DLC、试玩版", release)
        self.assertIn("版本", release)
        self.assertIn("无结果灵感推荐", release)
        self.assertIn("准确的 Steam AppID", release)
        self.assertIn("Steam Web API Key", release)
        self.assertIn("`/accountunbind`", release)
        self.assertIn("30 分钟", release)
        self.assertIn("QQ 官方、Telegram 和 Discord", release)
        self.assertIn("`/randomrec` 最多检查 50 款", release)
        self.assertIn("## 0.6.0", text)
        self.assertIn("astrbot_plugin_steam_game_recommender", text)
        self.assertIn("0–100 推荐分", text)
        self.assertIn("Steam 口碑", text)
        for internal_term in (
            "Steam 类型确认",
            "显式专用模型",
            "安全迁移",
            "对数知名度",
            "贝叶斯口碑",
            "并发",
            "Wilson",
            "响应契约",
        ):
            self.assertNotIn(internal_term, text)
        for english_phrase in (
            "# Changelog",
            "## Unreleased",
            "Improved keyword",
            "Strengthened filtering",
            "Added Steam price",
            "Fixed recommendation",
        ):
            self.assertNotIn(english_phrase, text)


if __name__ == "__main__":
    unittest.main()
