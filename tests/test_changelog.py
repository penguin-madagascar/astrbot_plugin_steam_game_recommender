from __future__ import annotations

import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


class ChangelogTest(unittest.TestCase):
    def test_changelog_tracks_unreleased_and_current_version(self) -> None:
        changelog = ROOT / "CHANGELOG.md"

        self.assertTrue(changelog.exists())
        text = changelog.read_text(encoding="utf-8")
        self.assertIn("# 更新日志", text)
        self.assertIn("## 未发布", text)
        self.assertIn("## 0.6.1 - 2026-07-13", text)
        self.assertIn("`/randomrec`", text)
        self.assertIn("`/随机推荐`", text)
        self.assertIn("README", text)
        self.assertIn("面向使用者", text)
        self.assertIn("标签 35%", text)
        self.assertIn("正向参考 25%", text)
        self.assertIn("知名度 15%", text)
        self.assertIn("删除数据完整度评分", text)
        self.assertIn("普通与强要求两级", text)
        self.assertIn("四个分组", text)
        self.assertIn("自动迁移", text)
        self.assertIn("## 0.6.0", text)
        self.assertIn("astrbot_plugin_steam_game_recommender", text)
        self.assertIn("连续分数", text)
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
