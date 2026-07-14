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
        self.assertIn("## 0.7.0 - 2026-07-14", text)
        self.assertIn("## 0.6.1 - 2026-07-13", text)
        self.assertIn("`/randomrec`", text)
        self.assertIn("`/随机推荐`", text)
        self.assertIn("README", text)
        self.assertIn("面向使用者", text)
        release = text.split("## 0.7.0 - 2026-07-14", 1)[1].split(
            "## 0.6.1 - 2026-07-13",
            1,
        )[0]
        self.assertLessEqual(release.count("\n- "), 3)
        self.assertIn("Steam 类型确认", release)
        self.assertIn("同作不同版本", release)
        self.assertIn("显式专用模型", release)
        self.assertIn("推荐评分与 Dashboard", release)
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
