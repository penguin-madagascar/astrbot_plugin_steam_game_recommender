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
