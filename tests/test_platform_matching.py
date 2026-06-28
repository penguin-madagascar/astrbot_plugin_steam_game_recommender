from __future__ import annotations

import unittest

from astrbot_plugin_game_recommender.services.platforms import (
    candidate_matches_platform,
    is_switch2_only,
    platform_families_for,
)
from astrbot_plugin_game_recommender.storage.models import GameCandidate


class PlatformMatchingTest(unittest.TestCase):
    def test_steam_request_requires_steam_store_or_url_evidence(self) -> None:
        pc_only = GameCandidate(title="PC Game", platforms=["PC"])
        steam_store = GameCandidate(title="Steam Game", platforms=["PC"], stores=["Steam"])
        steam_url = GameCandidate(
            title="Steam URL Game",
            platforms=["PC"],
            raw_url="https://store.steampowered.com/app/123",
        )

        self.assertFalse(candidate_matches_platform(pc_only, "steam"))
        self.assertTrue(candidate_matches_platform(steam_store, "steam"))
        self.assertTrue(candidate_matches_platform(steam_url, "steam"))

    def test_pc_request_accepts_pc_platform_or_steam_store_evidence(self) -> None:
        pc_only = GameCandidate(title="PC Game", platforms=["PC"])
        steam_only = GameCandidate(title="Steam Game", stores=["Steam"])

        self.assertTrue(candidate_matches_platform(pc_only, "pc"))
        self.assertTrue(candidate_matches_platform(steam_only, "pc"))

    def test_attached_steam_candidate_contributes_platform_families(self) -> None:
        rawg_candidate = GameCandidate(title="Switch Game", platforms=["Nintendo Switch"])
        steam_candidate = GameCandidate(title="Switch Game", platforms=["PC"], stores=["Steam"])

        families = platform_families_for(rawg_candidate, steam_candidate)

        self.assertEqual(families, ["steam", "pc", "nintendo switch"])

    def test_switch_2_only_keeps_family_match_but_is_detectable(self) -> None:
        self.assertTrue(is_switch2_only(["Nintendo Switch 2"]))
        self.assertTrue(is_switch2_only(["Switch 2"]))
        self.assertFalse(is_switch2_only(["Nintendo Switch", "Nintendo Switch 2"]))


if __name__ == "__main__":
    unittest.main()
