from __future__ import annotations

import unittest

from astrbot_plugin_steam_game_recommender.services.recommendation_intent import (
    ReferencePolarity,
    ReferenceQuery,
)
from astrbot_plugin_steam_game_recommender.services.reference_matching import (
    match_reference_query,
    title_base_key,
)
from astrbot_plugin_steam_game_recommender.storage.models import SteamSearchHit


class ReferenceMatchingTest(unittest.TestCase):
    def test_groups_localized_titles_by_appid_and_prefers_base_edition(self) -> None:
        reference = ReferenceQuery(
            "黑暗之魂",
            ("黑暗之魂", "Dark Souls"),
            ReferencePolarity.POSITIVE,
        )
        hits = [
            SteamSearchHit(appid=1, title="黑暗之魂：重制版"),
            SteamSearchHit(appid=1, title="DARK SOULS™: REMASTERED"),
            SteamSearchHit(appid=2, title="DARK SOULS™ III"),
        ]

        match = match_reference_query(reference, hits)

        self.assertIsNotNone(match)
        assert match is not None
        self.assertEqual(match.hit.appid, 1)
        self.assertEqual(match.matched_alias, "Dark Souls")
        self.assertEqual(match.match_kind, "base")
        self.assertEqual(match.confidence, 0.96)

    def test_exact_match_has_highest_priority(self) -> None:
        reference = ReferenceQuery("Portal", ("Portal",), ReferencePolarity.POSITIVE)

        match = match_reference_query(
            reference,
            [
                SteamSearchHit(appid=1, title="Portal Remastered"),
                SteamSearchHit(appid=2, title="Portal"),
            ],
        )

        self.assertIsNotNone(match)
        assert match is not None
        self.assertEqual(match.hit.appid, 2)
        self.assertEqual(match.match_kind, "exact")

    def test_tied_best_appids_are_ambiguous(self) -> None:
        reference = ReferenceQuery("Shared Name", ("Shared Name",), ReferencePolarity.POSITIVE)

        match = match_reference_query(
            reference,
            [
                SteamSearchHit(appid=1, title="Shared Name"),
                SteamSearchHit(appid=2, title="Shared Name"),
            ],
        )

        self.assertIsNone(match)

    def test_fuzzy_match_requires_threshold_and_margin(self) -> None:
        reference = ReferenceQuery(
            "Galactic Frontier",
            ("Galactic Frontier",),
            ReferencePolarity.POSITIVE,
        )

        confident = match_reference_query(
            reference,
            [
                SteamSearchHit(appid=1, title="Galactic Frontiers"),
                SteamSearchHit(appid=2, title="Galactic Farmer"),
            ],
        )
        ambiguous = match_reference_query(
            ReferenceQuery("Space Quest", ("Space Quest",), ReferencePolarity.POSITIVE),
            [
                SteamSearchHit(appid=3, title="Space Quests"),
                SteamSearchHit(appid=4, title="Space Quest X"),
            ],
        )
        weak = match_reference_query(
            reference,
            [SteamSearchHit(appid=5, title="Unrelated Adventure")],
        )

        self.assertIsNotNone(confident)
        assert confident is not None
        self.assertEqual(confident.match_kind, "fuzzy")
        self.assertIsNone(ambiguous)
        self.assertIsNone(weak)

    def test_edition_stripping_never_removes_sequel_numbers(self) -> None:
        self.assertEqual(title_base_key("Saga 2 Remastered"), "saga2")
        self.assertEqual(title_base_key("Saga III Definitive Edition"), "sagaiii")
        self.assertNotEqual(title_base_key("Saga 2"), title_base_key("Saga 3"))


if __name__ == "__main__":
    unittest.main()
