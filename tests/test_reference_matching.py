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
        self.assertIn(match.matched_alias, {"黑暗之魂", "Dark Souls"})
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
                SteamSearchHit(appid=4, title="Space Guest"),
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
        self.assertEqual(title_base_key("Saga III Definitive Edition"), "saga3")
        self.assertEqual(title_base_key("Control Ultimate Edition"), "control")
        self.assertEqual(title_base_key("Example Director's Cut"), "example")
        self.assertNotEqual(title_base_key("Saga 2"), title_base_key("Saga 3"))

    def test_roman_and_arabic_sequel_numbers_are_equivalent(self) -> None:
        for query, observed in (
            ("Dark Souls 2", "DARK SOULS II"),
            ("Resident Evil 4", "Resident Evil IV"),
            ("Final Fantasy 7", "FINAL FANTASY VII"),
            ("Final Fantasy 16", "FINAL FANTASY XVI"),
        ):
            with self.subTest(query=query, observed=observed):
                reference = ReferenceQuery(
                    query,
                    (query,),
                    ReferencePolarity.POSITIVE,
                )

                match = match_reference_query(
                    reference,
                    [SteamSearchHit(appid=8, title=observed)],
                )

                self.assertIsNotNone(match)
                assert match is not None
                self.assertEqual(match.match_kind, "exact")

    def test_fuzzy_matching_never_turns_a_base_title_into_a_sequel(self) -> None:
        for base_title, sequel_title in (
            ("Portal", "Portal 2"),
            ("Half-Life", "Half-Life 2"),
            ("Dark Souls", "Dark Souls II"),
            ("Saga 2", "Saga 3"),
        ):
            with self.subTest(sequel_title=sequel_title):
                reference = ReferenceQuery(
                    base_title,
                    (base_title,),
                    ReferencePolarity.POSITIVE,
                )

                match = match_reference_query(
                    reference,
                    [SteamSearchHit(appid=9, title=sequel_title)],
                )

                self.assertIsNone(match)

    def test_complete_edition_beats_a_similarly_named_sequel(self) -> None:
        reference = ReferenceQuery(
            "Control",
            ("Control",),
            ReferencePolarity.POSITIVE,
        )

        match = match_reference_query(
            reference,
            [
                SteamSearchHit(appid=10, title="Control Ultimate Edition"),
                SteamSearchHit(appid=11, title="Control 2"),
            ],
        )

        self.assertIsNotNone(match)
        assert match is not None
        self.assertEqual(match.hit.appid, 10)
        self.assertEqual(match.match_kind, "base")

    def test_exact_matching_preserves_non_latin_unicode_titles(self) -> None:
        reference = ReferenceQuery(
            "ドラゴンクエスト",
            ("ドラゴンクエスト",),
            ReferencePolarity.POSITIVE,
        )

        match = match_reference_query(
            reference,
            [SteamSearchHit(appid=7, title="ドラゴンクエスト")],
        )

        self.assertIsNotNone(match)
        assert match is not None
        self.assertEqual(match.match_kind, "exact")


if __name__ == "__main__":
    unittest.main()
