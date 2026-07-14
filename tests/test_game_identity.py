from __future__ import annotations

import unittest

from astrbot_plugin_steam_game_recommender.services.game_identity import (
    deduplicate_game_editions,
    game_family_key,
    is_edition_title,
)
from astrbot_plugin_steam_game_recommender.services.similarity_ranker import (
    SteamTagProfile,
    rank_steam_candidates,
)
from astrbot_plugin_steam_game_recommender.services.steam_index import (
    exclude_previously_shown,
)
from astrbot_plugin_steam_game_recommender.storage.models import (
    GameCandidate,
    RankedGame,
    ScoreBreakdown,
)


class GameFamilyKeyTest(unittest.TestCase):
    def test_strips_explicit_english_and_chinese_edition_suffixes(self) -> None:
        cases = {
            "Control Ultimate Edition": "control",
            "Skyrim Special Edition": "skyrim",
            "Divinity Definitive Edition": "divinity",
            "Nioh Complete Edition": "nioh",
            "Forza Horizon Deluxe Edition": "forza horizon",
            "BioShock Remastered": "bioshock",
            "Death Stranding Director's Cut": "death stranding",
            "Ghost of Tsushima DIRECTOR’S CUT": "ghost of tsushima",
            "Metro Redux": "metro",
            "The Elder Scrolls V: Skyrim VR": "the elder scrolls v skyrim",
            "巫师 3：狂猎 完全版": "巫师 3 狂猎",
            "死亡搁浅 导演剪辑版": "死亡搁浅",
            "生化危机 4 重制版": "生化危机 4",
        }

        for title, expected in cases.items():
            with self.subTest(title=title):
                self.assertEqual(game_family_key(title), expected)
                self.assertTrue(is_edition_title(title))

    def test_preserves_sequel_numbers_roman_numerals_and_regular_subtitles(self) -> None:
        titles = [
            "Portal 2",
            "Final Fantasy VII",
            "The Witcher 3: Wild Hunt",
            "Life is Strange: True Colors",
            "Half-Life: Alyx",
        ]

        for title in titles:
            with self.subTest(title=title):
                self.assertFalse(is_edition_title(title))

        self.assertNotEqual(game_family_key("Portal"), game_family_key("Portal 2"))
        self.assertNotEqual(
            game_family_key("Final Fantasy VI"),
            game_family_key("Final Fantasy VII"),
        )


class EditionDeduplicationTest(unittest.TestCase):
    def test_never_replaces_a_higher_tier_edition_with_lower_tier_standard_game(self) -> None:
        games = [
            RankedGame(
                appid=2,
                title="Control Ultimate Edition",
                score=30,
                score_breakdown=ScoreBreakdown(
                    relevance_tier="A", layer_score=0.30, retrieval_rank=2
                ),
            ),
            RankedGame(
                appid=1,
                title="Control",
                score=99,
                score_breakdown=ScoreBreakdown(
                    relevance_tier="B", layer_score=0.99, retrieval_rank=1
                ),
            ),
        ]

        selected = deduplicate_game_editions(games)

        self.assertEqual([game.appid for game in selected], [2])

    def test_prefers_standard_game_even_when_an_edition_scores_higher(self) -> None:
        games = [
            ranked(2, "Control Ultimate Edition", 95),
            ranked(1, "Control", 70),
            ranked(3, "Portal 2", 80),
        ]

        selected = deduplicate_game_editions(games)

        self.assertEqual([game.appid for game in selected], [3, 1])

    def test_keeps_highest_scoring_edition_when_no_standard_game_exists(self) -> None:
        games = [
            ranked(1, "Skyrim Special Edition", 75),
            ranked(2, "Skyrim VR", 88),
        ]

        selected = deduplicate_game_editions(games)

        self.assertEqual([game.appid for game in selected], [2])

    def test_preferred_owned_appid_wins_and_keeps_its_own_score_order(self) -> None:
        games = [
            ranked(2, "Control Ultimate Edition", 95),
            ranked(3, "Portal 2", 90),
            ranked(1, "Control", 70),
        ]

        selected = deduplicate_game_editions(games, preferred_appids=[2])

        self.assertEqual([game.appid for game in selected], [2, 3])

    def test_retry_exclusion_removes_every_edition_in_shown_family(self) -> None:
        games = [
            ranked(1, "Control", 90),
            ranked(2, "Control Ultimate Edition", 85),
            ranked(3, "Portal 2", 80),
        ]

        filtered = exclude_previously_shown(
            games,
            excluded_appids=[],
            excluded_titles=["Control Ultimate Edition"],
        )

        self.assertEqual([game.title for game in filtered], ["Portal 2"])

    def test_resolved_reference_excludes_only_exact_seed_appid(self) -> None:
        ranked_games = rank_steam_candidates(
            [
                candidate(1, "Control"),
                candidate(2, "Control Ultimate Edition"),
                candidate(3, "Portal 2"),
            ],
            SteamTagProfile(reference_titles=["Control"], reference_appids=[1]),
            min_review_count=0,
        )

        self.assertEqual(
            {game.title for game in ranked_games},
            {"Control Ultimate Edition", "Portal 2"},
        )

    def test_localized_reference_keeps_other_family_appids_eligible(self) -> None:
        ranked_games = rank_steam_candidates(
            [
                candidate(2, "Control Ultimate Edition"),
                candidate(3, "Portal 2"),
            ],
            SteamTagProfile(
                reference_titles=["控制"],
                reference_appids=[1],
                positive_reference_candidates=[candidate(1, "Control")],
            ),
            min_review_count=0,
        )

        self.assertEqual(
            {game.title for game in ranked_games},
            {"Control Ultimate Edition", "Portal 2"},
        )


def ranked(appid: int, title: str, score: int) -> RankedGame:
    return RankedGame(
        appid=appid,
        title=title,
        app_type="game",
        tags=["Puzzle"],
        review_total=500,
        review_positive_ratio=0.8,
        score=score,
    )


def candidate(appid: int, title: str) -> GameCandidate:
    return GameCandidate(
        appid=appid,
        title=title,
        app_type="game",
        tags=["Puzzle"],
        review_total=500,
        review_positive_ratio=0.8,
    )


if __name__ == "__main__":
    unittest.main()
