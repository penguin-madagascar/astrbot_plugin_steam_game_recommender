from __future__ import annotations

import unittest

from astrbot_plugin_steam_game_recommender.services.played_filter import (
    LibraryFilterModeError,
    filter_games_by_library_mode,
    parse_library_filter_command,
    resolve_library_filter_mode,
)
from astrbot_plugin_steam_game_recommender.storage.models import RankedGame, SteamOwnedGame


class LibraryFilterCommandTest(unittest.TestCase):
    def test_parses_chinese_and_english_prefix_arguments(self) -> None:
        self.assertEqual(
            parse_library_filter_command("排除已有 Steam 合作解谜").mode,
            "exclude_owned",
        )
        self.assertEqual(
            parse_library_filter_command("exclude-owned Steam co-op puzzle").mode,
            "exclude_owned",
        )
        self.assertEqual(
            parse_library_filter_command("仅查看已有 Steam 合作解谜").mode,
            "only_owned",
        )
        self.assertEqual(
            parse_library_filter_command("only-owned Steam co-op puzzle").mode,
            "only_owned",
        )

    def test_strips_only_prefix_argument_from_query(self) -> None:
        parsed = parse_library_filter_command("排除已有 Steam 合作解谜")
        unchanged = parse_library_filter_command("Steam 合作解谜，排除已有的不要")

        self.assertEqual(parsed.query, "Steam 合作解谜")
        self.assertIsNone(unchanged.mode)
        self.assertEqual(unchanged.query, "Steam 合作解谜，排除已有的不要")

    def test_rejects_conflicting_command_or_inferred_modes(self) -> None:
        with self.assertRaises(LibraryFilterModeError):
            parse_library_filter_command("排除已有 仅查看已有 Steam 合作")

        with self.assertRaises(LibraryFilterModeError):
            resolve_library_filter_mode("exclude_owned", "only_owned")


class LibraryFilterTest(unittest.TestCase):
    def test_exclude_owned_filters_all_owned_games_regardless_playtime(self) -> None:
        games = [
            RankedGame(title="Played Game", appid=1, score=10),
            RankedGame(title="Unplayed Owned Game", appid=2, score=9),
            RankedGame(title="Unknown Appid Game", score=8),
            RankedGame(title="Not Owned Game", appid=3, score=7),
        ]
        owned_games = [
            SteamOwnedGame(appid=1, name="Played Game", playtime_forever=30),
            SteamOwnedGame(appid=2, name="Unplayed Owned Game", playtime_forever=0),
        ]

        filtered, removed_count = filter_games_by_library_mode(
            games,
            owned_games,
            "exclude_owned",
        )

        self.assertEqual(
            [game.title for game in filtered],
            ["Unknown Appid Game", "Not Owned Game"],
        )
        self.assertEqual(removed_count, 2)

    def test_only_owned_keeps_only_owned_games_with_appid(self) -> None:
        games = [
            RankedGame(title="Played Game", appid=1, score=10),
            RankedGame(title="Unknown Appid Game", score=8),
            RankedGame(title="Not Owned Game", appid=3, score=7),
        ]
        owned_games = [SteamOwnedGame(appid=1, name="Played Game", playtime_forever=0)]

        filtered, removed_count = filter_games_by_library_mode(
            games,
            owned_games,
            "only_owned",
        )

        self.assertEqual([game.title for game in filtered], ["Played Game"])
        self.assertEqual(removed_count, 2)


if __name__ == "__main__":
    unittest.main()
