from __future__ import annotations

import unittest

from astrbot_plugin_game_recommender.services.played_filter import (
    filter_played_games,
    wants_played_game_exclusion,
)
from astrbot_plugin_game_recommender.storage.models import RankedGame, SteamOwnedGame


class PlayedFilterTriggerTest(unittest.TestCase):
    def test_detects_explicit_played_exclusion_phrases(self) -> None:
        self.assertTrue(wants_played_game_exclusion("推荐几个合作游戏，排除已玩"))
        self.assertTrue(wants_played_game_exclusion("Steam 解谜，过滤玩过的"))
        self.assertTrue(wants_played_game_exclusion("别推荐已玩游戏"))

    def test_does_not_trigger_without_explicit_phrase(self) -> None:
        self.assertFalse(wants_played_game_exclusion("推荐几个 Steam 合作游戏"))
        self.assertFalse(wants_played_game_exclusion("想找类似已玩过的双人成行"))


class PlayedFilterTest(unittest.TestCase):
    def test_filters_only_owned_games_with_positive_playtime(self) -> None:
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

        filtered, removed_count = filter_played_games(games, owned_games)

        self.assertEqual(
            [game.title for game in filtered],
            ["Unplayed Owned Game", "Unknown Appid Game", "Not Owned Game"],
        )
        self.assertEqual(removed_count, 1)


if __name__ == "__main__":
    unittest.main()
