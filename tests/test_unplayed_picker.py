from __future__ import annotations

import random
import unittest
from typing import Any

from astrbot_plugin_game_recommender.storage.models import (
    GameCandidate,
    SteamOwnedGame,
)


class UnplayedPickerTest(unittest.IsolatedAsyncioTestCase):
    async def test_randomly_picks_unplayed_game_that_meets_review_floor(self) -> None:
        from astrbot_plugin_game_recommender.services.unplayed_picker import (
            pick_random_unplayed_game,
        )

        client = FakeSteamClient(
            reviews={
                1: FakeReview(total_reviews=2000, positive_ratio=0.90),
                2: FakeReview(total_reviews=3000, positive_ratio=0.92),
            }
        )

        result = await pick_random_unplayed_game(
            [
                SteamOwnedGame(appid=1, name="First Candidate", playtime_forever=0),
                SteamOwnedGame(appid=2, name="Random Winner", playtime_forever=0),
            ],
            client,
            min_review_count=50,
            min_positive_ratio=0.65,
            rng=random.Random(1),
        )

        self.assertEqual(result.game.appid, 2)
        self.assertEqual(result.game.title, "Random Winner")
        self.assertEqual(result.game.review_total, 3000)
        self.assertEqual(result.game.review_positive_ratio, 0.92)
        self.assertEqual(result.owned_game.playtime_forever, 0)

    async def test_skips_played_and_low_review_games(self) -> None:
        from astrbot_plugin_game_recommender.services.unplayed_picker import (
            pick_random_unplayed_game,
        )

        client = FakeSteamClient(
            reviews={
                1: FakeReview(total_reviews=5000, positive_ratio=0.95),
                2: FakeReview(total_reviews=10, positive_ratio=0.99),
                3: FakeReview(total_reviews=600, positive_ratio=0.40),
                4: FakeReview(total_reviews=800, positive_ratio=0.80),
            }
        )

        result = await pick_random_unplayed_game(
            [
                SteamOwnedGame(appid=1, name="Already Played", playtime_forever=120),
                SteamOwnedGame(appid=2, name="Too Few Reviews", playtime_forever=0),
                SteamOwnedGame(appid=3, name="Low Ratio", playtime_forever=0),
                SteamOwnedGame(appid=4, name="Good Backlog", playtime_forever=0),
            ],
            client,
            min_review_count=50,
            min_positive_ratio=0.65,
            rng=random.Random(1),
        )

        self.assertEqual(result.game.appid, 4)
        self.assertNotIn(1, client.detail_appids)

    async def test_raises_when_no_unplayed_game_passes_review_floor(self) -> None:
        from astrbot_plugin_game_recommender.services.unplayed_picker import (
            UnplayedRecommendationError,
            pick_random_unplayed_game,
        )

        client = FakeSteamClient(
            reviews={
                1: FakeReview(total_reviews=20, positive_ratio=0.95),
                2: FakeReview(total_reviews=500, positive_ratio=0.20),
            }
        )

        with self.assertRaisesRegex(UnplayedRecommendationError, "未游玩且评价过线"):
            await pick_random_unplayed_game(
                [
                    SteamOwnedGame(appid=1, name="Too Few Reviews", playtime_forever=0),
                    SteamOwnedGame(appid=2, name="Low Ratio", playtime_forever=0),
                ],
                client,
                min_review_count=50,
                min_positive_ratio=0.65,
                rng=random.Random(3),
            )


class FakeReview:
    def __init__(self, total_reviews: int, positive_ratio: float | None) -> None:
        self.total_reviews = total_reviews
        self.positive_ratio = positive_ratio
        self.recent_positive_ratio = positive_ratio


class FakeSteamClient:
    def __init__(self, reviews: dict[int, FakeReview]) -> None:
        self.reviews = reviews
        self.detail_appids: list[int] = []

    async def get_review_summary(self, appid: int) -> FakeReview:
        return self.reviews[appid]

    async def get_game_detail(self, appid: int) -> GameCandidate:
        self.detail_appids.append(appid)
        return GameCandidate(
            title=game_title(appid),
            appid=appid,
            platforms=["PC"],
            genres=["Adventure"],
            tags=["Single-player"],
            stores=["Steam"],
            raw_url=f"https://store.steampowered.com/app/{appid}/",
        )


def game_title(appid: int) -> str:
    names: dict[int, str] = {
        1: "First Candidate",
        2: "Random Winner",
        3: "Low Ratio",
        4: "Good Backlog",
    }
    return names.get(appid, f"App {appid}")


if __name__ == "__main__":
    unittest.main()
