from __future__ import annotations

import asyncio
import random
import unittest

from astrbot_plugin_steam_game_recommender.clients.steam import SteamTransientError
from astrbot_plugin_steam_game_recommender.storage.models import (
    GameCandidate,
    SteamOwnedGame,
)


class UnplayedPickerTest(unittest.IsolatedAsyncioTestCase):
    def test_numeric_helpers_reject_booleans_fractions_and_non_finite_values(self) -> None:
        from astrbot_plugin_steam_game_recommender.services.unplayed_picker import (
            optional_float,
            optional_int,
            review_passes,
        )

        for value in (True, False, 1.5, float("nan"), float("inf"), "1.0"):
            with self.subTest(helper="int", value=value):
                self.assertIsNone(optional_int(value))
        for value in (True, False, float("nan"), float("inf"), float("-inf")):
            with self.subTest(helper="float", value=value):
                self.assertIsNone(optional_float(value))

        self.assertFalse(review_passes(FakeReview(True, 0.9), 0, 0.65))
        self.assertFalse(review_passes(FakeReview(100, True), 0, 0.65))
        self.assertFalse(review_passes(FakeReview(100, float("inf")), 0, 0.65))
        self.assertFalse(review_passes(FakeReview(100, 1.01), 0, 0.65))

    async def test_randomly_picks_unplayed_game_that_meets_review_floor(self) -> None:
        from astrbot_plugin_steam_game_recommender.services.unplayed_picker import (
            format_unplayed_recommendation,
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
        message = format_unplayed_recommendation(
            result,
            "它主打冒险玩法。较高好评率和充足评测量说明口碑与知名度较稳。",
        )
        self.assertEqual(
            message,
            "《Random Winner》\n它主打冒险玩法。较高好评率和充足评测量说明口碑与知名度较稳。",
        )
        self.assertNotIn("推荐分", message)
        self.assertNotIn("价格", message)
        self.assertNotIn("http", message)

    async def test_skips_played_and_low_review_games(self) -> None:
        from astrbot_plugin_steam_game_recommender.services.unplayed_picker import (
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
        from astrbot_plugin_steam_game_recommender.services.unplayed_picker import (
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

    async def test_skips_non_game_details_before_returning_random_pick(self) -> None:
        from astrbot_plugin_steam_game_recommender.services.unplayed_picker import (
            pick_random_unplayed_game,
        )

        client = FakeSteamClient(
            reviews={
                5: FakeReview(total_reviews=500, positive_ratio=0.9),
                6: FakeReview(total_reviews=500, positive_ratio=0.9),
            },
            app_types={5: "dlc", 6: "game"},
        )

        result = await pick_random_unplayed_game(
            [
                SteamOwnedGame(appid=5, name="Expansion", playtime_forever=0),
                SteamOwnedGame(appid=6, name="Base Game", playtime_forever=0),
            ],
            client,
            rng=NoShuffleRandom(),
        )

        self.assertEqual(result.game.appid, 6)
        self.assertEqual(client.detail_appids, [5, 6])

    async def test_compresses_owned_editions_before_checking_random_candidates(self) -> None:
        from astrbot_plugin_steam_game_recommender.services.unplayed_picker import (
            pick_random_unplayed_game,
        )

        client = FakeSteamClient(
            reviews={
                7: FakeReview(total_reviews=5, positive_ratio=0.9),
                9: FakeReview(total_reviews=500, positive_ratio=0.9),
            }
        )

        result = await pick_random_unplayed_game(
            [
                SteamOwnedGame(appid=7, name="Control", playtime_forever=0),
                SteamOwnedGame(
                    appid=8,
                    name="Control Ultimate Edition",
                    playtime_forever=0,
                ),
                SteamOwnedGame(appid=9, name="Portal 2", playtime_forever=0),
            ],
            client,
            rng=NoShuffleRandom(),
        )

        self.assertEqual(result.game.appid, 9)
        self.assertNotIn(8, client.review_appids)
        self.assertNotIn(8, client.detail_appids)

    async def test_any_played_edition_marks_the_whole_family_as_played(self) -> None:
        from astrbot_plugin_steam_game_recommender.services.unplayed_picker import (
            pick_random_unplayed_game,
        )

        family = [
            SteamOwnedGame(appid=7, name="Control", playtime_forever=0),
            SteamOwnedGame(
                appid=8,
                name="Control Ultimate Edition",
                playtime_forever=120,
            ),
        ]
        for owned_games in (
            [*family, SteamOwnedGame(appid=9, name="Portal 2", playtime_forever=0)],
            [
                *reversed(family),
                SteamOwnedGame(appid=9, name="Portal 2", playtime_forever=0),
            ],
        ):
            with self.subTest(owned_appids=[game.appid for game in owned_games]):
                client = FakeSteamClient(
                    reviews={9: FakeReview(total_reviews=500, positive_ratio=0.9)}
                )

                result = await pick_random_unplayed_game(
                    owned_games,
                    client,
                    rng=NoShuffleRandom(),
                )

                self.assertEqual(result.game.appid, 9)
                self.assertEqual(client.review_appids, [9])

    async def test_detail_failure_skips_candidate_and_checks_the_next_game(self) -> None:
        from astrbot_plugin_steam_game_recommender.services.unplayed_picker import (
            pick_random_unplayed_game,
        )

        client = FakeSteamClient(
            reviews={
                10: FakeReview(total_reviews=500, positive_ratio=0.9),
                9: FakeReview(total_reviews=500, positive_ratio=0.9),
            },
            detail_failures={10},
        )

        result = await pick_random_unplayed_game(
            [
                SteamOwnedGame(appid=10, name="Unavailable Game", playtime_forever=0),
                SteamOwnedGame(appid=9, name="Portal 2", playtime_forever=0),
            ],
            client,
            rng=NoShuffleRandom(),
        )

        self.assertEqual(result.game.appid, 9)
        self.assertEqual(client.detail_appids, [10, 9])

    async def test_review_failure_skips_candidate_and_checks_the_next_game(self) -> None:
        from astrbot_plugin_steam_game_recommender.services.unplayed_picker import (
            pick_random_unplayed_game,
        )

        client = FakeSteamClient(
            reviews={9: FakeReview(total_reviews=500, positive_ratio=0.9)},
            review_failures={10},
        )

        result = await pick_random_unplayed_game(
            [
                SteamOwnedGame(appid=10, name="Unavailable Review", playtime_forever=0),
                SteamOwnedGame(appid=9, name="Portal 2", playtime_forever=0),
            ],
            client,
            rng=NoShuffleRandom(),
        )

        self.assertEqual(result.game.appid, 9)
        self.assertEqual(client.review_appids, [10, 9])

    async def test_all_review_failures_report_service_unavailable(self) -> None:
        from astrbot_plugin_steam_game_recommender.services.unplayed_picker import (
            UnplayedRecommendationError,
            pick_random_unplayed_game,
        )

        client = FakeSteamClient(
            reviews={},
            review_failures={9, 10},
        )

        with self.assertRaisesRegex(UnplayedRecommendationError, "评测服务暂不可用"):
            await pick_random_unplayed_game(
                [
                    SteamOwnedGame(appid=10, name="Unavailable A", playtime_forever=0),
                    SteamOwnedGame(appid=9, name="Unavailable B", playtime_forever=0),
                ],
                client,
                rng=NoShuffleRandom(),
            )

    async def test_programming_error_is_not_silently_skipped(self) -> None:
        from astrbot_plugin_steam_game_recommender.services.unplayed_picker import (
            pick_random_unplayed_game,
        )

        client = ProgrammingErrorReviewClient()

        with self.assertRaisesRegex(RuntimeError, "decoder bug"):
            await pick_random_unplayed_game(
                [SteamOwnedGame(appid=1, name="Broken", playtime_forever=0)],
                client,
                rng=NoShuffleRandom(),
            )

    async def test_ready_result_does_not_hide_peer_programming_error(self) -> None:
        from astrbot_plugin_steam_game_recommender.services.unplayed_picker import (
            pick_random_unplayed_game,
        )

        client = ReadyAndProgrammingErrorClient()

        with self.assertRaisesRegex(RuntimeError, "decoder bug in peer"):
            await pick_random_unplayed_game(
                [
                    SteamOwnedGame(appid=1, name="Ready", playtime_forever=0),
                    SteamOwnedGame(appid=2, name="Broken", playtime_forever=0),
                ],
                client,
                rng=NoShuffleRandom(),
            )

    async def test_samples_at_most_fifty_games_from_a_large_library(self) -> None:
        from astrbot_plugin_steam_game_recommender.services.unplayed_picker import (
            UnplayedRecommendationError,
            pick_random_unplayed_game,
        )

        owned_games = [
            SteamOwnedGame(appid=appid, name=f"App {appid}", playtime_forever=0)
            for appid in range(1, 1001)
        ]
        client = FakeSteamClient(
            reviews={
                game.appid: FakeReview(total_reviews=1, positive_ratio=0.9)
                for game in owned_games
            }
        )

        with self.assertRaises(UnplayedRecommendationError):
            await pick_random_unplayed_game(
                owned_games,
                client,
                rng=NoShuffleRandom(),
            )

        self.assertEqual(len(client.review_appids), 50)
        self.assertEqual(client.review_appids, list(range(1, 51)))

    async def test_checks_reviews_with_at_most_five_concurrent_requests(self) -> None:
        from astrbot_plugin_steam_game_recommender.services.unplayed_picker import (
            UnplayedRecommendationError,
            pick_random_unplayed_game,
        )

        client = ConcurrentReviewClient()
        owned_games = [
            SteamOwnedGame(appid=appid, name=f"App {appid}", playtime_forever=0)
            for appid in range(1, 11)
        ]

        with self.assertRaises(UnplayedRecommendationError):
            await pick_random_unplayed_game(
                owned_games,
                client,
                rng=NoShuffleRandom(),
            )

        self.assertEqual(client.max_active_reviews, 5)

    async def test_stops_the_scan_at_the_total_deadline(self) -> None:
        from astrbot_plugin_steam_game_recommender.services.unplayed_picker import (
            UnplayedRecommendationError,
            pick_random_unplayed_game,
        )

        client = SlowReviewClient()

        with self.assertRaisesRegex(UnplayedRecommendationError, "超时"):
            await pick_random_unplayed_game(
                [SteamOwnedGame(appid=1, name="Slow Game", playtime_forever=0)],
                client,
                rng=NoShuffleRandom(),
                timeout_seconds=0.01,
            )

    async def test_returns_completed_match_without_waiting_for_slow_batch_peer(
        self,
    ) -> None:
        from astrbot_plugin_steam_game_recommender.services.unplayed_picker import (
            pick_random_unplayed_game,
        )

        result = await pick_random_unplayed_game(
            [
                SteamOwnedGame(appid=1, name="Ready Game", playtime_forever=0),
                SteamOwnedGame(appid=2, name="Slow Game", playtime_forever=0),
            ],
            ReadyThenSlowClient(),
            rng=NoShuffleRandom(),
            concurrency=2,
            timeout_seconds=0.05,
        )

        self.assertEqual(result.game.appid, 1)

    async def test_non_finite_picker_options_use_safe_defaults(self) -> None:
        from astrbot_plugin_steam_game_recommender.services.unplayed_picker import (
            pick_random_unplayed_game,
        )

        client = FakeSteamClient(
            reviews={1: FakeReview(total_reviews=500, positive_ratio=0.9)}
        )

        result = await pick_random_unplayed_game(
            [SteamOwnedGame(appid=1, name="Good Game", playtime_forever=0)],
            client,
            min_review_count=float("nan"),
            min_positive_ratio=float("nan"),
            rng=NoShuffleRandom(),
            sample_limit=float("nan"),
            concurrency=float("inf"),
            timeout_seconds=float("nan"),
        )

        self.assertEqual(result.game.appid, 1)


class FakeReview:
    def __init__(self, total_reviews: int, positive_ratio: float | None) -> None:
        self.total_reviews = total_reviews
        self.positive_ratio = positive_ratio
        self.recent_positive_ratio = positive_ratio


class FakeSteamClient:
    def __init__(
        self,
        reviews: dict[int, FakeReview],
        app_types: dict[int, str | None] | None = None,
        detail_failures: set[int] | None = None,
        review_failures: set[int] | None = None,
    ) -> None:
        self.reviews = reviews
        self.app_types = app_types or {}
        self.detail_failures = detail_failures or set()
        self.review_failures = review_failures or set()
        self.review_appids: list[int] = []
        self.detail_appids: list[int] = []

    async def get_review_summary(self, appid: int) -> FakeReview:
        self.review_appids.append(appid)
        if appid in self.review_failures:
            raise SteamTransientError("review unavailable")
        return self.reviews[appid]

    async def get_game_detail(self, appid: int) -> GameCandidate:
        self.detail_appids.append(appid)
        if appid in self.detail_failures:
            raise SteamTransientError("detail unavailable")
        return GameCandidate(
            title=game_title(appid),
            appid=appid,
            app_type=self.app_types.get(appid, "game"),
            platforms=["PC"],
            genres=["Adventure"],
            tags=["Single-player"],
            stores=["Steam"],
            raw_url=f"https://store.steampowered.com/app/{appid}/",
        )


class ConcurrentReviewClient:
    def __init__(self) -> None:
        self.active_reviews = 0
        self.max_active_reviews = 0

    async def get_review_summary(self, _appid: int) -> FakeReview:
        self.active_reviews += 1
        self.max_active_reviews = max(self.max_active_reviews, self.active_reviews)
        await asyncio.sleep(0.001)
        self.active_reviews -= 1
        return FakeReview(total_reviews=1, positive_ratio=0.9)

    async def get_game_detail(self, _appid: int) -> GameCandidate:
        raise AssertionError("low-review games must not load details")


class SlowReviewClient:
    async def get_review_summary(self, _appid: int) -> FakeReview:
        await asyncio.sleep(1)
        return FakeReview(total_reviews=1, positive_ratio=0.9)

    async def get_game_detail(self, _appid: int) -> GameCandidate:
        raise AssertionError("low-review games must not load details")


class ReadyThenSlowClient:
    async def get_review_summary(self, appid: int) -> FakeReview:
        if appid == 2:
            await asyncio.sleep(1)
        return FakeReview(total_reviews=500, positive_ratio=0.9)

    async def get_game_detail(self, appid: int) -> GameCandidate:
        return GameCandidate(
            title=game_title(appid),
            appid=appid,
            app_type="game",
            platforms=["PC"],
            stores=["Steam"],
        )


class ProgrammingErrorReviewClient:
    async def get_review_summary(self, _appid: int) -> FakeReview:
        raise RuntimeError("decoder bug")

    async def get_game_detail(self, _appid: int) -> GameCandidate:
        raise AssertionError("review failure must not load details")


class ReadyAndProgrammingErrorClient:
    async def get_review_summary(self, appid: int) -> FakeReview:
        if appid == 2:
            raise RuntimeError("decoder bug in peer")
        return FakeReview(total_reviews=500, positive_ratio=0.9)

    async def get_game_detail(self, appid: int) -> GameCandidate:
        return GameCandidate(
            title=game_title(appid),
            appid=appid,
            app_type="game",
            platforms=["PC"],
            stores=["Steam"],
        )


class NoShuffleRandom:
    def shuffle(self, values: list[SteamOwnedGame]) -> None:
        del values


def game_title(appid: int) -> str:
    names: dict[int, str] = {
        1: "First Candidate",
        2: "Random Winner",
        3: "Low Ratio",
        4: "Good Backlog",
        5: "Expansion",
        6: "Base Game",
        7: "Control",
        8: "Control Ultimate Edition",
        9: "Portal 2",
        10: "Unavailable Game",
    }
    return names.get(appid, f"App {appid}")


if __name__ == "__main__":
    unittest.main()
