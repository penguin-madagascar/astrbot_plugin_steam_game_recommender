from __future__ import annotations

# ruff: noqa: E402, I001

import unittest
from types import SimpleNamespace
from typing import Any
from unittest.mock import patch

try:
    _astrbot_stubs = __import__("tests.test_prepare_recommendation")
except ModuleNotFoundError:
    _astrbot_stubs = __import__("test_prepare_recommendation")

from astrbot_plugin_steam_game_recommender import main as main_module
from astrbot_plugin_steam_game_recommender.main import (
    PreparedRecommendation,
    SteamGameRecommenderPlugin,
)
from astrbot_plugin_steam_game_recommender.services.ranking_precedence import (
    effective_score,
)
from astrbot_plugin_steam_game_recommender.services.steam_price_bridge import (
    SteamPriceBridge,
)
from astrbot_plugin_steam_game_recommender.storage.models import (
    GamePreference,
    GamePriceSummary,
    RankedGame,
    ScoreBreakdown,
)


class PipelineEvent:
    unified_msg_origin = "fixture:test"


class StaticPressureIndex:
    def __init__(self, games: list[RankedGame]) -> None:
        self.games = games
        self.requested_limits: list[tuple[int, int | None]] = []

    async def recommend(self, _preference, *, limit: int, requested_limit=None, **_kwargs):
        self.requested_limits.append((limit, requested_limit))
        return list(self.games[:limit])


class RecordingPriceBridge(SteamPriceBridge):
    def __init__(self, summaries: dict[str, GamePriceSummary]) -> None:
        super().__init__(
            client=object(),
            config={"default_region": "CN"},
            service_factory=lambda _config, _client: object(),
        )
        self.summaries = summaries
        self.lookup_calls: list[str] = []

    async def lookup(
        self,
        title: str,
        country: str | None = None,
    ) -> GamePriceSummary | None:
        del country
        self.lookup_calls.append(title)
        return self.summaries[title]


def pressure_games() -> list[RankedGame]:
    games: list[RankedGame] = []
    for offset in range(60):
        appid = offset + 1
        if offset < 20:
            tier = "A"
            verification = "verified"
            layer_score = 0.80 - offset * 0.001
        elif offset < 40:
            tier = "A"
            verification = "technical_failure"
            layer_score = 0.99 - (offset - 20) * 0.001
        else:
            tier = "B"
            verification = "verified"
            layer_score = 0.99 - (offset - 40) * 0.001
        games.append(
            RankedGame(
                appid=appid,
                title=f"Pool Game {appid}",
                app_type="game",
                platforms=["PC"],
                stores=["Steam"],
                score=round(layer_score * 100),
                core_feature_verification=verification,
                score_breakdown=ScoreBreakdown(
                    relevance_tier=tier,
                    layer_score=layer_score,
                    retrieval_rank=appid,
                ),
            )
        )
    return games


def price_summary(current_amount: float, historic_low_amount: float) -> GamePriceSummary:
    return GamePriceSummary(
        region="CN",
        currency="CNY",
        current_price=f"¥{current_amount:g}",
        current_amount=current_amount,
        historic_low=f"¥{historic_low_amount:g}",
        historic_low_amount=historic_low_amount,
    )


def pressure_plugin(
    games: list[RankedGame],
) -> tuple[SteamGameRecommenderPlugin, StaticPressureIndex, RecordingPriceBridge]:
    index = StaticPressureIndex(games)
    summaries = {
        game.title: (
            price_summary(40, 30)
            if game.appid == 20 or int(game.appid or 0) > 20
            else price_summary(120, 110)
        )
        for game in games
    }
    price_bridge = RecordingPriceBridge(summaries)
    plugin = object.__new__(SteamGameRecommenderPlugin)
    plugin.context = object()
    plugin.provider_id = ""
    plugin.fallback_provider_id = ""
    plugin.semantic_verification_batch_size = 10
    plugin.cache = object()
    plugin.steam_client = SimpleNamespace(language="schinese")
    plugin.steam_index = index
    plugin.price_bridge = price_bridge

    async def no_owned_games(_event, required):
        del required
        return []

    async def no_profile(_event, _owned_games):
        return {}

    plugin._owned_games_for_recommendation = no_owned_games
    plugin._user_profile_tag_weights = no_profile
    return plugin, index, price_bridge


class CandidatePoolPressureTest(unittest.IsolatedAsyncioTestCase):
    async def test_budget_prices_all_sixty_without_breaking_precedence(self) -> None:
        games = pressure_games()
        plugin, index, price_bridge = pressure_plugin(games)
        preference = GamePreference(
            platforms=["steam"],
            budget=50,
            budget_currency="CNY",
            region="CN",
        )

        async def identity_reasons(_context, _event, _provider_id, ranked_games):
            return list(ranked_games)

        with patch.object(
            main_module,
            "generate_recommendation_reasons",
            identity_reasons,
        ):
            run = await plugin._run_recommendation(
                PipelineEvent(),
                PreparedRecommendation("budget pressure", preference, 10),
            )

        self.assertEqual(index.requested_limits, [(60, 10)])
        self.assertEqual(len(price_bridge.lookup_calls), 60)
        self.assertEqual(set(price_bridge.lookup_calls), {game.title for game in games})
        self.assertEqual(len(run.ranked_games), 10)
        self.assertIn(20, [game.appid for game in run.ranked_games])
        self.assertTrue(
            all(
                game.score_breakdown.relevance_tier == "A"
                and game.core_feature_verification == "verified"
                for game in run.ranked_games
            )
        )
        self.assertTrue(
            all(
                game.score
                == round(
                    effective_score(
                        game.score_breakdown,
                        fallback_score=game.score,
                    )
                )
                for game in run.ranked_games
            )
        )

    async def test_no_budget_prices_only_the_initial_display_window(self) -> None:
        games = pressure_games()
        plugin, index, price_bridge = pressure_plugin(games)
        initial_window = games[:10]

        async def identity_reasons(_context, _event, _provider_id, ranked_games):
            return list(ranked_games)

        with patch.object(
            main_module,
            "generate_recommendation_reasons",
            identity_reasons,
        ):
            run = await plugin._run_recommendation(
                PipelineEvent(),
                PreparedRecommendation(
                    "no budget pressure",
                    GamePreference(platforms=["steam"]),
                    10,
                ),
            )

        self.assertEqual(index.requested_limits, [(60, 10)])
        self.assertEqual(
            price_bridge.lookup_calls,
            [game.title for game in initial_window],
        )
        self.assertEqual(
            [game.appid for game in run.ranked_games],
            [game.appid for game in initial_window],
        )
        self.assertEqual(len(run.ranked_games), 10)


if __name__ == "__main__":
    unittest.main()
