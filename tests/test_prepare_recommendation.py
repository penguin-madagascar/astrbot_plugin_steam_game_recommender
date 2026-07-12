from __future__ import annotations

import sys
import types
import unittest

api_module = types.ModuleType("astrbot.api")
api_module.AstrBotConfig = dict
api_module.logger = types.SimpleNamespace(
    debug=lambda *_args, **_kwargs: None,
    exception=lambda *_args, **_kwargs: None,
    info=lambda *_args, **_kwargs: None,
    warning=lambda *_args, **_kwargs: None,
)
event_module = types.ModuleType("astrbot.api.event")
event_module.AstrMessageEvent = object
event_module.filter = types.SimpleNamespace(command=lambda *_args, **_kwargs: lambda func: func)
star_module = types.ModuleType("astrbot.api.star")
star_module.Context = object
star_module.Star = object
star_module.StarTools = types.SimpleNamespace(get_data_dir=lambda _name: "/tmp")
star_module.register = lambda *_args, **_kwargs: lambda cls: cls
command_module = types.ModuleType("astrbot.core.star.filter.command")
command_module.GreedyStr = str
sys.modules.setdefault("astrbot", types.ModuleType("astrbot"))
api_stub = sys.modules.setdefault("astrbot.api", api_module)
api_stub.AstrBotConfig = getattr(api_stub, "AstrBotConfig", dict)
api_stub.logger = getattr(api_stub, "logger", api_module.logger)
event_stub = sys.modules.setdefault("astrbot.api.event", event_module)
event_stub.AstrMessageEvent = getattr(event_stub, "AstrMessageEvent", object)
event_stub.filter = getattr(event_stub, "filter", event_module.filter)
star_stub = sys.modules.setdefault("astrbot.api.star", star_module)
star_stub.Context = getattr(star_stub, "Context", object)
star_stub.Star = getattr(star_stub, "Star", object)
star_stub.StarTools = getattr(star_stub, "StarTools", star_module.StarTools)
star_stub.register = getattr(star_stub, "register", star_module.register)
sys.modules.setdefault("astrbot.core", types.ModuleType("astrbot.core"))
sys.modules.setdefault("astrbot.core.star", types.ModuleType("astrbot.core.star"))
sys.modules.setdefault("astrbot.core.star.filter", types.ModuleType("astrbot.core.star.filter"))
command_stub = sys.modules.setdefault("astrbot.core.star.filter.command", command_module)
command_stub.GreedyStr = getattr(command_stub, "GreedyStr", str)

try:
    from astrbot_plugin_game_recommender.main import GameRecommenderPlugin, PreparedRecommendation
    from astrbot_plugin_game_recommender.services.played_filter import LibraryFilterModeError
    from astrbot_plugin_game_recommender.services.steam_index import STEAM_ONLY_SCOPE_WARNING
    from astrbot_plugin_game_recommender.storage.models import (
        GameCandidate,
        GamePreference,
        RankedGame,
        SteamAccountBinding,
        SteamOwnedGame,
    )
except ModuleNotFoundError as exc:
    if exc.name in {"astrbot", "pydantic"}:
        raise unittest.SkipTest(f"{exc.name} is not installed in this environment") from exc
    raise


class FakePreferenceParser:
    def __init__(self, preference: GamePreference) -> None:
        self.preference = preference
        self.seen_text = ""

    async def parse_preference(self, _event, text: str) -> GamePreference:
        self.seen_text = text
        return self.preference


class PrepareRecommendationLlmFallbackTest(unittest.IsolatedAsyncioTestCase):
    async def test_prepare_rejects_non_steam_platform_when_llm_fallback_is_disabled(self) -> None:
        preference = GamePreference(
            platforms=["nintendo switch"],
            genres_like=["party"],
            result_count=3,
        )
        plugin = object.__new__(GameRecommenderPlugin)
        plugin.max_results = 5
        plugin.enable_llm_fallback = False
        plugin.preference_parser = FakePreferenceParser(preference)

        with self.assertRaises(LibraryFilterModeError) as raised:
            await plugin._prepare_recommendation(object(), "Switch 聚会游戏")

        self.assertIn("仅支持 Steam", str(raised.exception))

    async def test_prepare_allows_non_steam_platform_when_llm_fallback_is_enabled(self) -> None:
        preference = GamePreference(
            platforms=["nintendo switch"],
            genres_like=["party"],
            result_count=3,
        )
        plugin = object.__new__(GameRecommenderPlugin)
        plugin.max_results = 5
        plugin.enable_llm_fallback = True
        plugin.preference_parser = FakePreferenceParser(preference)

        prepared = await plugin._prepare_recommendation(object(), "Switch 聚会游戏")

        self.assertEqual(prepared.preference.platforms, ["nintendo switch"])
        self.assertEqual(prepared.result_limit, 3)
        self.assertIn(STEAM_ONLY_SCOPE_WARNING, prepared.preference.parse_warnings)

    async def test_run_uses_llm_fallback_without_calling_steam_index_for_non_steam_only(
        self,
    ) -> None:
        plugin = object.__new__(GameRecommenderPlugin)
        plugin.enable_llm_fallback = True
        plugin.provider_id = "provider-1"
        plugin.context = FakeLlmContext(
            "LLM 兜底建议（未经过 Steam 索引验证）\n1. 《Mario Kart 8 Deluxe》：适合轻松多人竞速。"
        )
        plugin.steam_index = RaisingSteamIndex()
        plugin.price_bridge = RaisingPriceBridge()

        async def fake_profile(_event):
            raise AssertionError("pure non-Steam fallback must not load Steam profile")

        plugin._user_profile_tag_weights = fake_profile
        prepared = PreparedRecommendation(
            raw_query="Switch 聚会游戏",
            preference=GamePreference(
                platforms=["nintendo switch"],
                genres_like=["party"],
                parse_warnings=[STEAM_ONLY_SCOPE_WARNING],
                result_count=2,
            ),
            result_limit=2,
        )

        run = await plugin._run_recommendation(FakeEvent(), prepared)

        self.assertEqual(run.ranked_games, [])
        self.assertEqual(len(plugin.context.calls), 1)
        self.assertIn("LLM 兜底建议（未经过 Steam 索引验证）", run.messages[0])
        self.assertIn("Mario Kart 8 Deluxe", run.messages[0])


class RecommendationPipelineTest(unittest.IsolatedAsyncioTestCase):
    async def test_library_snapshot_is_loaded_once_for_profile_and_filtering(self) -> None:
        plugin = object.__new__(GameRecommenderPlugin)
        plugin.enable_llm_fallback = False
        plugin.provider_id = ""
        plugin.context = FakeLlmContext("")
        plugin.cache = BoundAccountCache()
        plugin.steam_client = CountingOwnedGamesClient()
        plugin.steam_index = OwnedAwareSteamIndex()
        plugin.price_bridge = IdentityPriceBridge()
        prepared = PreparedRecommendation(
            raw_query="排除已有的合作游戏",
            preference=GamePreference(
                platforms=["steam"],
                genres_like=["co-op"],
                library_filter_mode="exclude_owned",
                result_count=2,
            ),
            result_limit=2,
        )

        run = await plugin._run_recommendation(FakeEvent(), prepared)

        self.assertEqual(plugin.steam_client.owned_calls, 1)
        self.assertEqual([game.title for game in run.ranked_games], ["Fresh Game"])


class FakeEvent:
    unified_msg_origin = "qq:test"
    sender_id = "test"
    platform = "qq"


class FakeLlmResponse:
    def __init__(self, text: str) -> None:
        self.completion_text = text


class FakeLlmContext:
    def __init__(self, response: str) -> None:
        self.response = response
        self.calls: list[dict] = []

    async def llm_generate(self, **kwargs):
        self.calls.append(kwargs)
        return FakeLlmResponse(self.response)


class RaisingSteamIndex:
    async def recommend(self, **_kwargs):
        raise AssertionError("pure non-Steam fallback must not call Steam index")


class RaisingPriceBridge:
    def is_available(self) -> bool:
        raise AssertionError("pure non-Steam fallback must not inspect price bridge")

    async def enrich_ranked_games(self, *_args, **_kwargs):
        raise AssertionError("pure non-Steam fallback must not enrich prices")


class IdentityPriceBridge:
    def is_available(self) -> bool:
        return False

    async def enrich_ranked_games(self, games, _preference):
        return games


class BoundAccountCache:
    async def get_steam_account_binding(self, _platform, _user_id):
        return SteamAccountBinding(
            chat_user_id="test",
            steam_id64="76561198000000000",
            account_kind="steamid64",
            display_value="76561198000000000",
        )


class CountingOwnedGamesClient:
    def __init__(self) -> None:
        self.owned_calls = 0

    def has_web_api_key(self) -> bool:
        return True

    async def get_owned_games(self, _account_id):
        self.owned_calls += 1
        return [SteamOwnedGame(appid=1, name="Owned Game", playtime_forever=120)]


class OwnedAwareSteamIndex:
    async def load_entries(self):
        return [GameCandidate(appid=1, title="Owned Game", tags=["Co-op"])]

    async def recommend(self, _preference, **_kwargs):
        return [
            RankedGame(appid=1, title="Owned Game", score=90),
            RankedGame(appid=2, title="Fresh Game", score=80),
        ]


if __name__ == "__main__":
    unittest.main()
