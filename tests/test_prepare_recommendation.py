from __future__ import annotations

import sys
import types
import unittest
from unittest.mock import patch


class FakeStar:
    def __init__(self, context) -> None:
        self.context = context


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
star_module.Star = FakeStar
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
    from astrbot_plugin_steam_game_recommender import main as main_module
    from astrbot_plugin_steam_game_recommender.main import (
        PreparedRecommendation,
        SteamGameRecommenderPlugin,
    )
    from astrbot_plugin_steam_game_recommender.services.played_filter import LibraryFilterModeError
    from astrbot_plugin_steam_game_recommender.services.steam_index import STEAM_ONLY_SCOPE_WARNING
    from astrbot_plugin_steam_game_recommender.storage.models import (
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


class PluginDashboardConfigTest(unittest.TestCase):
    def test_nested_dashboard_config_is_wired_to_runtime_services(self) -> None:
        config = {
            "model_and_access": {
                "llm_provider_id": "provider/nested",
                "llm_fallback_provider_id": "provider/fallback",
                "steam_api_key": "nested-steam-key",
            },
            "price_and_region": {
                "default_region": "jp",
            },
            "recommendation_and_scoring": {
                "max_results": 8,
                "tag_coverage_weight": 45,
                "positive_reference_weight": 20,
                "library_profile_weight": 5,
                "review_reputation_weight": 20,
                "popularity_weight": 10,
                "steam_index_ttl_hours": 96,
                "steam_min_review_count": 80,
                "steam_min_positive_ratio": 0.72,
            },
            "cache_and_network": {
                "cache_ttl_hours": 48,
                "timeout_seconds": 21,
            },
            "llm_provider_id": "provider/legacy-flat",
            "default_region": "CN",
            "max_results": 2,
            "steam_api_key": "legacy-steam-key",
            "timeout_seconds": 1,
        }
        http_client = object()
        cache = object()

        with (
            patch.object(
                main_module.httpx,
                "AsyncClient",
                return_value=http_client,
            ) as async_client_class,
            patch.object(
                main_module,
                "SQLiteCacheRepository",
                return_value=cache,
            ),
            patch.object(main_module, "SteamClient") as steam_client_class,
            patch.object(main_module, "PreferenceParser"),
            patch.object(main_module, "SteamGameIndexService") as index_service_class,
            patch.object(main_module, "SteamPriceBridge") as price_bridge_class,
        ):
            price_bridge_class.return_value.is_available.return_value = False
            plugin = SteamGameRecommenderPlugin(object(), config)

        self.assertEqual(plugin.provider_id, "provider/nested")
        self.assertEqual(plugin.fallback_provider_id, "provider/fallback")
        self.assertEqual(plugin.default_region, "JP")
        self.assertEqual(plugin.max_results, 8)
        self.assertEqual(
            async_client_class.call_args.kwargs["timeout"].connect,
            21,
        )
        self.assertEqual(
            steam_client_class.call_args.kwargs,
            {
                "client": http_client,
                "cache": cache,
                "cache_ttl_hours": 48,
                "default_country": "JP",
                "language": "schinese",
                "steam_api_key": "nested-steam-key",
            },
        )
        self.assertEqual(index_service_class.call_args.kwargs["ttl_hours"], 96)
        self.assertEqual(
            set(index_service_class.call_args.kwargs),
            {"steam_client", "cache", "ttl_hours"},
        )
        price_bridge_class.assert_called_once_with(
            http_client,
            {"default_region": "JP"},
        )


class PrepareRecommendationLlmFallbackTest(unittest.IsolatedAsyncioTestCase):
    async def test_prepare_applies_query_region_and_region_local_budget_currency(self) -> None:
        preference = GamePreference(platforms=["steam"], budget=50, result_count=3)
        parser = FakePreferenceParser(preference)
        plugin = object.__new__(SteamGameRecommenderPlugin)
        plugin.max_results = 5
        plugin.default_region = "CN"
        plugin.fallback_provider_id = ""
        plugin.preference_parser = parser

        prepared = await plugin._prepare_recommendation(
            object(),
            "-US Steam 双人合作，预算 50",
        )

        self.assertEqual(parser.seen_text, "Steam 双人合作，预算 50")
        self.assertEqual(prepared.preference.region, "US")
        self.assertEqual(prepared.preference.budget_currency, "USD")

    async def test_prepare_rejects_non_steam_platform_without_fallback_provider(self) -> None:
        preference = GamePreference(
            platforms=["nintendo switch"],
            genres_like=["party"],
            result_count=3,
        )
        plugin = object.__new__(SteamGameRecommenderPlugin)
        plugin.max_results = 5
        plugin.fallback_provider_id = ""
        plugin.preference_parser = FakePreferenceParser(preference)

        with self.assertRaises(LibraryFilterModeError) as raised:
            await plugin._prepare_recommendation(object(), "Switch 聚会游戏")

        self.assertIn("仅支持 Steam", str(raised.exception))

    async def test_prepare_allows_non_steam_platform_with_explicit_fallback_provider(self) -> None:
        preference = GamePreference(
            platforms=["nintendo switch"],
            genres_like=["party"],
            result_count=3,
        )
        plugin = object.__new__(SteamGameRecommenderPlugin)
        plugin.max_results = 5
        plugin.fallback_provider_id = "provider/fallback"
        plugin.preference_parser = FakePreferenceParser(preference)

        prepared = await plugin._prepare_recommendation(object(), "Switch 聚会游戏")

        self.assertEqual(prepared.preference.platforms, ["nintendo switch"])
        self.assertEqual(prepared.result_limit, 3)
        self.assertIn(STEAM_ONLY_SCOPE_WARNING, prepared.preference.parse_warnings)

    async def test_run_uses_llm_fallback_without_calling_steam_index_for_non_steam_only(
        self,
    ) -> None:
        plugin = object.__new__(SteamGameRecommenderPlugin)
        plugin.fallback_provider_id = "provider/fallback"
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
        plugin = object.__new__(SteamGameRecommenderPlugin)
        plugin.fallback_provider_id = ""
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

    async def test_only_owned_passes_owned_appids_as_edition_preference(self) -> None:
        plugin = object.__new__(SteamGameRecommenderPlugin)
        plugin.fallback_provider_id = ""
        plugin.provider_id = ""
        plugin.context = FakeLlmContext("")
        plugin.cache = BoundAccountCache()
        plugin.steam_client = CountingOwnedGamesClient(
            [
                SteamOwnedGame(
                    appid=2,
                    name="Control Ultimate Edition",
                    playtime_forever=0,
                )
            ]
        )
        index = OwnedEditionSteamIndex()
        plugin.steam_index = index
        plugin.price_bridge = IdentityPriceBridge()
        prepared = PreparedRecommendation(
            raw_query="仅查看已有的动作游戏",
            preference=GamePreference(
                platforms=["steam"],
                genres_like=["action"],
                library_filter_mode="only_owned",
                result_count=2,
            ),
            result_limit=2,
        )

        run = await plugin._run_recommendation(FakeEvent(), prepared)

        self.assertEqual(index.preferred_appids, [2])
        self.assertEqual([game.appid for game in run.ranked_games], [2])

    async def test_final_output_guard_drops_unconfirmed_or_non_game_candidates(self) -> None:
        plugin = object.__new__(SteamGameRecommenderPlugin)
        plugin.fallback_provider_id = ""
        plugin.provider_id = ""
        plugin.context = FakeLlmContext("")
        plugin.cache = BoundAccountCache()
        plugin.steam_client = CountingOwnedGamesClient([])
        plugin.steam_index = TypeLeakingSteamIndex()
        plugin.price_bridge = IdentityPriceBridge()
        prepared = PreparedRecommendation(
            raw_query="Steam 动作游戏",
            preference=GamePreference(platforms=["steam"], genres_like=["action"]),
            result_limit=3,
        )

        run = await plugin._run_recommendation(FakeEvent(), prepared)

        self.assertEqual([game.title for game in run.ranked_games], ["Base Game"])


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
    def __init__(self, games=None) -> None:
        self.owned_calls = 0
        self.games = games or [
            SteamOwnedGame(appid=1, name="Owned Game", playtime_forever=120)
        ]

    def has_web_api_key(self) -> bool:
        return True

    async def get_owned_games(self, _account_id):
        self.owned_calls += 1
        return self.games


class OwnedAwareSteamIndex:
    async def load_entries(self):
        return [
            GameCandidate(
                appid=1,
                title="Owned Game",
                app_type="game",
                tags=["Co-op"],
            )
        ]

    async def recommend(self, _preference, **_kwargs):
        return [
            RankedGame(appid=1, title="Owned Game", app_type="game", score=90),
            RankedGame(appid=2, title="Fresh Game", app_type="game", score=80),
        ]


class OwnedEditionSteamIndex:
    def __init__(self) -> None:
        self.preferred_appids = None

    async def load_entries(self):
        return []

    async def recommend(self, _preference, **kwargs):
        self.preferred_appids = kwargs.get("preferred_appids")
        return [
            RankedGame(appid=1, title="Control", app_type="game", score=90),
            RankedGame(
                appid=2,
                title="Control Ultimate Edition",
                app_type="game",
                score=80,
            ),
        ]


class TypeLeakingSteamIndex:
    async def load_entries(self):
        return []

    async def recommend(self, _preference, **_kwargs):
        return [
            RankedGame(appid=1, title="Expansion", app_type="dlc", score=95),
            RankedGame(appid=2, title="Unknown", score=90),
            RankedGame(appid=3, title="Base Game", app_type="game", score=85),
        ]


if __name__ == "__main__":
    unittest.main()
