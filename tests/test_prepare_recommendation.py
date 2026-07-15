from __future__ import annotations

import sys
import types
import unittest
import json
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch


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
        RecommendationRun,
        SteamGameRecommenderPlugin,
    )
    from astrbot_plugin_steam_game_recommender.services.played_filter import LibraryFilterModeError
    from astrbot_plugin_steam_game_recommender.services.preference_parser import ParseOutcome
    from astrbot_plugin_steam_game_recommender.services.recommendation_memory import (
        RecommendationMemory,
    )
    from astrbot_plugin_steam_game_recommender.services.run_notices import RunNotice
    from astrbot_plugin_steam_game_recommender.services.steam_index import STEAM_ONLY_SCOPE_WARNING
    from astrbot_plugin_steam_game_recommender.services.semantic_feature_verifier import verdict_cache_key
    from astrbot_plugin_steam_game_recommender.storage.models import (
        GameCandidate,
        GamePreference,
        RankedGame,
        ScoreBreakdown,
        SteamAccountBinding,
        SteamOwnedGame,
    )
except ModuleNotFoundError as exc:
    if exc.name in {"astrbot", "pydantic"}:
        raise unittest.SkipTest(f"{exc.name} is not installed in this environment") from exc
    raise


class FakePreferenceParser:
    def __init__(
        self,
        preference: GamePreference,
        notices: tuple[RunNotice, ...] = (),
    ) -> None:
        self.preference = preference
        self.notices = notices
        self.seen_text = ""

    async def parse_preference(self, _event, text: str) -> ParseOutcome:
        self.seen_text = text
        return ParseOutcome(self.preference, "llm", self.notices)


class PluginDashboardConfigTest(unittest.TestCase):
    def test_nested_dashboard_config_is_wired_to_runtime_services(self) -> None:
        config = {
            "model_and_access": {
                "llm_provider_id": "provider/nested",
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
        self.assertFalse(hasattr(plugin, "fallback_provider_id"))
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


class PrepareRecommendationTest(unittest.IsolatedAsyncioTestCase):
    async def test_prepare_applies_query_region_and_region_local_budget_currency(self) -> None:
        preference = GamePreference(platforms=["steam"], budget=50, result_count=3)
        parser = FakePreferenceParser(preference)
        plugin = object.__new__(SteamGameRecommenderPlugin)
        plugin.max_results = 5
        plugin.default_region = "CN"
        plugin.preference_parser = parser

        prepared = await plugin._prepare_recommendation(
            object(),
            "-US Steam 双人合作，预算 50",
        )

        self.assertEqual(parser.seen_text, "Steam 双人合作，预算 50")
        self.assertEqual(prepared.preference.region, "US")
        self.assertEqual(prepared.preference.budget_currency, "USD")

    async def test_prepare_keeps_non_steam_scope_warning_for_empty_explanation(self) -> None:
        preference = GamePreference(
            platforms=["nintendo switch"],
            genres_like=["party"],
            result_count=3,
        )
        plugin = object.__new__(SteamGameRecommenderPlugin)
        plugin.max_results = 5
        plugin.preference_parser = FakePreferenceParser(preference)

        prepared = await plugin._prepare_recommendation(object(), "Switch 聚会游戏")

        self.assertIn(STEAM_ONLY_SCOPE_WARNING, prepared.preference.parse_warnings)
        self.assertEqual(prepared.result_limit, 3)

    async def test_prepare_carries_parser_notice_without_putting_it_in_warnings(self) -> None:
        preference = GamePreference(
            platforms=["nintendo switch"],
            genres_like=["party"],
            result_count=3,
        )
        plugin = object.__new__(SteamGameRecommenderPlugin)
        plugin.max_results = 5
        notice = RunNotice("parser", "error", "偏好模型不可用")
        plugin.preference_parser = FakePreferenceParser(preference, (notice,))

        prepared = await plugin._prepare_recommendation(object(), "Switch 聚会游戏")

        self.assertEqual(prepared.run_notices, (notice,))
        self.assertNotIn("偏好模型不可用", prepared.preference.parse_warnings)
        self.assertIn(STEAM_ONLY_SCOPE_WARNING, prepared.preference.parse_warnings)

    async def test_non_steam_only_returns_verified_empty_explanation_without_llm(
        self,
    ) -> None:
        plugin = object.__new__(SteamGameRecommenderPlugin)
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
        self.assertEqual(plugin.context.calls, [])
        self.assertEqual(len(run.messages), 1)
        self.assertIn("暂时没有找到满足当前条件的游戏", run.messages[0])
        self.assertIn(STEAM_ONLY_SCOPE_WARNING, run.messages[0])

    async def test_retry_parser_notice_is_request_local_and_not_replayed(self) -> None:
        notice = RunNotice(
            "preference_parser_unavailable",
            "error",
            "偏好解析模型暂时不可用，已使用关键词规则继续处理。",
        )
        memory = RecommendationMemory(
            chat_platform="qq",
            chat_user_id="test",
            raw_query="合作游戏",
            preference=GamePreference(platforms=["steam"], result_count=5),
            result_limit=5,
            shown_appids=[],
            shown_titles=[],
            created_at=1,
        )
        plugin = object.__new__(SteamGameRecommenderPlugin)
        plugin.cache = object()
        prepared_runs: list[PreparedRecommendation] = []

        async def prepare(_event, text, default_region=None):
            self.assertEqual(text, "改成三款解谜游戏")
            self.assertIsNone(default_region)
            return PreparedRecommendation(
                raw_query=text,
                preference=GamePreference(
                    platforms=["steam"],
                    genres_like=["puzzle"],
                    result_count=3,
                ),
                result_limit=3,
                run_notices=(notice,),
            )

        async def execute(_event, prepared, **_kwargs):
            prepared_runs.append(prepared)
            return RecommendationRun(
                messages=[
                    *(item.text for item in prepared.run_notices),
                    f"结果数量：{prepared.result_limit}",
                ],
                ranked_games=[],
                preference=prepared.preference,
                result_limit=prepared.result_limit,
                raw_query=prepared.raw_query,
                run_notices=prepared.run_notices,
            )

        async def save(*_args, **_kwargs):
            return None

        plugin._prepare_recommendation = prepare
        plugin._run_recommendation = execute
        plugin._save_retry_memory = save

        with patch.object(
            main_module,
            "load_recommendation_memory",
            AsyncMock(return_value=memory),
        ):
            first_messages = await plugin._retry_recommendation_messages(
                FakeEvent(),
                "改成三款解谜游戏",
            )
            second_messages = await plugin._retry_recommendation_messages(FakeEvent())

        self.assertEqual(first_messages[0], notice.text)
        self.assertEqual(first_messages[-1], "结果数量：3")
        self.assertEqual(second_messages, ["结果数量：5"])
        self.assertEqual(prepared_runs[0].run_notices, (notice,))
        self.assertEqual(prepared_runs[1].run_notices, ())
        self.assertNotIn(notice.text, memory.preference.parse_warnings)


class RecommendationPipelineTest(unittest.IsolatedAsyncioTestCase):
    async def test_library_snapshot_is_loaded_once_for_profile_and_filtering(self) -> None:
        plugin = object.__new__(SteamGameRecommenderPlugin)
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


class SemanticFeatureMainPipelineTest(unittest.IsolatedAsyncioTestCase):
    async def test_current_session_provider_verifies_top_twenty_and_keys_cache(self) -> None:
        feature = {
            "constraint_id": "branching",
            "source_span": "必须有分支剧情",
            "normalized_text": "branching story",
            "role": "core",
            "polarity": "positive",
        }
        preference = GamePreference(
            platforms=["steam"],
            soft_features=[feature],
            result_count=5,
        )
        games = [semantic_ranked_game(appid) for appid in range(1, 22)]
        context = SemanticLlmContext(
            provider_id="provider/session-model",
            response={
                "verdicts": [
                    {
                        "appid": appid,
                        "constraint_id": "branching",
                        "polarity": "positive",
                        "status": "satisfied",
                        "evidence_quote": f"Game {appid}",
                    }
                    for appid in range(1, 21)
                ]
            },
        )
        cache = SemanticMemoryCache()
        plugin = semantic_pipeline_plugin(context, cache, games)
        prepared = PreparedRecommendation(
            raw_query="必须有分支剧情",
            preference=preference,
            result_limit=30,
        )

        async def identity_reasons(_context, _event, _provider_id, ranked_games):
            return ranked_games

        with (
            patch.object(main_module, "generate_recommendation_reasons", identity_reasons),
            patch.object(main_module.logger, "debug") as debug_log,
        ):
            run = await plugin._run_recommendation(FakeEvent(), prepared)

        self.assertEqual(context.provider_requests, ["qq:test"])
        self.assertEqual(len(context.calls), 1)
        payload = json.loads(context.calls[0]["prompt"].split("INPUT=", 1)[1])
        self.assertEqual(
            [candidate["appid"] for candidate in payload["candidates"]],
            list(range(1, 21)),
        )
        self.assertEqual([game.appid for game in run.ranked_games], list(range(1, 21)))
        self.assertEqual(run.run_notices, ())
        self.assertEqual(preference.parse_warnings, [])
        expected_keys = {
            verdict_cache_key(
                preference.soft_features[0],
                game,
                provider_id="provider/session-model",
                locale="schinese",
            )
            for game in games[:20]
        }
        self.assertEqual({key for key, _payload in cache.writes}, expected_keys)
        log_args = debug_log.call_args.args
        rendered_log = log_args[0] % log_args[1:]
        self.assertIn("semantic_candidates=20", rendered_log)
        self.assertIn("semantic_notices=none", rendered_log)
        self.assertIn("semantic_features_ms=", rendered_log)

    async def test_contract_failure_is_request_local_and_core_returns_no_games(self) -> None:
        preference = GamePreference(
            platforms=["steam"],
            soft_features=[
                {
                    "constraint_id": "branching",
                    "source_span": "必须有分支剧情",
                    "normalized_text": "branching story",
                    "role": "core",
                    "polarity": "positive",
                }
            ],
        )
        context = SemanticLlmContext(
            provider_id="provider/session-model",
            response={"verdicts": []},
        )
        cache = SemanticMemoryCache()
        plugin = semantic_pipeline_plugin(context, cache, [semantic_ranked_game(1)])
        prepared = PreparedRecommendation("分支剧情", preference, 5)

        async def identity_reasons(_context, _event, _provider_id, ranked_games):
            return ranked_games

        with (
            patch.object(main_module, "generate_recommendation_reasons", identity_reasons),
            patch.object(main_module.logger, "debug") as debug_log,
        ):
            run = await plugin._run_recommendation(FakeEvent(), prepared)

        self.assertEqual(run.ranked_games, [])
        self.assertEqual(
            [notice.code for notice in run.run_notices],
            ["semantic_feature_contract_failure"],
        )
        self.assertEqual(preference.parse_warnings, [])
        self.assertEqual(cache.writes, [])
        self.assertEqual(len(context.calls), 1)
        self.assertNotIn("LLM 兜底建议", run.messages[0])
        log_args = debug_log.call_args.args
        rendered_log = log_args[0] % log_args[1:]
        self.assertIn("semantic_candidates=1", rendered_log)
        self.assertIn(
            "semantic_notices=semantic_feature_contract_failure",
            rendered_log,
        )

    async def test_notice_is_sent_as_independent_first_message(self) -> None:
        plugin = object.__new__(SteamGameRecommenderPlugin)
        preference = GamePreference()
        run = RecommendationRun(
            messages=["notice", "recommendations"],
            ranked_games=[],
            preference=preference,
            result_limit=5,
            raw_query="query",
            run_notices=(
                RunNotice("semantic_feature_contract_failure", "warning", "notice"),
            ),
        )
        saved: list[RecommendationRun] = []

        async def prepare(_event, _query):
            return PreparedRecommendation("query", preference, 5)

        async def execute(_event, _prepared):
            return run

        async def save(_event, completed):
            saved.append(completed)

        plugin._prepare_recommendation = prepare
        plugin._run_recommendation = execute
        plugin._save_recent_recommendation = save
        event = PlainResultEvent()

        results = [item async for item in plugin.recommend_games(event, "query")]

        self.assertEqual(results, [("plain", "notice\n\nrecommendations")])
        self.assertEqual(saved, [run])
        self.assertEqual(preference.parse_warnings, [])


class FakeEvent:
    unified_msg_origin = "qq:test"
    sender_id = "test"
    platform = "qq"


class PlainResultEvent(FakeEvent):
    def plain_result(self, text: str):
        return ("plain", text)


class SemanticMemoryCache:
    def __init__(self) -> None:
        self.payloads: dict[str, object] = {}
        self.writes: list[tuple[str, object]] = []

    async def get_json(self, key: str, _ttl_hours: int):
        return self.payloads.get(key)

    async def set_json(self, key: str, payload: object) -> None:
        self.payloads[key] = payload
        self.writes.append((key, payload))


class SemanticLlmContext:
    def __init__(self, *, provider_id: str, response: dict) -> None:
        self.provider_id = provider_id
        self.response = response
        self.calls: list[dict] = []
        self.provider_requests: list[str] = []

    async def get_current_chat_provider_id(self, *, umo):
        self.provider_requests.append(umo)
        return self.provider_id

    async def llm_generate(self, **kwargs):
        self.calls.append(kwargs)
        return SimpleNamespace(completion_text=json.dumps(self.response, ensure_ascii=False))


class StaticSemanticSteamIndex:
    def __init__(self, games: list[RankedGame]) -> None:
        self.games = games

    async def recommend(self, _preference, **_kwargs):
        return list(self.games)


def semantic_ranked_game(appid: int) -> RankedGame:
    return RankedGame.from_candidate(
        GameCandidate(
            appid=appid,
            title=f"Game {appid}",
            app_type="game",
            short_description="A story with branching choices.",
        ),
        80 - appid,
        ScoreBreakdown(
            relevance_tier="broad",
            semantic_score=0.5,
            quality_score=0.5,
            layer_score=0.5,
            positive_score=50,
            retrieval_rank=appid,
        ),
        [],
    )


def semantic_pipeline_plugin(
    context: SemanticLlmContext,
    cache: SemanticMemoryCache,
    games: list[RankedGame],
) -> SteamGameRecommenderPlugin:
    plugin = object.__new__(SteamGameRecommenderPlugin)
    plugin.provider_id = ""
    plugin.context = context
    plugin.cache = cache
    plugin.steam_client = SimpleNamespace(language="schinese")
    plugin.steam_index = StaticSemanticSteamIndex(games)
    plugin.price_bridge = IdentityPriceBridge()

    async def no_owned_games(_event, required):
        return []

    async def no_profile(_event, _owned_games):
        return {}

    plugin._owned_games_for_recommendation = no_owned_games
    plugin._user_profile_tag_weights = no_profile
    return plugin


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
