from __future__ import annotations

import sys
import time
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
    from astrbot_plugin_steam_game_recommender.clients.steam import SteamApiError
    from astrbot_plugin_steam_game_recommender.main import (
        PreparedRecommendation,
        RecommendationRun,
        SteamGameRecommenderPlugin,
        safe_bounded_float,
        safe_bounded_int,
    )
    from astrbot_plugin_steam_game_recommender.services.played_filter import (
        LIBRARY_FILTER_EXCLUDE_OWNED,
        LibraryFilterModeError,
    )
    from astrbot_plugin_steam_game_recommender.services.llm_fallback import (
        UnverifiedGameSuggestion,
    )
    from astrbot_plugin_steam_game_recommender.services.preference_parser import ParseOutcome
    from astrbot_plugin_steam_game_recommender.services.recommendation_memory import (
        RecommendationMemory,
        RecommendationResultSummary,
        dump_memory,
        recommendation_memory_key,
    )
    from astrbot_plugin_steam_game_recommender.services.run_notices import RunNotice
    from astrbot_plugin_steam_game_recommender.services.steam_index import (
        STEAM_INDEX_FALLBACK_WARNING,
        STEAM_ONLY_SCOPE_WARNING,
        STEAM_TAG_RECALL_DEGRADED_WARNING,
    )
    from astrbot_plugin_steam_game_recommender.services.steam_recall import (
        RecallHealth,
        RecallUnavailableError,
    )
    from astrbot_plugin_steam_game_recommender.services.semantic_feature_verifier import (
        RankedFeatureVerificationOutcome,
        verdict_cache_key,
    )
    from astrbot_plugin_steam_game_recommender.storage.models import (
        GameCandidate,
        GamePreference,
        RankedGame,
        ScoreBreakdown,
        SteamAccountBinding,
        SteamOwnedGame,
        SteamSearchHit,
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
    def test_numeric_config_helpers_reject_non_finite_and_clamp_extremes(self) -> None:
        self.assertEqual(safe_bounded_int("invalid", 15, minimum=1, maximum=120), 15)
        self.assertEqual(safe_bounded_int(-99, 15, minimum=1, maximum=120), 1)
        self.assertEqual(safe_bounded_int(999, 15, minimum=1, maximum=120), 120)
        self.assertEqual(
            safe_bounded_float(float("nan"), 0.65, minimum=0.0, maximum=1.0),
            0.65,
        )
        self.assertEqual(
            safe_bounded_float(float("inf"), 0.65, minimum=0.0, maximum=1.0),
            0.65,
        )
        self.assertEqual(
            safe_bounded_float(-1, 0.65, minimum=0.0, maximum=1.0),
            0.0,
        )

    def test_nested_dashboard_config_is_wired_to_runtime_services(self) -> None:
        config = {
            "model_and_access": {
                "llm_provider_id": "provider/nested",
                "llm_fallback_provider_id": "provider/fallback",
                "semantic_verification_batch_size": "99",
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
                "reuse_identical_query_cache": "false",
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
        self.assertEqual(getattr(plugin, "semantic_verification_batch_size", None), 10)
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
        self.assertIs(
            index_service_class.call_args.kwargs["reuse_cache"],
            False,
        )
        self.assertEqual(
            set(index_service_class.call_args.kwargs),
            {"steam_client", "cache", "ttl_hours", "reuse_cache"},
        )
        price_bridge_class.assert_called_once_with(
            http_client,
            {"default_region": "JP"},
        )

    def test_semantic_verification_batch_size_defaults_and_clamps_safely(self) -> None:
        def build(value=...):
            model_config = {} if value is ... else {"semantic_verification_batch_size": value}
            with (
                patch.object(main_module.httpx, "AsyncClient", return_value=object()),
                patch.object(main_module, "SQLiteCacheRepository", return_value=object()),
                patch.object(main_module, "SteamClient"),
                patch.object(main_module, "PreferenceParser"),
                patch.object(main_module, "SteamGameIndexService"),
                patch.object(main_module, "SteamPriceBridge") as price_bridge_class,
            ):
                price_bridge_class.return_value.is_available.return_value = False
                return SteamGameRecommenderPlugin(
                    object(),
                    {"model_and_access": model_config},
                )

        self.assertEqual(getattr(build(), "semantic_verification_batch_size", None), 5)
        self.assertEqual(
            getattr(build("not-an-int"), "semantic_verification_batch_size", None),
            5,
        )
        self.assertEqual(getattr(build(0), "semantic_verification_batch_size", None), 1)
        self.assertEqual(getattr(build(11), "semantic_verification_batch_size", None), 10)

    def test_identical_query_cache_reuse_defaults_false_and_accepts_explicit_true(
        self,
    ) -> None:
        def build(cache_config):
            with (
                patch.object(main_module.httpx, "AsyncClient", return_value=object()),
                patch.object(main_module, "SQLiteCacheRepository", return_value=object()),
                patch.object(main_module, "SteamClient"),
                patch.object(main_module, "PreferenceParser"),
                patch.object(main_module, "SteamGameIndexService") as index_service,
                patch.object(main_module, "SteamPriceBridge") as price_bridge,
            ):
                price_bridge.return_value.is_available.return_value = False
                plugin = SteamGameRecommenderPlugin(
                    object(),
                    {"cache_and_network": cache_config},
                )
            return plugin, index_service.call_args.kwargs

        default_plugin, default_kwargs = build({})
        enabled_plugin, enabled_kwargs = build(
            {"reuse_identical_query_cache": True}
        )

        self.assertIs(default_plugin.reuse_identical_query_cache, False)
        self.assertIs(default_kwargs["reuse_cache"], False)
        self.assertIs(enabled_plugin.reuse_identical_query_cache, True)
        self.assertIs(enabled_kwargs["reuse_cache"], True)

    def test_fallback_provider_defaults_empty_and_does_not_read_flat_config(self) -> None:
        with (
            patch.object(main_module.httpx, "AsyncClient", return_value=object()),
            patch.object(main_module, "SQLiteCacheRepository", return_value=object()),
            patch.object(main_module, "SteamClient"),
            patch.object(main_module, "PreferenceParser"),
            patch.object(main_module, "SteamGameIndexService"),
            patch.object(main_module, "SteamPriceBridge") as price_bridge,
        ):
            price_bridge.return_value.is_available.return_value = False
            default_plugin = SteamGameRecommenderPlugin(object(), {})
            flat_plugin = SteamGameRecommenderPlugin(
                object(),
                {"llm_fallback_provider_id": "provider/legacy-flat"},
            )

        self.assertEqual(default_plugin.fallback_provider_id, "")
        self.assertEqual(flat_plugin.fallback_provider_id, "")


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

    async def test_non_steam_empty_keeps_rule_result_when_fallback_is_disabled(
        self,
    ) -> None:
        plugin = object.__new__(SteamGameRecommenderPlugin)
        plugin.provider_id = "provider-1"
        plugin.fallback_provider_id = ""
        plugin.context = FakeLlmContext(
            json.dumps(
                {
                    "suggestions": [
                        {"title": "Game A", "reason": "符合多人偏好。"}
                    ]
                },
                ensure_ascii=False,
            )
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
    async def test_healthy_empty_result_stays_a_normal_empty_result(self) -> None:
        context = RaisingLlmContext()
        plugin = empty_pipeline_plugin(context)
        plugin.fallback_provider_id = ""
        preference = GamePreference(
            platforms=["steam"],
            genres_like=["strategy"],
        )

        run = await plugin._run_recommendation(
            FakeEvent(),
            PreparedRecommendation("Steam strategy", preference, 2),
        )

        self.assertEqual(run.ranked_games, [])
        self.assertEqual(preference.parse_warnings, [])
        self.assertEqual(context.calls, 0)
        rendered = "\n".join(run.messages)
        self.assertIn("暂时没有找到满足当前条件的游戏", rendered)
        self.assertNotIn(STEAM_INDEX_FALLBACK_WARNING, rendered)
        self.assertNotIn(STEAM_TAG_RECALL_DEGRADED_WARNING, rendered)

    async def test_only_typed_steam_errors_use_the_steam_failure_message(self) -> None:
        errors = [
            RecallUnavailableError(RecallHealth()),
            SteamApiError("https://api.steampowered.com?key=secret-key"),
            RuntimeError("provider token=secret-key /private/provider/path"),
        ]

        for error in errors:
            with self.subTest(error=type(error).__name__):
                plugin = object.__new__(SteamGameRecommenderPlugin)

                async def prepare(_event, raw_query):
                    return PreparedRecommendation(
                        raw_query,
                        GamePreference(platforms=["steam"]),
                        2,
                    )

                async def execute(_event, _prepared):
                    raise error

                plugin._prepare_recommendation = prepare
                plugin._run_recommendation = execute
                plugin._save_recent_recommendation = AsyncMock()

                results = [
                    result
                    async for result in plugin.recommend_games(
                        PlainResultEvent(),
                        "Steam query",
                    )
                ]

                expected_message = (
                    "Steam 查询暂时不可用，请稍后重试。"
                    if isinstance(error, SteamApiError)
                    else "游戏推荐暂时失败，请稍后重试。"
                )
                self.assertEqual(results, [("plain", expected_message)])
                self.assertNotIn("secret-key", results[0][1])
                self.assertNotIn("/private/", results[0][1])
                plugin._save_recent_recommendation.assert_not_awaited()

    async def test_command_failure_log_does_not_include_exception_text(self) -> None:
        secret = "provider token=secret-key /private/provider/path"
        plugin = object.__new__(SteamGameRecommenderPlugin)
        plugin._prepare_recommendation = AsyncMock(
            side_effect=RuntimeError(secret)
        )
        warnings: list[tuple[object, ...]] = []

        with patch.object(
            main_module.logger,
            "warning",
            side_effect=lambda *args, **_kwargs: warnings.append(args),
        ):
            results = [
                result
                async for result in plugin.recommend_games(
                    PlainResultEvent(),
                    "Steam query",
                )
            ]

        logged = " ".join(str(value) for call in warnings for value in call)
        self.assertEqual(
            results,
            [("plain", "游戏推荐暂时失败，请稍后重试。")],
        )
        self.assertNotIn(secret, logged)
        self.assertNotIn("secret-key", logged)
        self.assertIn("error_type=%s", logged)
        self.assertIn("RuntimeError", logged)

    async def test_partial_tag_recall_degradation_keeps_ranked_results(self) -> None:
        context = RaisingLlmContext()
        plugin = empty_pipeline_plugin(
            context,
            games=[RankedGame(appid=1, title="Verified Game", app_type="game", score=80)],
        )
        plugin.fallback_provider_id = ""
        preference = GamePreference(
            platforms=["steam"],
            parse_warnings=[STEAM_TAG_RECALL_DEGRADED_WARNING],
        )

        run = await plugin._run_recommendation(
            FakeEvent(),
            PreparedRecommendation("Steam query", preference, 2),
        )

        self.assertEqual([game.appid for game in run.ranked_games], [1])
        rendered = "\n".join(run.messages)
        self.assertIn(STEAM_TAG_RECALL_DEGRADED_WARNING, rendered)
        self.assertNotIn(STEAM_INDEX_FALLBACK_WARNING, rendered)
        self.assertEqual(context.calls, 0)

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


class LlmFallbackMainPipelineTest(unittest.IsolatedAsyncioTestCase):
    async def test_explicit_provider_adds_structured_fallback_after_non_steam_rule_result(
        self,
    ) -> None:
        context = FakeLlmContext(
            json.dumps(
                {
                    "suggestions": [
                        {"title": "Game A", "reason": "符合多人合作偏好。"}
                    ]
                },
                ensure_ascii=False,
            )
        )
        plugin = empty_pipeline_plugin(context)
        prepared = PreparedRecommendation(
            raw_query="非 Steam 多人合作",
            preference=GamePreference(
                platforms=["other"],
                parse_warnings=[STEAM_ONLY_SCOPE_WARNING],
            ),
            result_limit=2,
            run_notices=(RunNotice("parser", "warning", "解析通知"),),
        )

        run = await plugin._run_recommendation(FakeEvent(), prepared)

        self.assertEqual(len(context.calls), 1)
        self.assertEqual(
            context.calls[0]["chat_provider_id"],
            "provider/explicit-fallback",
        )
        self.assertEqual(run.ranked_games, [])
        self.assertEqual(
            run.unverified_suggestions,
            (
                UnverifiedGameSuggestion(
                    "Game A",
                    "符合多人合作偏好。",
                    title_verified=True,
                ),
            ),
        )
        self.assertTrue(run.used_unverified_fallback)
        self.assertEqual(run.messages[0], "解析通知")
        self.assertIn("暂时没有找到满足当前条件的游戏", run.messages[1])
        self.assertIn(STEAM_ONLY_SCOPE_WARNING, run.messages[1])
        self.assertEqual(
            run.messages[2],
            "⚠️ LLM 兜底建议（名称经 Steam 目录确认，需求匹配未验证）",
        )
        self.assertEqual(
            run.messages[3],
            "1. 模型候选（名称经 Steam 目录确认）：“Game A”\n系统说明："
            "Steam 仅确认了该名称对应游戏；模型认为它可能符合需求，"
            "需求匹配未经过 Steam 数据验证。",
        )

    async def test_healthy_language_constrained_steam_empty_result_uses_fallback(
        self,
    ) -> None:
        context = FakeLlmContext(
            json.dumps(
                {
                    "suggestions": [
                        {"title": "Game A", "reason": "符合解谜偏好。"}
                    ]
                },
                ensure_ascii=False,
            )
        )
        plugin = empty_pipeline_plugin(context)

        run = await plugin._run_recommendation(
            FakeEvent(),
            PreparedRecommendation(
                "Steam 解谜",
                GamePreference(
                    platforms=["steam"],
                    genres_like=["puzzle"],
                    required_languages=["schinese"],
                ),
                2,
            ),
        )

        self.assertEqual(len(context.calls), 1)
        self.assertTrue(run.used_unverified_fallback)

    async def test_semantic_filter_empty_result_uses_fallback(self) -> None:
        context = FakeLlmContext(
            json.dumps(
                {
                    "suggestions": [
                        {"title": "Game A", "reason": "符合分支剧情偏好。"}
                    ]
                },
                ensure_ascii=False,
            )
        )
        plugin = empty_pipeline_plugin(
            context,
            games=[RankedGame(appid=1, title="Candidate", app_type="game", score=80)],
        )
        plugin.provider_id = "provider/semantic-verifier"
        preference = GamePreference(
            platforms=["steam"],
            soft_features=[
                {
                    "constraint_id": "branching",
                    "source_span": "分支剧情",
                    "normalized_text": "branching story",
                    "role": "required",
                    "polarity": "positive",
                }
            ],
        )

        with patch.object(
            main_module,
            "verify_ranked_features",
            AsyncMock(return_value=RankedFeatureVerificationOutcome()),
        ):
            run = await plugin._run_recommendation(
                FakeEvent(),
                PreparedRecommendation("必须有分支剧情", preference, 2),
            )

        self.assertEqual(len(context.calls), 1)
        self.assertTrue(run.used_unverified_fallback)

    async def test_library_filter_empty_result_uses_fallback(self) -> None:
        context = FakeLlmContext(
            json.dumps(
                {
                    "suggestions": [
                        {"title": "Game A", "reason": "符合动作玩法偏好。"}
                    ]
                },
                ensure_ascii=False,
            )
        )
        plugin = empty_pipeline_plugin(
            context,
            games=[RankedGame(appid=1, title="Owned Game", app_type="game", score=80)],
        )

        async def owned_games(_event, required):
            self.assertTrue(required)
            return [SteamOwnedGame(appid=1, name="Owned Game")]

        plugin._owned_games_for_recommendation = owned_games
        run = await plugin._run_recommendation(
            FakeEvent(),
            PreparedRecommendation(
                "排除已有的动作游戏",
                GamePreference(
                    platforms=["steam"],
                    genres_like=["action"],
                    library_filter_mode=LIBRARY_FILTER_EXCLUDE_OWNED,
                ),
                2,
            ),
        )

        self.assertEqual(len(context.calls), 1)
        self.assertTrue(run.used_unverified_fallback)

    async def test_budget_price_filter_empty_result_uses_fallback(self) -> None:
        context = FakeLlmContext(
            json.dumps(
                {
                    "suggestions": [
                        {"title": "Game A", "reason": "符合预算需求。"}
                    ]
                },
                ensure_ascii=False,
            )
        )
        plugin = empty_pipeline_plugin(
            context,
            games=[RankedGame(appid=1, title="Candidate", app_type="game", score=80)],
        )
        plugin.price_bridge = EmptyPriceBridge()

        run = await plugin._run_recommendation(
            FakeEvent(),
            PreparedRecommendation(
                "预算内的动作游戏",
                GamePreference(
                    platforms=["steam"],
                    genres_like=["action"],
                    budget=50,
                ),
                2,
            ),
        )

        self.assertEqual(len(context.calls), 1)
        self.assertTrue(run.used_unverified_fallback)

    async def test_contract_failure_repairs_once_then_keeps_rule_empty_result(
        self,
    ) -> None:
        context = FakeLlmContext("not json")
        plugin = empty_pipeline_plugin(context)

        run = await plugin._run_recommendation(
            FakeEvent(),
            PreparedRecommendation("Steam 解谜", GamePreference(platforms=["steam"]), 2),
        )

        self.assertEqual(len(context.calls), 2)
        self.assertEqual(run.unverified_suggestions, ())
        self.assertEqual(
            [notice.code for notice in run.run_notices],
            ["llm_fallback_unavailable"],
        )
        self.assertEqual(
            run.messages[0],
            "LLM 兜底服务暂时不可用，本次仅保留规则空结果。",
        )
        self.assertIn("暂时没有找到满足当前条件的游戏", run.messages[1])
        self.assertNotIn("LLM 兜底建议", "\n".join(run.messages))

    async def test_provider_failure_calls_once_then_keeps_rule_empty_result(
        self,
    ) -> None:
        context = FailingProviderContext()
        plugin = empty_pipeline_plugin(context)

        run = await plugin._run_recommendation(
            FakeEvent(),
            PreparedRecommendation("Steam 解谜", GamePreference(platforms=["steam"]), 2),
        )

        self.assertEqual(context.calls, 1)
        self.assertEqual(run.unverified_suggestions, ())
        self.assertEqual(
            [notice.code for notice in run.run_notices],
            ["llm_fallback_unavailable"],
        )
        self.assertNotIn("LLM 兜底建议", "\n".join(run.messages))

    async def test_verified_ranked_result_never_calls_fallback(self) -> None:
        context = RaisingLlmContext()
        plugin = empty_pipeline_plugin(
            context,
            games=[RankedGame(appid=1, title="Verified Game", app_type="game", score=80)],
        )

        run = await plugin._run_recommendation(
            FakeEvent(),
            PreparedRecommendation(
                "Steam action",
                GamePreference(platforms=["steam"], genres_like=["action"]),
                2,
            ),
        )

        self.assertEqual([game.title for game in run.ranked_games], ["Verified Game"])
        self.assertEqual(context.calls, 0)
        self.assertFalse(run.used_unverified_fallback)

    async def test_core_unknown_result_never_calls_empty_result_fallback(self) -> None:
        preference = GamePreference(
            platforms=["steam"],
            soft_features=[
                {
                    "constraint_id": "branching",
                    "source_span": "分支剧情",
                    "normalized_text": "branching story",
                    "role": "core",
                    "polarity": "positive",
                }
            ],
        )
        context = SemanticLlmContext(
            provider_id="provider/semantic-verifier",
            response={
                "verdicts": [
                    {
                        "appid": 1,
                        "constraint_id": "branching",
                        "polarity": "positive",
                        "status": "unknown",
                        "evidence_quote": "",
                    }
                ]
            },
        )
        plugin = semantic_pipeline_plugin(
            context,
            SemanticMemoryCache(),
            [semantic_ranked_game(1)],
        )
        plugin.fallback_provider_id = "provider/explicit-fallback"
        fallback_generator = AsyncMock()

        async def identity_reasons(_context, _event, _provider_id, ranked_games):
            return ranked_games

        with (
            patch.object(main_module, "generate_recommendation_reasons", identity_reasons),
            patch.object(
                main_module,
                "generate_unverified_game_suggestions",
                fallback_generator,
            ),
        ):
            run = await plugin._run_recommendation(
                FakeEvent(),
                PreparedRecommendation("分支剧情", preference, 2),
            )

        fallback_generator.assert_not_awaited()
        self.assertEqual([game.appid for game in run.ranked_games], [1])
        self.assertEqual(
            [notice.code for notice in run.run_notices],
            ["semantic_feature_core_unknown"],
        )
        self.assertEqual(run.unverified_suggestions, ())
        self.assertFalse(run.used_unverified_fallback)
        rendered = "\n".join(run.messages)
        self.assertIn("不推荐理由", rendered)
        self.assertIn("分支剧情", rendered)
        self.assertNotIn("暂时没有找到满足当前条件的游戏", rendered)
        self.assertNotIn("LLM 兜底建议", rendered)

    async def test_typed_steam_and_unknown_pipeline_errors_do_not_trigger_fallback(
        self,
    ) -> None:
        errors = [
            RecallUnavailableError(RecallHealth()),
            SteamApiError("steam failed"),
            RuntimeError("programming error"),
        ]

        for error in errors:
            with self.subTest(error=type(error).__name__):
                context = RaisingLlmContext()
                plugin = empty_pipeline_plugin(context)
                plugin.steam_index = ErrorSteamIndex(error)

                with self.assertRaises(type(error)):
                    await plugin._run_recommendation(
                        FakeEvent(),
                        PreparedRecommendation(
                            "Steam query",
                            GamePreference(platforms=["steam"]),
                            2,
                        ),
                    )

                self.assertEqual(context.calls, 0)

    async def test_identical_empty_runs_call_provider_each_time(self) -> None:
        context = FakeLlmContext(
            json.dumps(
                {
                    "suggestions": [
                        {"title": "Game A", "reason": "符合玩法偏好。"}
                    ]
                },
                ensure_ascii=False,
            )
        )
        plugin = empty_pipeline_plugin(context)
        prepared = PreparedRecommendation(
            "same query",
            GamePreference(platforms=["other"]),
            1,
        )

        await plugin._run_recommendation(FakeEvent(), prepared)
        await plugin._run_recommendation(FakeEvent(), prepared)

        self.assertEqual(len(context.calls), 2)


class LlmFallbackMemoryIsolationTest(unittest.IsolatedAsyncioTestCase):
    async def test_initial_fallback_result_does_not_write_recommendation_memory(
        self,
    ) -> None:
        plugin = object.__new__(SteamGameRecommenderPlugin)
        plugin.cache = object()
        run = RecommendationRun(
            messages=["fallback"],
            ranked_games=[],
            preference=GamePreference(platforms=["steam"]),
            result_limit=1,
            raw_query="query",
            unverified_suggestions=(
                UnverifiedGameSuggestion("Fallback Game", "模型匹配理由。"),
            ),
        )

        with patch.object(
            main_module,
            "save_recommendation_memory",
            AsyncMock(),
        ) as save_memory:
            await plugin._save_recent_recommendation(FakeEvent(), run)

        save_memory.assert_not_awaited()

    async def test_retry_fallback_preserves_verified_memory_and_exclusions(
        self,
    ) -> None:
        memory = RecommendationMemory(
            chat_platform="qq",
            chat_user_id="test",
            raw_query="verified query",
            preference=GamePreference(platforms=["steam"], genres_like=["action"]),
            result_limit=2,
            shown_appids=[42],
            shown_titles=["verified game"],
            created_at=time.time(),
            last_results=[
                RecommendationResultSummary(
                    appid=42,
                    title="Verified Game",
                    tags=["action"],
                )
            ],
        )
        cache = SemanticMemoryCache()
        key = recommendation_memory_key("qq", "test")
        original_payload = dump_memory(memory)
        cache.payloads[key] = original_payload
        plugin = object.__new__(SteamGameRecommenderPlugin)
        plugin.cache = cache
        excluded_calls: list[tuple[list[int], list[str]]] = []

        async def execute(_event, prepared, **kwargs):
            excluded_calls.append(
                (list(kwargs["excluded_appids"]), list(kwargs["excluded_titles"]))
            )
            return RecommendationRun(
                messages=["fallback"],
                ranked_games=[],
                preference=prepared.preference,
                result_limit=prepared.result_limit,
                raw_query=prepared.raw_query,
                unverified_suggestions=(
                    UnverifiedGameSuggestion("Fallback Game", "模型匹配理由。"),
                ),
            )

        plugin._run_recommendation = execute
        plugin._save_retry_memory = AsyncMock()

        await plugin._retry_recommendation_messages(FakeEvent())
        await plugin._retry_recommendation_messages(FakeEvent())

        plugin._save_retry_memory.assert_not_awaited()
        self.assertEqual(
            excluded_calls,
            [([42], ["verified game"]), ([42], ["verified game"])],
        )
        self.assertNotIn("Fallback Game", excluded_calls[1][1])
        self.assertEqual(cache.payloads[key], original_payload)
        self.assertEqual(cache.writes, [])


class SemanticFeatureMainPipelineTest(unittest.IsolatedAsyncioTestCase):
    async def test_main_passes_actual_result_limit_to_semantic_verification(
        self,
    ) -> None:
        preference = GamePreference(
            platforms=["steam"],
            soft_features=[
                {
                    "constraint_id": "branching",
                    "source_span": "分支剧情",
                    "normalized_text": "branching story",
                    "role": "optional",
                    "polarity": "positive",
                }
            ],
        )
        games = [semantic_ranked_game(1)]
        context = SemanticLlmContext(
            provider_id="provider/session-model",
            response={"verdicts": []},
        )
        plugin = semantic_pipeline_plugin(context, SemanticMemoryCache(), games)
        prepared = PreparedRecommendation("分支剧情", preference, 7)
        semantic_result = RankedFeatureVerificationOutcome(games=tuple(games))
        semantic_verify = AsyncMock(return_value=semantic_result)

        async def identity_reasons(_context, _event, _provider_id, ranked_games):
            return ranked_games

        with (
            patch.object(main_module, "verify_ranked_features", semantic_verify),
            patch.object(main_module, "generate_recommendation_reasons", identity_reasons),
        ):
            await plugin._run_recommendation(FakeEvent(), prepared)

        self.assertEqual(semantic_verify.await_args.kwargs["result_limit"], 7)

    async def test_main_passes_identical_query_cache_policy_to_semantic_verifier(
        self,
    ) -> None:
        preference = GamePreference(
            platforms=["steam"],
            soft_features=[
                {
                    "constraint_id": "branching",
                    "source_span": "分支剧情",
                    "normalized_text": "branching story",
                    "role": "optional",
                    "polarity": "positive",
                }
            ],
        )
        games = [semantic_ranked_game(1)]
        context = SemanticLlmContext(
            provider_id="provider/session-model",
            response={"verdicts": []},
        )
        plugin = semantic_pipeline_plugin(context, SemanticMemoryCache(), games)
        plugin.reuse_identical_query_cache = False
        prepared = PreparedRecommendation("分支剧情", preference, 5)
        semantic_result = SimpleNamespace(
            games=tuple(games),
            notices=(),
            candidate_count=1,
        )

        async def identity_reasons(_context, _event, _provider_id, ranked_games):
            return ranked_games

        with (
            patch.object(main_module, "SemanticFeatureVerifier") as verifier_class,
            patch.object(
                main_module,
                "verify_ranked_features",
                AsyncMock(return_value=semantic_result),
            ),
            patch.object(main_module, "generate_recommendation_reasons", identity_reasons),
        ):
            await plugin._run_recommendation(FakeEvent(), prepared)

        self.assertIs(verifier_class.call_args.kwargs["reuse_cache"], False)

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
                        "evidence_quote": "branching choices",
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
            result_limit=20,
        )

        async def identity_reasons(_context, _event, _provider_id, ranked_games):
            return ranked_games

        with (
            patch.object(main_module, "generate_recommendation_reasons", identity_reasons),
            patch.object(main_module.logger, "debug") as debug_log,
        ):
            run = await plugin._run_recommendation(FakeEvent(), prepared)

        self.assertEqual(context.provider_requests, ["qq:test"])
        self.assertEqual(len(context.calls), 4)
        payloads = [
            json.loads(call["prompt"].split("INPUT=", 1)[1])
            for call in context.calls
        ]
        self.assertEqual(
            [
                [candidate["appid"] for candidate in payload["candidates"]]
                for payload in payloads
            ],
            [
                list(range(1, 6)),
                list(range(6, 11)),
                list(range(11, 16)),
                list(range(16, 21)),
            ],
        )
        self.assertTrue(
            all(
                len(json.dumps(payload, ensure_ascii=False, separators=(",", ":")))
                <= 48_000
                for payload in payloads
            )
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

    async def test_contract_failure_is_request_local_and_core_is_kept_with_caution(self) -> None:
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

        self.assertEqual([game.appid for game in run.ranked_games], [1])
        self.assertEqual(
            [notice.code for notice in run.run_notices],
            [
                "semantic_feature_contract_failure",
                "semantic_feature_required_unverified",
            ],
        )
        self.assertEqual(preference.parse_warnings, [])
        self.assertEqual(cache.writes, [])
        self.assertEqual(len(context.calls), 2)
        rendered_messages = "\n".join(run.messages)
        self.assertNotIn("LLM 兜底建议", rendered_messages)
        self.assertNotIn("格式无效，本次未采用整批结果", rendered_messages)
        self.assertIn("强提示", rendered_messages)
        self.assertIn("必须有分支剧情", rendered_messages)
        self.assertIn("不推荐理由", rendered_messages)
        self.assertIn("响应契约异常", rendered_messages)
        log_args = debug_log.call_args.args
        rendered_log = log_args[0] % log_args[1:]
        self.assertIn("semantic_candidates=1", rendered_log)
        self.assertIn(
            "semantic_notices=semantic_feature_contract_failure,"
            "semantic_feature_required_unverified",
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
    plugin.semantic_verification_batch_size = 5
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


class RaisingLlmContext:
    def __init__(self) -> None:
        self.calls = 0

    async def llm_generate(self, **_kwargs):
        self.calls += 1
        raise AssertionError("fallback provider must not be called")


class FailingProviderContext:
    def __init__(self) -> None:
        self.calls = 0

    async def llm_generate(self, **_kwargs):
        self.calls += 1
        raise RuntimeError("provider unavailable")


class ErrorSteamIndex:
    def __init__(self, error: Exception) -> None:
        self.error = error

    async def recommend(self, _preference, **_kwargs):
        raise self.error


def empty_pipeline_plugin(
    context,
    games: list[RankedGame] | None = None,
) -> SteamGameRecommenderPlugin:
    plugin = object.__new__(SteamGameRecommenderPlugin)
    plugin.provider_id = ""
    plugin.fallback_provider_id = "provider/explicit-fallback"
    plugin.semantic_verification_batch_size = 5
    plugin.context = context
    plugin.cache = object()
    plugin.steam_client = FallbackDirectorySteamClient()
    plugin.steam_index = StaticSemanticSteamIndex(games or [])
    plugin.price_bridge = IdentityPriceBridge()

    async def no_owned_games(_event, required):
        del required
        return []

    async def no_profile(_event, _owned_games):
        return {}

    plugin._owned_games_for_recommendation = no_owned_games
    plugin._user_profile_tag_weights = no_profile
    return plugin


class FallbackDirectorySteamClient:
    language = "schinese"

    def __init__(self) -> None:
        self._next_appid = 1
        self._titles: dict[int, str] = {}

    async def search_game_refs(self, *, search: str, **_kwargs):
        appid = self._next_appid
        self._next_appid += 1
        self._titles[appid] = search
        return [SteamSearchHit(appid=appid, title=search)]

    async def get_game_detail(self, appid: int):
        return GameCandidate(
            appid=appid,
            title=self._titles[appid],
            app_type="game",
        )


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


class EmptyPriceBridge:
    async def enrich_ranked_games(self, _games, _preference):
        return []


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
