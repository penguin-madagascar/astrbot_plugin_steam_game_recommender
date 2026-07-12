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
    from astrbot_plugin_game_recommender.services.diversity import (
        DIVERSITY_HIGH,
        DIVERSITY_STRICT,
    )
    from astrbot_plugin_game_recommender.services.played_filter import LibraryFilterModeError
    from astrbot_plugin_game_recommender.services.steam_index import STEAM_ONLY_SCOPE_WARNING
    from astrbot_plugin_game_recommender.storage.models import GamePreference, RankedGame
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


class PrepareRecommendationDiversityTest(unittest.IsolatedAsyncioTestCase):
    async def test_prepare_uses_llm_diversity_mode_over_query_terms(self) -> None:
        preference = GamePreference(
            platforms=["steam"],
            genres_like=["co-op"],
            diversity_mode=DIVERSITY_HIGH,
            result_count=5,
        )
        plugin = object.__new__(GameRecommenderPlugin)
        plugin.max_results = 5
        plugin.preference_parser = FakePreferenceParser(preference)

        prepared = await plugin._prepare_recommendation(
            object(),
            "严格匹配 Steam 合作解谜",
        )

        self.assertEqual(prepared.diversity_mode, DIVERSITY_HIGH)
        self.assertEqual(plugin.preference_parser.seen_text, "严格匹配 Steam 合作解谜")

    async def test_prepare_defaults_to_strict_when_llm_field_is_missing(self) -> None:
        preference = GamePreference(platforms=["steam"], genres_like=["co-op"], result_count=5)
        plugin = object.__new__(GameRecommenderPlugin)
        plugin.max_results = 5
        plugin.preference_parser = FakePreferenceParser(preference)

        prepared = await plugin._prepare_recommendation(object(), "Steam 合作解谜")

        self.assertEqual(prepared.diversity_mode, DIVERSITY_STRICT)


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

        self.assertIn("仅覆盖 Steam/PC", str(raised.exception))

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
            diversity_mode=DIVERSITY_STRICT,
            result_limit=2,
        )

        run = await plugin._run_recommendation(FakeEvent(), prepared)

        self.assertEqual(run.ranked_games, [])
        self.assertEqual(len(plugin.context.calls), 1)
        self.assertIn("LLM 兜底建议（未经过 Steam 索引验证）", run.messages[0])
        self.assertIn("Mario Kart 8 Deluxe", run.messages[0])


class EmbeddingPipelineTest(unittest.IsolatedAsyncioTestCase):
    async def test_run_requests_top_twenty_and_applies_embedding_reranker(self) -> None:
        plugin = object.__new__(GameRecommenderPlugin)
        plugin.enable_llm_fallback = False
        plugin.provider_id = ""
        plugin.context = FakeLlmContext("")
        plugin.steam_index = RecordingSteamIndex()
        plugin.price_bridge = IdentityPriceBridge()
        plugin.embedding_reranker = RecordingEmbeddingReranker()

        async def empty_profile(_event):
            return {}

        plugin._user_profile_tag_weights = empty_profile
        prepared = PreparedRecommendation(
            raw_query="Steam 合作解谜",
            preference=GamePreference(
                platforms=["steam"],
                genres_like=["co-op", "puzzle"],
                result_count=2,
            ),
            diversity_mode=DIVERSITY_STRICT,
            result_limit=2,
        )

        run = await plugin._run_recommendation(FakeEvent(), prepared)

        self.assertEqual(plugin.steam_index.seen_limit, 20)
        self.assertEqual(plugin.embedding_reranker.seen_query, "Steam 合作解谜")
        self.assertEqual([game.title for game in run.ranked_games], ["Second", "First"])


class FakeEvent:
    unified_msg_origin = "qq:test"


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


class RecordingSteamIndex:
    def __init__(self) -> None:
        self.seen_limit = 0

    async def recommend(self, _preference, limit: int, **_kwargs):
        self.seen_limit = limit
        return [
            RankedGame(title="First", score=80, tier="strong"),
            RankedGame(title="Second", score=70, tier="strong"),
        ]


class RecordingEmbeddingReranker:
    def __init__(self) -> None:
        self.seen_query = ""

    async def rerank(self, _preference, raw_query: str, games):
        self.seen_query = raw_query
        return list(reversed(games))


class IdentityPriceBridge:
    def is_available(self) -> bool:
        return False

    async def enrich_ranked_games(self, games, _preference):
        return games


if __name__ == "__main__":
    unittest.main()
