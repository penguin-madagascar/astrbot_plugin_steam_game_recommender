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
    from astrbot_plugin_game_recommender.main import GameRecommenderPlugin
    from astrbot_plugin_game_recommender.services.diversity import (
        DIVERSITY_HIGH,
        DIVERSITY_STRICT,
    )
    from astrbot_plugin_game_recommender.storage.models import GamePreference
except ModuleNotFoundError as exc:
    if exc.name in {"astrbot", "pydantic"}:
        raise unittest.SkipTest(f"{exc.name} is not installed in this environment")
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


if __name__ == "__main__":
    unittest.main()
