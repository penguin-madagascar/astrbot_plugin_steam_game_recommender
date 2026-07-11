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

from astrbot_plugin_game_recommender.services.diversity import (  # noqa: E402
    DIVERSITY_BALANCED,
    DIVERSITY_HIGH,
    DIVERSITY_STRICT,
    select_results_by_diversity,
)
from astrbot_plugin_game_recommender.services.preference_parser import (  # noqa: E402
    parse_preference_json,
)
from astrbot_plugin_game_recommender.services.similarity_ranker import (  # noqa: E402
    build_profile_from_preference,
    rank_steam_candidates,
)
from astrbot_plugin_game_recommender.storage.models import (  # noqa: E402
    GameCandidate,
    GamePreference,
)


class DiversityPreferenceParsingTest(unittest.TestCase):
    def test_llm_json_accepts_diversity_mode(self) -> None:
        preference = parse_preference_json(
            """
            {
              "platforms": ["steam"],
              "genres_like": ["co-op"],
              "diversity_mode": "high",
              "result_count": 5
            }
            """
        )

        self.assertEqual(preference.diversity_mode, DIVERSITY_HIGH)

    def test_invalid_diversity_mode_defaults_to_strict(self) -> None:
        preference = parse_preference_json(
            """
            {
              "platforms": ["steam"],
              "genres_like": ["co-op"],
              "diversity_mode": "surprising",
              "result_count": 5
            }
            """
        )

        self.assertEqual(preference.diversity_mode, DIVERSITY_STRICT)

    def test_missing_diversity_mode_defaults_to_strict(self) -> None:
        preference = parse_preference_json(
            """
            {
              "platforms": ["steam"],
              "genres_like": ["co-op"],
              "result_count": 5
            }
            """
        )

        self.assertEqual(preference.diversity_mode, DIVERSITY_STRICT)


class DiversitySelectionTest(unittest.TestCase):
    def test_strict_keeps_primary_rank_order(self) -> None:
        ranked = sample_ranked_games()

        selected = select_results_by_diversity(ranked, limit=4, mode=DIVERSITY_STRICT)

        self.assertEqual(
            [game.title for game in selected],
            ["Farm Co-op A", "Farm Co-op B", "Story Co-op", "Lower Match Builder"],
        )

    def test_balanced_matches_existing_same_primary_score_rerank(self) -> None:
        ranked = sample_ranked_games()

        selected = select_results_by_diversity(ranked, limit=4, mode=DIVERSITY_BALANCED)

        self.assertEqual(
            [game.title for game in selected],
            ["Farm Co-op A", "Story Co-op", "Farm Co-op B", "Lower Match Builder"],
        )

    def test_high_can_diversify_within_tier_but_not_across_tier(self) -> None:
        ranked = rank_steam_candidates(
            [
                steam_game("Farm Strong", ["Co-op", "Puzzle", "Relaxing", "Farming", "Crafting"]),
                steam_game("Farm Strong 2", ["Co-op", "Puzzle", "Relaxing", "Farming", "Crafting"]),
                steam_game("Story Strong", ["Co-op", "Puzzle", "Relaxing", "Story Rich"]),
                steam_game("Backup Builder", ["Relaxing", "Building", "Automation"]),
            ],
            build_profile_from_preference(
                GamePreference(platforms=["steam"], genres_like=["co-op", "puzzle", "relaxing"])
            ),
        )

        selected = select_results_by_diversity(ranked, limit=4, mode=DIVERSITY_HIGH)

        self.assertEqual(selected[0].tier, "strong")
        self.assertEqual(selected[1].tier, "strong")
        self.assertEqual(selected[2].tier, "strong")
        self.assertEqual(selected[3].tier, "recommended")
        self.assertEqual(
            [game.title for game in selected[:3]],
            ["Farm Strong", "Story Strong", "Farm Strong 2"],
        )


def sample_ranked_games():
    return rank_steam_candidates(
        [
            steam_game("Farm Co-op A", ["Co-op", "Puzzle", "Farming", "Crafting"]),
            steam_game("Farm Co-op B", ["Co-op", "Puzzle", "Farming", "Crafting"]),
            steam_game("Story Co-op", ["Co-op", "Puzzle", "Story Rich", "Choices Matter"]),
            steam_game("Lower Match Builder", ["Co-op", "Building", "Automation"]),
        ],
        build_profile_from_preference(
            GamePreference(platforms=["steam"], genres_like=["co-op", "puzzle"])
        ),
    )


def steam_game(title: str, tags: list[str]) -> GameCandidate:
    return GameCandidate(
        title=title,
        appid=abs(hash(title)) % 1000000,
        platforms=["PC"],
        tags=tags,
        stores=["Steam"],
        review_total=500,
        review_positive_ratio=0.8,
    )


if __name__ == "__main__":
    unittest.main()
