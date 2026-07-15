from __future__ import annotations

# ruff: noqa: E402, I001

import json
import unittest
from pathlib import Path
from types import SimpleNamespace
from typing import Any
from unittest.mock import patch

try:
    _astrbot_stubs = __import__("tests.test_prepare_recommendation")
except ModuleNotFoundError:
    _astrbot_stubs = __import__("test_prepare_recommendation")

from astrbot_plugin_steam_game_recommender import main as main_module
from astrbot_plugin_steam_game_recommender.main import SteamGameRecommenderPlugin
from astrbot_plugin_steam_game_recommender.services.preference_parser import (
    PreferenceParser,
)
from astrbot_plugin_steam_game_recommender.services.steam_index import (
    STEAM_INDEX_FALLBACK_WARNING,
    STEAM_TAG_RECALL_DEGRADED_WARNING,
)

try:
    from tests.e2e_recommendation_harness import (
        FrozenSteamClient,
        MemoryIndexCache,
        RecordingSteamGameIndexService,
    )
except ModuleNotFoundError:
    from e2e_recommendation_harness import (
        FrozenSteamClient,
        MemoryIndexCache,
        RecordingSteamGameIndexService,
    )


FIXTURE_PATH = (
    Path(__file__).parent
    / "fixtures"
    / "e2e_semantic_recommendation_scenarios.json"
)


class PipelineEvent:
    unified_msg_origin = "fixture:test"


class StructuredPipelineContext:
    def __init__(self, scenario: dict[str, Any]) -> None:
        self.scenario = scenario
        self.parser_prompts: list[str] = []
        self.semantic_payloads: list[dict[str, Any]] = []

    async def llm_generate(self, **kwargs: Any) -> SimpleNamespace:
        prompt = str(kwargs.get("prompt") or "")
        if "INPUT=" not in prompt:
            raw_query = str(self.scenario["raw_query"])
            if raw_query not in prompt:
                raise AssertionError("parser prompt omitted the original query")
            self.parser_prompts.append(prompt)
            response = self.scenario["parser_payload"]
        else:
            payload = json.loads(prompt.split("INPUT=", 1)[1])
            self.semantic_payloads.append(payload)
            fixture_verdicts = self.scenario["semantic_verdicts"]
            verdicts = []
            for request in payload["requests"]:
                key = "|".join(
                    (
                        str(request["appid"]),
                        request["constraint_id"],
                        request["polarity"],
                    )
                )
                if key not in fixture_verdicts:
                    raise AssertionError(f"fixture has no verdict for request {key}")
                verdicts.append({**request, **fixture_verdicts[key]})
            response = {"verdicts": verdicts}
        return SimpleNamespace(
            completion_text=json.dumps(response, ensure_ascii=False)
        )


class FullPipelineIndex(RecordingSteamGameIndexService):
    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.ranked_results = ()

    async def recommend(self, *args: Any, **kwargs: Any):
        ranked = await super().recommend(*args, **kwargs)
        self.ranked_results = tuple(ranked)
        return ranked


class IdentityPriceBridge:
    async def enrich_ranked_games(self, games, _preference):
        return list(games)


def load_fixture() -> dict[str, Any]:
    return json.loads(FIXTURE_PATH.read_text(encoding="utf-8"))


def build_plugin(
    fixture: dict[str, Any],
    scenario: dict[str, Any],
) -> tuple[SteamGameRecommenderPlugin, StructuredPipelineContext]:
    context = StructuredPipelineContext(scenario)
    cache = MemoryIndexCache()
    client = FrozenSteamClient(fixture)
    index = FullPipelineIndex(client, cache, clock=lambda: 1_700_000_000.0)
    plugin = object.__new__(SteamGameRecommenderPlugin)
    plugin.context = context
    plugin.provider_id = "fixture-provider"
    plugin.fallback_provider_id = ""
    plugin.semantic_verification_batch_size = 10
    plugin.reuse_identical_query_cache = False
    plugin.max_results = 3
    plugin.default_region = "CN"
    plugin.cache = cache
    plugin.steam_client = client
    plugin.preference_parser = PreferenceParser(context, plugin.provider_id)
    plugin.steam_index = index
    plugin.price_bridge = IdentityPriceBridge()

    async def no_owned_games(_event, required):
        del required
        return []

    async def no_profile(_event, _owned_games):
        return {}

    plugin._owned_games_for_recommendation = no_owned_games
    plugin._user_profile_tag_weights = no_profile
    return plugin, context


class SemanticRecommendationEndToEndTest(unittest.IsolatedAsyncioTestCase):
    async def test_raw_queries_reach_real_recall_semantic_contract_and_formatter(
        self,
    ) -> None:
        fixture = load_fixture()

        async def identity_reasons(_context, _event, _provider_id, ranked_games):
            return list(ranked_games)

        for scenario in fixture["scenarios"]:
            with self.subTest(scenario=scenario["id"]):
                plugin, context = build_plugin(fixture, scenario)
                prepared = await plugin._prepare_recommendation(
                    PipelineEvent(),
                    scenario["raw_query"],
                )
                with patch.object(
                    main_module,
                    "generate_recommendation_reasons",
                    identity_reasons,
                ):
                    run = await plugin._run_recommendation(
                        PipelineEvent(),
                        prepared,
                    )

                expected = scenario["expected"]
                self.assertEqual(len(context.parser_prompts), 1)
                self.assertEqual(prepared.raw_query, scenario["raw_query"])
                self.assertTrue(prepared.preference.soft_features)
                self.assertTrue(
                    all(
                        feature.source_span in scenario["raw_query"]
                        for feature in prepared.preference.soft_features
                    )
                )
                self.assertGreaterEqual(
                    len(set(plugin.steam_client.storefront_tag_calls)),
                    expected["minimum_tag_sources"],
                )
                self.assertGreaterEqual(
                    len(plugin.steam_client.storefront_intersection_calls),
                    expected["minimum_intersection_sources"],
                )
                self.assertTrue(plugin.steam_index.ranked_results)
                self.assertTrue(
                    set(expected["relevance_tiers"])
                    <= {
                        game.score_breakdown.relevance_tier
                        for game in plugin.steam_index.ranked_results
                    }
                )
                self.assertEqual(
                    len(context.semantic_payloads),
                    expected["semantic_batches"],
                )
                request_keys = {
                    "|".join(
                        (
                            str(request["appid"]),
                            request["constraint_id"],
                            request["polarity"],
                        )
                    )
                    for payload in context.semantic_payloads
                    for request in payload["requests"]
                }
                self.assertEqual(request_keys, set(scenario["semantic_verdicts"]))
                self.assertEqual(
                    [game.appid for game in run.ranked_games],
                    expected["output_appids"],
                )
                rendered = "\n".join(run.messages)
                games_by_appid = {
                    int(game["appid"]): game for game in fixture["games"]
                }
                for appid in expected["output_appids"]:
                    self.assertIn(games_by_appid[appid]["title"], rendered)
                for appid in expected["excluded_appids"]:
                    self.assertNotIn(games_by_appid[appid]["title"], rendered)
                self.assertNotIn("暂时没有找到满足当前条件的游戏", rendered)
                self.assertNotIn(STEAM_INDEX_FALLBACK_WARNING, rendered)
                self.assertNotIn(STEAM_TAG_RECALL_DEGRADED_WARNING, rendered)


if __name__ == "__main__":
    unittest.main()
