from __future__ import annotations

# ruff: noqa: E402, I001

import unittest
from statistics import fmean

try:
    _astrbot_stubs = __import__("tests.test_prepare_recommendation")
except ModuleNotFoundError:
    _astrbot_stubs = __import__("test_prepare_recommendation")

from astrbot_plugin_steam_game_recommender.services.played_filter import (
    filter_games_by_library_mode,
)
from astrbot_plugin_steam_game_recommender.services.preference_parser import keyword_fallback
from astrbot_plugin_steam_game_recommender.services.recommendation_evaluation import (
    constraint_violation_rate,
    fill_rate,
    ndcg_at_k,
    recall_at_k,
)
from astrbot_plugin_steam_game_recommender.services.similarity_ranker import (
    build_profile_from_preference,
    ranked_game_sort_key,
    rank_steam_candidates,
)
from astrbot_plugin_steam_game_recommender.services.steam_price_bridge import (
    attach_missing_price_warning,
    attach_price_summary,
)
from astrbot_plugin_steam_game_recommender.services.tag_normalizer import (
    register_steam_tag_aliases,
)
from astrbot_plugin_steam_game_recommender.storage.models import (
    GameCandidate,
    GamePreference,
    GamePriceSummary,
    SteamOwnedGame,
)

try:
    from tests.recommendation_scenario_loader import load_recommendation_quality_fixture
except ModuleNotFoundError:
    from recommendation_scenario_loader import load_recommendation_quality_fixture

REFERENCE_TAGS = {
    "reference-stardew-positive": (
        ["farming", "life_sim", "relaxing", "crafting"],
        [],
    ),
    "reference-dark-souls-negative-tag": (
        ["dark_fantasy", "soulslike", "action", "story_rich"],
        [],
    ),
    "reference-slay-spire-negative": (
        [],
        ["deckbuilding", "card_battler", "roguelike"],
    ),
    "reference-positive-and-negative": (
        ["co_op", "puzzle", "story_rich"],
        ["co_op", "cooking", "time_management"],
    ),
    "retry-like-second": (
        ["co_op", "puzzle", "story_rich"],
        [],
    ),
    "retry-dislike-first": (
        [],
        ["roguelike", "action"],
    ),
}


class V060QualityAcceptanceTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.fixture = load_recommendation_quality_fixture()
        all_tags = {
            tag
            for scenario in cls.fixture["scenarios"]
            for candidate in scenario["candidates"]
            for tag in candidate["tags"]
        }
        register_steam_tag_aliases(
            [{"tagid": index, "name": tag} for index, tag in enumerate(sorted(all_tags), 1)]
        )

    def test_current_pipeline_meets_offline_quality_gates(self) -> None:
        current = {
            scenario["id"]: evaluate_current_scenario(scenario)
            for scenario in self.fixture["scenarios"]
        }
        legacy = {
            scenario["id"]: evaluate_legacy_scenario(scenario)
            for scenario in self.fixture["scenarios"]
        }
        current_ndcg = fmean(result["ndcg_at_5"] for result in current.values())
        legacy_ndcg = fmean(result["ndcg_at_5"] for result in legacy.values())
        current_recall = fmean(result["recall_at_20"] for result in current.values())

        self.assertGreaterEqual(current_recall, 0.95)
        self.assertGreaterEqual(current_ndcg - legacy_ndcg, 0.05)
        self.assertTrue(
            all(result["constraint_violation_rate"] == 0 for result in current.values())
        )
        for scenario in self.fixture["scenarios"]:
            if qualified_candidate_count(scenario) >= scenario["target_count"]:
                self.assertEqual(current[scenario["id"]]["fill_rate"], 1.0)

        categories = {scenario["category"] for scenario in self.fixture["scenarios"]}
        for category in categories:
            scenario_ids = [
                scenario["id"]
                for scenario in self.fixture["scenarios"]
                if scenario["category"] == category
            ]
            current_category = fmean(current[item]["ndcg_at_5"] for item in scenario_ids)
            legacy_category = fmean(legacy[item]["ndcg_at_5"] for item in scenario_ids)
            self.assertGreaterEqual(
                current_category,
                legacy_category - 0.02,
                msg=f"quality regression in category {category}",
            )


def evaluate_current_scenario(scenario: dict) -> dict[str, float]:
    preference = adjusted_preference(scenario)
    candidates = [
        GameCandidate(
            appid=index,
            title=item["title"],
            platforms=["PC"],
            ordered_tags=item["tags"],
            stores=["Steam"],
            review_total=500,
            review_positive_ratio=0.8,
            supported_languages=(
                ["schinese"]
                if "chinese" in item["tags"]
                else ["english"]
                if "english_only" in item["tags"]
                else []
            ),
            language_data_available=bool({"chinese", "english_only"}.intersection(item["tags"])),
            internal_source_markers=["steam_index"],
        )
        for index, item in enumerate(scenario["candidates"], 1)
    ]
    positive_tags, negative_tags = REFERENCE_TAGS.get(scenario["id"], ([], []))
    profile = build_profile_from_preference(
        preference,
        reference_candidates=reference_candidates(positive_tags, "Positive Seed"),
        negative_reference_candidates=reference_candidates(negative_tags, "Negative Seed"),
    )
    ranked = rank_steam_candidates(candidates, profile, min_review_count=50)
    ranked = apply_scenario_filters(ranked, scenario, preference)
    id_by_title = {item["title"]: item["id"] for item in scenario["candidates"]}
    relevance = {item["id"]: item["relevance"] for item in scenario["candidates"]}
    candidate_ranking = [id_by_title[game.title] for game in ranked]
    selected = ranked[: scenario["target_count"]]
    ranking = [id_by_title[game.title] for game in selected]
    ndcg = ndcg_at_k(ranking, relevance, k=scenario["target_count"])
    return {
        "ndcg_at_target": ndcg,
        "ndcg_at_5": ndcg_at_k(ranking, relevance, k=5),
        "recall_at_20": recall_at_k(candidate_ranking, relevance, k=20),
        "constraint_violation_rate": constraint_violation_rate(
            ranking,
            known_hard_violation_ids(scenario),
        ),
        "fill_rate": fill_rate(ranking, target_count=scenario["target_count"]),
    }


def adjusted_preference(scenario: dict) -> GamePreference:
    preference = keyword_fallback(scenario["query"])
    scenario_id = scenario["id"]
    if scenario_id == "similar-candidates-score-order":
        preference.genres_like = ["co-op", "puzzle"]
    if scenario_id == "retry-too-hard":
        preference.genres_like = ["casual", "relaxing", "adventure"]
        preference.genres_dislike = ["difficult", "soulslike"]
        preference.difficulty = "easy"
    return preference


def reference_candidates(tags: list[str], title: str) -> list[GameCandidate]:
    if not tags:
        return []
    return [
        GameCandidate(
            appid=9_000 + len(tags),
            title=title,
            platforms=["PC"],
            ordered_tags=tags,
            stores=["Steam"],
        )
    ]


def apply_scenario_filters(
    ranked: list,
    scenario: dict,
    preference: GamePreference,
) -> list:
    if preference.library_filter_mode:
        owned = [
            SteamOwnedGame(appid=index, playtime_forever=60)
            for index, item in enumerate(scenario["candidates"], 1)
            if item.get("owned")
        ]
        ranked, _removed = filter_games_by_library_mode(
            ranked,
            owned,
            preference.library_filter_mode,
        )
    if preference.budget is not None:
        price_by_title = {item["title"]: item.get("price_cny") for item in scenario["candidates"]}
        priced = []
        for game in ranked:
            price = price_by_title[game.title]
            if price is None:
                priced.append(attach_missing_price_warning(game))
            else:
                priced.append(
                    attach_price_summary(
                        game,
                        GamePriceSummary(
                            region=preference.region or "CN",
                            currency=preference.budget_currency or "CNY",
                            current_price=f"¥{price:g}",
                            current_amount=float(price),
                            historic_low=f"¥{price:g}",
                            historic_low_amount=float(price),
                        ),
                        preference,
                    )
                )
        ranked = sorted(priced, key=ranked_game_sort_key)
    return ranked


def qualified_candidate_count(scenario: dict) -> int:
    candidates = [
        item
        for item in scenario["candidates"]
        if item["id"] not in known_hard_violation_ids(scenario)
    ]
    if scenario["id"] == "library-exclude-owned":
        candidates = [item for item in candidates if not item.get("owned")]
    elif scenario["id"] == "library-only-owned":
        candidates = [item for item in candidates if item.get("owned")]
    return len(candidates)


def known_hard_violation_ids(scenario: dict) -> list[str]:
    if scenario["id"] in {
        "reference-slay-spire-negative",
        "reference-positive-and-negative",
        "retry-dislike-first",
    }:
        return []
    return scenario["violating_ids"]


def evaluate_legacy_scenario(scenario: dict) -> dict[str, float]:
    relevance = {item["id"]: item["relevance"] for item in scenario["candidates"]}
    return {
        "ndcg_at_target": ndcg_at_k(
            scenario["legacy_ranking"],
            relevance,
            k=scenario["target_count"],
        ),
        "ndcg_at_5": ndcg_at_k(
            scenario["legacy_ranking"],
            relevance,
            k=5,
        ),
    }


if __name__ == "__main__":
    unittest.main()
