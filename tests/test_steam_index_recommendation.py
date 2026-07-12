from __future__ import annotations

import importlib
import importlib.util
import unittest
from typing import Any

from astrbot_plugin_game_recommender.storage.models import GameCandidate, GamePreference


class SteamIndexModelTest(unittest.TestCase):
    def test_game_candidate_declares_steam_index_fields(self) -> None:
        fields = model_fields(GameCandidate)

        for field in (
            "appid",
            "review_total",
            "review_positive_ratio",
            "review_recent_ratio",
            "release_date",
            "index_source",
        ):
            self.assertIn(field, fields)


class TagNormalizerTest(unittest.TestCase):
    def test_maps_chinese_terms_and_steam_categories_to_canonical_tags(self) -> None:
        normalizer = optional_import("astrbot_plugin_game_recommender.services.tag_normalizer")

        self.assertEqual(normalizer.normalize_tag("本地合作"), "local_coop")
        self.assertEqual(normalizer.normalize_tag("Shared/Split Screen Co-op"), "local_coop")
        self.assertEqual(normalizer.normalize_tag("轻松"), "relaxing")
        self.assertEqual(normalizer.normalize_tag("Simplified Chinese"), "chinese")
        self.assertEqual(normalizer.normalize_tag("开放世界"), "open_world")
        self.assertEqual(normalizer.normalize_tag("Story Rich"), "story_rich")
        self.assertEqual(normalizer.normalize_tag("魂类"), "soulslike")
        self.assertEqual(normalizer.normalize_tag("类魂"), "soulslike")

        tags = normalizer.canonical_tags_from_terms(
            ["双人", "Local Co-op", "休闲", "Puzzle", "恐怖", "剧情向"]
        )
        self.assertEqual(
            tags,
            ["co_op", "local_coop", "casual", "puzzle", "horror", "story_rich"],
        )

    def test_registers_english_steam_tags_and_maps_chinese_aliases_to_them(self) -> None:
        normalizer = optional_import("astrbot_plugin_game_recommender.services.tag_normalizer")

        normalizer.register_steam_tag_aliases(
            [
                {"tagid": 87918, "name": "Farming Sim"},
                {"tagid": 10235, "name": "Life Sim"},
                {"tagid": 3964, "name": "Pixel Graphics"},
            ]
        )

        self.assertEqual(normalizer.normalize_tag("Farming Sim"), "farming_sim")
        self.assertEqual(normalizer.normalize_tag("农场模拟"), "farming_sim")
        self.assertEqual(normalizer.normalize_tag("生活模拟"), "life_sim")
        self.assertEqual(normalizer.normalize_tag("像素图形"), "pixel_graphics")

    def test_maps_specific_mechanic_tags_without_broad_false_positive(self) -> None:
        normalizer = optional_import("astrbot_plugin_game_recommender.services.tag_normalizer")

        self.assertEqual(normalizer.normalize_tag("Deckbuilding"), "deckbuilding")
        self.assertEqual(
            normalizer.normalize_tag("Open World Survival Craft"),
            "open_world_survival_craft",
        )
        self.assertEqual(normalizer.normalize_tag("Choices Matter"), "choices_matter")

        terms = normalizer.extract_description_terms(
            "Includes Steam Trading Cards, profile backgrounds, and soundtrack."
        )

        self.assertNotIn("deckbuilding", normalizer.canonical_tags_from_terms(terms))
        self.assertNotIn("card_battler", normalizer.canonical_tags_from_terms(terms))


class SimilarityRankerTest(unittest.TestCase):
    def test_high_tag_overlap_beats_high_review_low_overlap(self) -> None:
        ranker = optional_import("astrbot_plugin_game_recommender.services.similarity_ranker")
        profile = ranker.SteamTagProfile(
            include_tags=["co_op", "local_coop", "puzzle", "casual", "relaxing"],
            exclude_tags=[],
            reference_titles=[],
        )
        candidates = [
            steam_index_game(
                "High Review Generic Co-op",
                tags=["co_op", "multiplayer"],
                review_total=50000,
                review_positive_ratio=0.95,
            ),
            steam_index_game(
                "Lower Review Better Match",
                tags=["co_op", "local_coop", "puzzle", "casual", "relaxing"],
                review_total=600,
                review_positive_ratio=0.78,
            ),
        ]

        ranked = ranker.rank_steam_candidates(
            candidates,
            profile,
            min_review_count=50,
            min_positive_ratio=0.65,
        )

        self.assertEqual(
            [game.title for game in ranked],
            ["Lower Review Better Match", "High Review Generic Co-op"],
        )
        self.assertGreater(ranked[0].facts.match_score, ranked[1].facts.match_score)
        self.assertTrue(any("相似标签" in reason for reason in ranked[0].fit_points))

    def test_excludes_disliked_tags_and_singleplayer_only_candidates(self) -> None:
        ranker = optional_import("astrbot_plugin_game_recommender.services.similarity_ranker")
        profile = ranker.SteamTagProfile(
            include_tags=["co_op", "puzzle"],
            required_tags=["co_op"],
            exclude_tags=["horror", "soulslike"],
            reference_titles=[],
        )
        candidates = [
            steam_index_game("Safe Co-op Puzzle", tags=["co_op", "puzzle"]),
            steam_index_game("Scary Co-op Puzzle", tags=["co_op", "puzzle", "horror"]),
            steam_index_game("Solo Puzzle", tags=["singleplayer", "puzzle"]),
        ]

        ranked = ranker.rank_steam_candidates(
            candidates,
            profile,
            min_review_count=50,
            min_positive_ratio=0.65,
        )

        self.assertEqual([game.title for game in ranked], ["Safe Co-op Puzzle"])

    def test_reference_game_tags_expand_profile_without_seed_titles(self) -> None:
        ranker = optional_import("astrbot_plugin_game_recommender.services.similarity_ranker")
        profile = ranker.build_profile_from_preference(
            GamePreference(
                platforms=["steam"],
                genres_like=["co-op"],
                reference_games_like=["Reference Farm"],
            ),
            reference_candidates=[
                steam_index_game(
                    "Reference Farm",
                    tags=["co_op", "farming", "crafting", "relaxing"],
                )
            ],
        )

        self.assertIn("farming", profile.include_tags)
        self.assertIn("crafting", profile.include_tags)
        self.assertIn("relaxing", profile.include_tags)

    def test_aaa_profile_prioritizes_broad_blockbuster_matches(self) -> None:
        ranker = optional_import("astrbot_plugin_game_recommender.services.similarity_ranker")
        profile = ranker.build_profile_from_preference(
            GamePreference(
                extra_tags=["aaa", "open world", "story rich"],
                genres_like=["action", "adventure", "rpg"],
            )
        )

        self.assertIn("open_world", profile.include_tags)
        self.assertIn("story_rich", profile.include_tags)

        ranked = ranker.rank_steam_candidates(
            [
                steam_index_game(
                    "High Review Generic Action",
                    tags=["Action", "Adventure", "RPG"],
                    review_total=90000,
                    review_positive_ratio=0.96,
                ),
                steam_index_game(
                    "Focused AAA Adventure",
                    tags=["Action", "Adventure", "RPG", "Open World", "Story Rich"],
                    review_total=500,
                    review_positive_ratio=0.80,
                ),
            ],
            profile,
            min_review_count=50,
            min_positive_ratio=0.65,
        )

        self.assertEqual(
            [game.title for game in ranked],
            ["Focused AAA Adventure", "High Review Generic Action"],
        )


class SteamIndexServiceTest(unittest.IsolatedAsyncioTestCase):
    def test_aaa_request_uses_blockbuster_search_terms(self) -> None:
        index_module = optional_import("astrbot_plugin_game_recommender.services.steam_index")
        ranker = optional_import("astrbot_plugin_game_recommender.services.similarity_ranker")
        preference = GamePreference(extra_tags=["aaa"])

        terms = index_module.search_terms_for(
            preference,
            ranker.build_profile_from_preference(preference),
        )

        self.assertEqual(
            terms[:5],
            ["popular", "action adventure", "open world", "story rich", "rpg"],
        )
        self.assertNotIn("popular co-op", terms)

    async def test_enrich_candidate_persists_description_mechanic_tags(self) -> None:
        index_module = optional_import("astrbot_plugin_game_recommender.services.steam_index")
        service = index_module.SteamGameIndexService(
            steam_client=NoLiveSearchSteamClient(),
            cache=MemoryCache({}),
            min_review_count=0,
        )

        enriched = await service.enrich_candidate(
            GameCandidate(
                title="Workshop Isles",
                platforms=["PC"],
                genres=["Simulation"],
                tags=["Single-player"],
                stores=["Steam"],
                description=(
                    "A cozy open world survival craft game about automation, "
                    "base building, and farming with friends."
                ),
            )
        )

        self.assertIn("open_world_survival_craft", enriched.inferred_tags)
        self.assertIn("automation", enriched.inferred_tags)
        self.assertIn("building", enriched.inferred_tags)
        self.assertIn("farming", enriched.inferred_tags)
        self.assertIn("relaxing", enriched.inferred_tags)

    async def test_description_inference_never_becomes_hard_constraint_evidence(self) -> None:
        index_module = optional_import("astrbot_plugin_game_recommender.services.steam_index")
        constraints = optional_import(
            "astrbot_plugin_game_recommender.services.constraint_evaluator"
        )
        service = index_module.SteamGameIndexService(
            steam_client=NoLiveSearchSteamClient(),
            cache=MemoryCache({}),
            min_review_count=0,
        )

        enriched = await service.enrich_candidate(
            GameCandidate(
                title="Unverified Description",
                platforms=["PC"],
                tags=["Co-op"],
                stores=["Steam"],
                description="A horror story that mentions Chinese only in prose.",
            )
        )
        assessment = constraints.evaluate_candidate_constraints(
            enriched,
            required_tags=["chinese"],
            exclude_tags=["horror"],
        )

        self.assertNotIn("horror", enriched.tags)
        self.assertNotIn("chinese", enriched.tags)
        self.assertIn("horror", enriched.inferred_tags)
        self.assertIn("chinese", enriched.inferred_tags)
        self.assertEqual(assessment.status, "unknown")
        self.assertEqual(assessment.violations, [])

    async def test_enrich_candidate_loads_english_steam_tags_and_store_page_tags(self) -> None:
        index_module = optional_import("astrbot_plugin_game_recommender.services.steam_index")
        service = index_module.SteamGameIndexService(
            steam_client=TagAwareSteamClient(),
            cache=MemoryCache({}),
            min_review_count=0,
        )

        enriched = await service.enrich_candidate(
            GameCandidate(
                title="Stardew Valley",
                appid=413150,
                platforms=["PC"],
                genres=["RPG", "Simulation"],
                tags=["Single-player"],
                stores=["Steam"],
            )
        )

        self.assertIn("farming sim", enriched.ordered_tags)
        self.assertIn("life sim", enriched.ordered_tags)
        self.assertIn("tag_enrichment:steam_popular_tags", enriched.source_reasons)
        self.assertIn("tag_enrichment:steam_store_page_tags", enriched.source_reasons)

    async def test_steam_store_tags_improve_recommendation_ranking(self) -> None:
        index_module = optional_import("astrbot_plugin_game_recommender.services.steam_index")
        cache = MemoryCache(
            {
                "steam_index:v2": [
                    dump_model(
                        steam_index_game(
                            "Generic Multiplayer",
                            tags=["Multiplayer"],
                            review_total=90000,
                            review_positive_ratio=0.96,
                        )
                    ),
                    dump_model(
                        steam_index_game(
                            "Farm Life Match",
                            tags=["Farming Sim", "Life Sim", "Relaxing", "Multiplayer"],
                            review_total=500,
                            review_positive_ratio=0.80,
                        )
                    ),
                ]
            }
        )
        service = index_module.SteamGameIndexService(
            steam_client=TagAwareSteamClient(),
            cache=cache,
            min_review_count=50,
            min_positive_ratio=0.65,
        )

        ranked = await service.recommend(
            GamePreference(extra_tags=["农场模拟", "放松", "多人"]),
            limit=2,
        )

        self.assertEqual(
            [game.title for game in ranked],
            ["Farm Life Match", "Generic Multiplayer"],
        )

    async def test_recommend_keeps_cached_results_when_supplemental_search_is_empty(self) -> None:
        index_module = optional_import("astrbot_plugin_game_recommender.services.steam_index")
        cache = MemoryCache(
            {
                "steam_index:v2": [
                    dump_model(
                        steam_index_game(
                            "Generic High Review",
                            tags=["co_op", "multiplayer"],
                            review_total=90000,
                            review_positive_ratio=0.96,
                        )
                    ),
                    dump_model(
                        steam_index_game(
                            "Better Local Puzzle",
                            tags=["co_op", "local_coop", "puzzle", "casual", "chinese"],
                            review_total=800,
                            review_positive_ratio=0.80,
                        )
                    ),
                ]
            }
        )
        service = index_module.SteamGameIndexService(
            steam_client=NoLiveSearchSteamClient(),
            cache=cache,
            ttl_hours=168,
            min_review_count=50,
            min_positive_ratio=0.65,
        )

        ranked = await service.recommend(
            GamePreference(
                platforms=["steam"],
                genres_like=["co-op", "local co-op", "puzzle", "casual"],
                language="中文",
                result_count=2,
            ),
            limit=2,
        )

        self.assertEqual(
            [game.title for game in ranked],
            ["Better Local Puzzle", "Generic High Review"],
        )
        self.assertEqual(ranked[0].index_source, "steam_index")


def optional_import(module_name: str):
    spec = importlib.util.find_spec(module_name)
    if spec is None:
        raise AssertionError(f"{module_name} module is missing")
    return importlib.import_module(module_name)


def steam_index_game(
    title: str,
    tags: list[str],
    review_total: int = 500,
    review_positive_ratio: float = 0.75,
) -> GameCandidate:
    return GameCandidate(
        title=title,
        appid=abs(hash(title)) % 1000000,
        platforms=["PC"],
        genres=[],
        tags=tags,
        stores=["Steam"],
        raw_url=f"https://store.steampowered.com/app/{abs(hash(title)) % 1000000}/",
        review_total=review_total,
        review_positive_ratio=review_positive_ratio,
        review_recent_ratio=review_positive_ratio,
        index_source="steam_index",
    )


def model_fields(model: type) -> set[str]:
    fields = getattr(model, "model_fields", None)
    if fields is not None:
        return set(fields)
    return set(getattr(model, "__fields__", {}))


def dump_model(model: Any) -> dict[str, Any]:
    dumper = getattr(model, "model_dump", None)
    return dumper() if dumper else model.dict()


class MemoryCache:
    def __init__(self, payloads: dict[str, Any]) -> None:
        self.payloads = payloads
        self.writes: dict[str, Any] = {}

    async def get_json(self, key: str, _ttl_hours: int) -> Any | None:
        payload = self.payloads.get(key)
        if key == "steam_index:v2" and isinstance(payload, list):
            return {
                "version": 2,
                "entries": [{"candidate": entry, "refreshed_at": 1.0} for entry in payload],
                "search_coverage": {},
            }
        return payload

    async def set_json(self, key: str, payload: Any) -> None:
        self.writes[key] = payload
        self.payloads[key] = payload


class NoLiveSearchSteamClient:
    async def search_games(self, **_kwargs: Any) -> list[GameCandidate]:
        return []


class TagAwareSteamClient(NoLiveSearchSteamClient):
    async def get_popular_tags(self) -> list[dict[str, Any]]:
        return [
            {"tagid": 87918, "name": "Farming Sim"},
            {"tagid": 10235, "name": "Life Sim"},
            {"tagid": 3964, "name": "Pixel Graphics"},
            {"tagid": 3859, "name": "Multiplayer"},
            {"tagid": 1654, "name": "Relaxing"},
        ]

    async def get_store_page_tags(self, appid: int) -> list[str]:
        del appid
        return ["Farming Sim", "Pixel Graphics", "Multiplayer", "Life Sim", "RPG", "Relaxing"]

    async def get_review_summary(self, appid: int):
        del appid
        return None
