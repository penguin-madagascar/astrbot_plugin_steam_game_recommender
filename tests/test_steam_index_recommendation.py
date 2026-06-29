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

        tags = normalizer.canonical_tags_from_terms(
            ["双人", "Local Co-op", "休闲", "Puzzle", "恐怖"]
        )
        self.assertEqual(tags, ["co_op", "local_coop", "casual", "puzzle", "horror"])


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


class SteamIndexServiceTest(unittest.IsolatedAsyncioTestCase):
    async def test_recommend_uses_cached_index_without_live_search(self) -> None:
        index_module = optional_import("astrbot_plugin_game_recommender.services.steam_index")
        cache = MemoryCache(
            {
                "steam_index:entries": [
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
        return self.payloads.get(key)

    async def set_json(self, key: str, payload: Any) -> None:
        self.writes[key] = payload
        self.payloads[key] = payload


class NoLiveSearchSteamClient:
    async def search_games(self, **_kwargs: Any) -> list[GameCandidate]:
        raise AssertionError("cached index recommendations must not call live Steam search")
