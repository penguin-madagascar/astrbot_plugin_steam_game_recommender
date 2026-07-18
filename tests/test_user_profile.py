from __future__ import annotations

import unittest

from astrbot_plugin_steam_game_recommender.services.similarity_ranker import (
    SteamTagProfile,
    rank_steam_candidates,
)
from astrbot_plugin_steam_game_recommender.services.steam_index import (
    STEAM_INDEX_SCHEMA_VERSION,
    SteamGameIndexService,
)
from astrbot_plugin_steam_game_recommender.storage.models import (
    GameCandidate,
    GamePreference,
    SteamAccountBinding,
    SteamOwnedGame,
)


class UserProfileWeightsTest(unittest.TestCase):
    def test_playtime_profile_weights_known_owned_game_tags(self) -> None:
        from astrbot_plugin_steam_game_recommender.services.user_profile import (
            build_user_tag_weights,
        )

        weights = build_user_tag_weights(
            [
                SteamOwnedGame(appid=1, name="Farm Workshop", playtime_forever=2400),
                SteamOwnedGame(appid=2, name="Arena Shooter", playtime_forever=30),
                SteamOwnedGame(appid=3, name="Unindexed Game", playtime_forever=9000),
            ],
            [
                steam_game(1, "Farm Workshop", ["Farming", "Crafting", "Building"]),
                steam_game(2, "Arena Shooter", ["Shooter", "PvP"]),
            ],
        )

        self.assertGreater(weights["farming"], weights["shooter"])
        self.assertGreater(weights["crafting"], weights["pvp"])
        self.assertNotIn("singleplayer", weights)


class UserProfileRankerTest(unittest.TestCase):
    def test_profile_bonus_reorders_only_equal_primary_matches(self) -> None:
        profile = SteamTagProfile(include_tags=["co_op", "puzzle"])
        ranked = rank_steam_candidates(
            [
                steam_game(1, "Generic Co-op Puzzle", ["Co-op", "Puzzle"]),
                steam_game(2, "Craft Co-op Puzzle", ["Co-op", "Puzzle", "Farming", "Crafting"]),
            ],
            profile,
            profile_tag_weights={"farming": 1.0, "crafting": 0.9},
        )

        self.assertEqual(
            [game.title for game in ranked],
            [
                "Craft Co-op Puzzle",
                "Generic Co-op Puzzle",
            ],
        )
        self.assertGreater(ranked[0].score_breakdown.library_profile or 0, 0)
        self.assertEqual(
            ranked[0].score_breakdown.tag_coverage,
            ranked[1].score_breakdown.tag_coverage,
        )

    def test_profile_bonus_does_not_outrank_better_primary_match_or_exclusions(self) -> None:
        profile = SteamTagProfile(
            include_tags=["co_op", "puzzle", "relaxing"],
            exclude_tags=["horror"],
        )
        ranked = rank_steam_candidates(
            [
                steam_game(1, "Focused Co-op Puzzle", ["Co-op", "Puzzle", "Relaxing"]),
                steam_game(2, "Profile Favorite Co-op", ["Co-op", "Farming", "Crafting"]),
                steam_game(3, "Scary Profile Favorite", ["Co-op", "Puzzle", "Horror", "Farming"]),
            ],
            profile,
            profile_tag_weights={"farming": 1.0, "crafting": 0.9},
        )

        self.assertEqual(
            [game.title for game in ranked],
            [
                "Focused Co-op Puzzle",
                "Profile Favorite Co-op",
            ],
        )


class UserProfileSteamIndexTest(unittest.IsolatedAsyncioTestCase):
    async def test_recommend_passes_profile_weights_to_ranker(self) -> None:
        service = SteamGameIndexService(
            steam_client=NoLiveSearchSteamClient(),
            cache=MemoryCache(
                [
                    steam_game(1, "Generic Co-op Puzzle", ["Co-op", "Puzzle"]),
                    steam_game(2, "Craft Co-op Puzzle", ["Co-op", "Puzzle", "Farming"]),
                ]
            ),
            clock=lambda: 1.0,
        )

        ranked = await service.recommend(
            GamePreference(platforms=["steam"], genres_like=["co-op", "puzzle"]),
            limit=2,
            profile_tag_weights={"farming": 1.0},
        )

        self.assertEqual(
            [game.title for game in ranked],
            [
                "Craft Co-op Puzzle",
                "Generic Co-op Puzzle",
            ],
        )

    async def test_service_uses_fixed_policy_without_weight_configuration(self) -> None:
        service = SteamGameIndexService(
            steam_client=NoLiveSearchSteamClient(),
            cache=MemoryCache(
                [
                    steam_game(1, "Exact Small Match", ["Puzzle"], reviews=1),
                    steam_game(2, "Huge Wrong Match", ["Action"], reviews=1_000_000),
                ]
            ),
            clock=lambda: 1.0,
        )

        ranked = await service.recommend(
            GamePreference(platforms=["steam"], genres_like=["puzzle"]),
            limit=2,
        )

        self.assertEqual(ranked[0].title, "Exact Small Match")
        self.assertFalse(hasattr(service, "positive_component_weights"))


class BoundUserProfileLoaderTest(unittest.IsolatedAsyncioTestCase):
    async def test_loads_weights_for_bound_account_with_api_key(self) -> None:
        from astrbot_plugin_steam_game_recommender.services.user_profile import (
            load_bound_user_tag_weights,
        )

        weights = await load_bound_user_tag_weights(
            chat_platform="qq",
            chat_user_id="user-1",
            cache=FakeBindingCache(
                SteamAccountBinding(
                    chat_platform="qq",
                    chat_user_id="user-1",
                    steam_id64="76561198000000000",
                    account_kind="steam_id64",
                    display_value="76561198000000000",
                )
            ),
            steam_client=FakeOwnedGamesClient(has_key=True),
            index_entries=[
                steam_game(1, "Farm Workshop", ["Farming", "Crafting"]),
                steam_game(2, "Arena Shooter", ["Shooter"]),
            ],
        )

        self.assertGreater(weights["farming"], weights["shooter"])

    async def test_missing_binding_or_api_key_returns_empty_weights(self) -> None:
        from astrbot_plugin_steam_game_recommender.services.user_profile import (
            load_bound_user_tag_weights,
        )

        no_binding = await load_bound_user_tag_weights(
            chat_platform="qq",
            chat_user_id="user-1",
            cache=FakeBindingCache(None),
            steam_client=FakeOwnedGamesClient(has_key=True),
            index_entries=[steam_game(1, "Farm Workshop", ["Farming"])],
        )
        no_api_key = await load_bound_user_tag_weights(
            chat_platform="qq",
            chat_user_id="user-1",
            cache=FakeBindingCache(
                SteamAccountBinding(
                    chat_platform="qq",
                    chat_user_id="user-1",
                    steam_id64="76561198000000000",
                    account_kind="steam_id64",
                    display_value="76561198000000000",
                )
            ),
            steam_client=FakeOwnedGamesClient(has_key=False),
            index_entries=[steam_game(1, "Farm Workshop", ["Farming"])],
        )

        self.assertEqual(no_binding, {})
        self.assertEqual(no_api_key, {})


def steam_game(
    appid: int,
    title: str,
    tags: list[str],
    reviews: int = 500,
) -> GameCandidate:
    return GameCandidate(
        title=title,
        appid=appid,
        app_type="game",
        platforms=["PC"],
        tags=tags,
        stores=["Steam"],
        review_total=reviews,
        review_positive_ratio=0.8,
    )


class MemoryCache:
    def __init__(self, entries: list[GameCandidate]) -> None:
        self.entries = entries

    async def get_json(self, _key: str, _ttl_hours: int):
        return {
            "schema_version": STEAM_INDEX_SCHEMA_VERSION,
            "entries": [
                {"candidate": entry.model_dump(), "refreshed_at": 1.0} for entry in self.entries
            ],
            "search_coverage": {},
        }

    async def set_json(self, _key: str, _payload) -> None:
        return None


class NoLiveSearchSteamClient:
    async def search_games(self, **_kwargs):
        return []


class FakeBindingCache:
    def __init__(self, binding: SteamAccountBinding | None) -> None:
        self.binding = binding

    async def get_steam_account_binding(self, chat_platform: str, chat_user_id: str):
        if self.binding is None:
            return None
        if (
            self.binding.chat_platform == chat_platform
            and self.binding.chat_user_id == chat_user_id
        ):
            return self.binding
        return None


class FakeOwnedGamesClient:
    def __init__(self, has_key: bool) -> None:
        self.has_key = has_key

    def has_web_api_key(self) -> bool:
        return self.has_key

    async def get_owned_games(
        self,
        _steam_id64: str,
        *,
        binding_identity=None,
    ) -> list[SteamOwnedGame]:
        del binding_identity
        return [
            SteamOwnedGame(appid=1, name="Farm Workshop", playtime_forever=2400),
            SteamOwnedGame(appid=2, name="Arena Shooter", playtime_forever=30),
        ]


if __name__ == "__main__":
    unittest.main()
