from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

try:
    __import__("tests.test_prepare_recommendation")
except ModuleNotFoundError:
    __import__("test_prepare_recommendation")

from astrbot_plugin_steam_game_recommender import main as main_module
from astrbot_plugin_steam_game_recommender.main import SteamGameRecommenderPlugin
from astrbot_plugin_steam_game_recommender.services.account_binding import (
    AccountBindingError,
)
from astrbot_plugin_steam_game_recommender.services.recommendation_memory import (
    recommendation_owner_scope,
)
from astrbot_plugin_steam_game_recommender.services.unplayed_picker import (
    UnplayedRecommendationError,
)
from astrbot_plugin_steam_game_recommender.storage.models import (
    GameCandidate,
    SteamAccountBinding,
    SteamOwnedGame,
)
from astrbot_plugin_steam_game_recommender.storage.repository import (
    SQLiteCacheRepository,
)


class RandomRecommendationCommandTest(unittest.IsolatedAsyncioTestCase):
    async def test_command_sends_one_plain_message_without_forward_record_or_extra_fields(
        self,
    ) -> None:
        plugin = object.__new__(SteamGameRecommenderPlugin)
        plugin.cache = BoundCache()
        plugin.steam_client = UnplayedSteamClient()
        plugin.context = ReasonContext()
        plugin.provider_id = "provider-1"
        plugin.recommendation_config = {
            "steam_min_review_count": 50,
            "steam_min_positive_ratio": 0.65,
        }
        plugin.config = {
            "steam_min_review_count": 50_000,
            "steam_min_positive_ratio": 0.99,
        }
        event = PlainEvent()

        results = [item async for item in plugin.recommend_random_game(event)]

        self.assertEqual(len(results), 1)
        self.assertEqual(len(event.plain_messages), 1)
        self.assertEqual(event.chain_calls, 0)
        message = event.plain_messages[0]
        self.assertTrue(message.startswith("《Backlog Game》\n"))
        for excluded in ("推荐分", "价格", "购买链接", "http", "数据来源"):
            self.assertNotIn(excluded, message)

    async def test_unplayed_error_text_is_not_reflected_to_chat(self) -> None:
        secret = "provider-token-secret /private/provider/path?token=abcdef"
        plugin = object.__new__(SteamGameRecommenderPlugin)
        plugin.cache = BoundCache()
        plugin.steam_client = UnplayedSteamClient()
        plugin.recommendation_config = {}

        with patch.object(
            main_module,
            "pick_random_unplayed_game",
            side_effect=UnplayedRecommendationError(secret),
        ):
            results = [
                item async for item in plugin.recommend_random_game(PlainEvent())
            ]

        self.assertEqual(len(results), 1)
        self.assertNotIn(secret, results[0][1])
        self.assertNotIn("/private", results[0][1])


class AccountUnbindCommandTest(unittest.IsolatedAsyncioTestCase):
    async def test_account_error_text_is_not_reflected_to_chat(self) -> None:
        secret = "provider-token-secret /private/provider/path?token=abcdef"
        plugin = object.__new__(SteamGameRecommenderPlugin)
        plugin.cache = FailingAccountCache(secret)

        results = [item async for item in plugin.account_bind(PlainEvent(), "")]

        self.assertEqual(len(results), 1)
        self.assertNotIn(secret, results[0][1])
        self.assertNotIn("/private", results[0][1])

    async def test_unbind_deletes_binding_memory_and_personal_library_cache(self) -> None:
        plugin = object.__new__(SteamGameRecommenderPlugin)
        plugin.cache = DeletingBoundCache()
        event = PlainEvent()

        results = [item async for item in plugin.account_unbind(event)]

        self.assertEqual(
            results,
            [("plain", "Steam 账号已解除绑定，相关个人缓存也已删除。")],
        )
        self.assertEqual(plugin.cache.deleted_bindings, [("qq", "test")])
        self.assertEqual(
            plugin.cache.deleted_owners,
            [
                recommendation_owner_scope("qq", "test"),
                "steam-account:76561198000000000",
            ],
        )

    async def test_single_platform_instance_lazily_migrates_legacy_binding(self) -> None:
        plugin = object.__new__(SteamGameRecommenderPlugin)
        plugin.context = PlatformContext("onebot-instance-2")
        plugin.cache = LegacyBindingCache()

        results = [
            item
            async for item in plugin.account_bind(OneBotInstanceEvent(), "")
        ]

        self.assertEqual(
            results,
            [("plain", "当前绑定 Steam ID：76561198000000000（steam_id64）。")],
        )
        self.assertEqual(
            plugin.cache.migrations,
            [("aiocqhttp", "onebot-instance-2", "test")],
        )

    async def test_multiple_platform_instances_require_a_new_binding(self) -> None:
        plugin = object.__new__(SteamGameRecommenderPlugin)
        plugin.context = PlatformContext("onebot-instance-2", "onebot-instance-3")
        plugin.cache = LegacyBindingCache()

        results = [
            item
            async for item in plugin.account_bind(OneBotInstanceEvent(), "")
        ]

        self.assertIn("请使用 /accountbind 重新绑定", results[0][1])
        self.assertEqual(plugin.cache.migrations, [])

    async def test_legacy_claimed_by_another_instance_cannot_migrate(self) -> None:
        plugin = object.__new__(SteamGameRecommenderPlugin)
        plugin.context = PlatformContext("onebot-instance-b")
        plugin.cache = ClaimedLegacyBindingCache("onebot-instance-a")

        results = [
            item
            async for item in plugin.account_bind(OneBotInstanceBEvent(), "")
        ]

        self.assertIn("请使用 /accountbind 重新绑定", results[0][1])
        self.assertEqual(plugin.cache.migrations, [])

    async def test_unbind_clears_current_memory_when_legacy_belongs_elsewhere(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            cache = SQLiteCacheRepository(Path(tmpdir) / "cache.sqlite3")
            await cache.upsert_steam_account_binding(
                SteamAccountBinding(
                    chat_platform="aiocqhttp",
                    chat_user_id="test",
                    steam_id64="76561198000000000",
                    account_kind="steam_id64",
                    display_value="76561198000000000",
                    metadata={
                        "migrated_to_platform_instance": "onebot-instance-a"
                    },
                )
            )
            owner = recommendation_owner_scope("onebot-instance-b", "test")
            await cache.set_json(
                "instance-b-memory",
                {"private": True},
                ttl_hours=1,
                owner_scope=owner,
            )
            plugin = object.__new__(SteamGameRecommenderPlugin)
            plugin.context = PlatformContext(
                "onebot-instance-a",
                "onebot-instance-b",
            )
            plugin.cache = cache

            results = [
                item async for item in plugin.account_unbind(OneBotInstanceBEvent())
            ]

            self.assertEqual(
                results,
                [("plain", "当前没有绑定 Steam 账号；相关推荐记录已清理。")],
            )
            self.assertIsNone(await cache.get_json("instance-b-memory", 1))
            self.assertIsNotNone(
                await cache.get_steam_account_binding("aiocqhttp", "test")
            )

    async def test_explicit_rebind_claims_an_unowned_ambiguous_legacy(self) -> None:
        plugin = object.__new__(SteamGameRecommenderPlugin)
        plugin.context = PlatformContext("onebot-instance-2", "onebot-instance-3")
        plugin.cache = LegacyBindingCache()

        results = [
            item
            async for item in plugin.account_bind(
                OneBotInstanceEvent(),
                "76561198000000001",
            )
        ]

        self.assertIn("账号绑定成功", results[0][1])
        self.assertEqual(
            plugin.cache.legacy_claims,
            [("onebot-instance-2", "aiocqhttp")],
        )

    async def test_unbind_removes_the_retained_legacy_row_after_migration(self) -> None:
        plugin = object.__new__(SteamGameRecommenderPlugin)
        plugin.cache = MigratedDeletingCache()

        results = [item async for item in plugin.account_unbind(OneBotInstanceEvent())]

        self.assertEqual(
            results,
            [("plain", "Steam 账号已解除绑定，相关个人缓存也已删除。")],
        )
        self.assertEqual(
            plugin.cache.deleted_bindings,
            [
                ("onebot-instance-2", "test"),
                ("aiocqhttp", "test"),
            ],
        )

    async def test_unbind_lazily_migrates_and_removes_a_single_legacy_binding(
        self,
    ) -> None:
        plugin = object.__new__(SteamGameRecommenderPlugin)
        plugin.context = PlatformContext("onebot-instance-2")
        plugin.cache = LegacyDeletingCache()

        results = [item async for item in plugin.account_unbind(OneBotInstanceEvent())]

        self.assertEqual(
            results,
            [("plain", "Steam 账号已解除绑定，相关个人缓存也已删除。")],
        )
        self.assertEqual(
            plugin.cache.migrations,
            [("aiocqhttp", "onebot-instance-2", "test")],
        )
        self.assertEqual(
            plugin.cache.deleted_bindings,
            [
                ("onebot-instance-2", "test"),
                ("aiocqhttp", "test"),
            ],
        )

    async def test_rebind_keeps_migration_marker_and_clears_old_library_cache(
        self,
    ) -> None:
        plugin = object.__new__(SteamGameRecommenderPlugin)
        plugin.context = PlatformContext("onebot-instance-2")
        plugin.cache = MigratedRebindingCache()

        results = [
            item
            async for item in plugin.account_bind(
                OneBotInstanceEvent(),
                "76561198000000001",
            )
        ]

        self.assertIn("账号绑定成功", results[0][1])
        self.assertIn(
            "steam-account:76561198000000000",
            plugin.cache.deleted_owners,
        )
        self.assertEqual(
            plugin.cache.saved.metadata["migrated_from_platform"],
            "aiocqhttp",
        )


class PlainEvent:
    unified_msg_origin = "qq:test"
    sender_id = "test"
    platform = "qq"

    def __init__(self) -> None:
        self.plain_messages: list[str] = []
        self.chain_calls = 0

    def plain_result(self, text: str):
        self.plain_messages.append(text)
        return ("plain", text)

    def chain_result(self, _chain):
        self.chain_calls += 1
        return ("chain", _chain)


class OneBotInstanceEvent(PlainEvent):
    unified_msg_origin = "onebot-instance-2:GroupMessage:group-9"

    def get_platform_name(self) -> str:
        return "aiocqhttp"

    def get_platform_id(self) -> str:
        return "onebot-instance-2"

    def get_sender_id(self) -> str:
        return "test"


class OneBotInstanceBEvent(OneBotInstanceEvent):
    unified_msg_origin = "onebot-instance-b:GroupMessage:group-9"

    def get_platform_id(self) -> str:
        return "onebot-instance-b"


class BoundCache:
    async def get_steam_account_binding(self, _platform: str, _user_id: str):
        return SteamAccountBinding(
            chat_user_id="test",
            steam_id64="76561198000000000",
            account_kind="steamid64",
            display_value="76561198000000000",
        )


class FailingAccountCache:
    def __init__(self, message: str) -> None:
        self.message = message

    async def get_steam_account_binding(self, _platform: str, _user_id: str):
        raise AccountBindingError(self.message)


class DeletingBoundCache(BoundCache):
    def __init__(self) -> None:
        self.deleted_bindings: list[tuple[str, str]] = []
        self.deleted_owners: list[str] = []

    async def delete_steam_account_binding_family(
        self,
        platform: str,
        user_id: str,
    ):
        binding = await self.get_steam_account_binding(platform, user_id)
        if binding is None:
            return []
        self.deleted_bindings.append((platform, user_id))
        return [binding]

    async def delete_steam_account_data(
        self,
        platform: str,
        user_id: str,
        *,
        recommendation_owner_scope: str,
    ):
        deleted = await self.delete_steam_account_binding_family(
            platform,
            user_id,
        )
        self.deleted_owners.append(recommendation_owner_scope)
        for binding in deleted:
            owner_scope = f"steam-account:{binding.steam_id64}"
            if owner_scope not in self.deleted_owners:
                self.deleted_owners.append(owner_scope)
        return deleted

    async def delete_owner_scope(self, owner_scope: str):
        self.deleted_owners.append(owner_scope)
        return 1


class MigratedDeletingCache(DeletingBoundCache):
    async def get_steam_account_binding(self, platform: str, _user_id: str):
        if platform == "aiocqhttp":
            return SteamAccountBinding(
                chat_platform="aiocqhttp",
                chat_user_id="test",
                steam_id64="76561198000000000",
                account_kind="steam_id64",
                display_value="76561198000000000",
                metadata={
                    "migrated_to_platform_instance": "onebot-instance-2"
                },
            )
        return SteamAccountBinding(
            chat_platform="onebot-instance-2",
            chat_user_id="test",
            steam_id64="76561198000000000",
            account_kind="steam_id64",
            display_value="76561198000000000",
            metadata={"migrated_from_platform": "aiocqhttp"},
        )

    async def delete_steam_account_binding_family(
        self,
        platform: str,
        user_id: str,
    ):
        current = await self.get_steam_account_binding(platform, user_id)
        legacy = await self.get_steam_account_binding("aiocqhttp", user_id)
        self.deleted_bindings.extend(
            [(platform, user_id), ("aiocqhttp", user_id)]
        )
        return [current, legacy]


class LegacyBindingCache:
    def __init__(self) -> None:
        self.binding = SteamAccountBinding(
            chat_platform="aiocqhttp",
            chat_user_id="test",
            steam_id64="76561198000000000",
            account_kind="steam_id64",
            display_value="76561198000000000",
        )
        self.migrations: list[tuple[str, str, str]] = []
        self.legacy_claims: list[tuple[str, str]] = []
        self.saved = None

    async def get_steam_account_binding(self, platform: str, _user_id: str):
        return self.binding if platform == "aiocqhttp" else None

    async def migrate_steam_account_binding(
        self,
        legacy_platform: str,
        platform_instance: str,
        user_id: str,
    ):
        self.migrations.append((legacy_platform, platform_instance, user_id))
        dumper = getattr(self.binding, "model_dump", None)
        data = dumper() if dumper else self.binding.dict()
        data["chat_platform"] = platform_instance
        validator = getattr(SteamAccountBinding, "model_validate", None)
        return validator(data) if validator else SteamAccountBinding.parse_obj(data)

    async def upsert_steam_account_binding_claiming_legacy(
        self,
        binding,
        *,
        legacy_platform: str,
    ):
        self.legacy_claims.append((binding.chat_platform, legacy_platform))
        data = binding.dict()
        data["metadata"] = {
            **binding.metadata,
            "migrated_from_platform": legacy_platform,
        }
        self.saved = SteamAccountBinding.parse_obj(data)
        return self.saved


class ClaimedLegacyBindingCache(LegacyBindingCache):
    def __init__(self, claimed_by: str) -> None:
        super().__init__()
        data = self.binding.dict()
        data["metadata"] = {
            "migrated_to_platform_instance": claimed_by,
        }
        self.binding = SteamAccountBinding.parse_obj(data)


class LegacyDeletingCache(LegacyBindingCache):
    def __init__(self) -> None:
        super().__init__()
        self.current = None
        self.deleted_bindings: list[tuple[str, str]] = []
        self.deleted_owners: list[str] = []

    async def get_steam_account_binding(self, platform: str, _user_id: str):
        if platform == "onebot-instance-2":
            return self.current
        return self.binding if platform == "aiocqhttp" else None

    async def migrate_steam_account_binding(
        self,
        legacy_platform: str,
        platform_instance: str,
        user_id: str,
    ):
        migrated = await super().migrate_steam_account_binding(
            legacy_platform,
            platform_instance,
            user_id,
        )
        dumper = getattr(migrated, "model_dump", None)
        data = dumper() if dumper else migrated.dict()
        data["metadata"] = {"migrated_from_platform": legacy_platform}
        validator = getattr(SteamAccountBinding, "model_validate", None)
        self.current = (
            validator(data) if validator else SteamAccountBinding.parse_obj(data)
        )
        return self.current

    async def delete_steam_account_binding_family(
        self,
        platform: str,
        user_id: str,
    ):
        self.deleted_bindings.extend(
            [(platform, user_id), ("aiocqhttp", user_id)]
        )
        return [self.current, self.binding]

    async def delete_owner_scope(self, owner_scope: str):
        self.deleted_owners.append(owner_scope)
        return 1

    async def delete_steam_account_data(
        self,
        platform: str,
        user_id: str,
        *,
        recommendation_owner_scope: str,
    ):
        deleted = await self.delete_steam_account_binding_family(
            platform,
            user_id,
        )
        self.deleted_owners.append(recommendation_owner_scope)
        for binding in deleted:
            owner_scope = f"steam-account:{binding.steam_id64}"
            if owner_scope not in self.deleted_owners:
                self.deleted_owners.append(owner_scope)
        return deleted


class MigratedRebindingCache(MigratedDeletingCache):
    def __init__(self) -> None:
        super().__init__()
        self.deleted_owners: list[str] = []
        self.saved = None

    async def upsert_steam_account_binding(self, binding):
        self.saved = binding
        return binding

    async def delete_owner_scope(self, owner_scope: str):
        self.deleted_owners.append(owner_scope)
        return 1


class PlatformContext:
    def __init__(self, *platform_ids: str) -> None:
        platforms = [PlatformInstance(value) for value in platform_ids]
        self.platform_manager = type(
            "PlatformManager",
            (),
            {"get_insts": lambda _self: platforms},
        )()


class PlatformInstance:
    def __init__(self, platform_id: str) -> None:
        self.platform_id = platform_id

    def meta(self):
        return type(
            "Meta",
            (),
            {"id": self.platform_id, "name": "aiocqhttp"},
        )()


class ReviewSummary:
    total_reviews = 20_000
    positive_ratio = 0.91
    recent_positive_ratio = 0.90


class UnplayedSteamClient:
    def has_web_api_key(self) -> bool:
        return True

    async def get_owned_games(
        self,
        _steam_id64: str,
        *,
        binding_identity=None,
    ):
        del binding_identity
        return [SteamOwnedGame(appid=77, name="Backlog Game", playtime_forever=0)]

    async def get_review_summary(self, _appid: int):
        return ReviewSummary()

    async def get_game_detail(self, _appid: int):
        return GameCandidate(
            title="Backlog Game",
            appid=77,
            app_type="game",
            genres=["Adventure"],
            tags=["Puzzle", "Story Rich"],
        )


class ReasonResponse:
    completion_text = json.dumps(
        {
            "appid": 77,
            "reason": "它以冒险解谜和剧情体验为主。较高好评率与充足评测量说明口碑和知名度都较稳。",
            "evidence_ids": ["gameplay", "reviews", "popularity"],
        },
        ensure_ascii=False,
    )


class ReasonContext:
    async def llm_generate(self, **_kwargs):
        return ReasonResponse()


if __name__ == "__main__":
    unittest.main()
