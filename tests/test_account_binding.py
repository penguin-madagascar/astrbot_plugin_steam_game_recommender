from __future__ import annotations

import sqlite3
import tempfile
import unittest
from pathlib import Path

from astrbot_plugin_steam_game_recommender.services.account_binding import (
    STEAM_ACCOUNT_ID_MAX,
    STEAMID64_BASE,
    AccountBindingError,
    account_identity_from_event,
    parse_account_binding_command,
    platform_instance_ids_for_name,
    platform_name_from_event,
    recommendation_scope_from_event,
)
from astrbot_plugin_steam_game_recommender.services.recommendation_memory import (
    recommendation_owner_scope,
)
from astrbot_plugin_steam_game_recommender.storage import repository as repository_module
from astrbot_plugin_steam_game_recommender.storage.models import SteamAccountBinding
from astrbot_plugin_steam_game_recommender.storage.repository import SQLiteCacheRepository


class AccountBindingParserTest(unittest.TestCase):
    def test_steam_id64_is_preserved_as_canonical_account_id(self) -> None:
        parsed = parse_account_binding_command("76561198000000000")

        self.assertEqual(parsed.steam_id64, "76561198000000000")
        self.assertEqual(parsed.account_kind, "steam_id64")
        self.assertEqual(parsed.display_value, "76561198000000000")

    def test_steam_friend_code_is_converted_to_steam_id64(self) -> None:
        parsed = parse_account_binding_command("1234-5678")

        self.assertEqual(parsed.steam_id64, str(STEAMID64_BASE + 12345678))
        self.assertEqual(parsed.account_kind, "steam_friend_code")
        self.assertEqual(parsed.display_value, "1234-5678")
        self.assertEqual(parsed.metadata["steam_friend_code"], "12345678")

    def test_steam_friend_code_accepts_spaces_and_hyphens(self) -> None:
        parsed = parse_account_binding_command("12 34-5678")

        self.assertEqual(parsed.steam_id64, str(STEAMID64_BASE + 12345678))
        self.assertEqual(parsed.account_kind, "steam_friend_code")

    def test_rejects_provider_prefix_and_invalid_account(self) -> None:
        with self.assertRaises(AccountBindingError):
            parse_account_binding_command("steam 123456")

        with self.assertRaises(AccountBindingError):
            parse_account_binding_command("not-a-number")

    def test_steam_id64_must_represent_a_personal_account(self) -> None:
        for value in (
            str(STEAMID64_BASE),
            str(STEAMID64_BASE + STEAM_ACCOUNT_ID_MAX + 1),
        ):
            with self.subTest(value=value), self.assertRaises(AccountBindingError):
                parse_account_binding_command(value)

        lower = parse_account_binding_command(str(STEAMID64_BASE + 1))
        upper = parse_account_binding_command(
            str(STEAMID64_BASE + STEAM_ACCOUNT_ID_MAX)
        )

        self.assertEqual(lower.account_kind, "steam_id64")
        self.assertEqual(upper.account_kind, "steam_id64")


class ChatIdentityTest(unittest.TestCase):
    def test_separates_platform_capability_account_and_retry_identities(self) -> None:
        event = IdentityEvent()

        self.assertEqual(platform_name_from_event(event), "aiocqhttp")
        self.assertEqual(
            account_identity_from_event(event),
            ("onebot-instance-2", "user-7"),
        )
        self.assertEqual(
            recommendation_scope_from_event(event),
            ("onebot-instance-2:GroupMessage:group-9", "user-7"),
        )

    def test_account_identity_falls_back_to_umo_platform_instance(self) -> None:
        event = IdentityEvent()
        event.get_platform_id = None

        self.assertEqual(
            account_identity_from_event(event),
            ("onebot-instance-2", "user-7"),
        )

    def test_retry_identity_requires_a_session_origin(self) -> None:
        event = IdentityEvent()
        event.unified_msg_origin = ""

        with self.assertRaisesRegex(AccountBindingError, "会话"):
            recommendation_scope_from_event(event)

    def test_platform_instance_inventory_requires_matching_adapter_type(self) -> None:
        context = PlatformContext(
            FakePlatform("onebot-instance-2", "aiocqhttp"),
            FakePlatform("telegram-instance-1", "telegram"),
        )

        self.assertEqual(
            platform_instance_ids_for_name(context, "aiocqhttp"),
            ["onebot-instance-2"],
        )
        self.assertEqual(
            platform_instance_ids_for_name(object(), "aiocqhttp"),
            None,
        )


class IdentityEvent:
    unified_msg_origin = "onebot-instance-2:GroupMessage:group-9"

    def get_platform_name(self) -> str:
        return "aiocqhttp"

    def get_platform_id(self) -> str:
        return "onebot-instance-2"

    def get_sender_id(self) -> str:
        return "user-7"


class FakePlatform:
    def __init__(self, platform_id: str, name: str) -> None:
        self.platform_id = platform_id
        self.name = name

    def meta(self):
        return type("Meta", (), {"id": self.platform_id, "name": self.name})()


class PlatformContext:
    def __init__(self, *platforms: FakePlatform) -> None:
        self.platform_manager = type(
            "PlatformManager",
            (),
            {"get_insts": lambda _self: list(platforms)},
        )()


class AccountBindingRepositoryTest(unittest.IsolatedAsyncioTestCase):
    async def test_upsert_and_get_account_binding(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            repo = SQLiteCacheRepository(Path(tmpdir) / "cache.sqlite3")

            saved = await repo.upsert_steam_account_binding(
                SteamAccountBinding(
                    chat_platform="qq",
                    chat_user_id="user-1",
                    steam_id64="76561198000000000",
                    account_kind="steam_id64",
                    display_value="76561198000000000",
                    metadata={"source": "test"},
                )
            )
            loaded = await repo.get_steam_account_binding("qq", "user-1")

            self.assertIsNotNone(loaded)
            self.assertEqual(loaded, saved)
            self.assertEqual(loaded.steam_id64, "76561198000000000")
            self.assertEqual(loaded.metadata["source"], "test")
            self.assertIsNotNone(loaded.created_at)
            self.assertIsNotNone(loaded.updated_at)

    async def test_upsert_replaces_the_users_existing_steam_binding(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            repo = SQLiteCacheRepository(Path(tmpdir) / "cache.sqlite3")
            await repo.upsert_steam_account_binding(
                SteamAccountBinding(
                    chat_platform="qq",
                    chat_user_id="user-1",
                    steam_id64="76561198000000000",
                    account_kind="steam_id64",
                    display_value="76561198000000000",
                )
            )
            await repo.upsert_steam_account_binding(
                SteamAccountBinding(
                    chat_platform="qq",
                    chat_user_id="user-1",
                    steam_id64="76561198012345678",
                    account_kind="steam_friend_code",
                    display_value="5207922950",
                )
            )

            steam = await repo.get_steam_account_binding("qq", "user-1")

            self.assertIsNotNone(steam)
            assert steam is not None
            self.assertEqual(steam.steam_id64, "76561198012345678")

    async def test_lazy_migration_preserves_a_marked_legacy_binding(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            repo = SQLiteCacheRepository(Path(tmpdir) / "cache.sqlite3")
            await repo.upsert_steam_account_binding(
                SteamAccountBinding(
                    chat_platform="aiocqhttp",
                    chat_user_id="user-1",
                    steam_id64="76561198000000000",
                    account_kind="steam_id64",
                    display_value="76561198000000000",
                )
            )

            migrated = await repo.migrate_steam_account_binding(
                "aiocqhttp",
                "onebot-instance-2",
                "user-1",
            )
            legacy = await repo.get_steam_account_binding("aiocqhttp", "user-1")
            current = await repo.get_steam_account_binding(
                "onebot-instance-2",
                "user-1",
            )

            self.assertEqual(migrated, current)
            self.assertIsNotNone(legacy)
            assert legacy is not None
            self.assertEqual(
                legacy.metadata["migrated_to_platform_instance"],
                "onebot-instance-2",
            )
            self.assertEqual(
                current.metadata["migrated_from_platform"],
                "aiocqhttp",
            )

    async def test_lazy_migration_cannot_take_over_another_instance_claim(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            repo = SQLiteCacheRepository(Path(tmpdir) / "cache.sqlite3")
            await repo.upsert_steam_account_binding(
                SteamAccountBinding(
                    chat_platform="aiocqhttp",
                    chat_user_id="user-1",
                    steam_id64="76561198000000000",
                    account_kind="steam_id64",
                    display_value="76561198000000000",
                    metadata={
                        "migrated_to_platform_instance": "onebot-instance-a"
                    },
                )
            )

            migrated = await repo.migrate_steam_account_binding(
                "aiocqhttp",
                "onebot-instance-b",
                "user-1",
            )

            legacy = await repo.get_steam_account_binding("aiocqhttp", "user-1")
            self.assertIsNone(migrated)
            self.assertIsNotNone(legacy)
            assert legacy is not None
            self.assertEqual(
                legacy.metadata["migrated_to_platform_instance"],
                "onebot-instance-a",
            )
            self.assertIsNone(
                await repo.get_steam_account_binding(
                    "onebot-instance-b",
                    "user-1",
                )
            )

    async def test_explicit_rebind_claims_unowned_legacy_for_atomic_deletion(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            repo = SQLiteCacheRepository(Path(tmpdir) / "cache.sqlite3")
            await repo.upsert_steam_account_binding(
                SteamAccountBinding(
                    chat_platform="aiocqhttp",
                    chat_user_id="user-1",
                    steam_id64="76561198000000000",
                    account_kind="steam_id64",
                    display_value="76561198000000000",
                )
            )
            saved = await repo.upsert_steam_account_binding_claiming_legacy(
                SteamAccountBinding(
                    chat_platform="onebot-instance-b",
                    chat_user_id="user-1",
                    steam_id64="76561198000000001",
                    account_kind="steam_id64",
                    display_value="76561198000000001",
                ),
                legacy_platform="aiocqhttp",
            )
            owner = recommendation_owner_scope("onebot-instance-b", "user-1")
            for key, owner_scope in (
                ("legacy-library", "steam-account:76561198000000000"),
                ("current-library", "steam-account:76561198000000001"),
                ("recommendation", owner),
                ("unrelated", "steam-account:76561198000000002"),
            ):
                await repo.set_json(
                    key,
                    {"key": key},
                    ttl_hours=24,
                    owner_scope=owner_scope,
                )

            deleted = await repo.delete_steam_account_data(
                "onebot-instance-b",
                "user-1",
                recommendation_owner_scope=owner,
            )

            self.assertEqual(
                saved.metadata["migrated_from_platform"],
                "aiocqhttp",
            )
            self.assertEqual(
                [binding.chat_platform for binding in deleted],
                ["onebot-instance-b", "aiocqhttp"],
            )
            self.assertIsNone(
                await repo.get_steam_account_binding("onebot-instance-b", "user-1")
            )
            self.assertIsNone(
                await repo.get_steam_account_binding("aiocqhttp", "user-1")
            )
            for key in ("legacy-library", "current-library", "recommendation"):
                self.assertIsNone(await repo.get_json(key, 24))
            self.assertEqual(
                await repo.get_json("unrelated", 24),
                {"key": "unrelated"},
            )

    async def test_explicit_rebind_does_not_claim_another_instance_legacy(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            repo = SQLiteCacheRepository(Path(tmpdir) / "cache.sqlite3")
            await repo.upsert_steam_account_binding(
                SteamAccountBinding(
                    chat_platform="aiocqhttp",
                    chat_user_id="user-1",
                    steam_id64="76561198000000000",
                    account_kind="steam_id64",
                    display_value="76561198000000000",
                    metadata={
                        "migrated_to_platform_instance": "onebot-instance-a"
                    },
                )
            )

            saved = await repo.upsert_steam_account_binding_claiming_legacy(
                SteamAccountBinding(
                    chat_platform="onebot-instance-b",
                    chat_user_id="user-1",
                    steam_id64="76561198000000001",
                    account_kind="steam_id64",
                    display_value="76561198000000001",
                ),
                legacy_platform="aiocqhttp",
            )

            legacy = await repo.get_steam_account_binding("aiocqhttp", "user-1")
            self.assertNotIn("migrated_from_platform", saved.metadata)
            self.assertIsNotNone(legacy)
            assert legacy is not None
            self.assertEqual(
                legacy.metadata["migrated_to_platform_instance"],
                "onebot-instance-a",
            )

    async def test_binding_family_delete_atomically_removes_migrated_pair(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            repo = SQLiteCacheRepository(Path(tmpdir) / "cache.sqlite3")
            await repo.upsert_steam_account_binding(
                SteamAccountBinding(
                    chat_platform="aiocqhttp",
                    chat_user_id="user-1",
                    steam_id64="76561198000000000",
                    account_kind="steam_id64",
                    display_value="76561198000000000",
                )
            )
            await repo.migrate_steam_account_binding(
                "aiocqhttp",
                "onebot-instance-2",
                "user-1",
            )

            deleted = await repo.delete_steam_account_binding_family(
                "onebot-instance-2",
                "user-1",
            )

            self.assertEqual(
                [binding.chat_platform for binding in deleted],
                ["onebot-instance-2", "aiocqhttp"],
            )
            self.assertTrue(
                all(
                    binding.steam_id64 == "76561198000000000"
                    for binding in deleted
                )
            )
            self.assertIsNone(
                await repo.get_steam_account_binding("onebot-instance-2", "user-1")
            )
            self.assertIsNone(
                await repo.get_steam_account_binding("aiocqhttp", "user-1")
            )

    async def test_binding_family_delete_keeps_unrelated_legacy_row(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            repo = SQLiteCacheRepository(Path(tmpdir) / "cache.sqlite3")
            await repo.upsert_steam_account_binding(
                SteamAccountBinding(
                    chat_platform="aiocqhttp",
                    chat_user_id="user-1",
                    steam_id64="76561198000000001",
                    account_kind="steam_id64",
                    display_value="76561198000000001",
                )
            )
            await repo.upsert_steam_account_binding(
                SteamAccountBinding(
                    chat_platform="onebot-instance-2",
                    chat_user_id="user-1",
                    steam_id64="76561198000000000",
                    account_kind="steam_id64",
                    display_value="76561198000000000",
                    metadata={"migrated_from_platform": "aiocqhttp"},
                )
            )

            deleted = await repo.delete_steam_account_binding_family(
                "onebot-instance-2",
                "user-1",
            )

            self.assertEqual(
                [binding.chat_platform for binding in deleted],
                ["onebot-instance-2"],
            )
            self.assertIsNotNone(
                await repo.get_steam_account_binding("aiocqhttp", "user-1")
            )

    async def test_binding_family_delete_removes_rebound_migrated_pair(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            repo = SQLiteCacheRepository(Path(tmpdir) / "cache.sqlite3")
            await repo.upsert_steam_account_binding(
                SteamAccountBinding(
                    chat_platform="aiocqhttp",
                    chat_user_id="user-1",
                    steam_id64="76561198000000000",
                    account_kind="steam_id64",
                    display_value="76561198000000000",
                    metadata={
                        "migrated_to_platform_instance": "onebot-instance-2"
                    },
                )
            )
            await repo.upsert_steam_account_binding(
                SteamAccountBinding(
                    chat_platform="onebot-instance-2",
                    chat_user_id="user-1",
                    steam_id64="76561198000000001",
                    account_kind="steam_id64",
                    display_value="76561198000000001",
                    metadata={"migrated_from_platform": "aiocqhttp"},
                )
            )

            deleted = await repo.delete_steam_account_binding_family(
                "onebot-instance-2",
                "user-1",
            )

            self.assertEqual(
                [binding.chat_platform for binding in deleted],
                ["onebot-instance-2", "aiocqhttp"],
            )

    async def test_binding_family_delete_rolls_back_when_any_delete_fails(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = Path(tmpdir) / "cache.sqlite3"
            repo = SQLiteCacheRepository(db_path)
            await repo.upsert_steam_account_binding(
                SteamAccountBinding(
                    chat_platform="aiocqhttp",
                    chat_user_id="user-1",
                    steam_id64="76561198000000000",
                    account_kind="steam_id64",
                    display_value="76561198000000000",
                )
            )
            await repo.migrate_steam_account_binding(
                "aiocqhttp",
                "onebot-instance-2",
                "user-1",
            )
            with sqlite3.connect(db_path) as connection:
                connection.execute(
                    """
                    CREATE TRIGGER reject_legacy_binding_delete
                    BEFORE DELETE ON steam_account_bindings
                    WHEN OLD.chat_platform = 'aiocqhttp'
                    BEGIN
                        SELECT RAISE(ABORT, 'synthetic delete failure');
                    END
                    """
                )

            with self.assertRaises(repository_module.CacheStorageError):
                await repo.delete_steam_account_binding_family(
                    "onebot-instance-2",
                    "user-1",
                )

            self.assertIsNotNone(
                await repo.get_steam_account_binding("onebot-instance-2", "user-1")
            )
            self.assertIsNotNone(
                await repo.get_steam_account_binding("aiocqhttp", "user-1")
            )


if __name__ == "__main__":
    unittest.main()
