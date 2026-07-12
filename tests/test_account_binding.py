from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from astrbot_plugin_game_recommender.services.account_binding import (
    STEAMID64_BASE,
    AccountBindingError,
    parse_account_binding_command,
)
from astrbot_plugin_game_recommender.storage.models import AccountBinding
from astrbot_plugin_game_recommender.storage.repository import SQLiteCacheRepository


class AccountBindingParserTest(unittest.TestCase):
    def test_steam_id64_is_preserved_as_canonical_account_id(self) -> None:
        parsed = parse_account_binding_command("steam 76561198000000000")

        self.assertEqual(parsed.provider, "steam")
        self.assertEqual(parsed.account_id, "76561198000000000")
        self.assertEqual(parsed.account_kind, "steam_id64")
        self.assertEqual(parsed.display_value, "76561198000000000")

    def test_steam_friend_code_is_converted_to_steam_id64(self) -> None:
        parsed = parse_account_binding_command("1234-5678")

        self.assertEqual(parsed.provider, "steam")
        self.assertEqual(parsed.account_id, str(STEAMID64_BASE + 12345678))
        self.assertEqual(parsed.account_kind, "steam_friend_code")
        self.assertEqual(parsed.display_value, "1234-5678")
        self.assertEqual(parsed.metadata["steam_friend_code"], "12345678")

    def test_steam_friend_code_accepts_spaces_and_hyphens(self) -> None:
        parsed = parse_account_binding_command("steam 12 34-5678")

        self.assertEqual(parsed.account_id, str(STEAMID64_BASE + 12345678))
        self.assertEqual(parsed.account_kind, "steam_friend_code")

    def test_rejects_unsupported_provider_and_invalid_account(self) -> None:
        with self.assertRaises(AccountBindingError):
            parse_account_binding_command("xbox 123456")

        with self.assertRaises(AccountBindingError):
            parse_account_binding_command("steam not-a-number")


class AccountBindingRepositoryTest(unittest.IsolatedAsyncioTestCase):
    async def test_upsert_and_get_account_binding(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            repo = SQLiteCacheRepository(Path(tmpdir) / "cache.sqlite3")

            saved = await repo.upsert_account_binding(
                AccountBinding(
                    chat_platform="qq",
                    chat_user_id="user-1",
                    provider="steam",
                    account_id="76561198000000000",
                    account_kind="steam_id64",
                    display_value="76561198000000000",
                    metadata={"source": "test"},
                )
            )
            loaded = await repo.get_account_binding("qq", "user-1", "steam")

            self.assertIsNotNone(loaded)
            self.assertEqual(loaded, saved)
            self.assertEqual(loaded.account_id, "76561198000000000")
            self.assertEqual(loaded.metadata["source"], "test")
            self.assertIsNotNone(loaded.created_at)
            self.assertIsNotNone(loaded.updated_at)

    async def test_upsert_replaces_same_provider_and_lists_multiple_providers(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            repo = SQLiteCacheRepository(Path(tmpdir) / "cache.sqlite3")
            await repo.upsert_account_binding(
                AccountBinding(
                    chat_platform="qq",
                    chat_user_id="user-1",
                    provider="steam",
                    account_id="76561198000000000",
                    account_kind="steam_id64",
                    display_value="76561198000000000",
                )
            )
            await repo.upsert_account_binding(
                AccountBinding(
                    chat_platform="qq",
                    chat_user_id="user-1",
                    provider="steam",
                    account_id="76561198012345678",
                    account_kind="steam_friend_code",
                    display_value="5207922950",
                )
            )
            await repo.upsert_account_binding(
                AccountBinding(
                    chat_platform="qq",
                    chat_user_id="user-1",
                    provider="future_provider",
                    account_id="future-id",
                    account_kind="future_kind",
                    display_value="future-id",
                )
            )

            bindings = await repo.list_account_bindings("qq", "user-1")

            self.assertEqual(
                [binding.provider for binding in bindings],
                ["future_provider", "steam"],
            )
            steam = await repo.get_account_binding("qq", "user-1", "steam")
            self.assertEqual(steam.account_id, "76561198012345678")


if __name__ == "__main__":
    unittest.main()
