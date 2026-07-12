from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from astrbot_plugin_steam_game_recommender.services.account_binding import (
    STEAMID64_BASE,
    AccountBindingError,
    parse_account_binding_command,
)
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


if __name__ == "__main__":
    unittest.main()
