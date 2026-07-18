from __future__ import annotations

import asyncio
import json
import sqlite3
import stat
import tempfile
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

from astrbot_plugin_steam_game_recommender.storage import repository as repository_module
from astrbot_plugin_steam_game_recommender.storage.models import SteamAccountBinding
from astrbot_plugin_steam_game_recommender.storage.repository import SQLiteCacheRepository


class SQLiteCacheRepositoryTest(unittest.IsolatedAsyncioTestCase):
    def test_directory_creation_failure_does_not_expose_local_path(self) -> None:
        secret = "/private/provider-token-secret/cache.sqlite3"

        with (
            patch.object(
                Path,
                "mkdir",
                side_effect=NotADirectoryError(secret),
            ),
            self.assertRaises(repository_module.CacheStorageError) as raised,
        ):
            SQLiteCacheRepository(Path(secret))

        self.assertEqual(raised.exception.code, "cache_directory_failure")
        self.assertNotIn(secret, str(raised.exception))

    async def test_legacy_cache_is_rebuilt_without_losing_account_bindings(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = Path(tmpdir) / "cache.sqlite3"
            create_legacy_database(db_path)

            repository = SQLiteCacheRepository(db_path)

            binding = await repository.get_steam_account_binding("qq", "user-1")
            self.assertIsNotNone(binding)
            assert binding is not None
            self.assertEqual(binding.steam_id64, "76561198000000000")
            with sqlite3.connect(db_path) as connection:
                columns = {
                    row[1] for row in connection.execute("PRAGMA table_info(cache)")
                }
                count = connection.execute("SELECT COUNT(*) FROM cache").fetchone()[0]
            self.assertEqual(
                columns,
                {
                    "key",
                    "payload",
                    "created_at",
                    "expires_at",
                    "last_accessed_at",
                    "payload_bytes",
                    "owner_scope",
                },
            )
            self.assertEqual(count, 0)

    async def test_legacy_cache_rebuild_rolls_back_as_one_transaction(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = Path(tmpdir) / "cache.sqlite3"
            create_legacy_database(db_path)

            with patch.object(
                SQLiteCacheRepository,
                "_create_cache_indexes",
                side_effect=sqlite3.OperationalError("synthetic migration failure"),
            ):
                with self.assertRaises(repository_module.CacheStorageError):
                    SQLiteCacheRepository(db_path)

            with sqlite3.connect(db_path) as connection:
                columns = {
                    row[1] for row in connection.execute("PRAGMA table_info(cache)")
                }
                cache_rows = connection.execute(
                    "SELECT key, payload FROM cache"
                ).fetchall()
                binding_count = connection.execute(
                    "SELECT COUNT(*) FROM steam_account_bindings"
                ).fetchone()[0]
            self.assertEqual(columns, {"key", "payload", "created_at"})
            self.assertEqual(cache_rows, [("legacy", "{}")])
            self.assertEqual(binding_count, 1)

    def test_connect_closes_connection_when_permission_setup_fails(self) -> None:
        repository = object.__new__(SQLiteCacheRepository)
        repository.db_path = Path("unused.sqlite3")
        connection = MagicMock()

        with (
            patch.object(repository_module.sqlite3, "connect", return_value=connection),
            patch.object(
                repository,
                "_secure_database_files",
                side_effect=repository_module.CacheStorageError(
                    "cache_permission_failure"
                ),
            ),
            self.assertRaises(repository_module.CacheStorageError),
        ):
            repository._connect()

        connection.close.assert_called_once_with()

    async def test_regular_connections_do_not_reset_journal_mode(self) -> None:
        statements: list[str] = []
        original_connect = sqlite3.connect

        def traced_connect(*args, **kwargs):
            connection = original_connect(*args, **kwargs)
            connection.set_trace_callback(statements.append)
            return connection

        with (
            tempfile.TemporaryDirectory() as tmpdir,
            patch.object(repository_module.sqlite3, "connect", side_effect=traced_connect),
        ):
            repository = SQLiteCacheRepository(Path(tmpdir) / "cache.sqlite3")
            statements.clear()

            await repository.set_json("key", {"value": 1}, ttl_hours=1)
            await repository.get_json("key", ttl_hours=1)

        self.assertFalse(
            any("journal_mode" in statement.casefold() for statement in statements)
        )

    async def test_repository_enables_incremental_page_reclamation(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            repository = SQLiteCacheRepository(Path(tmpdir) / "cache.sqlite3")

            with sqlite3.connect(repository.db_path) as connection:
                auto_vacuum = connection.execute("PRAGMA auto_vacuum").fetchone()[0]

        self.assertEqual(auto_vacuum, 2)

    async def test_single_row_supports_fresh_and_stale_age_windows(self) -> None:
        now = 1_700_000_000.0
        with (
            tempfile.TemporaryDirectory() as tmpdir,
            patch.object(repository_module.time, "time", return_value=now),
        ):
            repository = SQLiteCacheRepository(Path(tmpdir) / "cache.sqlite3")
            await repository.set_json(
                "game:1",
                {"title": "Cached"},
                ttl_seconds=6 * 3600,
            )

            with patch.object(
                repository_module.time,
                "time",
                return_value=now + 2 * 3600,
            ):
                fresh = await repository.get_json("game:1", ttl_hours=1)
                stale = await repository.get_json(
                    "game:1",
                    ttl_hours=1,
                    allow_stale_seconds=5 * 3600,
                )

            self.assertIsNone(fresh)
            self.assertEqual(stale, {"title": "Cached"})
            with sqlite3.connect(repository.db_path) as connection:
                row = connection.execute(
                    "SELECT COUNT(*), expires_at, payload_bytes FROM cache"
                ).fetchone()
            self.assertEqual(row[0], 1)
            self.assertEqual(row[1], now + 6 * 3600)
            self.assertEqual(
                row[2],
                len(json.dumps({"title": "Cached"}, ensure_ascii=False).encode()),
            )

    async def test_expired_rows_are_physically_deleted(self) -> None:
        now = 1_700_000_000.0
        with (
            tempfile.TemporaryDirectory() as tmpdir,
            patch.object(repository_module.time, "time", return_value=now),
        ):
            repository = SQLiteCacheRepository(Path(tmpdir) / "cache.sqlite3")
            await repository.set_json("expired", {"value": 1}, ttl_hours=1)

            with patch.object(
                repository_module.time,
                "time",
                return_value=now + 3601,
            ):
                self.assertIsNone(await repository.get_json("expired", ttl_hours=24))

            with sqlite3.connect(repository.db_path) as connection:
                count = connection.execute("SELECT COUNT(*) FROM cache").fetchone()[0]
            self.assertEqual(count, 0)

    async def test_owner_deletion_only_removes_matching_cache_rows(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            repository = SQLiteCacheRepository(Path(tmpdir) / "cache.sqlite3")
            await repository.set_json(
                "owned:a",
                {"games": [1]},
                ttl_hours=24,
                owner_scope="steam-account:1",
            )
            await repository.set_json(
                "owned:b",
                {"games": [2]},
                ttl_hours=24,
                owner_scope="steam-account:2",
            )
            await repository.set_json(
                "owned:a-child",
                {"games": [4]},
                ttl_hours=24,
                owner_scope="steam-account:1:child",
            )
            await repository.set_json("public", {"value": 3}, ttl_hours=24)

            deleted = await repository.delete_owner_scope("steam-account:1")

            self.assertEqual(deleted, 1)
            self.assertIsNone(await repository.get_json("owned:a", 24))
            self.assertEqual(await repository.get_json("owned:b", 24), {"games": [2]})
            self.assertEqual(
                await repository.get_json("owned:a-child", 24),
                {"games": [4]},
            )
            self.assertEqual(await repository.get_json("public", 24), {"value": 3})

    async def test_personal_cache_write_requires_the_exact_chat_binding(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            repository = SQLiteCacheRepository(Path(tmpdir) / "cache.sqlite3")
            await repository.upsert_steam_account_binding(
                SteamAccountBinding(
                    chat_platform="onebot-instance-a",
                    chat_user_id="user-1",
                    steam_id64="76561198000000000",
                    account_kind="steam_id64",
                    display_value="76561198000000000",
                )
            )

            foreign_write = await repository.set_json_if_steam_account_binding(
                "foreign",
                {"games": [1]},
                chat_platform="onebot-instance-b",
                chat_user_id="user-1",
                steam_id64="76561198000000000",
                ttl_hours=24,
                owner_scope="steam-account:76561198000000000",
            )
            bound_write = await repository.set_json_if_steam_account_binding(
                "bound",
                {"games": [1]},
                chat_platform="onebot-instance-a",
                chat_user_id="user-1",
                steam_id64="76561198000000000",
                ttl_hours=24,
                owner_scope="steam-account:76561198000000000",
            )

            self.assertIs(foreign_write, False)
            self.assertIsNone(await repository.get_json("foreign", 24))
            self.assertIs(bound_write, True)
            self.assertEqual(await repository.get_json("bound", 24), {"games": [1]})

    async def test_hard_limit_evicts_lru_rows_to_low_watermark(self) -> None:
        with (
            tempfile.TemporaryDirectory() as tmpdir,
            patch.object(repository_module, "CACHE_HARD_MAX_ROWS", 3),
            patch.object(repository_module, "CACHE_LOW_MAX_ROWS", 2),
        ):
            repository = SQLiteCacheRepository(Path(tmpdir) / "cache.sqlite3")
            for index in range(4):
                await repository.set_json(
                    f"key:{index}",
                    {"index": index},
                    ttl_hours=24,
                )

            with sqlite3.connect(repository.db_path) as connection:
                keys = {
                    row[0]
                    for row in connection.execute(
                        "SELECT key FROM cache ORDER BY last_accessed_at"
                    )
                }
            self.assertLessEqual(len(keys), 2)
            self.assertIn("key:3", keys)
            self.assertNotIn("key:0", keys)

    async def test_multiple_repository_instances_cannot_exceed_hard_limit(self) -> None:
        with (
            tempfile.TemporaryDirectory() as tmpdir,
            patch.object(repository_module, "CACHE_HARD_MAX_ROWS", 6),
            patch.object(repository_module, "CACHE_LOW_MAX_ROWS", 4),
        ):
            db_path = Path(tmpdir) / "cache.sqlite3"
            repositories = [
                SQLiteCacheRepository(db_path),
                SQLiteCacheRepository(db_path),
            ]

            await asyncio.gather(
                *(
                    repositories[index % 2].set_json(
                        f"concurrent:{index}",
                        {"index": index},
                        ttl_hours=24,
                    )
                    for index in range(30)
                )
            )

            with sqlite3.connect(db_path) as connection:
                count = connection.execute("SELECT COUNT(*) FROM cache").fetchone()[0]
            self.assertLessEqual(count, 6)

    async def test_byte_limit_evicts_to_make_room_for_incoming_payload(self) -> None:
        with (
            tempfile.TemporaryDirectory() as tmpdir,
            patch.object(repository_module, "CACHE_HARD_MAX_BYTES", 100),
            patch.object(repository_module, "CACHE_LOW_MAX_BYTES", 60),
        ):
            repository = SQLiteCacheRepository(Path(tmpdir) / "cache.sqlite3")
            for index in range(4):
                await repository.set_json(
                    f"byte:{index}",
                    {"value": "x" * 24},
                    ttl_hours=24,
                )

            with sqlite3.connect(repository.db_path) as connection:
                total_bytes = connection.execute(
                    "SELECT COALESCE(SUM(payload_bytes), 0) FROM cache"
                ).fetchone()[0]
                keys = {
                    row[0] for row in connection.execute("SELECT key FROM cache")
                }
            self.assertLessEqual(total_bytes, 100)
            self.assertIn("byte:3", keys)

    async def test_cleanup_deletes_all_expired_rows_in_bounded_transactions(self) -> None:
        now = 1_700_000_000.0
        statements: list[str] = []
        original_connect = sqlite3.connect

        def traced_connect(*args, **kwargs):
            connection = original_connect(*args, **kwargs)
            connection.set_trace_callback(statements.append)
            return connection

        with (
            tempfile.TemporaryDirectory() as tmpdir,
            patch.object(repository_module, "CACHE_DELETE_BATCH_SIZE", 2),
            patch.object(repository_module.time, "time", return_value=now),
        ):
            db_path = Path(tmpdir) / "cache.sqlite3"
            repository = SQLiteCacheRepository(db_path)
            with sqlite3.connect(db_path) as connection:
                connection.executemany(
                    """
                    INSERT INTO cache(
                        key, payload, created_at, expires_at,
                        last_accessed_at, payload_bytes, owner_scope
                    ) VALUES (?, '{}', ?, ?, ?, 2, '')
                    """,
                    [(f"expired:{index}", now - 10, now - 1, now - 10) for index in range(5)],
                )

            with (
                patch.object(
                    repository_module.sqlite3,
                    "connect",
                    side_effect=traced_connect,
                ),
                patch.object(repository_module.time, "time", return_value=now + 1),
            ):
                await repository.cleanup()

            with sqlite3.connect(db_path) as connection:
                count = connection.execute("SELECT COUNT(*) FROM cache").fetchone()[0]
            self.assertEqual(count, 0)
            delete_counts: list[int] = []
            current_deletes = 0
            for statement in statements:
                normalized = statement.strip().upper()
                if normalized.startswith("BEGIN"):
                    current_deletes = 0
                elif normalized.startswith("DELETE FROM CACHE"):
                    current_deletes += 1
                elif normalized == "COMMIT":
                    delete_counts.append(current_deletes)
            self.assertTrue(delete_counts)
            self.assertTrue(all(count <= 2 for count in delete_counts))

    async def test_directory_database_and_sidecars_are_private(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            data_dir = Path(tmpdir) / "public-data"
            data_dir.mkdir(mode=0o755)
            db_path = data_dir / "cache.sqlite3"

            repository = SQLiteCacheRepository(db_path)
            await repository.set_json("key", {"value": 1}, ttl_hours=1)

            self.assertEqual(stat.S_IMODE(data_dir.stat().st_mode), 0o700)
            self.assertEqual(stat.S_IMODE(db_path.stat().st_mode), 0o600)
            for suffix in ("-wal", "-shm"):
                sidecar = Path(f"{db_path}{suffix}")
                if await asyncio.to_thread(sidecar.exists):
                    sidecar_mode = await asyncio.to_thread(
                        lambda sidecar=sidecar: stat.S_IMODE(sidecar.stat().st_mode)
                    )
                    self.assertEqual(sidecar_mode, 0o600)

    async def test_invalid_ttl_raises_typed_storage_error(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            repository = SQLiteCacheRepository(Path(tmpdir) / "cache.sqlite3")

            with self.assertRaises(repository_module.CacheStorageError) as raised:
                await repository.set_json("key", {"value": 1}, ttl_hours=float("nan"))

            self.assertEqual(raised.exception.code, "invalid_cache_ttl")

    async def test_account_binding_can_be_deleted_for_unbind(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            repository = SQLiteCacheRepository(Path(tmpdir) / "cache.sqlite3")
            await repository.upsert_steam_account_binding(
                SteamAccountBinding(
                    chat_platform="bot-instance",
                    chat_user_id="user-1",
                    steam_id64="76561198000000000",
                    account_kind="steam_id64",
                    display_value="76561198000000000",
                )
            )

            deleted = await repository.delete_steam_account_binding(
                "bot-instance",
                "user-1",
            )

            self.assertTrue(deleted)
            self.assertIsNone(
                await repository.get_steam_account_binding(
                    "bot-instance",
                    "user-1",
                )
            )


def create_legacy_database(db_path: Path) -> None:
    with sqlite3.connect(db_path) as connection:
        connection.execute(
            """
            CREATE TABLE cache (
                key TEXT PRIMARY KEY,
                payload TEXT NOT NULL,
                created_at REAL NOT NULL
            )
            """
        )
        connection.execute(
            "INSERT INTO cache(key, payload, created_at) VALUES (?, ?, ?)",
            ("legacy", "{}", 1.0),
        )
        connection.execute(
            """
            CREATE TABLE steam_account_bindings (
                chat_platform TEXT NOT NULL,
                chat_user_id TEXT NOT NULL,
                steam_id64 TEXT NOT NULL,
                account_kind TEXT NOT NULL,
                display_value TEXT NOT NULL,
                metadata_json TEXT NOT NULL DEFAULT '{}',
                created_at REAL NOT NULL,
                updated_at REAL NOT NULL,
                PRIMARY KEY (chat_platform, chat_user_id)
            )
            """
        )
        connection.execute(
            """
            INSERT INTO steam_account_bindings(
                chat_platform,
                chat_user_id,
                steam_id64,
                account_kind,
                display_value,
                metadata_json,
                created_at,
                updated_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                "qq",
                "user-1",
                "76561198000000000",
                "steam_id64",
                "76561198000000000",
                "{}",
                1.0,
                1.0,
            ),
        )


if __name__ == "__main__":
    unittest.main()
