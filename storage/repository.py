from __future__ import annotations

import asyncio
import json
import math
import os
import sqlite3
import threading
import time
from pathlib import Path
from typing import Any

from .models import SteamAccountBinding

CACHE_DEFAULT_TTL_HOURS = 24.0
CACHE_HARD_MAX_BYTES = 256 * 1024 * 1024
CACHE_LOW_MAX_BYTES = 192 * 1024 * 1024
CACHE_HARD_MAX_ROWS = 10_000
CACHE_LOW_MAX_ROWS = 7_500
CACHE_CLEANUP_WRITE_INTERVAL = 100
CACHE_CLEANUP_TIME_INTERVAL_SECONDS = 5 * 60
CACHE_DELETE_BATCH_SIZE = 500

_CACHE_COLUMNS = {
    "key",
    "payload",
    "created_at",
    "expires_at",
    "last_accessed_at",
    "payload_bytes",
    "owner_scope",
}


class CacheStorageError(RuntimeError):
    """Stable cache failure that does not expose SQLite or payload details."""

    def __init__(self, code: str = "cache_storage_failure") -> None:
        self.code = code
        super().__init__(code)


class SQLiteCacheRepository:
    def __init__(self, db_path: Path) -> None:
        self.db_path = db_path
        self._lock = threading.RLock()
        self._successful_writes = 0
        self._last_cleanup_at = 0.0
        self._secure_directory()
        self._initialize()

    def _secure_directory(self) -> None:
        self.db_path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
        try:
            os.chmod(self.db_path.parent, 0o700)
        except OSError:
            raise CacheStorageError("cache_permission_failure") from None

    def _secure_database_files(self) -> None:
        for path in (
            self.db_path,
            Path(f"{self.db_path}-wal"),
            Path(f"{self.db_path}-shm"),
        ):
            if not path.exists():
                continue
            try:
                os.chmod(path, 0o600)
            except OSError:
                raise CacheStorageError("cache_permission_failure") from None

    def _connect(self) -> sqlite3.Connection:
        connection: sqlite3.Connection | None = None
        try:
            connection = sqlite3.connect(self.db_path, timeout=5.0)
            connection.execute("PRAGMA busy_timeout = 5000")
            self._secure_database_files()
            return connection
        except CacheStorageError:
            if connection is not None:
                connection.close()
            raise
        except sqlite3.Error:
            if connection is not None:
                connection.close()
            raise CacheStorageError() from None

    def _initialize(self) -> None:
        with self._lock:
            connection = self._connect()
            try:
                self._enable_incremental_auto_vacuum(connection)
                connection.execute("PRAGMA journal_mode = WAL")
                self._secure_database_files()
                connection.execute("BEGIN IMMEDIATE")
                self._create_binding_table(connection)
                self._migrate_cache_table(connection)
                self._create_cache_indexes(connection)
                connection.commit()
            except sqlite3.Error:
                connection.rollback()
                raise CacheStorageError("cache_initialization_failure") from None
            except BaseException:
                connection.rollback()
                raise
            finally:
                connection.close()
                self._secure_database_files()
            self._cleanup_sync(force=True)

    @staticmethod
    def _enable_incremental_auto_vacuum(connection: sqlite3.Connection) -> None:
        try:
            mode = int(connection.execute("PRAGMA auto_vacuum").fetchone()[0])
            if mode == 2:
                return
            connection.execute("PRAGMA auto_vacuum = INCREMENTAL")
            connection.execute("VACUUM")
        except sqlite3.Error:
            # Reclamation is best-effort; SQLite can continue reusing free pages.
            return

    @staticmethod
    def _create_binding_table(connection: sqlite3.Connection) -> None:
        connection.execute(
            """
            CREATE TABLE IF NOT EXISTS steam_account_bindings (
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

    @staticmethod
    def _migrate_cache_table(connection: sqlite3.Connection) -> None:
        exists = connection.execute(
            "SELECT 1 FROM sqlite_master WHERE type = 'table' AND name = 'cache'"
        ).fetchone()
        if exists:
            columns = {
                row[1] for row in connection.execute("PRAGMA table_info(cache)")
            }
            if columns != _CACHE_COLUMNS:
                connection.execute("DROP TABLE cache")
        connection.execute(
            """
            CREATE TABLE IF NOT EXISTS cache (
                key TEXT PRIMARY KEY,
                payload TEXT NOT NULL,
                created_at REAL NOT NULL,
                expires_at REAL NOT NULL DEFAULT 0,
                last_accessed_at REAL NOT NULL DEFAULT 0,
                payload_bytes INTEGER NOT NULL DEFAULT 0,
                owner_scope TEXT NOT NULL DEFAULT ''
            )
            """
        )

    @staticmethod
    def _create_cache_indexes(connection: sqlite3.Connection) -> None:
        connection.execute(
            "CREATE INDEX IF NOT EXISTS idx_cache_expires_at ON cache(expires_at)"
        )
        connection.execute(
            """
            CREATE INDEX IF NOT EXISTS idx_cache_last_accessed_at
            ON cache(last_accessed_at)
            """
        )
        connection.execute(
            "CREATE INDEX IF NOT EXISTS idx_cache_owner_scope ON cache(owner_scope)"
        )

    async def get_json(
        self,
        key: str,
        ttl_hours: int | float = CACHE_DEFAULT_TTL_HOURS,
        *,
        allow_stale_seconds: int | float = 0,
    ) -> Any | None:
        return await asyncio.to_thread(
            self._get_json_sync,
            key,
            ttl_hours,
            allow_stale_seconds,
        )

    def _get_json_sync(
        self,
        key: str,
        ttl_hours: int | float,
        allow_stale_seconds: int | float,
    ) -> Any | None:
        ttl = _validated_ttl_hours(ttl_hours)
        stale_seconds = _validated_non_negative_seconds(allow_stale_seconds)
        now = time.time()
        with self._lock:
            connection = self._connect()
            try:
                row = connection.execute(
                    "SELECT payload, created_at, expires_at FROM cache WHERE key = ?",
                    (key,),
                ).fetchone()
                if not row:
                    return None
                payload, created_at, expires_at = row
                if now > float(expires_at):
                    with connection:
                        connection.execute("DELETE FROM cache WHERE key = ?", (key,))
                    return None
                if now - float(created_at) > ttl * 3600 + stale_seconds:
                    return None
                try:
                    value = json.loads(payload)
                except (TypeError, json.JSONDecodeError):
                    with connection:
                        connection.execute("DELETE FROM cache WHERE key = ?", (key,))
                    return None
                with connection:
                    connection.execute(
                        "UPDATE cache SET last_accessed_at = ? WHERE key = ?",
                        (now, key),
                    )
                return value
            except sqlite3.Error:
                raise CacheStorageError("cache_read_failure") from None
            finally:
                connection.close()
                self._secure_database_files()

    async def set_json(
        self,
        key: str,
        payload: Any,
        ttl_hours: int | float = CACHE_DEFAULT_TTL_HOURS,
        *,
        ttl_seconds: int | float | None = None,
        owner_scope: str = "",
    ) -> None:
        resolved_ttl_hours = (
            _validated_positive_seconds(ttl_seconds) / 3600
            if ttl_seconds is not None
            else ttl_hours
        )
        await asyncio.to_thread(
            self._set_json_sync,
            key,
            payload,
            resolved_ttl_hours,
            owner_scope,
        )

    def _set_json_sync(
        self,
        key: str,
        payload: Any,
        ttl_hours: int | float,
        owner_scope: str,
    ) -> None:
        ttl = _validated_ttl_hours(ttl_hours)
        try:
            encoded = json.dumps(payload, ensure_ascii=False)
        except (TypeError, ValueError, OverflowError):
            raise CacheStorageError("cache_serialization_failure") from None
        payload_bytes = len(encoded.encode("utf-8"))
        if payload_bytes > CACHE_HARD_MAX_BYTES:
            raise CacheStorageError("cache_payload_too_large")
        now = time.time()
        expires_at = now + ttl * 3600
        resolved_owner = str(owner_scope or "").strip()

        with self._lock:
            connection = self._connect()
            try:
                self._write_with_capacity(
                    connection,
                    key=key,
                    encoded=encoded,
                    payload_bytes=payload_bytes,
                    owner_scope=resolved_owner,
                    now=now,
                    expires_at=expires_at,
                )
                self._successful_writes += 1
                should_cleanup = (
                    self._successful_writes % CACHE_CLEANUP_WRITE_INTERVAL == 0
                    or now - self._last_cleanup_at
                    >= CACHE_CLEANUP_TIME_INTERVAL_SECONDS
                )
                if should_cleanup:
                    self._cleanup_connection(connection, now=now, enforce_limits=True)
            except CacheStorageError:
                raise
            except sqlite3.Error:
                connection.rollback()
                raise CacheStorageError("cache_write_failure") from None
            finally:
                connection.close()
                self._secure_database_files()

    def _write_with_capacity(
        self,
        connection: sqlite3.Connection,
        *,
        key: str,
        encoded: str,
        payload_bytes: int,
        owner_scope: str,
        now: float,
        expires_at: float,
    ) -> None:
        while True:
            connection.execute("BEGIN IMMEDIATE")
            try:
                existing = connection.execute(
                    "SELECT payload_bytes FROM cache WHERE key = ?",
                    (key,),
                ).fetchone()
                current_rows, current_bytes = self._cache_totals(connection)
                existing_bytes = int(existing[0]) if existing else 0
                projected_rows = current_rows + (0 if existing else 1)
                projected_bytes = current_bytes - existing_bytes + payload_bytes
                if (
                    projected_rows <= CACHE_HARD_MAX_ROWS
                    and projected_bytes <= CACHE_HARD_MAX_BYTES
                ):
                    connection.execute(
                        """
                        INSERT INTO cache(
                            key,
                            payload,
                            created_at,
                            expires_at,
                            last_accessed_at,
                            payload_bytes,
                            owner_scope
                        )
                        VALUES (?, ?, ?, ?, ?, ?, ?)
                        ON CONFLICT(key) DO UPDATE SET
                            payload = excluded.payload,
                            created_at = excluded.created_at,
                            expires_at = excluded.expires_at,
                            last_accessed_at = excluded.last_accessed_at,
                            payload_bytes = excluded.payload_bytes,
                            owner_scope = excluded.owner_scope
                        """,
                        (
                            key,
                            encoded,
                            now,
                            expires_at,
                            now,
                            payload_bytes,
                            owner_scope,
                        ),
                    )
                    connection.commit()
                    return

                deleted = self._delete_expired_batch(connection, now)
                if not deleted:
                    target_rows = max(
                        CACHE_LOW_MAX_ROWS - (0 if existing else 1),
                        0,
                    )
                    target_bytes = max(
                        CACHE_LOW_MAX_BYTES - (payload_bytes - existing_bytes),
                        0,
                    )
                    deleted = self._evict_lru_batch(
                        connection,
                        target_rows=target_rows,
                        target_bytes=target_bytes,
                        protected_key=key if existing else None,
                    )
                if not deleted:
                    connection.rollback()
                    raise CacheStorageError("cache_capacity_failure")
                connection.commit()
            except BaseException:
                if connection.in_transaction:
                    connection.rollback()
                raise

    @staticmethod
    def _cache_totals(connection: sqlite3.Connection) -> tuple[int, int]:
        row = connection.execute(
            "SELECT COUNT(*), COALESCE(SUM(payload_bytes), 0) FROM cache"
        ).fetchone()
        return (int(row[0]), int(row[1])) if row else (0, 0)

    async def delete_cache_owner(self, owner_scope: str) -> int:
        return await asyncio.to_thread(self._delete_cache_owner_sync, owner_scope)

    async def delete_owner_scope(self, owner_scope: str) -> int:
        return await self.delete_cache_owner(owner_scope)

    def _delete_cache_owner_sync(self, owner_scope: str) -> int:
        resolved_owner = str(owner_scope or "").strip()
        if not resolved_owner:
            return 0
        with self._lock:
            connection = self._connect()
            try:
                deleted_total = 0
                while True:
                    connection.execute("BEGIN IMMEDIATE")
                    keys = connection.execute(
                        """
                        SELECT key FROM cache
                        WHERE owner_scope = ?
                        LIMIT ?
                        """,
                        (resolved_owner, CACHE_DELETE_BATCH_SIZE),
                    ).fetchall()
                    if not keys:
                        connection.commit()
                        return deleted_total
                    connection.executemany(
                        "DELETE FROM cache WHERE key = ?",
                        keys,
                    )
                    connection.commit()
                    deleted_total += len(keys)
            except sqlite3.Error:
                if connection.in_transaction:
                    connection.rollback()
                raise CacheStorageError("cache_owner_delete_failure") from None
            finally:
                connection.close()
                self._secure_database_files()

    async def cleanup(self) -> None:
        await asyncio.to_thread(self._cleanup_sync, force=True)

    def _cleanup_sync(self, *, force: bool = False) -> None:
        now = time.time()
        if not force and (
            now - self._last_cleanup_at < CACHE_CLEANUP_TIME_INTERVAL_SECONDS
        ):
            return
        with self._lock:
            connection = self._connect()
            try:
                self._cleanup_connection(connection, now=now, enforce_limits=True)
            except sqlite3.Error:
                raise CacheStorageError("cache_cleanup_failure") from None
            finally:
                connection.close()
                self._secure_database_files()

    def _cleanup_connection(
        self,
        connection: sqlite3.Connection,
        *,
        now: float,
        enforce_limits: bool,
    ) -> None:
        while True:
            connection.execute("BEGIN IMMEDIATE")
            try:
                deleted = self._delete_expired_batch(connection, now)
                connection.commit()
            except BaseException:
                if connection.in_transaction:
                    connection.rollback()
                raise
            if deleted < CACHE_DELETE_BATCH_SIZE:
                break

        if enforce_limits:
            while True:
                connection.execute("BEGIN IMMEDIATE")
                try:
                    rows, payload_bytes = self._cache_totals(connection)
                    if (
                        rows <= CACHE_HARD_MAX_ROWS
                        and payload_bytes <= CACHE_HARD_MAX_BYTES
                    ):
                        connection.commit()
                        break
                    deleted = self._evict_lru_batch(
                        connection,
                        target_rows=CACHE_LOW_MAX_ROWS,
                        target_bytes=CACHE_LOW_MAX_BYTES,
                    )
                    connection.commit()
                except BaseException:
                    if connection.in_transaction:
                        connection.rollback()
                    raise
                if not deleted:
                    break
        self._last_cleanup_at = now
        self._maybe_compact(connection)

    @staticmethod
    def _delete_expired_batch(
        connection: sqlite3.Connection,
        now: float,
    ) -> int:
        keys = connection.execute(
            "SELECT key FROM cache WHERE expires_at < ? LIMIT ?",
            (now, CACHE_DELETE_BATCH_SIZE),
        ).fetchall()
        if not keys:
            return 0
        connection.executemany(
            "DELETE FROM cache WHERE key = ?",
            keys,
        )
        return len(keys)

    def _evict_lru_batch(
        self,
        connection: sqlite3.Connection,
        *,
        target_rows: int,
        target_bytes: int,
        protected_key: str | None = None,
    ) -> int:
        rows, payload_bytes = self._cache_totals(connection)
        if rows <= target_rows and payload_bytes <= target_bytes:
            return 0
        where = "WHERE key != ?" if protected_key is not None else ""
        params: tuple[Any, ...]
        if protected_key is None:
            params = (CACHE_DELETE_BATCH_SIZE,)
        else:
            params = (protected_key, CACHE_DELETE_BATCH_SIZE)
        candidates = connection.execute(
            f"""
            SELECT key, payload_bytes FROM cache
            {where}
            ORDER BY last_accessed_at ASC, created_at ASC, key ASC
            LIMIT ?
            """,
            params,
        ).fetchall()
        if not candidates:
            return 0

        keys: list[tuple[str]] = []
        remaining_rows = rows
        remaining_bytes = payload_bytes
        for candidate_key, candidate_bytes in candidates:
            keys.append((str(candidate_key),))
            remaining_rows -= 1
            remaining_bytes -= int(candidate_bytes)
            if remaining_rows <= target_rows and remaining_bytes <= target_bytes:
                break
        connection.executemany("DELETE FROM cache WHERE key = ?", keys)
        return len(keys)

    @staticmethod
    def _maybe_compact(connection: sqlite3.Connection) -> None:
        page_count = int(connection.execute("PRAGMA page_count").fetchone()[0])
        free_pages = int(connection.execute("PRAGMA freelist_count").fetchone()[0])
        if page_count <= 0 or free_pages / page_count <= 0.25:
            return
        try:
            connection.execute("PRAGMA incremental_vacuum")
        except sqlite3.Error:
            return

    async def upsert_steam_account_binding(
        self,
        binding: SteamAccountBinding,
    ) -> SteamAccountBinding:
        return await asyncio.to_thread(self._upsert_steam_account_binding_sync, binding)

    def _upsert_steam_account_binding_sync(
        self,
        binding: SteamAccountBinding,
    ) -> SteamAccountBinding:
        now = time.time()
        with self._lock:
            connection = self._connect()
            try:
                with connection:
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
                        )
                        VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                        ON CONFLICT(chat_platform, chat_user_id) DO UPDATE SET
                            steam_id64 = excluded.steam_id64,
                            account_kind = excluded.account_kind,
                            display_value = excluded.display_value,
                            metadata_json = excluded.metadata_json,
                            updated_at = excluded.updated_at
                        """,
                        (
                            binding.chat_platform,
                            binding.chat_user_id,
                            binding.steam_id64,
                            binding.account_kind,
                            binding.display_value,
                            json.dumps(binding.metadata, ensure_ascii=False),
                            binding.created_at or now,
                            now,
                        ),
                    )
            except (sqlite3.Error, TypeError, ValueError, OverflowError):
                raise CacheStorageError("account_binding_write_failure") from None
            finally:
                connection.close()
                self._secure_database_files()
        saved = self._get_steam_account_binding_sync(
            binding.chat_platform,
            binding.chat_user_id,
        )
        if saved is None:
            raise CacheStorageError("account_binding_write_failure")
        return saved

    async def get_steam_account_binding(
        self,
        chat_platform: str,
        chat_user_id: str,
    ) -> SteamAccountBinding | None:
        return await asyncio.to_thread(
            self._get_steam_account_binding_sync,
            chat_platform,
            chat_user_id,
        )

    async def migrate_steam_account_binding(
        self,
        legacy_platform: str,
        target_platform_instance: str,
        chat_user_id: str,
    ) -> SteamAccountBinding | None:
        return await asyncio.to_thread(
            self._migrate_steam_account_binding_sync,
            legacy_platform,
            target_platform_instance,
            chat_user_id,
        )

    def _migrate_steam_account_binding_sync(
        self,
        legacy_platform: str,
        target_platform_instance: str,
        chat_user_id: str,
    ) -> SteamAccountBinding | None:
        source_platform = legacy_platform or "default"
        target_platform = target_platform_instance or "default"
        if source_platform == target_platform:
            return self._get_steam_account_binding_sync(
                source_platform,
                chat_user_id,
            )

        now = time.time()
        with self._lock:
            connection = self._connect()
            try:
                connection.execute("BEGIN IMMEDIATE")
                source_row = connection.execute(
                    """
                    SELECT
                        chat_platform,
                        chat_user_id,
                        steam_id64,
                        account_kind,
                        display_value,
                        metadata_json,
                        created_at,
                        updated_at
                    FROM steam_account_bindings
                    WHERE chat_platform = ? AND chat_user_id = ?
                    """,
                    (source_platform, chat_user_id),
                ).fetchone()
                source = steam_account_binding_from_row(source_row)
                if source is None:
                    connection.commit()
                    return None

                source_metadata = dict(source.metadata)
                source_metadata["migrated_to_platform_instance"] = target_platform
                target_metadata = dict(source.metadata)
                target_metadata.pop("migrated_to_platform_instance", None)
                target_metadata["migrated_from_platform"] = source_platform
                connection.execute(
                    """
                    UPDATE steam_account_bindings
                    SET metadata_json = ?, updated_at = ?
                    WHERE chat_platform = ? AND chat_user_id = ?
                    """,
                    (
                        json.dumps(source_metadata, ensure_ascii=False),
                        now,
                        source_platform,
                        chat_user_id,
                    ),
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
                    ON CONFLICT(chat_platform, chat_user_id) DO UPDATE SET
                        steam_id64 = excluded.steam_id64,
                        account_kind = excluded.account_kind,
                        display_value = excluded.display_value,
                        metadata_json = excluded.metadata_json,
                        updated_at = excluded.updated_at
                    """,
                    (
                        target_platform,
                        chat_user_id,
                        source.steam_id64,
                        source.account_kind,
                        source.display_value,
                        json.dumps(target_metadata, ensure_ascii=False),
                        source.created_at or now,
                        now,
                    ),
                )
                target_row = connection.execute(
                    """
                    SELECT
                        chat_platform,
                        chat_user_id,
                        steam_id64,
                        account_kind,
                        display_value,
                        metadata_json,
                        created_at,
                        updated_at
                    FROM steam_account_bindings
                    WHERE chat_platform = ? AND chat_user_id = ?
                    """,
                    (target_platform, chat_user_id),
                ).fetchone()
                connection.commit()
                return steam_account_binding_from_row(target_row)
            except (sqlite3.Error, TypeError, ValueError, OverflowError):
                if connection.in_transaction:
                    connection.rollback()
                raise CacheStorageError("account_binding_migration_failure") from None
            finally:
                connection.close()
                self._secure_database_files()

    async def delete_steam_account_binding(
        self,
        chat_platform: str,
        chat_user_id: str,
    ) -> bool:
        return await asyncio.to_thread(
            self._delete_steam_account_binding_sync,
            chat_platform,
            chat_user_id,
        )

    def _delete_steam_account_binding_sync(
        self,
        chat_platform: str,
        chat_user_id: str,
    ) -> bool:
        with self._lock:
            connection = self._connect()
            try:
                with connection:
                    cursor = connection.execute(
                        """
                        DELETE FROM steam_account_bindings
                        WHERE chat_platform = ? AND chat_user_id = ?
                        """,
                        (chat_platform or "default", chat_user_id),
                    )
                return cursor.rowcount > 0
            except sqlite3.Error:
                raise CacheStorageError("account_binding_delete_failure") from None
            finally:
                connection.close()
                self._secure_database_files()

    def _get_steam_account_binding_sync(
        self,
        chat_platform: str,
        chat_user_id: str,
    ) -> SteamAccountBinding | None:
        with self._lock:
            connection = self._connect()
            try:
                row = connection.execute(
                    """
                    SELECT
                        chat_platform,
                        chat_user_id,
                        steam_id64,
                        account_kind,
                        display_value,
                        metadata_json,
                        created_at,
                        updated_at
                    FROM steam_account_bindings
                    WHERE chat_platform = ? AND chat_user_id = ?
                    """,
                    (chat_platform or "default", chat_user_id),
                ).fetchone()
            except sqlite3.Error:
                raise CacheStorageError("account_binding_read_failure") from None
            finally:
                connection.close()
                self._secure_database_files()
        return steam_account_binding_from_row(row)


def _validated_ttl_hours(value: int | float) -> float:
    if isinstance(value, bool):
        raise CacheStorageError("invalid_cache_ttl")
    try:
        ttl = float(value)
    except (TypeError, ValueError, OverflowError):
        raise CacheStorageError("invalid_cache_ttl") from None
    if not math.isfinite(ttl) or ttl <= 0:
        raise CacheStorageError("invalid_cache_ttl")
    return ttl


def _validated_positive_seconds(value: int | float) -> float:
    seconds = _validated_non_negative_seconds(value)
    if seconds <= 0:
        raise CacheStorageError("invalid_cache_ttl")
    return seconds


def _validated_non_negative_seconds(value: int | float) -> float:
    if isinstance(value, bool):
        raise CacheStorageError("invalid_cache_ttl")
    try:
        seconds = float(value)
    except (TypeError, ValueError, OverflowError):
        raise CacheStorageError("invalid_cache_ttl") from None
    if not math.isfinite(seconds) or seconds < 0:
        raise CacheStorageError("invalid_cache_ttl")
    return seconds


def steam_account_binding_from_row(row: Any) -> SteamAccountBinding | None:
    if not row:
        return None
    (
        chat_platform,
        chat_user_id,
        steam_id64,
        account_kind,
        display_value,
        metadata_json,
        created_at,
        updated_at,
    ) = row
    try:
        metadata = json.loads(metadata_json or "{}")
    except json.JSONDecodeError:
        metadata = {}
    return SteamAccountBinding(
        chat_platform=chat_platform,
        chat_user_id=chat_user_id,
        steam_id64=steam_id64,
        account_kind=account_kind,
        display_value=display_value,
        metadata=metadata if isinstance(metadata, dict) else {},
        created_at=float(created_at),
        updated_at=float(updated_at),
    )
