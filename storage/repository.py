from __future__ import annotations

import asyncio
import json
import sqlite3
import time
from pathlib import Path
from typing import Any

from .models import AccountBinding


class SQLiteCacheRepository:
    def __init__(self, db_path: Path) -> None:
        self.db_path = db_path
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._initialize()

    def _connect(self) -> sqlite3.Connection:
        return sqlite3.connect(self.db_path)

    def _initialize(self) -> None:
        with self._connect() as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS cache (
                    key TEXT PRIMARY KEY,
                    payload TEXT NOT NULL,
                    created_at REAL NOT NULL
                )
                """
            )
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS account_bindings (
                    chat_platform TEXT NOT NULL,
                    chat_user_id TEXT NOT NULL,
                    provider TEXT NOT NULL,
                    account_id TEXT NOT NULL,
                    account_kind TEXT NOT NULL,
                    display_value TEXT NOT NULL,
                    metadata_json TEXT NOT NULL DEFAULT '{}',
                    created_at REAL NOT NULL,
                    updated_at REAL NOT NULL,
                    PRIMARY KEY (chat_platform, chat_user_id, provider)
                )
                """
            )

    async def get_json(self, key: str, ttl_hours: int) -> Any | None:
        return await asyncio.to_thread(self._get_json_sync, key, ttl_hours)

    def _get_json_sync(self, key: str, ttl_hours: int) -> Any | None:
        with self._connect() as conn:
            row = conn.execute(
                "SELECT payload, created_at FROM cache WHERE key = ?",
                (key,),
            ).fetchone()
            if not row:
                return None
            payload, created_at = row
            if time.time() - float(created_at) > max(ttl_hours, 0) * 3600:
                conn.execute("DELETE FROM cache WHERE key = ?", (key,))
                return None
            return json.loads(payload)

    async def set_json(self, key: str, payload: Any) -> None:
        await asyncio.to_thread(self._set_json_sync, key, payload)

    def _set_json_sync(self, key: str, payload: Any) -> None:
        with self._connect() as conn:
            conn.execute(
                """
                INSERT INTO cache(key, payload, created_at)
                VALUES (?, ?, ?)
                ON CONFLICT(key) DO UPDATE SET
                    payload = excluded.payload,
                    created_at = excluded.created_at
                """,
                (key, json.dumps(payload, ensure_ascii=False), time.time()),
            )

    async def upsert_account_binding(self, binding: AccountBinding) -> AccountBinding:
        return await asyncio.to_thread(self._upsert_account_binding_sync, binding)

    def _upsert_account_binding_sync(self, binding: AccountBinding) -> AccountBinding:
        now = time.time()
        with self._connect() as conn:
            conn.execute(
                """
                INSERT INTO account_bindings(
                    chat_platform,
                    chat_user_id,
                    provider,
                    account_id,
                    account_kind,
                    display_value,
                    metadata_json,
                    created_at,
                    updated_at
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(chat_platform, chat_user_id, provider) DO UPDATE SET
                    account_id = excluded.account_id,
                    account_kind = excluded.account_kind,
                    display_value = excluded.display_value,
                    metadata_json = excluded.metadata_json,
                    updated_at = excluded.updated_at
                """,
                (
                    binding.chat_platform,
                    binding.chat_user_id,
                    binding.provider,
                    binding.account_id,
                    binding.account_kind,
                    binding.display_value,
                    json.dumps(binding.metadata, ensure_ascii=False),
                    binding.created_at or now,
                    now,
                ),
            )
        saved = self._get_account_binding_sync(
            binding.chat_platform,
            binding.chat_user_id,
            binding.provider,
        )
        if saved is None:
            raise RuntimeError("account binding was not saved")
        return saved

    async def get_account_binding(
        self,
        chat_platform: str,
        chat_user_id: str,
        provider: str,
    ) -> AccountBinding | None:
        return await asyncio.to_thread(
            self._get_account_binding_sync,
            chat_platform,
            chat_user_id,
            provider,
        )

    def _get_account_binding_sync(
        self,
        chat_platform: str,
        chat_user_id: str,
        provider: str,
    ) -> AccountBinding | None:
        with self._connect() as conn:
            row = conn.execute(
                """
                SELECT
                    chat_platform,
                    chat_user_id,
                    provider,
                    account_id,
                    account_kind,
                    display_value,
                    metadata_json,
                    created_at,
                    updated_at
                FROM account_bindings
                WHERE chat_platform = ? AND chat_user_id = ? AND provider = ?
                """,
                (chat_platform or "default", chat_user_id, provider),
            ).fetchone()
        return account_binding_from_row(row)

    async def list_account_bindings(
        self,
        chat_platform: str,
        chat_user_id: str,
    ) -> list[AccountBinding]:
        return await asyncio.to_thread(
            self._list_account_bindings_sync,
            chat_platform,
            chat_user_id,
        )

    def _list_account_bindings_sync(
        self,
        chat_platform: str,
        chat_user_id: str,
    ) -> list[AccountBinding]:
        with self._connect() as conn:
            rows = conn.execute(
                """
                SELECT
                    chat_platform,
                    chat_user_id,
                    provider,
                    account_id,
                    account_kind,
                    display_value,
                    metadata_json,
                    created_at,
                    updated_at
                FROM account_bindings
                WHERE chat_platform = ? AND chat_user_id = ?
                ORDER BY provider
                """,
                (chat_platform or "default", chat_user_id),
            ).fetchall()
        return [binding for row in rows if (binding := account_binding_from_row(row))]


def account_binding_from_row(row: Any) -> AccountBinding | None:
    if not row:
        return None
    (
        chat_platform,
        chat_user_id,
        provider,
        account_id,
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
    return AccountBinding(
        chat_platform=chat_platform,
        chat_user_id=chat_user_id,
        provider=provider,
        account_id=account_id,
        account_kind=account_kind,
        display_value=display_value,
        metadata=metadata if isinstance(metadata, dict) else {},
        created_at=float(created_at),
        updated_at=float(updated_at),
    )
