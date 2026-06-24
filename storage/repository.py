from __future__ import annotations

import asyncio
import json
import sqlite3
import time
from pathlib import Path
from typing import Any


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

