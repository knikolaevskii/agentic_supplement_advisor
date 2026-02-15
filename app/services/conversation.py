"""SQLite-backed conversation metadata service."""

from __future__ import annotations

import sqlite3
import uuid
from datetime import datetime, timezone


class ConversationService:
    """Manages conversation metadata in a lightweight SQLite table."""

    def __init__(self, db_path: str = "data/chats.db") -> None:
        self._conn = sqlite3.connect(db_path, check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        self._conn.execute(
            """
            CREATE TABLE IF NOT EXISTS conversations (
                id         TEXT PRIMARY KEY,
                user_id    TEXT NOT NULL,
                title      TEXT NOT NULL,
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL
            )
            """
        )
        self._conn.commit()

    def _now(self) -> str:
        return datetime.now(timezone.utc).isoformat()

    def create(self, user_id: str, title: str) -> dict:
        conv_id = uuid.uuid4().hex[:12]
        now = self._now()
        self._conn.execute(
            "INSERT INTO conversations (id, user_id, title, created_at, updated_at) "
            "VALUES (?, ?, ?, ?, ?)",
            (conv_id, user_id, title, now, now),
        )
        self._conn.commit()
        return {
            "id": conv_id,
            "user_id": user_id,
            "title": title,
            "created_at": now,
            "updated_at": now,
        }

    def list_for_user(self, user_id: str) -> list[dict]:
        rows = self._conn.execute(
            "SELECT * FROM conversations WHERE user_id = ? ORDER BY updated_at DESC",
            (user_id,),
        ).fetchall()
        return [dict(r) for r in rows]

    def get(self, conv_id: str) -> dict | None:
        row = self._conn.execute(
            "SELECT * FROM conversations WHERE id = ?",
            (conv_id,),
        ).fetchone()
        return dict(row) if row else None

    def update_title(self, conv_id: str, title: str) -> None:
        self._conn.execute(
            "UPDATE conversations SET title = ?, updated_at = ? WHERE id = ?",
            (title, self._now(), conv_id),
        )
        self._conn.commit()

    def touch(self, conv_id: str) -> None:
        self._conn.execute(
            "UPDATE conversations SET updated_at = ? WHERE id = ?",
            (self._now(), conv_id),
        )
        self._conn.commit()

    def delete(self, conv_id: str) -> None:
        self._conn.execute(
            "DELETE FROM conversations WHERE id = ?",
            (conv_id,),
        )
        self._conn.commit()
