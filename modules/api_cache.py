from __future__ import annotations

import json
import sqlite3
from contextlib import closing
from pathlib import Path
from typing import Any


BASE_DIR = Path(__file__).resolve().parent.parent
CACHE_PATH = BASE_DIR / "data" / "moneycontrol_cache.sqlite3"


def _connect() -> sqlite3.Connection:
    CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
    connection = sqlite3.connect(CACHE_PATH)
    connection.execute(
        "CREATE TABLE IF NOT EXISTS api_cache (url TEXT PRIMARY KEY, payload TEXT NOT NULL, fetched_at TEXT NOT NULL)"
    )
    return connection


def get_cached_payload(url: str) -> dict[str, Any] | None:
    with closing(_connect()) as connection:
        row = connection.execute("SELECT payload FROM api_cache WHERE url = ?", (url,)).fetchone()
    if row is None:
        return None
    try:
        payload = json.loads(row[0])
    except (TypeError, ValueError):
        return None
    return payload if isinstance(payload, dict) else None


def store_payload(url: str, payload: dict[str, Any], fetched_at: str) -> None:
    with closing(_connect()) as connection:
        connection.execute(
            "INSERT OR REPLACE INTO api_cache (url, payload, fetched_at) VALUES (?, ?, ?)",
            (url, json.dumps(payload, separators=(",", ":")), fetched_at),
        )
        connection.commit()