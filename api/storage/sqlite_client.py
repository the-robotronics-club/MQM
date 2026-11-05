"""Bounded, transactional recent readings for a single free instance.

The free host's filesystem is ephemeral: this is demo history, not archival storage.
"""

from __future__ import annotations

import json
import os
import sqlite3
import time
from contextlib import contextmanager
from pathlib import Path

from api.config import get_settings


@contextmanager
def connection():
    path = Path(os.environ.get("SNAPSHOT_SQLITE_PATH", "/tmp/mqm/readings.sqlite3"))
    path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(path, timeout=10)
    try:
        with conn:
            conn.execute("PRAGMA journal_mode=WAL")
            conn.execute("""CREATE TABLE IF NOT EXISTS readings (
            camera_id TEXT NOT NULL, timestamp TEXT NOT NULL,
            received_at REAL NOT NULL, payload TEXT NOT NULL,
            PRIMARY KEY (camera_id, timestamp))""")
            conn.execute("CREATE INDEX IF NOT EXISTS reading_time ON readings(received_at)")
            yield conn
    finally:
        conn.close()


def save(camera_id: str, metrics: dict) -> None:
    with connection() as conn:
        conn.execute(
            "INSERT OR REPLACE INTO readings VALUES (?, ?, ?, ?)",
            (camera_id, metrics["timestamp"], time.time(), json.dumps(metrics)),
        )
        conn.execute("DELETE FROM readings WHERE received_at < ?", (time.time() - 86400,))


def latest(camera_ids: tuple[str, ...]) -> dict[str, dict]:
    cutoff = time.time() - get_settings().latest_ttl_seconds
    found = {}
    with connection() as conn:
        for camera_id in camera_ids:
            row = conn.execute(
                "SELECT payload FROM readings WHERE camera_id=? AND received_at>=? "
                "ORDER BY received_at DESC LIMIT 1",
                (camera_id, cutoff),
            ).fetchone()
            if row:
                found[camera_id] = json.loads(row[0])
    return found


def history(camera_id: str, minutes: int) -> list[dict]:
    with connection() as conn:
        rows = conn.execute(
            "SELECT payload FROM readings WHERE camera_id=? AND received_at>=? "
            "ORDER BY received_at",
            (camera_id, time.time() - minutes * 60),
        ).fetchall()
    return [json.loads(row[0]) for row in rows]


def check() -> bool:
    with connection() as conn:
        return conn.execute("SELECT 1").fetchone()[0] == 1
