"""Macro snapshot cache: one row per `macro` pipeline run, reused by
`analyze <TICKER>` (and re-runs of `macro` itself) when younger than 24h so
macro.build_macro_pack()'s ~15 fetches don't repeat on every ticker analysis
in a busy day.

Own SQLite file (reports/macro.sqlite), separate from track_accuracy.sqlite's
append-only report-scoring log — different lifecycle (latest-wins cache vs
permanent scoring history) and different writer (pipeline.py's macro flow vs
track_accuracy.py's offline scorer), same "small local db, not a shared
state store" posture as track_accuracy.py.
"""

from __future__ import annotations

import json
import sqlite3
from datetime import datetime, timedelta, timezone
from pathlib import Path

BASE = Path(__file__).resolve().parent
DB_PATH = BASE / "reports" / "macro.sqlite"


def _ensure_schema(conn: sqlite3.Connection) -> None:
    conn.execute("""
        CREATE TABLE IF NOT EXISTS macro_snapshots (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            as_of TEXT NOT NULL,
            pack_json TEXT NOT NULL,
            card_json TEXT NOT NULL,
            tilts_json TEXT NOT NULL,
            created_at TEXT NOT NULL
        )
    """)
    conn.commit()


def save_snapshot(pack: dict, card: dict, tilts: dict) -> int:
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(DB_PATH)
    try:
        _ensure_schema(conn)
        cur = conn.execute(
            "INSERT INTO macro_snapshots (as_of, pack_json, card_json, tilts_json, created_at) "
            "VALUES (?,?,?,?,?)",
            (pack["as_of"], json.dumps(pack, default=str), json.dumps(card, default=str),
             json.dumps(tilts, default=str), datetime.now(timezone.utc).isoformat()))
        conn.commit()
        return cur.lastrowid
    finally:
        conn.close()


def latest_snapshot(max_age_hours: float = 24.0) -> dict | None:
    """Most recent snapshot if younger than max_age_hours, else None (caller
    refreshes). Age is measured from created_at (when this process wrote the
    row), not the pack's own `as_of` string, so a malformed/clock-skewed
    as_of can never accidentally extend a stale snapshot's reused lifetime."""
    if not DB_PATH.exists():
        return None
    conn = sqlite3.connect(DB_PATH)
    try:
        _ensure_schema(conn)
        row = conn.execute(
            "SELECT id, as_of, pack_json, card_json, tilts_json, created_at "
            "FROM macro_snapshots ORDER BY id DESC LIMIT 1").fetchone()
    finally:
        conn.close()
    if row is None:
        return None
    row_id, as_of, pack_json, card_json, tilts_json, created_at = row
    age = datetime.now(timezone.utc) - datetime.fromisoformat(created_at)
    if age > timedelta(hours=max_age_hours):
        return None
    return {"id": row_id, "as_of": as_of, "pack": json.loads(pack_json),
            "card": json.loads(card_json), "tilts": json.loads(tilts_json),
            "age_hours": age.total_seconds() / 3600}


def snapshot_before(as_of: str) -> dict | None:
    """Newest stored snapshot with an earlier as_of, any age - the baseline for
    'what changed' in the macro report."""
    if not DB_PATH.exists():
        return None
    conn = sqlite3.connect(DB_PATH)
    try:
        _ensure_schema(conn)
        row = conn.execute(
            "SELECT as_of, card_json, tilts_json FROM macro_snapshots WHERE as_of < ? "
            "ORDER BY id DESC LIMIT 1", (as_of,)).fetchone()
    finally:
        conn.close()
    if row is None:
        return None
    return {"as_of": row[0], "card": json.loads(row[1]), "tilts": json.loads(row[2])}
