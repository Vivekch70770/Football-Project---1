"""SQLite storage for analysis jobs and match events."""
from __future__ import annotations

import json
import sqlite3
from contextlib import contextmanager
from datetime import datetime
from typing import Any, Dict, List, Optional

from config import settings

EVENT_COLUMNS = ["id", "type", "time_s", "confidence", "start_s", "end_s",
                 "track_ids", "source", "status", "note"]


@contextmanager
def _conn():
    settings.ensure_dirs()
    con = sqlite3.connect(settings.DB_PATH)
    con.row_factory = sqlite3.Row
    try:
        yield con
        con.commit()
    finally:
        con.close()


def init_db() -> None:
    with _conn() as c:
        c.executescript("""
        CREATE TABLE IF NOT EXISTS jobs (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT, created TEXT, status TEXT, video_path TEXT,
            fps REAL, frame_count INTEGER, width INTEGER, height INTEGER,
            params TEXT, error TEXT
        );
        CREATE TABLE IF NOT EXISTS events (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            job_id INTEGER, type TEXT, time_s REAL, confidence REAL,
            start_s REAL, end_s REAL, track_ids TEXT, source TEXT,
            status TEXT, note TEXT
        );
        """)


def create_job(name: str, video_path: str, meta, params: Dict[str, Any]) -> int:
    with _conn() as c:
        cur = c.execute(
            "INSERT INTO jobs (name, created, status, video_path, fps, frame_count, width, height, params) "
            "VALUES (?,?,?,?,?,?,?,?,?)",
            (name, datetime.now().isoformat(timespec="seconds"), "running", video_path,
             meta.fps, meta.frame_count, meta.width, meta.height, json.dumps(params)))
        return int(cur.lastrowid)


def update_job(job_id: int, **fields) -> None:
    if "params" in fields and not isinstance(fields["params"], str):
        fields["params"] = json.dumps(fields["params"])
    cols = ", ".join(f"{k}=?" for k in fields)
    with _conn() as c:
        c.execute(f"UPDATE jobs SET {cols} WHERE id=?", (*fields.values(), job_id))


def _job(row) -> Dict[str, Any]:
    d = dict(row)
    d["params"] = json.loads(d["params"]) if d.get("params") else {}
    return d


def get_job(job_id: int) -> Optional[Dict[str, Any]]:
    with _conn() as c:
        row = c.execute("SELECT * FROM jobs WHERE id=?", (job_id,)).fetchone()
    return _job(row) if row else None


def list_jobs() -> List[Dict[str, Any]]:
    with _conn() as c:
        rows = c.execute("SELECT * FROM jobs ORDER BY id DESC").fetchall()
    return [_job(r) for r in rows]


# --------------------------------------------------------------------- events
def add_events(job_id: int, events: List[Dict[str, Any]]) -> None:
    with _conn() as c:
        for e in events:
            c.execute(
                "INSERT INTO events (job_id,type,time_s,confidence,start_s,end_s,track_ids,source,status,note) "
                "VALUES (?,?,?,?,?,?,?,?,?,?)",
                (job_id, e["type"], float(e["time_s"]), e.get("confidence"), float(e["start_s"]),
                 float(e["end_s"]), e.get("track_ids", ""), e.get("source", "manual"),
                 e.get("status", "pending"), e.get("note", "")))


def get_events(job_id: int) -> List[Dict[str, Any]]:
    with _conn() as c:
        rows = c.execute("SELECT * FROM events WHERE job_id=? ORDER BY time_s", (job_id,)).fetchall()
    return [dict(r) for r in rows]


def clear_auto_pending_events(job_id: int) -> None:
    """Remove un-reviewed automatic candidates; keeps manual / confirmed / rejected events."""
    with _conn() as c:
        c.execute("DELETE FROM events WHERE job_id=? AND source='auto' AND status='pending'", (job_id,))


def save_event_rows(job_id: int, rows: List[Dict[str, Any]]) -> None:
    """Replace all events of a job with the edited rows (used by the review table)."""
    with _conn() as c:
        c.execute("DELETE FROM events WHERE job_id=?", (job_id,))
    add_events(job_id, rows)
