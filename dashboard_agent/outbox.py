"""Hang doi luu tren dia (SQLite): ca nga chua gui duoc se duoc gui bu khi co mang lai."""
from __future__ import annotations

import json
import logging
import sqlite3
import threading
import time
from pathlib import Path

log = logging.getLogger("outbox")

MAX_ITEMS = 1000
MAX_BACKOFF_S = 300.0


class Outbox:
    def __init__(self, data_dir: Path):
        self.dir = data_dir
        self.snap_dir = data_dir / "snapshots"
        self.snap_dir.mkdir(parents=True, exist_ok=True)
        self.db = sqlite3.connect(str(data_dir / "outbox.sqlite3"), check_same_thread=False)
        self.lock = threading.Lock()
        self.wake = threading.Event()
        with self.lock:
            self.db.execute(
                "create table if not exists outbox ("
                " id text primary key, payload text not null, snapshot text,"
                " attempts integer not null default 0, next_try real not null default 0,"
                " created real not null)"
            )
            self.db.commit()

    def put_event(self, event_id: str, row: dict, jpg: bytes | None) -> None:
        snap = None
        if jpg:
            snap = str(self.snap_dir / f"{event_id}.jpg")
            Path(snap).write_bytes(jpg)
        with self.lock:
            self.db.execute(
                "insert or ignore into outbox (id, payload, snapshot, created) values (?, ?, ?, ?)",
                (event_id, json.dumps(row), snap, time.time()),
            )
            # Gioi han kich thuoc: bo nhung muc cu nhat
            extra = self.db.execute("select count(*) from outbox").fetchone()[0] - MAX_ITEMS
            if extra > 0:
                old = self.db.execute("select id, snapshot from outbox order by created limit ?", (extra,)).fetchall()
                for oid, osnap in old:
                    self.db.execute("delete from outbox where id = ?", (oid,))
                    if osnap:
                        Path(osnap).unlink(missing_ok=True)
                log.warning("Hang doi day, da bo %d ca nga cu nhat", extra)
            self.db.commit()
        self.wake.set()

    def due(self, limit: int = 20) -> list:
        with self.lock:
            return self.db.execute(
                "select id, payload, snapshot, attempts from outbox where next_try <= ? order by created limit ?",
                (time.time(), limit),
            ).fetchall()

    def done(self, event_id: str, snapshot: str | None) -> None:
        with self.lock:
            self.db.execute("delete from outbox where id = ?", (event_id,))
            self.db.commit()
        if snapshot:
            Path(snapshot).unlink(missing_ok=True)

    def retry_later(self, event_id: str, attempts: int) -> None:
        delay = min(MAX_BACKOFF_S, 2.0 * (2 ** min(attempts, 8)))
        with self.lock:
            self.db.execute("update outbox set attempts = ?, next_try = ? where id = ?",
                            (attempts + 1, time.time() + delay, event_id))
            self.db.commit()

    def size(self) -> int:
        with self.lock:
            return self.db.execute("select count(*) from outbox").fetchone()[0]
