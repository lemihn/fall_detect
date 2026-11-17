"""Phat hien ca te nga moi tu /status cua 4_realtime.py.

Quy tac: 4_realtime.py bat persons[i].telegram_sent = True dung luc gui Telegram
(nguoi nga nam bat dong >= telegram_delay giay) va giu True den khi nguoi do day.
Moi cap (person_id, fall_time) la 1 ca nga.
"""
from __future__ import annotations

import time
import uuid
from collections import OrderedDict
from datetime import datetime, timezone

# Namespace co dinh -> cung 1 ca nga luon ra cung 1 uuid (gui lai khong bi trung)
EVENT_NS = uuid.UUID("6f1c2a52-6d1e-4b8e-9a39-0c4f4d6c9a11")
FALL_TIME_FMT = "%H:%M:%S - %d/%m/%Y"


def event_id_for(device_id: str, person_id, fall_time: str) -> str:
    return str(uuid.uuid5(EVENT_NS, f"{device_id}|{person_id}|{fall_time}"))


def parse_fall_time(s: str | None) -> str | None:
    """'14:03:21 - 15/11/2026' (gio may Pi) -> ISO 8601 co mui gio."""
    if not s:
        return None
    try:
        naive = datetime.strptime(s, FALL_TIME_FMT)
    except ValueError:
        return None
    local = naive.astimezone()  # gan mui gio cua may dang chay
    return local.astimezone(timezone.utc).isoformat()


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


class FallEventDetector:
    def __init__(self, device_id: str, memory: int = 500):
        self.device_id = device_id
        self.seen: "OrderedDict[str, float]" = OrderedDict()
        self.memory = memory

    def check(self, status: dict) -> list:
        """Tra ve danh sach dong events moi (chua gui) tu 1 lan doc /status."""
        out = []
        for p in status.get("persons") or []:
            if not p.get("telegram_sent"):
                continue
            pid, ft = p.get("id"), p.get("fall_time")
            if pid is None or not ft:
                continue
            eid = event_id_for(self.device_id, pid, ft)
            if eid in self.seen:
                continue
            self.seen[eid] = time.time()
            while len(self.seen) > self.memory:
                self.seen.popitem(last=False)
            try:
                person_id = int(pid)
            except (TypeError, ValueError):
                person_id = None
            out.append({
                "id": eid,
                "device_id": self.device_id,
                "person_id": person_id,
                "fall_time": parse_fall_time(ft),
                "detected_at": now_iso(),
                "duration_s": float(p.get("since") or 0.0),
            })
        return out
