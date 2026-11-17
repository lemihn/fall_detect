"""Cac vong lap chay nen: doc /status cua Pi, doc /data cua ESP32, heartbeat, gui hang doi."""
from __future__ import annotations

import json
import logging
import threading
import time
from datetime import datetime, timezone

import requests

from config import AGENT_VERSION, Config
from event_detector import FallEventDetector, now_iso
from outbox import Outbox
from supabase_client import SupabaseClient, SupabaseError

log = logging.getLogger("monitor")

VALID_STATES = {"normal", "sit", "lie", "fall", "warmup"}
ESP32_STATES = {"normal", "warning", "fall"}


class Monitor:
    def __init__(self, cfg: Config, sb: SupabaseClient, outbox: Outbox, snapshot_fn):
        self.cfg = cfg
        self.sb = sb
        self.outbox = outbox
        self.snapshot_fn = snapshot_fn  # () -> bytes | None, lay anh hien truong
        self.detector = FallEventDetector(cfg.device_id)
        self.http = requests.Session()
        self.lock = threading.Lock()
        self.hb_now = threading.Event()

        self.pi_status: dict | None = None
        self.pi_ok_at = 0.0
        self.esp: dict | None = None
        self.esp_ok_at = 0.0
        self.stream_url: str | None = cfg.public_stream_url or None
        self.stream_updated_at: str | None = now_iso() if self.stream_url else None
        self.device_ready = False
        self.ensure_lock = threading.Lock()

    # ------------------------------------------------------------------ public
    def set_stream_url(self, url: str | None) -> None:
        with self.lock:
            if url == self.stream_url:
                return
            self.stream_url = url
            self.stream_updated_at = now_iso()
        log.info("Link camera: %s", url)
        self.hb_now.set()

    def start(self) -> None:
        for fn, name in ((self._status_loop, "status"), (self._esp_loop, "esp32"),
                         (self._heartbeat_loop, "heartbeat"), (self._outbox_loop, "outbox")):
            threading.Thread(target=fn, name=name, daemon=True).start()

    # ------------------------------------------------------------------ Pi /status
    def _status_loop(self) -> None:
        last_key = None
        was_ok = None
        while True:
            t0 = time.time()
            try:
                r = self.http.get(self.cfg.pi_url + "/status", timeout=2.0)
                r.raise_for_status()
                st = r.json()
                with self.lock:
                    self.pi_status, self.pi_ok_at = st, time.time()
                if was_ok is not True:
                    log.info("Da ket noi /status cua Pi (%s)", self.cfg.pi_url)
                was_ok = True
                key = (st.get("state"), st.get("alarm"), len(st.get("persons") or []))
                if key != last_key:
                    last_key = key
                    self.hb_now.set()  # trang thai doi -> bao len web ngay
                for ev in self.detector.check(st):
                    self._new_event(ev)
            except (requests.RequestException, ValueError) as e:
                if was_ok is not False:
                    log.warning("Khong doc duoc /status cua Pi: %s", e)
                    self.hb_now.set()
                was_ok = False
            time.sleep(max(0.0, self.cfg.status_poll_s - (time.time() - t0)))

    def _new_event(self, ev: dict) -> None:
        jpg = None
        try:
            jpg = self.snapshot_fn()
        except Exception as e:  # anh la phu, khong duoc lam mat ca nga
            log.warning("Khong lay duoc anh hien truong: %s", e)
        if jpg:
            day = datetime.now(timezone.utc).strftime("%Y-%m-%d")
            ev["snapshot_path"] = f"{self.cfg.device_id}/{day}/{ev['id']}.jpg"
        log.warning("PHAT HIEN CA NGA: nguoi %s, bat dong %.1fs, anh=%s",
                    ev["person_id"], ev["duration_s"], bool(jpg))
        self.outbox.put_event(ev["id"], ev, jpg)

    # ------------------------------------------------------------------ ESP32 /data
    def _esp_loop(self) -> None:
        if not self.cfg.esp32_ip:
            log.info("Khong cau hinh ESP32_IP -> bo qua theo doi hop bao dong")
            return
        url = f"http://{self.cfg.esp32_ip}/data"
        was_ok = None
        while True:
            try:
                r = self.http.get(url, timeout=3.0)
                r.raise_for_status()
                data = r.json()
                with self.lock:
                    self.esp, self.esp_ok_at = data, time.time()
                if was_ok is not True:
                    log.info("Da ket noi ESP32 (%s)", self.cfg.esp32_ip)
                    self.hb_now.set()
                was_ok = True
            except (requests.RequestException, ValueError) as e:
                if was_ok is not False:
                    log.warning("Khong doc duoc ESP32 %s: %s", self.cfg.esp32_ip, e)
                    self.hb_now.set()
                was_ok = False
            time.sleep(self.cfg.esp32_poll_s)

    # ------------------------------------------------------------------ heartbeat
    def build_heartbeat(self) -> dict:
        now = time.time()
        with self.lock:
            st = self.pi_status if now - self.pi_ok_at <= 5.0 else None
            esp = self.esp
            esp_ok = esp is not None and now - self.esp_ok_at <= max(15.0, 3 * self.cfg.esp32_poll_s)
            row = {
                "id": self.cfg.device_id,
                "last_seen": now_iso(),
                "stream_url": self.stream_url,
                "stream_updated_at": self.stream_updated_at,
                "esp32_online": esp_ok,
                "agent_version": AGENT_VERSION,
            }
        if st:
            state = st.get("state")
            try:
                alarm = int(st.get("alarm") or 0)
            except (TypeError, ValueError):
                alarm = 0
            row.update({
                "state": state if state in VALID_STATES else "unknown",
                "alarm": alarm if alarm in (0, 1, 2) else 0,
                "fps": float(st.get("fps") or 0.0),
                "camera": str(st.get("camera") or ""),
                "persons_count": len(st.get("persons") or []),
            })
        else:
            # Agent van chay nhung chuong trinh nhan dien khong tra loi
            row.update({"state": "unknown", "alarm": 0, "fps": None, "persons_count": 0})
        if esp_ok:
            sc = str(esp.get("stateClass") or "").lower()
            row["esp32_state"] = sc if sc in ESP32_STATES else None
            row["esp32_last_seen"] = datetime.fromtimestamp(self.esp_ok_at, timezone.utc).isoformat()
        return row

    def _ensure_device(self) -> bool:
        with self.ensure_lock:  # heartbeat va outbox cung goi -> chi tao 1 lan
            if self.device_ready:
                return True
            try:
                self.sb.ensure_device(self.cfg.device_id, self.cfg.device_name, self.cfg.device_location)
                self.device_ready = True
                log.info("Thiet bi '%s' da co tren Supabase", self.cfg.device_id)
            except SupabaseError as e:
                log.warning("Chua tao duoc thiet bi tren Supabase: %s", e)
            return self.device_ready

    def _heartbeat_loop(self) -> None:
        fails = 0
        while True:
            self.hb_now.wait(timeout=self.cfg.heartbeat_s)
            self.hb_now.clear()
            if not self._ensure_device():
                continue
            try:
                self.sb.upsert_device(self.build_heartbeat())
                if fails:
                    log.info("Heartbeat hoat dong lai sau %d lan loi", fails)
                fails = 0
            except SupabaseError as e:
                fails += 1
                if fails == 1 or fails % 30 == 0:
                    log.warning("Heartbeat loi (%d): %s", fails, e)
            time.sleep(0.5)  # chong gui doan dap khi trang thai doi lien tuc

    # ------------------------------------------------------------------ outbox
    def _outbox_loop(self) -> None:
        while True:
            self.outbox.wake.wait(timeout=5.0)
            self.outbox.wake.clear()
            if not self._ensure_device():
                continue
            for eid, payload, snap, attempts in self.outbox.due():
                try:
                    row = json.loads(payload)
                    if snap and row.get("snapshot_path"):
                        try:
                            with open(snap, "rb") as f:
                                self.sb.upload_snapshot(row["snapshot_path"], f.read())
                        except FileNotFoundError:
                            row["snapshot_path"] = None
                    self.sb.insert_event(row)
                    self.outbox.done(eid, snap)
                    log.info("Da gui ca nga %s len Supabase", eid)
                except SupabaseError as e:
                    if not e.retryable and attempts >= 5:
                        log.error("Bo ca nga %s sau %d lan loi: %s", eid, attempts + 1, e)
                        self.outbox.done(eid, snap)
                    else:
                        log.warning("Chua gui duoc ca nga %s (lan %d): %s", eid, attempts + 1, e)
                        self.outbox.retry_later(eid, attempts)
