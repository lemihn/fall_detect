"""Test agent khong can Pi that hay Supabase that: dung server gia lap tren localhost.

    python -m unittest discover -s dashboard_agent/tests -v
"""
from __future__ import annotations

import json
import socket
import sys
import tempfile
import threading
import time
import unittest
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import requests

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from config import Config  # noqa: E402
from event_detector import FallEventDetector, event_id_for, parse_fall_time  # noqa: E402
from gateway import Gateway  # noqa: E402
from mjpeg import MjpegParser, MjpegRelay  # noqa: E402
from monitor import Monitor  # noqa: E402
from outbox import Outbox  # noqa: E402
from supabase_client import SupabaseClient  # noqa: E402

JPG = b"\xff\xd8fakejpeg\xff\xd9"


def free_port() -> int:
    s = socket.socket()
    s.bind(("127.0.0.1", 0))
    port = s.getsockname()[1]
    s.close()
    return port


def serve(handler_cls) -> tuple:
    port = free_port()
    srv = ThreadingHTTPServer(("127.0.0.1", port), handler_cls)
    srv.daemon_threads = True
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    return srv, f"http://127.0.0.1:{port}"


# --------------------------------------------------------------------------- unit
class EventDetectorTest(unittest.TestCase):
    def test_only_after_telegram_sent_and_once(self):
        d = FallEventDetector("pi-x")
        ft = "14:03:21 - 15/11/2026"
        falling = {"persons": [{"id": 3, "state": "fall", "since": 4.0, "fall_time": ft, "telegram_sent": False}]}
        sent = {"persons": [{"id": 3, "state": "fall", "since": 10.2, "fall_time": ft, "telegram_sent": True}]}
        self.assertEqual(d.check(falling), [])
        evs = d.check(sent)
        self.assertEqual(len(evs), 1)
        self.assertEqual(evs[0]["person_id"], 3)
        self.assertAlmostEqual(evs[0]["duration_s"], 10.2)
        self.assertEqual(evs[0]["id"], event_id_for("pi-x", 3, ft))
        self.assertEqual(d.check(sent), [])  # cung ca nga -> khong lap lai

    def test_new_fall_same_person_is_new_event(self):
        d = FallEventDetector("pi-x")
        a = {"persons": [{"id": 1, "since": 11, "fall_time": "10:00:00 - 15/11/2026", "telegram_sent": True}]}
        b = {"persons": [{"id": 1, "since": 11, "fall_time": "10:05:00 - 15/11/2026", "telegram_sent": True}]}
        self.assertEqual(len(d.check(a)), 1)
        self.assertEqual(len(d.check(b)), 1)

    def test_parse_fall_time(self):
        self.assertIsNotNone(parse_fall_time("14:03:21 - 15/11/2026"))
        self.assertIsNone(parse_fall_time("bad"))
        self.assertIsNone(parse_fall_time(None))


class MjpegParserTest(unittest.TestCase):
    def test_split_frames_across_chunks(self):
        stream = b"".join(b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" + JPG + bytes([i]) + b"\r\n" for i in range(3))
        p = MjpegParser()
        frames = []
        for i in range(0, len(stream), 7):
            frames += p.feed(stream[i:i + 7])
        # frame cuoi chi tach duoc khi co boundary tiep theo
        self.assertEqual(frames, [JPG + b"\x00", JPG + b"\x01"])


# --------------------------------------------------------------------------- gia lap
class FakePi(BaseHTTPRequestHandler):
    status = {"state": "normal", "alarm": 0, "fps": 14.5, "camera": "0", "persons": []}
    calls: list = []

    def log_message(self, *a):
        pass

    def do_GET(self):
        FakePi.calls.append(self.path)
        if self.path == "/status":
            body = json.dumps(FakePi.status).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
        elif self.path == "/video_feed":
            self.send_response(200)
            self.send_header("Content-Type", "multipart/x-mixed-replace; boundary=frame")
            self.end_headers()
            try:
                for _ in range(200):
                    self.wfile.write(b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" + JPG + b"\r\n")
                    self.wfile.flush()
                    time.sleep(0.05)
            except OSError:
                pass
        elif self.path in ("/reset_alert", "/switch_camera/1"):
            body = b'{"ok": true}'
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
        else:
            self.send_response(404)
            self.send_header("Content-Length", "0")
            self.end_headers()


class FakeSupabase(BaseHTTPRequestHandler):
    devices: dict = {}
    events: dict = {}
    objects: dict = {}
    fail = False

    def log_message(self, *a):
        pass

    def _send(self, code, obj=None):
        body = json.dumps(obj).encode() if obj is not None else b""
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        if self.path.startswith("/auth/v1/user"):
            tok = self.headers.get("Authorization", "")
            return self._send(200, {"id": "u-member"}) if tok == "Bearer good" else (
                self._send(200, {"id": "u-outsider"}) if tok == "Bearer outsider" else self._send(401, {}))
        if self.path.startswith("/rest/v1/dashboard_users"):
            return self._send(200, [{"role": "admin"}] if "u-member" in self.path else [])
        self._send(404, {})

    def do_POST(self):
        assert self.headers.get("apikey") == "svc"
        data = self.rfile.read(int(self.headers.get("Content-Length") or 0))
        if FakeSupabase.fail:
            return self._send(503, {"message": "down"})
        if self.path.startswith("/rest/v1/devices"):
            row = json.loads(data)
            prefer = self.headers.get("Prefer", "")
            cur = FakeSupabase.devices.get(row["id"])
            if cur is None:
                FakeSupabase.devices[row["id"]] = dict(row)
            elif "merge-duplicates" in prefer:
                cur.update(row)
            return self._send(201)
        if self.path.startswith("/rest/v1/events"):
            row = json.loads(data)
            assert row["device_id"] in FakeSupabase.devices, "FK devices"
            FakeSupabase.events.setdefault(row["id"], row)
            return self._send(201)
        if self.path.startswith("/storage/v1/object/snapshots/"):
            FakeSupabase.objects[self.path.split("/snapshots/", 1)[1]] = data
            return self._send(200, {"Key": "ok"})
        self._send(404, {})


def wait_for(cond, timeout=8.0):
    end = time.time() + timeout
    while time.time() < end:
        if cond():
            return True
        time.sleep(0.05)
    return False


# --------------------------------------------------------------------------- tich hop
class AgentIntegrationTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.pi_srv, cls.pi_url = serve(FakePi)
        cls.sb_srv, cls.sb_url = serve(FakeSupabase)
        try:  # Windows khoa file sqlite dang mo -> bo qua loi khi don dep
            cls.tmp = tempfile.TemporaryDirectory(ignore_cleanup_errors=True)
        except TypeError:  # Python < 3.10
            cls.tmp = tempfile.TemporaryDirectory()
        cls.cfg = Config(
            supabase_url=cls.sb_url, supabase_service_key="svc", device_id="pi-test",
            device_name="Pi test", device_location="Lab", pi_url=cls.pi_url, esp32_ip="",
            gateway_host="127.0.0.1", gateway_port=free_port(), stream_max_fps=20,
            stream_max_viewers=1, allowed_origins=["*"], tunnel="off",
            status_poll_s=0.2, heartbeat_s=1.0, data_dir=Path(cls.tmp.name),
        )
        cls.sb = SupabaseClient(cls.cfg.supabase_url, "svc")
        cls.outbox = Outbox(cls.cfg.data_dir)
        cls.relay = MjpegRelay(cls.pi_url + "/video_feed")
        cls.monitor = Monitor(cls.cfg, cls.sb, cls.outbox, snapshot_fn=cls.relay.snapshot)
        cls.monitor.start()
        Gateway(cls.cfg, cls.sb, cls.relay).start()
        cls.gw = f"http://127.0.0.1:{cls.cfg.gateway_port}"
        wait_for(lambda: requests.get(cls.gw + "/health", timeout=1).ok if _port_open(cls.cfg.gateway_port) else False)

    def test_1_heartbeat_creates_device(self):
        self.assertTrue(wait_for(lambda: FakeSupabase.devices.get("pi-test", {}).get("last_seen")))
        dev = FakeSupabase.devices["pi-test"]
        self.assertEqual(dev["name"], "Pi test")
        self.assertEqual(dev["state"], "normal")
        self.assertEqual(dev["fps"], 14.5)

    def test_2_heartbeat_does_not_overwrite_admin_name(self):
        wait_for(lambda: "pi-test" in FakeSupabase.devices)
        FakeSupabase.devices["pi-test"]["name"] = "Ten admin dat"
        self.monitor.hb_now.set()
        time.sleep(1.5)
        self.assertEqual(FakeSupabase.devices["pi-test"]["name"], "Ten admin dat")

    def test_3_fall_event_with_snapshot(self):
        ft = "09:15:00 - 15/11/2026"
        FakePi.status = {"state": "fall", "alarm": 2, "fps": 14, "camera": "0",
                         "persons": [{"id": 7, "state": "fall", "since": 10.4, "fall_time": ft, "telegram_sent": True}]}
        eid = event_id_for("pi-test", 7, ft)
        self.assertTrue(wait_for(lambda: eid in FakeSupabase.events, timeout=12))
        ev = FakeSupabase.events[eid]
        self.assertEqual(ev["person_id"], 7)
        self.assertTrue(ev["snapshot_path"].startswith("pi-test/"))
        self.assertEqual(FakeSupabase.objects[ev["snapshot_path"]], JPG)
        self.assertTrue(wait_for(lambda: FakeSupabase.devices["pi-test"].get("state") == "fall"))
        self.assertEqual(self.outbox.size(), 0)

    def test_4_offline_then_resend(self):
        FakeSupabase.fail = True
        ft = "09:20:00 - 15/11/2026"
        FakePi.status = {"state": "fall", "alarm": 2, "fps": 14, "camera": "0",
                         "persons": [{"id": 8, "since": 11, "fall_time": ft, "telegram_sent": True}]}
        eid = event_id_for("pi-test", 8, ft)
        self.assertTrue(wait_for(lambda: self.outbox.size() == 1, timeout=10))
        time.sleep(0.5)
        self.assertNotIn(eid, FakeSupabase.events)
        FakeSupabase.fail = False
        # dua lich thu lai ve hien tai de khong phai doi backoff
        with self.outbox.lock:
            self.outbox.db.execute("update outbox set next_try = 0")
            self.outbox.db.commit()
        self.outbox.wake.set()
        self.assertTrue(wait_for(lambda: eid in FakeSupabase.events, timeout=10))
        self.assertEqual(self.outbox.size(), 0)

    def test_5_gateway_auth(self):
        self.assertEqual(requests.get(self.gw + "/snapshot", timeout=5).status_code, 401)
        self.assertEqual(requests.get(self.gw + "/snapshot?token=bad", timeout=5).status_code, 401)
        self.assertEqual(requests.get(self.gw + "/snapshot?token=outsider", timeout=5).status_code, 401)
        r = requests.get(self.gw + "/snapshot?token=good", timeout=10)
        self.assertEqual(r.status_code, 200)
        self.assertEqual(r.content, JPG)

    def test_6_stream_and_viewer_limit(self):
        with requests.get(self.gw + "/stream?token=good", stream=True, timeout=10) as r:
            self.assertEqual(r.status_code, 200)
            self.assertIn("multipart/x-mixed-replace", r.headers["Content-Type"])
            p, got = MjpegParser(), []
            for chunk in r.iter_content(1024):
                got += p.feed(chunk)
                if len(got) >= 2:
                    break
            self.assertEqual(got[0], JPG)
            # gioi han 1 nguoi xem
            self.assertEqual(requests.get(self.gw + "/stream?token=good", timeout=5).status_code, 503)

    def test_7_controls(self):
        self.assertEqual(requests.post(self.gw + "/api/reset_alert", timeout=5).status_code, 401)
        h = {"Authorization": "Bearer good"}
        self.assertEqual(requests.post(self.gw + "/api/reset_alert", headers=h, timeout=5).status_code, 200)
        r = requests.post(self.gw + "/api/switch_camera", headers=h, json={"camera": 1}, timeout=5)
        self.assertEqual(r.status_code, 200)
        self.assertIn("/switch_camera/1", FakePi.calls)
        r = requests.post(self.gw + "/api/switch_camera", headers=h, json={"camera": "../x"}, timeout=5)
        self.assertEqual(r.status_code, 400)


def _port_open(port: int) -> bool:
    with socket.socket() as s:
        return s.connect_ex(("127.0.0.1", port)) == 0


if __name__ == "__main__":
    unittest.main(verbosity=2)
