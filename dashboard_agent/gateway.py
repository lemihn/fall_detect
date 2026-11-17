"""Gateway HTTP: cho website xem camera live + dieu khien, chi khi co token dang nhap hop le.

Endpoint:
  GET  /health                     -> kiem tra song (khong can token)
  GET  /stream?token=...           -> video MJPEG (dung truc tiep trong <img src>)
  GET  /snapshot?token=...         -> 1 anh JPEG
  GET  /api/status                 -> /status cua Pi (Authorization: Bearer <token>)
  POST /api/reset_alert            -> tat bao dong
  POST /api/switch_camera          -> body {"camera": 1}
"""
from __future__ import annotations

import json
import logging
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import parse_qs, urlparse

import requests

from config import Config
from mjpeg import MjpegRelay
from supabase_client import SupabaseClient

log = logging.getLogger("gateway")

REVERIFY_S = 60.0


class Gateway:
    def __init__(self, cfg: Config, sb: SupabaseClient, relay: MjpegRelay):
        self.cfg = cfg
        self.sb = sb
        self.relay = relay
        self.http = requests.Session()
        self.viewer_lock = threading.Lock()
        self.viewers = 0

    def serve_forever(self) -> None:
        gw = self

        class Handler(_Handler):
            gateway = gw

        srv = ThreadingHTTPServer((self.cfg.gateway_host, self.cfg.gateway_port), Handler)
        srv.daemon_threads = True
        log.info("Gateway dang chay tai http://%s:%d", self.cfg.gateway_host, self.cfg.gateway_port)
        srv.serve_forever()

    def start(self) -> None:
        threading.Thread(target=self.serve_forever, name="gateway", daemon=True).start()

    def pi(self, path: str, timeout: float = 3.0) -> requests.Response:
        return self.http.get(self.cfg.pi_url + path, timeout=timeout)


class _Handler(BaseHTTPRequestHandler):
    gateway: Gateway
    protocol_version = "HTTP/1.1"
    server_version = "FallDashboardGateway"

    # ------------------------------------------------------------------ tien ich
    def log_message(self, fmt, *args):  # khong ghi query string (co token) vao log
        log.debug("%s %s", self.command, urlparse(self.path).path)

    def _cors(self) -> None:
        origins = self.gateway.cfg.allowed_origins
        origin = self.headers.get("Origin", "")
        if "*" in origins:
            self.send_header("Access-Control-Allow-Origin", "*")
        elif origin in origins:
            self.send_header("Access-Control-Allow-Origin", origin)
            self.send_header("Vary", "Origin")

    def _json(self, code: int, obj) -> None:
        body = json.dumps(obj, ensure_ascii=False).encode()
        self.send_response(code)
        self._cors()
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def _token(self) -> str:
        auth = self.headers.get("Authorization", "")
        if auth.lower().startswith("bearer "):
            return auth[7:].strip()
        return (parse_qs(urlparse(self.path).query).get("token") or [""])[0]

    def _auth(self):
        user = self.gateway.sb.verify_dashboard_token(self._token())
        if not user:
            self._json(401, {"error": "unauthorized"})
        return user

    # ------------------------------------------------------------------ routes
    def do_OPTIONS(self):
        self.send_response(204)
        self._cors()
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Authorization, Content-Type")
        self.send_header("Access-Control-Max-Age", "600")
        self.send_header("Content-Length", "0")
        self.end_headers()

    def do_GET(self):
        path = urlparse(self.path).path
        if path == "/health":
            return self._json(200, {"ok": True, "device_id": self.gateway.cfg.device_id})
        if path == "/stream":
            return self._stream()
        if path == "/snapshot":
            return self._snapshot()
        if path == "/api/status":
            return self._proxy_status()
        self._json(404, {"error": "not found"})

    def do_POST(self):
        path = urlparse(self.path).path
        if path == "/api/reset_alert":
            if not self._auth():
                return
            return self._call_pi("/reset_alert")
        if path == "/api/switch_camera":
            if not self._auth():
                return
            try:
                n = int(self.headers.get("Content-Length") or 0)
                body = json.loads(self.rfile.read(min(n, 4096)) or b"{}")
                cam = int(body.get("camera"))
                if not 0 <= cam <= 9:
                    raise ValueError
            except (ValueError, TypeError):
                return self._json(400, {"error": "camera phai la so 0-9"})
            return self._call_pi(f"/switch_camera/{cam}")
        self._json(404, {"error": "not found"})

    # ------------------------------------------------------------------ xu ly
    def _call_pi(self, pi_path: str):
        try:
            r = self.gateway.pi(pi_path)
            return self._json(r.status_code, r.json())
        except (requests.RequestException, ValueError) as e:
            return self._json(502, {"error": f"Pi khong tra loi: {e.__class__.__name__}"})

    def _proxy_status(self):
        if not self._auth():
            return
        return self._call_pi("/status")

    def _snapshot(self):
        if not self._auth():
            return
        jpg = self.gateway.relay.snapshot()
        if not jpg:
            return self._json(503, {"error": "camera offline"})
        self.send_response(200)
        self._cors()
        self.send_header("Content-Type", "image/jpeg")
        self.send_header("Content-Length", str(len(jpg)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(jpg)

    def _stream(self):
        gw = self.gateway
        token = self._token()
        if not self._auth():
            return
        with gw.viewer_lock:
            if gw.viewers >= gw.cfg.stream_max_viewers:
                return self._json(503, {"error": "qua nhieu nguoi xem"})
            gw.viewers += 1
        gw.relay.acquire()
        log.info("Nguoi xem moi (dang co %d)", gw.viewers)
        try:
            self.send_response(200)
            self._cors()
            self.send_header("Content-Type", "multipart/x-mixed-replace; boundary=frame")
            self.send_header("Cache-Control", "no-store")
            self.send_header("Connection", "close")
            self.end_headers()
            self.close_connection = True
            min_gap = 1.0 / gw.cfg.stream_max_fps
            seq, last_sent, verified_at = 0, 0.0, time.time()
            while True:
                if time.time() - verified_at > REVERIFY_S:
                    if not gw.sb.verify_dashboard_token(token):
                        log.info("Token het han -> ngat luong video")
                        return
                    verified_at = time.time()
                seq, jpg = gw.relay.wait_frame(seq, timeout=5.0)
                if jpg is None:
                    continue  # chua co anh moi, giu ket noi
                wait = min_gap - (time.time() - last_sent)
                if wait > 0:
                    time.sleep(wait)
                self.wfile.write(b"--frame\r\nContent-Type: image/jpeg\r\nContent-Length: "
                                 + str(len(jpg)).encode() + b"\r\n\r\n" + jpg + b"\r\n")
                self.wfile.flush()
                last_sent = time.time()
        except (BrokenPipeError, ConnectionResetError, ConnectionAbortedError, OSError):
            pass
        finally:
            gw.relay.release()
            with gw.viewer_lock:
                gw.viewers -= 1
            log.info("Nguoi xem roi di (con %d)", gw.viewers)
