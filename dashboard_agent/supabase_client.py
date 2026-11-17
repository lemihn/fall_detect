"""Goi Supabase qua REST (PostgREST, Storage, Auth) bang requests."""
from __future__ import annotations

import logging
import threading
import time
from hashlib import sha256

import requests

log = logging.getLogger("supabase")


class SupabaseError(Exception):
    def __init__(self, msg: str, status: int = 0, retryable: bool = True):
        super().__init__(msg)
        self.status = status
        self.retryable = retryable


class SupabaseClient:
    def __init__(self, url: str, service_key: str, timeout: float = 8.0):
        self.url = url
        self.key = service_key
        self.timeout = timeout
        self.http = requests.Session()
        self._auth_cache: dict = {}
        self._auth_lock = threading.Lock()

    # ------------------------------------------------------------------ helpers
    def _headers(self, extra: dict | None = None) -> dict:
        h = {"apikey": self.key}
        # Key kieu cu (JWT "eyJ...") can them Authorization. Key moi (sb_secret_...) chi can apikey.
        if self.key.startswith("eyJ"):
            h["Authorization"] = f"Bearer {self.key}"
        if extra:
            h.update(extra)
        return h

    def _check(self, r: requests.Response, what: str) -> requests.Response:
        if r.status_code < 300:
            return r
        retryable = r.status_code >= 500 or r.status_code in (408, 429)
        raise SupabaseError(f"{what}: HTTP {r.status_code} {r.text[:300]}", r.status_code, retryable)

    def _req(self, method: str, path: str, what: str, **kw) -> requests.Response:
        try:
            r = self.http.request(method, self.url + path, timeout=self.timeout, **kw)
        except requests.RequestException as e:
            raise SupabaseError(f"{what}: {e.__class__.__name__}: {e}") from e
        return self._check(r, what)

    # ------------------------------------------------------------------ devices
    def ensure_device(self, device_id: str, name: str, location: str) -> None:
        """Tao dong thiet bi neu chua co. Khong ghi de ten/vi tri admin da sua tren web."""
        self._req(
            "POST", "/rest/v1/devices?on_conflict=id", "ensure_device",
            headers=self._headers({"Prefer": "resolution=ignore-duplicates,return=minimal"}),
            json={"id": device_id, "name": name or device_id, "location": location},
        )

    def upsert_device(self, row: dict) -> None:
        """Heartbeat: chi cap nhat cac cot co trong row."""
        self._req(
            "POST", "/rest/v1/devices?on_conflict=id", "heartbeat",
            headers=self._headers({"Prefer": "resolution=merge-duplicates,return=minimal"}),
            json=row,
        )

    # ------------------------------------------------------------------ events
    def insert_event(self, row: dict) -> None:
        """Them ca nga. id do agent sinh -> gui lai nhieu lan cung khong bi trung."""
        self._req(
            "POST", "/rest/v1/events?on_conflict=id", "insert_event",
            headers=self._headers({"Prefer": "resolution=ignore-duplicates,return=minimal"}),
            json=row,
        )

    def upload_snapshot(self, path: str, jpg: bytes) -> None:
        self._req(
            "POST", f"/storage/v1/object/snapshots/{path}", "upload_snapshot",
            headers=self._headers({"Content-Type": "image/jpeg", "x-upsert": "true"}),
            data=jpg,
        )

    # ------------------------------------------------------------------ auth (cho gateway)
    def verify_dashboard_token(self, token: str) -> dict | None:
        """Tra ve {'user_id','role'} neu token hop le va nguoi dung co trong dashboard_users."""
        if not token or len(token) > 4096:
            return None
        key = sha256(token.encode()).hexdigest()
        now = time.time()
        with self._auth_lock:
            hit = self._auth_cache.get(key)
            if hit and hit[0] > now:
                return hit[1]
            if len(self._auth_cache) > 500:
                self._auth_cache = {k: v for k, v in self._auth_cache.items() if v[0] > now}
        result = None
        try:
            r = self.http.get(self.url + "/auth/v1/user", timeout=self.timeout,
                              headers={"apikey": self.key, "Authorization": f"Bearer {token}"})
            if r.status_code == 200:
                uid = r.json().get("id")
                if uid:
                    q = self._req("GET", f"/rest/v1/dashboard_users?user_id=eq.{uid}&select=role",
                                  "check_member", headers=self._headers())
                    rows = q.json()
                    if rows:
                        result = {"user_id": uid, "role": rows[0].get("role", "viewer")}
        except (requests.RequestException, SupabaseError, ValueError) as e:
            log.warning("Khong kiem tra duoc token: %s", e)
            return None  # loi mang: khong cache
        with self._auth_lock:
            self._auth_cache[key] = (now + (60 if result else 10), result)
        return result
