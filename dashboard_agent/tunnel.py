"""Chay Cloudflare quick tunnel de dua gateway ra Internet, khong can mo port modem.

Moi lan cloudflared khoi dong lai, URL (https://xxx.trycloudflare.com) se doi;
URL moi duoc bao ve qua on_url() -> agent ghi vao devices.stream_url.
"""
from __future__ import annotations

import logging
import re
import subprocess
import threading
import time

log = logging.getLogger("tunnel")

URL_RE = re.compile(r"https://[a-z0-9-]+\.trycloudflare\.com")


class QuickTunnel:
    def __init__(self, cloudflared: str, local_url: str, on_url):
        self.cmd = [cloudflared, "tunnel", "--no-autoupdate", "--url", local_url]
        self.on_url = on_url

    def start(self) -> None:
        threading.Thread(target=self._run, name="tunnel", daemon=True).start()

    def _run(self) -> None:
        backoff = 5.0
        while True:
            started = time.time()
            try:
                proc = subprocess.Popen(self.cmd, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE,
                                        text=True, encoding="utf-8", errors="replace")
            except FileNotFoundError:
                log.error("Khong tim thay cloudflared (%s). Xem README de cai dat. "
                          "Camera live tu xa se khong hoat dong.", self.cmd[0])
                return
            log.info("Dang mo Cloudflare tunnel...")
            assert proc.stderr is not None
            for line in proc.stderr:
                m = URL_RE.search(line)
                if m:
                    self.on_url(m.group(0))
                    backoff = 5.0
            code = proc.wait()
            self.on_url(None)
            if time.time() - started > 120:
                backoff = 5.0
            log.warning("cloudflared dung (ma %s), thu lai sau %.0fs", code, backoff)
            time.sleep(backoff)
            backoff = min(backoff * 2, 300.0)
