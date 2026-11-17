"""Dashboard agent - chay tren Pi, song song voi detect_v3/model/4_realtime.py (KHONG sua code do).

    python dashboard_agent/agent.py

Lam 3 viec:
  1. Doc /status cua 4_realtime.py -> heartbeat + ca te nga (kem anh) len Supabase.
  2. Doc /data cua ESP32 -> tinh trang hop bao dong.
  3. Gateway video co kiem tra dang nhap + Cloudflare tunnel -> xem camera tu xa.
"""
from __future__ import annotations

import argparse
import logging
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from config import AGENT_VERSION, Config  # noqa: E402
from gateway import Gateway  # noqa: E402
from mjpeg import MjpegRelay  # noqa: E402
from monitor import Monitor  # noqa: E402
from outbox import Outbox  # noqa: E402
from supabase_client import SupabaseClient  # noqa: E402
from tunnel import QuickTunnel  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser(description="Fall detection dashboard agent")
    ap.add_argument("--env", type=Path, default=None, help="duong dan file .env (mac dinh dashboard_agent/.env)")
    ap.add_argument("-v", "--verbose", action="store_true")
    args = ap.parse_args()

    logging.basicConfig(level=logging.DEBUG if args.verbose else logging.INFO,
                        format="%(asctime)s %(levelname)-7s [%(name)s] %(message)s", datefmt="%H:%M:%S")
    logging.getLogger("urllib3").setLevel(logging.WARNING)
    log = logging.getLogger("agent")

    cfg = Config.from_env(args.env)
    cfg.data_dir.mkdir(parents=True, exist_ok=True)
    log.info("Dashboard agent v%s | thiet bi=%s | Pi=%s | ESP32=%s | tunnel=%s",
             AGENT_VERSION, cfg.device_id, cfg.pi_url, cfg.esp32_ip or "-", cfg.tunnel)

    sb = SupabaseClient(cfg.supabase_url, cfg.supabase_service_key)
    outbox = Outbox(cfg.data_dir)
    if outbox.size():
        log.info("Con %d ca nga chua gui tu lan truoc, se gui bu", outbox.size())
    relay = MjpegRelay(cfg.pi_url + "/video_feed")
    monitor = Monitor(cfg, sb, outbox, snapshot_fn=relay.snapshot)

    monitor.start()
    Gateway(cfg, sb, relay).start()
    if cfg.tunnel == "quick" and not cfg.public_stream_url:
        QuickTunnel(cfg.cloudflared_path, f"http://127.0.0.1:{cfg.gateway_port}", monitor.set_stream_url).start()

    try:
        while True:
            time.sleep(3600)
    except KeyboardInterrupt:
        log.info("Dung agent")


if __name__ == "__main__":
    main()
