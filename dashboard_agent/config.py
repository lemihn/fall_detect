"""Doc cau hinh tu bien moi truong va file .env (khong can thu vien ngoai)."""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

HERE = Path(__file__).resolve().parent
AGENT_VERSION = "0.1.0"


def load_dotenv(path: Path) -> None:
    """Nap file .env don gian: KEY=VALUE, bo qua dong trong / comment. Khong ghi de bien da co."""
    if not path.exists():
        return
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, val = line.split("=", 1)
        key, val = key.strip(), val.strip()
        if len(val) >= 2 and val[0] == val[-1] and val[0] in "\"'":
            val = val[1:-1]
        os.environ.setdefault(key, val)


def _env(name: str, default: str = "") -> str:
    return os.environ.get(name, default).strip()


def _float(name: str, default: float) -> float:
    try:
        return float(_env(name, str(default)))
    except ValueError:
        return default


def _int(name: str, default: int) -> int:
    try:
        return int(_env(name, str(default)))
    except ValueError:
        return default


@dataclass
class Config:
    supabase_url: str
    supabase_service_key: str
    device_id: str
    device_name: str
    device_location: str
    pi_url: str
    esp32_ip: str
    gateway_host: str
    gateway_port: int
    stream_max_fps: float
    stream_max_viewers: int
    allowed_origins: list = field(default_factory=list)
    tunnel: str = "quick"
    cloudflared_path: str = "cloudflared"
    public_stream_url: str = ""
    status_poll_s: float = 1.0
    esp32_poll_s: float = 5.0
    heartbeat_s: float = 10.0
    data_dir: Path = HERE / "data"

    @classmethod
    def from_env(cls, env_file: Path | None = None) -> "Config":
        load_dotenv(env_file or HERE / ".env")
        cfg = cls(
            supabase_url=_env("SUPABASE_URL").rstrip("/"),
            supabase_service_key=_env("SUPABASE_SERVICE_KEY"),
            device_id=_env("DEVICE_ID", "pi-01"),
            device_name=_env("DEVICE_NAME", ""),
            device_location=_env("DEVICE_LOCATION", ""),
            pi_url=_env("PI_URL", "http://127.0.0.1:5000").rstrip("/"),
            esp32_ip=_env("ESP32_IP", ""),
            gateway_host=_env("GATEWAY_HOST", "127.0.0.1"),
            gateway_port=_int("GATEWAY_PORT", 8080),
            stream_max_fps=max(1.0, _float("STREAM_MAX_FPS", 10.0)),
            stream_max_viewers=max(1, _int("STREAM_MAX_VIEWERS", 4)),
            allowed_origins=[o.strip() for o in _env("ALLOWED_ORIGINS", "*").split(",") if o.strip()],
            tunnel=_env("TUNNEL", "quick").lower(),
            cloudflared_path=_env("CLOUDFLARED_PATH", "cloudflared"),
            public_stream_url=_env("PUBLIC_STREAM_URL", "").rstrip("/"),
            status_poll_s=max(0.2, _float("STATUS_POLL_S", 1.0)),
            esp32_poll_s=max(1.0, _float("ESP32_POLL_S", 5.0)),
            heartbeat_s=max(2.0, _float("HEARTBEAT_S", 10.0)),
            data_dir=Path(_env("DATA_DIR", str(HERE / "data"))),
        )
        missing = [n for n, v in (("SUPABASE_URL", cfg.supabase_url),
                                  ("SUPABASE_SERVICE_KEY", cfg.supabase_service_key)) if not v]
        if missing:
            raise SystemExit(f"Thieu cau hinh: {', '.join(missing)}. Xem dashboard_agent/.env.example")
        if cfg.tunnel not in ("quick", "off"):
            raise SystemExit("TUNNEL chi nhan 'quick' hoac 'off'")
        return cfg
