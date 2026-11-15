"""Doc luong MJPEG tu /video_feed cua 4_realtime.py MOT lan, chia cho nhieu nguoi xem."""
from __future__ import annotations

import logging
import threading
import time
import urllib.request

log = logging.getLogger("mjpeg")

BOUNDARY = b"--frame"
MAX_BUF = 8 * 1024 * 1024
IDLE_STOP_S = 10.0


class MjpegParser:
    """Tach anh JPEG tu luong multipart cua Flask:  --frame\\r\\nContent-Type: image/jpeg\\r\\n\\r\\n<jpg>\\r\\n"""

    def __init__(self):
        self.buf = b""

    def feed(self, chunk: bytes) -> list:
        self.buf += chunk
        frames = []
        while True:
            start = self.buf.find(BOUNDARY)
            if start < 0:
                self.buf = self.buf[-len(BOUNDARY):]
                break
            hdr_end = self.buf.find(b"\r\n\r\n", start)
            if hdr_end < 0:
                self.buf = self.buf[start:]
                break
            data_start = hdr_end + 4
            nxt = self.buf.find(b"\r\n" + BOUNDARY, data_start)
            if nxt < 0:
                self.buf = self.buf[start:]
                break
            frames.append(self.buf[data_start:nxt])
            self.buf = self.buf[nxt + 2:]
        if len(self.buf) > MAX_BUF:  # du lieu hong -> bo
            self.buf = b""
        return frames


class MjpegRelay:
    """Chi doc /video_feed khi co nguoi xem hoac can chup anh; tu dung khi khong ai dung."""

    def __init__(self, video_url: str):
        self.url = video_url
        self.cond = threading.Condition()
        self.frame: bytes | None = None
        self.frame_at = 0.0
        self.seq = 0
        self.viewers = 0
        self.demand_at = 0.0
        self.running = False

    # ------------------------------------------------------------------ dung chung
    def _ensure_running(self) -> None:
        with self.cond:
            self.demand_at = time.time()
            if self.running:
                return
            self.running = True
        threading.Thread(target=self._reader, name="mjpeg-reader", daemon=True).start()

    def acquire(self) -> None:
        with self.cond:
            self.viewers += 1
        self._ensure_running()

    def release(self) -> None:
        with self.cond:
            self.viewers = max(0, self.viewers - 1)
            self.demand_at = time.time()

    def wait_frame(self, after_seq: int, timeout: float = 5.0):
        """Doi anh moi hon after_seq. Tra ve (seq, jpg) hoac (after_seq, None) neu het gio."""
        with self.cond:
            self.cond.wait_for(lambda: self.seq != after_seq, timeout=timeout)
            if self.seq == after_seq:
                return after_seq, None
            return self.seq, self.frame

    def snapshot(self, max_age: float = 2.0, timeout: float = 6.0) -> bytes | None:
        with self.cond:
            if self.frame is not None and time.time() - self.frame_at <= max_age:
                return self.frame
            seq = self.seq
        self._ensure_running()
        return self.wait_frame(seq, timeout)[1]

    # ------------------------------------------------------------------ doc nen
    def _idle(self) -> bool:
        with self.cond:
            return self.viewers == 0 and time.time() - self.demand_at > IDLE_STOP_S

    def _reader(self) -> None:
        log.info("Bat dau doc video tu %s", self.url)
        try:
            while not self._idle():
                try:
                    # read1(): tra du lieu ngay khi co, khong doi du kich thuoc chunk
                    with urllib.request.urlopen(self.url, timeout=10.0) as r:
                        parser = MjpegParser()
                        while True:
                            chunk = r.read1(64 * 1024)
                            if not chunk:
                                break
                            for jpg in parser.feed(chunk)[-1:]:  # chi giu anh moi nhat
                                with self.cond:
                                    self.frame, self.frame_at = jpg, time.time()
                                    self.seq += 1
                                    self.cond.notify_all()
                            if self._idle():
                                return
                except (OSError, ValueError) as e:  # URLError, timeout, mat ket noi
                    log.warning("Mat ket noi video: %s", e)
                    time.sleep(1.0)
        finally:
            with self.cond:
                self.running = False
            log.info("Dung doc video (khong con nguoi xem)")
