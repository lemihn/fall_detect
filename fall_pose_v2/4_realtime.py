"""
Buoc 4: chay THUC TE (webcam / video) -> 4 trang thai: normal | sit | lie | fall  (+ gui ESP32, web dashboard)

    python fall_pose_v2/4_realtime.py --source 0 --esp32-ip 192.168.1.25
    python fall_pose_v2/4_realtime.py --source video.mp4
    python fall_pose_v2/4_realtime.py --selftest video.mp4     # chay khong GUI/khong server, in dong thoi gian trang thai

Khac ban cu: KHONG con luat tay (ti le khung, goc nghieng, van toc). Mo hinh quyet dinh; may trang thai
AlertStateMachine chi lam muot + xac nhan + giu canh bao khi nguoi con nam tren san.
"""
import argparse
import os
import queue
import sys
import threading
import time
from collections import deque
from pathlib import Path

import cv2
import numpy as np
import requests
import torch
from flask import Flask, Response, jsonify
from ultralytics import YOLO

sys.path.insert(0, str(Path(__file__).resolve().parent))
from fall_common import (CLASS_NAMES, FALL, GRID_HZ, SEQ_STEPS, SKELETON, YOLO_IMGSZ,  # noqa: E402
                         AlertStateMachine, load_model, pick_from_result, window_features)

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
WARMUP_S = 1.0          # can it nhat 1 giay lich su moi du doan
STALE_S = 0.35          # mau cu hon nguong nay coi nhu "khong co nguoi"
COLORS = {"normal": (0, 200, 0), "sit": (255, 160, 0), "lie": (255, 0, 160), "fall": (0, 0, 255)}


# ----------------------------------------------------------------------------
class Pipeline:
    """Pose -> bo dem theo THOI GIAN -> cua so 15Hz -> mo hinh -> may trang thai."""

    def __init__(self, model_path, weights, device, sm_kwargs):
        self.device = device
        self.pose = YOLO(weights)
        self.model, self.ck = load_model(model_path, device)
        self.sm = AlertStateMachine(**sm_kwargs)
        self.reset()

    def reset(self):
        self.sm.reset()
        self.buf = deque()                       # (t, present, kp, kc, bx)
        self.last_tick = -1e9
        self.probs = np.array([1, 0, 0, 0], dtype=np.float32)
        self.out = dict(state="normal", alarm=0, posture=0, since=0.0)
        self.last_det = None

    def _window(self, t):
        times = np.array([b[0] for b in self.buf])
        tt = t - (SEQ_STEPS - 1 - np.arange(SEQ_STEPS)) / GRID_HZ
        idx = np.clip(np.searchsorted(times, tt, side="right") - 1, 0, len(times) - 1)
        stale = (tt - times[idx]) > STALE_S
        kp = np.stack([self.buf[i][2] for i in idx])
        kc = np.stack([self.buf[i][3] for i in idx])
        bx = np.stack([self.buf[i][4] for i in idx])
        pr = np.array([self.buf[i][1] for i in idx]) & ~stale
        return window_features(kp, kc, bx, pr)

    def process(self, frame, t):
        H, W = frame.shape[:2]
        res = self.pose.predict(frame, imgsz=YOLO_IMGSZ, conf=0.25, verbose=False)[0]
        got = pick_from_result(res, W, H)
        if got is None:
            z17, z4 = np.zeros((17, 2), np.float32), np.zeros(4, np.float32)
            self.buf.append((t, False, z17, np.zeros(17, np.float32), z4))
        else:
            kp, kc, bx = got
            self.buf.append((t, True, kp, kc, bx))
            self.last_det = (kp, kc)
        while self.buf and t - self.buf[0][0] > 3.0:
            self.buf.popleft()

        warm = (t - self.buf[0][0]) >= WARMUP_S
        if warm and (t - self.last_tick) >= 1.0 / GRID_HZ - 1e-6:
            self.last_tick = t
            x = torch.from_numpy(self._window(t)[None]).to(self.device)
            with torch.no_grad():
                self.probs = torch.softmax(self.model(x), 1)[0].cpu().numpy()
            self.out = self.sm.update(t, self.probs, got is not None)
        return got is not None, warm

    def draw(self, frame, present, warm):
        st = self.out["state"]
        col = COLORS[st]
        if present and self.last_det is not None:
            kp, kc = self.last_det
            for a, b in SKELETON:
                if kc[a] > 0.3 and kc[b] > 0.3:
                    cv2.line(frame, tuple(int(v) for v in kp[a]), tuple(int(v) for v in kp[b]), col, 2)
            for i in range(17):
                if kc[i] > 0.3:
                    cv2.circle(frame, tuple(int(v) for v in kp[i]), 3, (255, 255, 255), -1)
        label = st.upper() if warm else "WARMING UP"
        if self.out["alarm"] == 2:
            label = f"FALL - KHAN CAP ({self.out['since']:.0f}s)"
        cv2.rectangle(frame, (0, 0), (frame.shape[1], 64), (0, 0, 0), -1)
        cv2.putText(frame, label, (12, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.9, col, 2)
        txt = " ".join(f"{n}:{p:.2f}" for n, p in zip(CLASS_NAMES, self.probs))
        cv2.putText(frame, txt, (12, 54), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        if st == "fall":
            cv2.rectangle(frame, (0, 0), (frame.shape[1] - 1, frame.shape[0] - 1), (0, 0, 255), 6)


# ----------------------------------------------------------------------------
class Esp32Sender:
    """Gui trang thai sang ESP32 o thread rieng, KHONG bao gio chan vong lap AI."""

    def __init__(self, ip):
        self.ip, self.q, self.last = ip, queue.Queue(maxsize=1), (None, 0.0)
        if ip:
            threading.Thread(target=self._worker, daemon=True).start()

    def send(self, state):
        if not self.ip:
            return
        s = "fall" if state == "fall" else "normal"          # firmware chi biet normal/warning/fall
        now = time.time()
        if s == self.last[0] and now - self.last[1] < 1.0:
            return
        self.last = (s, now)
        try:
            self.q.get_nowait()
        except queue.Empty:
            pass
        self.q.put_nowait(s)

    def _worker(self):
        while True:
            s = self.q.get()
            try:
                requests.get(f"http://{self.ip}/set?state={s}", timeout=1.0)
            except Exception:
                pass


# ----------------------------------------------------------------------------
class Shared:
    def __init__(self):
        self.lock = threading.Lock()
        self.cond = threading.Condition(self.lock)
        self.jpeg = None
        self.status = dict(state="normal", alarm=0, fps=0.0, probs=[1, 0, 0, 0], camera="-")
        self.reset_req = False
        self.switch_req = None


def open_capture(src):
    cap = cv2.VideoCapture(src)
    if (not cap.isOpened()) and isinstance(src, int):
        cap = cv2.VideoCapture(src, cv2.CAP_DSHOW)
    if cap.isOpened() and isinstance(src, int):
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    return cap


def ai_thread(args, shared, device):
    pipe = Pipeline(args.model, args.weights, device,
                    dict(fall_thr=args.fall_thr, fall_confirm=args.fall_confirm))
    esp = Esp32Sender(args.esp32_ip)
    src = args.source
    cap = open_capture(src)
    print(f">>> Camera/video {src}: {'OK' if cap.isOpened() else 'KHONG MO DUOC'}")
    is_file = isinstance(src, str)
    vid_fps = (cap.get(cv2.CAP_PROP_FPS) or 30.0) if is_file else 30.0
    frame_idx = 0
    prev = time.time()
    fps = 0.0
    while True:
        with shared.lock:
            reset, sw = shared.reset_req, shared.switch_req
            shared.reset_req, shared.switch_req = False, None
        if reset:
            pipe.reset()
        if sw is not None and sw != src:
            cap.release()
            cap, src, is_file = open_capture(sw), sw, isinstance(sw, str)
            vid_fps = (cap.get(cv2.CAP_PROP_FPS) or 30.0) if is_file else 30.0
            frame_idx = 0
            pipe.reset()
        ok, frame = cap.read() if cap.isOpened() else (False, None)
        if not ok:
            if is_file:
                cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                frame_idx = 0
                pipe.reset()
            time.sleep(0.05)
            continue
        now = time.time()
        t_sim = (frame_idx / vid_fps) if is_file else now
        frame_idx += 1
        present, warm = pipe.process(frame, t_sim)
        fps = 0.9 * fps + 0.1 / max(now - prev, 1e-3)
        prev = now
        pipe.draw(frame, present, warm)
        esp.send(pipe.out["state"])
        if args.show:
            cv2.imshow("Fall Detection v2 (Nhan Q de thoat)", frame)
            if (cv2.waitKey(1) & 0xFF) in (ord("q"), ord("Q"), 27):
                os._exit(0)
        ok, jpg = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 80])
        with shared.cond:
            shared.jpeg = jpg.tobytes()
            shared.status = dict(state=pipe.out["state"], alarm=pipe.out["alarm"], fps=round(fps, 1),
                                 probs=[round(float(p), 3) for p in pipe.probs], camera=str(src))
            shared.cond.notify_all()


PAGE = """<!doctype html><meta charset=utf-8><title>Fall detection</title>
<body style="background:#111827;color:#f3f4f6;font-family:sans-serif;text-align:center">
<h2>Giám sát té ngã — normal / sit / lie / fall</h2>
<div id=banner style="margin:8px auto;padding:10px;max-width:640px;border-radius:8px;font-size:22px">...</div>
<img src="/video_feed" style="max-width:96%;border-radius:10px"><br><br>
<button onclick="fetch('/reset_alert')">Reset cảnh báo</button>
<button onclick="fetch('/switch_camera/0')">Camera 0</button>
<button onclick="fetch('/switch_camera/1')">Camera 1</button>
<script>
const C={normal:'#166534',sit:'#9a6700',lie:'#86198f',fall:'#b91c1c'};
setInterval(()=>fetch('/status').then(r=>r.json()).then(s=>{
 const b=document.getElementById('banner');b.style.background=C[s.state];
 b.innerText=s.state.toUpperCase()+(s.alarm==2?' — KHẨN CẤP':'')+'   ('+s.fps+' fps)';}),400);
</script></body>"""


def make_app(shared):
    app = Flask(__name__)

    @app.route("/")
    def index():
        return PAGE

    @app.route("/status")
    def status():
        with shared.lock:
            return jsonify(shared.status)

    @app.route("/reset_alert")
    def reset_alert():
        with shared.lock:
            shared.reset_req = True
        return jsonify(ok=True)

    @app.route("/switch_camera/<int:i>")
    def switch_camera(i):
        with shared.lock:
            shared.switch_req = i
        return jsonify(ok=True, camera=i)

    @app.route("/video_feed")
    def video_feed():
        def gen():
            last = None
            while True:
                with shared.cond:
                    shared.cond.wait_for(lambda: shared.jpeg is not last, timeout=2.0)
                    jpg = shared.jpeg
                if jpg is None or jpg is last:
                    continue
                last = jpg
                yield b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" + jpg + b"\r\n"
        return Response(gen(), mimetype="multipart/x-mixed-replace; boundary=frame")

    return app


# ----------------------------------------------------------------------------
def selftest(args, device):
    """Chay dong bo tren video, thoi gian gia lap theo fps video -> in dong thoi gian trang thai."""
    pipe = Pipeline(args.model, args.weights, device,
                    dict(fall_thr=args.fall_thr, fall_confirm=args.fall_confirm))
    cap = cv2.VideoCapture(args.selftest)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    i, seg = 0, []
    t0 = time.time()
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        t = i / fps
        present, warm = pipe.process(frame, t)
        st = pipe.out["state"] if warm else "warmup"
        if not seg or seg[-1][2] != st:
            seg.append([t, t, st])
        seg[-1][1] = t
        i += 1
    print(f"{i} frame, {time.time() - t0:.1f}s xu ly")
    for a, b, st in seg:
        print(f"  {a:6.2f}s - {b:6.2f}s : {st}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", default="0", help="so (webcam) hoac duong dan video / URL")
    ap.add_argument("--model", type=Path, default=HERE / "models" / "fall_net.pth")
    ap.add_argument("--weights", default=str(ROOT / "yolov8n-pose.pt"))
    ap.add_argument("--esp32-ip", default=os.environ.get("ESP32_IP", ""))
    ap.add_argument("--host", default="0.0.0.0")
    ap.add_argument("--port", type=int, default=5000)
    ap.add_argument("--fall-thr", type=float, default=0.60)
    ap.add_argument("--fall-confirm", type=int, default=3)
    ap.add_argument("--show", action="store_true", help="hien cua so video OpenCV truc tiep")
    ap.add_argument("--selftest", default=None)
    args = ap.parse_args()
    if not Path(args.weights).exists():
        args.weights = "yolov8n-pose.pt"
    args.source = int(args.source) if str(args.source).isdigit() else args.source
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if args.selftest:
        return selftest(args, device)

    shared = Shared()
    threading.Thread(target=ai_thread, args=(args, shared, device), daemon=True).start()
    print(f">>> Mo trinh duyet: http://127.0.0.1:{args.port}")
    make_app(shared).run(host=args.host, port=args.port, debug=False, threaded=True)


if __name__ == "__main__":
    main()
