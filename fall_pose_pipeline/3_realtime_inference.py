import os
import sys
import math
import time
import json
import threading
from collections import deque
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import cv2
import numpy as np
import requests
import torch
import torch.nn as nn
from ultralytics import YOLO

try:
    from flask import Flask, Response, jsonify
    HAS_FLASK = True
except ImportError:
    HAS_FLASK = False

# ==========================================
# CAU HINH HE THONG (DYNAMIC PATHS)
# ==========================================
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
BASE_DIR = os.path.dirname(SCRIPT_DIR)

MODEL_POSE_PATH = os.path.join(BASE_DIR, "yolov8n-pose.pt")
MODEL_LSTM_PATH = os.path.join(SCRIPT_DIR, "fall_lstm_model.pth")

# Nguon camera (1 cho webcam phu, 0 cho webcam mac dinh, hoac duong dan file video)
CAMERA_SOURCE = 1
ESP32_IP = "192.168.1.25"  # Dia chi IP cua ESP32

# Cau hinh suy luan & Bo loc nhay cho Webcam
SEQ_LEN = 20
YOLO_IMGSZ = 320                # Tang toc FPS tren CPU (~40+ FPS)
POSE_CONF_THRESHOLD = 0.45      # Nguong tin cay YOLO-Pose de khong bat nham do vat/quan ao trong tu
MIN_VALID_KPTS = 5              # So diem khop ro rang toi thieu de xac nhan la nguoi that
FALL_CONFIRM_FRAMES = 4         # Xac nhan khi phat hien nga (~0.15 - 0.2s)
WARNING_CONFIRM_FRAMES = 3      # Xac nhan khi mat thang bang / cui sau
MAX_COUNTER_CAP = 10            # Gioi han tran bo dem de phuc hoi nhanh khi dung day
IMMOBILE_ALERT_SECONDS = 3.0    # Thoi gian bat dong sau khi nga de bao khan cap

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
CLASS_NAMES = ["normal", "fall", "warning"]


# ==========================================
# DINH NGHIA MANG LSTM
# ==========================================
class FallDetectionLSTM(nn.Module):
    def __init__(self, input_size=36, hidden_size=64, num_layers=2, num_classes=3, dropout=0.3):
        super(FallDetectionLSTM, self).__init__()
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
            bidirectional=False
        )
        self.classifier = nn.Sequential(
            nn.Linear(hidden_size, 32),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(32, num_classes)
        )

    def forward(self, x):
        lstm_out, _ = self.lstm(x)
        last_out = lstm_out[:, -1, :]
        return self.classifier(last_out)


# ==========================================
# TRICH XUAT FEATURE & DONG HOC CO THE (BIOMECHANICS)
# ==========================================
def is_valid_kpt(pt, conf=1.0):
    return pt[0] > 2.0 and pt[1] > 2.0 and conf >= 0.20


def get_valid_midpoint(kpts_xy, kpts_conf, idx1, idx2):
    v1 = is_valid_kpt(kpts_xy[idx1], kpts_conf[idx1])
    v2 = is_valid_kpt(kpts_xy[idx2], kpts_conf[idx2])
    if v1 and v2:
        return (
            (float(kpts_xy[idx1][0]) + float(kpts_xy[idx2][0])) / 2.0,
            (float(kpts_xy[idx1][1]) + float(kpts_xy[idx2][1])) / 2.0,
        )
    if v1:
        return float(kpts_xy[idx1][0]), float(kpts_xy[idx1][1])
    if v2:
        return float(kpts_xy[idx2][0]), float(kpts_xy[idx2][1])
    return None


def extract_frame_features_and_kinematics(kpts_xy, kpts_conf, box_xyxy, frame_shape):
    """
    1. Tao vector 36 chieu tuong thich 100% voi trong so fall_lstm_model.pth da huan luyen.
    2. Tinh toan them cac chi so Co sinh hoc (Biomechanics) cho Webcam:
       - body_tilt_deg: Goc nghieng truc co the (tinh ca Dau-Vai va Vai-Hong cho webcam goc gan)
       - aspect_ratio: Ty le Rong/Cao cua bounding box
       - upper_y_norm: Vi tri do cao cua Dau/Vai tren khung hinh (0.0 = dinh, 1.0 = day man hinh)
       - head_below_shoulder: Dau bi chúc xuong ngang hoac thap hon vai
    """
    frame_h, frame_w = frame_shape[:2]
    x1, y1, x2, y2 = box_xyxy
    w = max(float(x2 - x1), 1.0)
    h = max(float(y2 - y1), 1.0)
    scale = max(w, h, 1.0)

    cx = (x1 + x2) / 2.0
    cy = (y1 + y2) / 2.0

    # 34 chieu keypoint chuan hoa dung theo cong thuc luc train LSTM
    norm_kpts = []
    for (kx, ky) in kpts_xy:
        norm_x = (float(kx) - cx) / scale
        norm_y = (float(ky) - cy) / scale
        norm_kpts.extend([norm_x, norm_y])

    aspect_ratio = w / h

    # Tinh trung diem Dau (Mui/Mat/Tai 0..4), Vai (5, 6), Hong (11, 12)
    head_pts = [kpts_xy[i] for i in range(5) if is_valid_kpt(kpts_xy[i], kpts_conf[i])]
    head_pt = (
        float(np.mean([p[0] for p in head_pts])),
        float(np.mean([p[1] for p in head_pts]))
    ) if head_pts else None

    sh_pt = get_valid_midpoint(kpts_xy, kpts_conf, 5, 6)
    hip_pt = get_valid_midpoint(kpts_xy, kpts_conf, 11, 12)

    # 1. Goc nghieng Vai - Hong (Torso angle)
    angles_rad = []
    if sh_pt is not None and hip_pt is not None:
        dx = sh_pt[0] - hip_pt[0]
        dy = sh_pt[1] - hip_pt[1]
        if math.hypot(dx, dy) > 15.0:
            angles_rad.append(math.atan2(abs(dx), max(dy, 1e-4)))

    # 2. Goc nghieng Dau - Vai (Ho tro webcam goc gan khi khong thay Hong)
    if head_pt is not None and sh_pt is not None:
        dx_hs = head_pt[0] - sh_pt[0]
        dy_hs = sh_pt[1] - head_pt[1]  # Binh thuong dau o cao hon vai -> dy_hs > 0
        if math.hypot(dx_hs, dy_hs) > 10.0:
            angles_rad.append(math.atan2(abs(dx_hs), max(dy_hs, 1e-4)))

    # 3. Goc nghieng Dau - Hong
    if head_pt is not None and hip_pt is not None:
        dx_hh = head_pt[0] - hip_pt[0]
        dy_hh = hip_pt[1] - head_pt[1]
        if math.hypot(dx_hh, dy_hh) > 20.0:
            angles_rad.append(math.atan2(abs(dx_hh), max(dy_hh, 1e-4)))

    torso_angle = max(angles_rad) if angles_rad else math.atan2(abs(float(kpts_xy[5][0] - kpts_xy[11][0])), abs(float(kpts_xy[5][1] - kpts_xy[11][1])) + 1e-4)

    # Vi tri do cao phan than tren (Dau/Vai) so voi chieu cao khung hinh (0.0 -> 1.0)
    if head_pt is not None:
        upper_y_norm = head_pt[1] / max(frame_h, 1.0)
    elif sh_pt is not None:
        upper_y_norm = sh_pt[1] / max(frame_h, 1.0)
    else:
        upper_y_norm = cy / max(frame_h, 1.0)

    head_below_shoulder = False
    if head_pt is not None and sh_pt is not None:
        # Neu dau ha thap ngang bang hoac thap hon vai
        if (head_pt[1] - sh_pt[1]) > -0.04 * h:
            head_below_shoulder = True

    feat = np.array(norm_kpts + [aspect_ratio, torso_angle], dtype=np.float32)
    kinematics = {
        "aspect_ratio": aspect_ratio,
        "torso_angle": torso_angle,
        "tilt_deg": math.degrees(torso_angle),
        "upper_y_norm": upper_y_norm,
        "box_cy_norm": cy / max(frame_h, 1.0),
        "box_h_norm": h / max(frame_h, 1.0),
        "head_below_shoulder": head_below_shoulder,
    }
    return feat, kinematics


def select_primary_person(results, frame_shape):
    """Chon nguoi co dien tich bounding box lon nhat va du so diem khop hop le (loc bo quan ao/do vat)."""
    if results.keypoints is None or results.boxes is None:
        return None, None, None

    num_det = min(len(results.keypoints), len(results.boxes))
    if num_det == 0:
        return None, None, None

    frame_h, frame_w = frame_shape[:2]
    min_box_area = 0.015 * frame_w * frame_h

    boxes_xyxy = results.boxes.xyxy.cpu().numpy()
    kpts_all = results.keypoints.xy.cpu().numpy()
    confs_all = (
        results.keypoints.conf.cpu().numpy()
        if results.keypoints.conf is not None
        else np.ones((num_det, 17), dtype=np.float32)
    )

    best_idx = -1
    best_area = -1.0
    for i in range(num_det):
        if kpts_all[i].shape[0] < 17:
            continue
        x1, y1, x2, y2 = boxes_xyxy[i]
        area = max(0.0, x2 - x1) * max(0.0, y2 - y1)
        if area < min_box_area:
            continue
        # Kiem tra so luong khop xuong ro rang (tranh bat nham quan ao treo trong tu)
        valid_kpt_count = sum(1 for k in range(17) if kpts_all[i][k][0] > 2.0 and kpts_all[i][k][1] > 2.0 and confs_all[i][k] >= 0.35)
        if valid_kpt_count < MIN_VALID_KPTS:
            continue
        if area > best_area:
            best_area = area
            best_idx = i

    if best_idx == -1:
        return None, None, None

    return kpts_all[best_idx], confs_all[best_idx], boxes_xyxy[best_idx]


# ==========================================
# GUI TRANG THAI SANG ESP32 (NON-BLOCKING ASYNC)
# ==========================================
iot_lock = threading.Lock()
last_sent_state = ""
last_send_time = 0.0
send_interval = 1.0
is_sending_esp32 = False


def _async_send_worker(state, ip):
    global is_sending_esp32
    try:
        url = f"http://{ip}/set?state={state}"
        requests.get(url, timeout=1.0)
        print(f"[IoT Bridge] Da gui den ESP32: {state}")
    except Exception:
        pass
    finally:
        with iot_lock:
            is_sending_esp32 = False


def send_state_to_esp32_async(state):
    global last_sent_state, last_send_time, is_sending_esp32
    now = time.time()
    with iot_lock:
        if is_sending_esp32:
            return
        if state == last_sent_state and (now - last_send_time) < send_interval:
            return
        last_sent_state = state
        last_send_time = now
        is_sending_esp32 = True

    t = threading.Thread(target=_async_send_worker, args=(state, ESP32_IP), daemon=True)
    t.start()


# ==========================================
# WEB STREAM SERVER (FLASK HOAC BUILT-IN HTTP SERVER)
# ==========================================
state_lock = threading.Lock()
latest_frame = None
current_state = "normal"
current_confidence = 0.0
current_fps = 0.0
current_alert = "An toan"
active_camera_source = CAMERA_SOURCE
requested_camera_source = None
reset_latch_requested = False

HTML_PAGE = """<!DOCTYPE html>
<html>
<head>
    <meta charset="utf-8">
    <title>Fall Detection AI (YOLO-Pose + LSTM)</title>
    <style>
        body { text-align:center; background:#111827; color:#f3f4f6; font-family:sans-serif; margin:0; padding:20px; }
        .cam-btn { background:#1f2937; color:#f3f4f6; border:1px solid #4b5563; padding:8px 16px; margin:0 6px; border-radius:8px; cursor:pointer; font-size:14px; }
        .cam-btn:hover { background:#374151; }
        .reset-btn { background:#7f1d1d; color:#fecaca; border:1px solid #ef4444; padding:8px 16px; margin:0 6px; border-radius:8px; cursor:pointer; font-size:14px; }
        .reset-btn:hover { background:#991b1b; }
    </style>
</head>
<body>
    <h1>HỆ THỐNG GIÁM SÁT TÉ NGÃ THỜI GIAN THỰC</h1>
    <p>AI: YOLO-Pose + LSTM + Biomechanics | IoT: ESP32 Dashboard</p>
    <div style="margin-bottom: 14px;">
        <button class="cam-btn" onclick="switchCam(0)">Camera 0 (Mặc định)</button>
        <button class="cam-btn" onclick="switchCam(1)">Camera 1 (Webcam phụ)</button>
        <button class="reset-btn" onclick="resetAlert()">Reset Cảnh Báo</button>
        <span id="cam-msg" style="margin-left:10px; color:#9ca3af; font-size:14px;"></span>
    </div>
    <img src="/video_feed" style="width:80%; max-width:960px; border-radius:12px; border:2px solid #374151;">
    <script>
        function switchCam(idx) {
            document.getElementById('cam-msg').innerText = 'Đang chuyển sang Camera ' + idx + '...';
            fetch('/switch_camera/' + idx)
                .then(r => r.json())
                .then(d => { document.getElementById('cam-msg').innerText = 'Đang dùng Camera ' + d.requested_camera; });
        }
        function resetAlert() {
            fetch('/reset_alert')
                .then(r => r.json())
                .then(() => { document.getElementById('cam-msg').innerText = 'Đã reset trạng thái về Normal'; });
        }
    </script>
</body>
</html>
"""


def get_status_dict():
    with state_lock:
        return {
            "state": current_state,
            "alert": current_alert,
            "confidence": float(current_confidence),
            "fps": float(current_fps),
            "camera": active_camera_source,
            "esp32_ip": ESP32_IP
        }


def request_camera_switch(cam_idx: int):
    global requested_camera_source
    with state_lock:
        requested_camera_source = cam_idx
    return {"status": "ok", "requested_camera": cam_idx}


def request_reset_alert():
    global reset_latch_requested
    with state_lock:
        reset_latch_requested = True
    return {"status": "ok"}


if HAS_FLASK:
    app = Flask(__name__)

    @app.route("/")
    def index():
        return HTML_PAGE

    @app.route("/video_feed")
    def video_feed():
        def generate():
            while True:
                with state_lock:
                    frame_to_encode = None if latest_frame is None else latest_frame.copy()
                if frame_to_encode is None:
                    time.sleep(0.02)
                    continue
                ret, buffer = cv2.imencode(".jpg", frame_to_encode)
                if not ret:
                    continue
                yield (b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" + buffer.tobytes() + b"\r\n")
                time.sleep(0.03)
        return Response(generate(), mimetype="multipart/x-mixed-replace; boundary=frame")

    @app.route("/status")
    def status():
        return jsonify(get_status_dict())

    @app.route("/switch_camera/<int:cam_idx>")
    def switch_cam_route(cam_idx):
        return jsonify(request_camera_switch(cam_idx))

    @app.route("/reset_alert")
    def reset_alert_route():
        return jsonify(request_reset_alert())


class StreamRequestHandler(BaseHTTPRequestHandler):
    def log_message(self, format, *args):
        pass

    def do_GET(self):
        if self.path == "/" or self.path.startswith("/index"):
            body = HTML_PAGE.encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
        elif self.path.startswith("/switch_camera/"):
            try:
                cam_idx = int(self.path.split("/switch_camera/")[1].split("?")[0])
            except ValueError:
                cam_idx = 0
            body = json.dumps(request_camera_switch(cam_idx)).encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
        elif self.path.startswith("/reset_alert"):
            body = json.dumps(request_reset_alert()).encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
        elif self.path.startswith("/status"):
            body = json.dumps(get_status_dict()).encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
        elif self.path.startswith("/video_feed"):
            self.send_response(200)
            self.send_header("Age", "0")
            self.send_header("Cache-Control", "no-cache, private")
            self.send_header("Pragma", "no-cache")
            self.send_header("Content-Type", "multipart/x-mixed-replace; boundary=frame")
            self.end_headers()
            try:
                while True:
                    with state_lock:
                        frame_to_encode = None if latest_frame is None else latest_frame.copy()
                    if frame_to_encode is None:
                        time.sleep(0.02)
                        continue
                    ret, buffer = cv2.imencode(".jpg", frame_to_encode)
                    if not ret:
                        continue
                    jpg_bytes = buffer.tobytes()
                    self.wfile.write(b"--frame\r\n")
                    self.wfile.write(b"Content-Type: image/jpeg\r\n")
                    self.wfile.write(f"Content-Length: {len(jpg_bytes)}\r\n\r\n".encode("utf-8"))
                    self.wfile.write(jpg_bytes)
                    self.wfile.write(b"\r\n")
                    time.sleep(0.03)
            except (BrokenPipeError, ConnectionResetError, ConnectionAbortedError):
                pass
        else:
            self.send_error(404)


def run_web_server(host="0.0.0.0", port=5000):
    print(f">>> Web Server dang chay tai: http://localhost:{port}")
    if HAS_FLASK:
        app.run(host=host, port=port, debug=False, threaded=True)
    else:
        server = ThreadingHTTPServer((host, port), StreamRequestHandler)
        server.serve_forever()


# ==========================================
# MO CAMERA VOI CO CHE FALLBACK AN TOAN
# ==========================================
def open_camera(source):
    print(f">>> Dang mo camera source = {source} (Media Foundation / Default Backend)...")
    cap = cv2.VideoCapture(source)
    if cap.isOpened():
        cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        ret, test_frame = cap.read()
        if ret and test_frame is not None:
            return cap, source
        cap.release()

    if isinstance(source, int) and source != 0:
        print(f"[CANH BAO] Camera index {source} khong phan hoi, chuyen sang Camera index 0...")
        cap = cv2.VideoCapture(0)
        if cap.isOpened():
            cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
            return cap, 0
    return cap, source


# ==========================================
# AI MAIN LOOP (HYBRID LSTM + KINEMATIC FALL ENGINE)
# ==========================================
def ai_loop():
    global latest_frame, current_state, current_confidence, current_fps, current_alert
    global active_camera_source, requested_camera_source, reset_latch_requested

    print(f">>> Thiet bi suy luan: {DEVICE}")
    print(f">>> Dang tai mo hinh Pose: {MODEL_POSE_PATH}")
    pose_model = YOLO(MODEL_POSE_PATH)

    lstm_model = FallDetectionLSTM(input_size=36, hidden_size=64, num_layers=2, num_classes=3).to(DEVICE)
    if os.path.exists(MODEL_LSTM_PATH):
        ckpt = torch.load(MODEL_LSTM_PATH, map_location=DEVICE)
        state_dict = ckpt["model_state_dict"] if isinstance(ckpt, dict) and "model_state_dict" in ckpt else ckpt
        lstm_model.load_state_dict(state_dict)
        print(">>> Da load thanh cong trong so LSTM tu:", MODEL_LSTM_PATH)
    else:
        print(">>> [CANH BAO] Chua tim thay trong so LSTM tai:", MODEL_LSTM_PATH)

    lstm_model.eval()

    cap, active_camera_source = open_camera(CAMERA_SOURCE)
    if not cap.isOpened():
        print(f">>> [LOI] Khong the mo nguon camera/video: {CAMERA_SOURCE}")
        return

    pose_history = deque(maxlen=SEQ_LEN)
    motion_history = deque(maxlen=30)

    last_valid_feat = None
    missing_counter = 0

    fall_counter = 0
    warning_counter = 0
    fall_start_time = None
    prev_time = time.time()

    print(f">>> AI Loop da san sang tren Camera {active_camera_source}. Nhan Ctrl + C de thoat.")
    while True:
        with state_lock:
            req_cam = requested_camera_source
            requested_camera_source = None
            do_reset = reset_latch_requested
            reset_latch_requested = False

        if do_reset:
            pose_history.clear()
            motion_history.clear()
            fall_counter = 0
            warning_counter = 0
            fall_start_time = None

        if req_cam is not None and req_cam != active_camera_source:
            print(f">>> Dang chuyen doi sang Camera {req_cam}...")
            cap.release()
            new_cap, new_src = open_camera(req_cam)
            if new_cap.isOpened():
                cap = new_cap
                active_camera_source = new_src
                pose_history.clear()
                motion_history.clear()
                fall_counter = 0
                warning_counter = 0
                fall_start_time = None
                print(f">>> Da chuyen sang Camera {active_camera_source} thanh cong.")
            else:
                cap, active_camera_source = open_camera(active_camera_source)

        ret, frame = cap.read()
        if not ret or frame is None:
            if isinstance(active_camera_source, str):
                cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            time.sleep(0.03)
            continue

        now = time.time()
        results = pose_model(frame, imgsz=YOLO_IMGSZ, verbose=False, conf=POSE_CONF_THRESHOLD)[0]
        kpts, kpts_conf, box = select_primary_person(results, frame.shape)

        raw_state = "normal"
        conf = 0.0
        tilt_deg = 0.0
        aspect_ratio = 0.0
        drop_speed = 0.0
        person_present = (kpts is not None and box is not None)

        if person_present:
            feat, kin = extract_frame_features_and_kinematics(kpts, kpts_conf, box, frame.shape)
            last_valid_feat = feat
            missing_counter = 0
            frame = results.plot()

            tilt_deg = kin["tilt_deg"]
            aspect_ratio = kin["aspect_ratio"]
            upper_y = kin["upper_y_norm"]

            motion_history.append((now, upper_y, tilt_deg))
            # Tinh do sut giam do cao (drop_speed) trong 0.20s - 0.90s gan nhat
            recent_y = [item[1] for item in motion_history if 0.15 <= (now - item[0]) <= 0.90]
            if recent_y:
                drop_speed = upper_y - min(recent_y)  # > 0 nghia la dau/vai dang roi xuong phia duoi man hinh

            if len(pose_history) == 0:
                for _ in range(SEQ_LEN):
                    pose_history.append(feat)
            else:
                pose_history.append(feat)

            # 1. Du doan tu mang LSTM
            inp = torch.tensor(np.array([pose_history], dtype=np.float32), device=DEVICE)
            with torch.no_grad():
                logits = lstm_model(inp)
                probs = torch.softmax(logits, dim=1)[0].cpu().numpy()

            p_norm, p_fall, p_warn = float(probs[0]), float(probs[1]), float(probs[2])

            # 2. Hop nhat voi Phan tich Co sinh hoc & Van toc roi (Kinematic Rules)
            bio_fall_score = 0.0
            bio_warn_score = 0.0

            # Truong hop A: Nguoi nam ngang hoac nghieng rap nguoi (> 48 do) hoac khung hinh nam ngang
            if tilt_deg >= 48.0 or aspect_ratio >= 1.15:
                bio_fall_score = max(bio_fall_score, min(0.98, 0.72 + (tilt_deg - 48.0) * 0.01 + max(0.0, aspect_ratio - 1.0) * 0.25))
            elif tilt_deg >= 30.0 or aspect_ratio >= 0.90:
                bio_warn_score = max(bio_warn_score, min(0.92, 0.65 + (tilt_deg - 30.0) * 0.012))

            # Truong hop B: Dau/Vai roi nhanh xuong phia duoi khung hinh
            if drop_speed >= 0.16 and (upper_y >= 0.50 or tilt_deg >= 28.0 or kin["head_below_shoulder"]):
                bio_fall_score = max(bio_fall_score, min(0.98, 0.78 + drop_speed * 0.8))
            elif drop_speed >= 0.11:
                bio_warn_score = max(bio_warn_score, min(0.90, 0.68 + drop_speed * 0.8))

            # Truong hop C: Dau chuc thap ngang/duoi vai hoac nam sat san nha (upper_y >= 0.65)
            if kin["head_below_shoulder"] and (tilt_deg >= 35.0 or aspect_ratio >= 0.85 or upper_y >= 0.55):
                bio_fall_score = max(bio_fall_score, 0.88)
            elif upper_y >= 0.68 and (drop_speed >= 0.08 or tilt_deg >= 30.0 or aspect_ratio >= 0.85):
                bio_fall_score = max(bio_fall_score, 0.85)

            # Ket hop diem so LSTM va Biomechanics (ngan LSTM bao nham khi nguoi dang dung thang)
            if tilt_deg < 22.0 and aspect_ratio < 0.75 and upper_y < 0.55 and drop_speed < 0.06:
                p_fall = 0.0
                p_warn = min(p_warn, 0.30)

            final_fall_score = max(p_fall, bio_fall_score)
            final_warn_score = max(p_warn, bio_warn_score)

            if final_fall_score >= 0.55:
                raw_state = "fall"
                conf = final_fall_score
            elif final_warn_score >= 0.52:
                raw_state = "warning"
                conf = final_warn_score
            else:
                raw_state = "normal"
                conf = max(p_norm, 1.0 - max(final_fall_score, final_warn_score))

            # Phuc hoi nhanh khi nguoi dung thang day
            if tilt_deg < 22.0 and aspect_ratio < 0.80 and upper_y < 0.55 and raw_state == "normal":
                fall_counter = max(0, fall_counter - 3)
                warning_counter = max(0, warning_counter - 3)

        else:
            # KHONG CO NGUOI TRONG KHUNG HINH -> Khong bao gio bao dong gia!
            missing_counter += 1
            if missing_counter > 5:
                pose_history.clear()
                motion_history.clear()
                last_valid_feat = None
                fall_counter = 0
                warning_counter = 0
                fall_start_time = None

        # Bo dem xac nhan trang thai (Debouncing)
        if raw_state == "fall":
            fall_counter = min(fall_counter + 2, MAX_COUNTER_CAP)
            warning_counter = max(0, warning_counter - 1)
        elif raw_state == "warning":
            warning_counter = min(warning_counter + 2, MAX_COUNTER_CAP)
            fall_counter = max(0, fall_counter - 1)
        else:
            fall_counter = max(0, fall_counter - 2)
            warning_counter = max(0, warning_counter - 2)

        final_state = "normal"
        alert_msg = "An toan" if person_present else "An toan (Khong co nguoi)"

        if person_present and fall_counter >= FALL_CONFIRM_FRAMES:
            final_state = "fall"
            if fall_start_time is None:
                fall_start_time = now
            immobile_duration = now - fall_start_time
            if immobile_duration >= IMMOBILE_ALERT_SECONDS:
                alert_msg = f"NGA BAT DONG ({immobile_duration:.1f}s) - KHAN CAP!"
            else:
                alert_msg = "PHAT HIEN TE NGA!"
        elif person_present and warning_counter >= WARNING_CONFIRM_FRAMES:
            final_state = "warning"
            fall_start_time = None
            alert_msg = "Canh bao mat thang bang / nghieng nguoi"
        else:
            fall_start_time = None

        # Tinh FPS
        fps = 1.0 / max(now - prev_time, 0.001)
        prev_time = now

        # Gui den ESP32 (Non-blocking)
        send_state_to_esp32_async(final_state)

        # Ve HUD len video
        state_color = (0, 255, 0)
        if final_state == "fall":
            state_color = (0, 0, 255)
        elif final_state == "warning":
            state_color = (0, 255, 255)

        cv2.rectangle(frame, (10, 10), (590, 128), (0, 0, 0), -1)
        cv2.putText(frame, f"STATUS: {final_state.upper()} ({conf:.2f})", (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, state_color, 2)
        cv2.putText(frame, f"ALERT: {alert_msg}", (20, 72),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.58, state_color, 2)
        cv2.putText(
            frame,
            f"FPS: {fps:.1f} | Cam: {active_camera_source} | Tilt: {tilt_deg:.0f}deg | AR: {aspect_ratio:.2f} | Drop: {drop_speed:.2f}",
            (20, 106),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.50,
            (255, 255, 255),
            1,
        )

        with state_lock:
            current_state = final_state
            current_confidence = round(conf, 2)
            current_fps = round(fps, 1)
            current_alert = alert_msg
            latest_frame = frame.copy()

    cap.release()


# ==========================================
# MAIN
# ==========================================
if __name__ == "__main__":
    t = threading.Thread(target=ai_loop, daemon=True)
    t.start()
    run_web_server(host="0.0.0.0", port=5000)
