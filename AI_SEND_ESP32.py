from ultralytics import YOLO
import cv2
import time
import os
import requests
from flask import Flask, Response, jsonify
import threading

# =========================
# DUONG DAN MODEL
# =========================
MODEL_PATH = r"E:\KHKT\DE_TAI\HE_THONG_TE_NGA\train_3class_plus_all\train_3class_plus_all\weights\best.pt"

# =========================
# NGUON CAMERA / VIDEO
# =========================

# Dung webcam laptop:
# CAMERA_SOURCE = 0

# Dung video test:
CAMERA_SOURCE = r"E:\KHKT\DE_TAI\HE_THONG_TE_NGA\chute01\chute01\cam1.avi"

# Dung camera IP RTSP:
# CAMERA_SOURCE = "rtsp://admin:123456@192.168.1.50:554/stream1"

# Dung camera IP HTTP:
# CAMERA_SOURCE = "http://192.168.1.50:8080/video"

# =========================
# IP ESP32
# =========================
ESP32_IP = "192.168.1.25"   # sửa thành IP ESP32 của bạn

# =========================
# CAI DAT AI
# =========================
CONF_THRESHOLD = 0.45
IMG_SIZE = 640

FALL_CONFIRM_FRAMES = 12
WARNING_CONFIRM_FRAMES = 10

MIN_BOX_AREA_RATIO = 0.003
MAX_BOX_AREA_RATIO = 0.65
EDGE_MARGIN = 10
FALL_MIN_CONF = 0.45
MAX_FALL_ASPECT_RATIO = 5.0

# =========================
# FLASK SERVER DE XEM CAMERA AI
# =========================
app = Flask(__name__)

# =========================
# KIEM TRA MODEL
# =========================
if not os.path.exists(MODEL_PATH):
    print("Khong tim thay model:")
    print(MODEL_PATH)
    exit()

# =========================
# LOAD MODEL
# =========================
print("Dang load model...")
model = YOLO(MODEL_PATH)
names = model.names

print("Danh sach class:", names)

# =========================
# MO CAMERA / VIDEO
# =========================
cap = cv2.VideoCapture(CAMERA_SOURCE)

if not cap.isOpened():
    print("Khong mo duoc camera/video.")
    print("Kiem tra CAMERA_SOURCE:", CAMERA_SOURCE)
    exit()

# =========================
# BIEN TRANG THAI
# =========================
fall_count = 0
warning_count = 0
normal_count = 0

last_sent_state = ""
last_send_time = 0
send_interval = 1.0

current_state = "normal"
current_confidence = 0.0
current_alert = "Khong co canh bao"
current_fps = 0.0

prev_time = time.time()

# Frame moi nhat de stream web
latest_frame = None
frame_lock = threading.Lock()


# =========================
# GUI TRANG THAI VE ESP32
# =========================
def send_state_to_esp32(state):
    global last_sent_state, last_send_time

    now = time.time()

    # Khong gui lien tuc qua nhanh
    if state == last_sent_state and now - last_send_time < send_interval:
        return

    try:
        url = f"http://{ESP32_IP}/set?state={state}"
        requests.get(url, timeout=1)

        print("Gui ESP32:", state)

        last_sent_state = state
        last_send_time = now

    except Exception as e:
        print("Khong gui duoc ESP32:", e)


# =========================
# LOC BOX SAI
# =========================
def is_valid_box(x1, y1, x2, y2, frame_w, frame_h, class_name, conf):
    box_w = x2 - x1
    box_h = y2 - y1

    if box_w <= 0 or box_h <= 0:
        return False, 0.0, "box loi"

    box_area = box_w * box_h
    frame_area = frame_w * frame_h
    area_ratio = box_area / frame_area
    aspect_ratio = box_w / max(box_h, 1)

    touch_left = x1 <= EDGE_MARGIN
    touch_top = y1 <= EDGE_MARGIN
    touch_right = x2 >= frame_w - EDGE_MARGIN
    touch_bottom = y2 >= frame_h - EDGE_MARGIN

    if area_ratio < MIN_BOX_AREA_RATIO:
        return False, area_ratio, "box qua nho"

    if area_ratio > MAX_BOX_AREA_RATIO:
        return False, area_ratio, "box qua lon"

    edge_count = 0

    if touch_left:
        edge_count += 1
    if touch_top:
        edge_count += 1
    if touch_right:
        edge_count += 1
    if touch_bottom:
        edge_count += 1

    if edge_count >= 2:
        return False, area_ratio, "cham nhieu mep"

    if class_name == "fall":
        if conf < FALL_MIN_CONF:
            return False, area_ratio, "fall conf thap"

        if aspect_ratio > MAX_FALL_ASPECT_RATIO:
            return False, area_ratio, "fall box qua dai"

    return True, area_ratio, "hop le"


# =========================
# HAM XU LY AI
# =========================
def ai_loop():
    global cap
    global fall_count, warning_count, normal_count
    global current_state, current_confidence, current_alert, current_fps
    global latest_frame, prev_time

    print("AI dang chay...")
    print("Nhan Ctrl + C de dung.")

    while True:
        ret, frame = cap.read()

        # Neu doc het video thi tua lai dau
        if not ret:
            print("Khong doc duoc frame. Dang thu mo lai...")
            cap.release()
            time.sleep(1)
            cap = cv2.VideoCapture(CAMERA_SOURCE)
            continue

        frame_h, frame_w = frame.shape[:2]

        results = model(
            frame,
            conf=CONF_THRESHOLD,
            imgsz=IMG_SIZE,
            verbose=False
        )

        raw_state = "normal"
        detected_state = "normal"
        best_conf = 0.0
        valid_detection = False

        # =========================
        # DOC KET QUA YOLO
        # =========================
        for result in results:
            for box in result.boxes:
                cls_id = int(box.cls[0])
                conf = float(box.conf[0])
                class_name = names[cls_id]

                x1, y1, x2, y2 = map(int, box.xyxy[0])

                valid_box, area_ratio, reason = is_valid_box(
                    x1, y1, x2, y2,
                    frame_w,
                    frame_h,
                    class_name,
                    conf
                )

                if not valid_box:
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (140, 140, 140), 1)

                    cv2.putText(
                        frame,
                        f"ignored {class_name} {conf:.2f}",
                        (max(x1, 10), max(y1 - 10, 25)),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.5,
                        (140, 140, 140),
                        1
                    )
                    continue

                valid_detection = True

                if conf > best_conf:
                    best_conf = conf

                if class_name == "fall":
                    raw_state = "fall"
                elif class_name == "warning" and raw_state != "fall":
                    raw_state = "warning"
                elif class_name == "normal" and raw_state not in ["fall", "warning"]:
                    raw_state = "normal"

                if class_name == "fall":
                    color = (0, 0, 255)
                elif class_name == "warning":
                    color = (0, 255, 255)
                else:
                    color = (0, 255, 0)

                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

                label = f"{class_name} {conf:.2f}"
                cv2.putText(
                    frame,
                    label,
                    (max(x1, 10), max(y1 - 10, 25)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.65,
                    color,
                    2
                )

        if not valid_detection:
            raw_state = "normal"
            best_conf = 0.0

        # =========================
        # XAC NHAN TRANG THAI THEO FRAME
        # =========================
        if raw_state == "fall":
            fall_count += 1
            warning_count = 0
            normal_count = 0

        elif raw_state == "warning":
            warning_count += 1
            fall_count = 0
            normal_count = 0

        else:
            normal_count += 1
            fall_count = 0
            warning_count = 0

        if fall_count >= FALL_CONFIRM_FRAMES:
            detected_state = "fall"
        elif warning_count >= WARNING_CONFIRM_FRAMES:
            detected_state = "warning"
        else:
            detected_state = "normal"

        # =========================
        # FPS
        # =========================
        now = time.time()
        fps = 1 / max(now - prev_time, 0.0001)
        prev_time = now

        current_state = detected_state
        current_confidence = best_conf
        current_fps = fps

        if detected_state == "fall":
            state_color = (0, 0, 255)
            current_alert = "Canh bao! Phat hien te nga"
        elif detected_state == "warning":
            state_color = (0, 255, 255)
            current_alert = "Phat hien tu the bat thuong"
        else:
            state_color = (0, 255, 0)
            current_alert = "Khong co canh bao"

        # =========================
        # GUI ESP32
        # =========================
        send_state_to_esp32(detected_state)

        # =========================
        # VE THONG TIN LEN FRAME
        # =========================
        cv2.rectangle(frame, (10, 10), (900, 120), (0, 0, 0), -1)

        cv2.putText(
            frame,
            f"STATE: {detected_state.upper()} | RAW: {raw_state.upper()} | CONF: {best_conf:.2f}",
            (20, 45),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.85,
            state_color,
            2
        )

        cv2.putText(
            frame,
            f"FPS: {fps:.1f} | FALL_COUNT: {fall_count} | WARNING_COUNT: {warning_count}",
            (20, 80),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            (255, 255, 255),
            2
        )

        cv2.putText(
            frame,
            f"ESP32: {ESP32_IP}",
            (20, 108),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (255, 255, 255),
            2
        )

        with frame_lock:
            latest_frame = frame.copy()


# =========================
# STREAM VIDEO AI LEN WEB
# =========================
def generate_frames():
    global latest_frame

    while True:
        with frame_lock:
            if latest_frame is None:
                continue

            frame = latest_frame.copy()

        ret, buffer = cv2.imencode(".jpg", frame)

        if not ret:
            continue

        frame_bytes = buffer.tobytes()

        yield (
            b"--frame\r\n"
            b"Content-Type: image/jpeg\r\n\r\n" + frame_bytes + b"\r\n"
        )


# =========================
# FLASK ROUTES
# =========================
@app.route("/")
def index():
    return """
    <html>
    <head>
        <meta charset="UTF-8">
        <title>AI Fall Detection</title>
        <style>
            body {
                margin: 0;
                background: #eef2f7;
                font-family: Arial, sans-serif;
                text-align: center;
            }

            header {
                background: #0f172a;
                color: white;
                padding: 18px;
            }

            img {
                width: 92%;
                max-width: 980px;
                margin-top: 20px;
                border-radius: 14px;
                box-shadow: 0 4px 16px rgba(0,0,0,0.2);
            }

            .box {
                width: 92%;
                max-width: 980px;
                margin: 20px auto;
                background: white;
                border-radius: 14px;
                padding: 16px;
            }
        </style>
    </head>

    <body>
        <header>
            <h1>AI PHÁT HIỆN TÉ NGÃ NGƯỜI GIÀ</h1>
            <p>YOLO + Camera/Video + ESP32 Dashboard</p>
        </header>

        <img src="/video_feed">

        <div class="box">
            <h3>Trạng thái AI</h3>
            <p>Xem JSON tại <b>/status</b></p>
        </div>
    </body>
    </html>
    """


@app.route("/video_feed")
def video_feed():
    return Response(
        generate_frames(),
        mimetype="multipart/x-mixed-replace; boundary=frame"
    )


@app.route("/status")
def status():
    return jsonify({
        "state": current_state,
        "confidence": round(current_confidence, 2),
        "alert": current_alert,
        "fps": round(current_fps, 1),
        "esp32_ip": ESP32_IP
    })


# =========================
# MAIN
# =========================
if __name__ == "__main__":
    ai_thread = threading.Thread(target=ai_loop)
    ai_thread.daemon = True
    ai_thread.start()

    print("AI Web Server dang chay...")
    print("Mo tren may nay: http://127.0.0.1:5000")
    print("Video feed: http://127.0.0.1:5000/video_feed")

    app.run(host="0.0.0.0", port=5000, debug=False)