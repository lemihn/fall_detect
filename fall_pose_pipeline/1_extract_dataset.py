import os
import re
import math
import json
import time
from collections import Counter
import numpy as np
from ultralytics import YOLO

# ==========================================
# CAU HINH (DYNAMIC PATHS)
# ==========================================
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
BASE_DIR = os.path.dirname(SCRIPT_DIR)

IMG_DIR = os.path.join(BASE_DIR, "fall_detect_new", "images", "train")
LBL_DIR = os.path.join(BASE_DIR, "fall_detect_new", "labels", "train")
OUTPUT_DIR = SCRIPT_DIR
MODEL_POSE_PATH = os.path.join(BASE_DIR, "yolov8n-pose.pt")

SEQ_LEN = 20       # Do dai chuoi frame (20 frames ~ 0.66 - 0.8 giay)
STEP = 2           # Do truot (stride) giua cac chuoi
VAL_RATIO = 0.25   # 25% su kien dung de kiem thu (Val)
RANDOM_SEED = 42

os.makedirs(OUTPUT_DIR, exist_ok=True)


# ==========================================
# HAM TRICH XUAT FEATURE KHUNG XUONG AN TOAN (36 dims)
# ==========================================
def is_valid_kpt(pt):
    return pt[0] > 1.0 and pt[1] > 1.0


def get_valid_midpoint(kpts_xy, idx1, idx2):
    v1 = is_valid_kpt(kpts_xy[idx1])
    v2 = is_valid_kpt(kpts_xy[idx2])
    if v1 and v2:
        return (kpts_xy[idx1][0] + kpts_xy[idx2][0]) / 2.0, (kpts_xy[idx1][1] + kpts_xy[idx2][1]) / 2.0
    if v1:
        return float(kpts_xy[idx1][0]), float(kpts_xy[idx1][1])
    if v2:
        return float(kpts_xy[idx2][0]), float(kpts_xy[idx2][1])
    return None


def extract_frame_features(kpts_xy, box_xyxy):
    """
    kpts_xy: (17, 2)
    box_xyxy: [x1, y1, x2, y2]
    Output: vector 36 chieu da chuan hoa (xu ly an toan cac khop bi khuat = (0, 0))
    """
    x1, y1, x2, y2 = box_xyxy
    w = max(float(x2 - x1), 1.0)
    h = max(float(y2 - y1), 1.0)
    scale = max(w, h, 1.0)

    cx = (x1 + x2) / 2.0
    cy = (y1 + y2) / 2.0

    # 17 normalized keypoints (x, y) -> 34 values
    norm_kpts = []
    for i in range(17):
        if i < len(kpts_xy) and is_valid_kpt(kpts_xy[i]):
            norm_x = (float(kpts_xy[i][0]) - cx) / scale
            norm_y = (float(kpts_xy[i][1]) - cy) / scale
        else:
            norm_x, norm_y = 0.0, 0.0
        norm_kpts.extend([norm_x, norm_y])

    aspect_ratio = w / h

    # Torso angle so voi phuong thang dung: Shoulder center (5, 6), Hip center (11, 12)
    sh_pt = get_valid_midpoint(kpts_xy, 5, 6) if len(kpts_xy) >= 13 else None
    hip_pt = get_valid_midpoint(kpts_xy, 11, 12) if len(kpts_xy) >= 13 else None

    if sh_pt is not None and hip_pt is not None:
        dx = sh_pt[0] - hip_pt[0]
        dy = sh_pt[1] - hip_pt[1]
        torso_angle = math.atan2(abs(dx), abs(dy) + 1e-4)
    else:
        torso_angle = math.atan2(w, h + 1e-4) if aspect_ratio > 1.15 else 0.0

    features = norm_kpts + [aspect_ratio, torso_angle]
    return np.array(features, dtype=np.float32)


def select_primary_person(res):
    if res.keypoints is None or res.boxes is None:
        return None, None
    num_det = min(len(res.keypoints), len(res.boxes))
    if num_det == 0:
        return None, None

    boxes_xyxy = res.boxes.xyxy.cpu().numpy()
    kpts_all = res.keypoints.xy.cpu().numpy()

    best_idx = -1
    best_area = -1.0
    for i in range(num_det):
        if kpts_all[i].shape[0] < 17:
            continue
        x1, y1, x2, y2 = boxes_xyxy[i]
        area = max(0.0, x2 - x1) * max(0.0, y2 - y1)
        if area > best_area:
            best_area = area
            best_idx = i

    if best_idx == -1:
        return None, None
    return kpts_all[best_idx], boxes_xyxy[best_idx]


# ==========================================
# KIEM TOAN CHONG RO RI DU LIEU (ZERO DATA LEAKAGE)
# ==========================================
def extract_event_id(seq_name):
    """Lay ID su kien goc (vd: 'fall-09-cam0' -> 'fall-09') de cam0 va cam1 khong bi tach sang 2 tap."""
    m = re.match(r"((?:adl|fall)-\d+)", seq_name)
    return m.group(1) if m else seq_name


def verify_zero_leakage(train_sequences, val_sequences):
    """Xac nhan khong co chuoi hoac su kien nao trung lap giua Train va Val."""
    seq_overlap = set(train_sequences).intersection(set(val_sequences))
    if seq_overlap:
        raise ValueError(f"PHAT HIEN DATA LEAKAGE (Sequence Overlap): {seq_overlap}")

    train_events = {extract_event_id(s) for s in train_sequences}
    val_events = {extract_event_id(s) for s in val_sequences}
    event_overlap = train_events.intersection(val_events)
    if event_overlap:
        raise ValueError(f"PHAT HIEN DATA LEAKAGE GIUA CAC GOC CAMERA (Event Overlap): {event_overlap}")

    print(">>> XAC NHAN: Khong co ro ri du lieu giua Train va Val (Zero Sequence & Camera Leakage).")


# ==========================================
# PHAN TICH DANH SACH SEQUENCE
# ==========================================
def get_sequences():
    img_files = os.listdir(IMG_DIR)
    lbl_files = set(os.listdir(LBL_DIR))

    seq_dict = {}

    for f in img_files:
        # Ten file co dang: <seq>-<cam>-rgb-<frame>...
        m = re.match(r"((adl|fall)-\d+-(cam\d+))-rgb-(\d+)", f)
        if not m:
            continue

        seq_name = m.group(1)
        frame_num = int(m.group(4))

        if seq_name not in seq_dict:
            seq_dict[seq_name] = []

        stem = os.path.splitext(f)[0]
        lbl_file = stem + ".txt"

        label = 0  # Mac dinh la Normal
        has_label = False

        if seq_name.startswith("adl"):
            label = 0
            has_label = True
        elif lbl_file in lbl_files:
            try:
                with open(os.path.join(LBL_DIR, lbl_file), "r", encoding="utf-8") as fp:
                    first_line = fp.readline().strip()
                    if first_line:
                        label = int(first_line.split()[0])
                        has_label = True
            except Exception:
                pass

        if has_label:
            seq_dict[seq_name].append({
                "file": f,
                "frame_num": frame_num,
                "label": label
            })

    # Sap xep tung sequence theo thu tu frame_num
    valid_seqs = {}
    for seq_name, frames in seq_dict.items():
        if len(frames) >= SEQ_LEN:
            frames.sort(key=lambda x: x["frame_num"])
            valid_seqs[seq_name] = frames

    return valid_seqs


# ==========================================
# MAIN EXTRACTION
# ==========================================
def main():
    print(">>> Dang load model YOLOv8n-pose...")
    model = YOLO(MODEL_POSE_PATH)

    print(">>> Dang quet cac chuoi video...")
    seq_dict = get_sequences()

    # Gom nhom theo Event ID (vd: fall-09 gom ca cam0 va cam1) de chong ro ri cheo camera
    adl_events = sorted({extract_event_id(s) for s in seq_dict if s.startswith("adl")})
    fall_events = sorted({extract_event_id(s) for s in seq_dict if s.startswith("fall")})

    np.random.seed(RANDOM_SEED)
    np.random.shuffle(adl_events)
    np.random.shuffle(fall_events)

    n_val_adl = max(1, int(len(adl_events) * VAL_RATIO))
    n_val_fall = max(1, int(len(fall_events) * VAL_RATIO))

    val_event_ids = set(adl_events[:n_val_adl] + fall_events[:n_val_fall])

    val_seq_names = sorted([s for s in seq_dict if extract_event_id(s) in val_event_ids])
    train_seq_names = sorted([s for s in seq_dict if extract_event_id(s) not in val_event_ids])

    verify_zero_leakage(train_seq_names, val_seq_names)
    print(f"Train sequences: {len(train_seq_names)} | Val sequences: {len(val_seq_names)}")

    val_seq_set = set(val_seq_names)
    X_train, y_train = [], []
    X_val, y_val = [], []

    start_time = time.time()
    total_seqs = len(seq_dict)

    for idx, (seq_name, frames) in enumerate(seq_dict.items()):
        is_val = seq_name in val_seq_set

        seq_features = []
        seq_labels = []
        last_valid_feat = np.zeros(36, dtype=np.float32)

        for item in frames:
            img_path = os.path.join(IMG_DIR, item["file"])
            res = model(img_path, verbose=False)[0]
            kpts, box = select_primary_person(res)

            if kpts is not None and box is not None:
                feat = extract_frame_features(kpts, box)
                last_valid_feat = feat
            else:
                feat = last_valid_feat

            seq_features.append(feat)
            seq_labels.append(item["label"])

        # Cat thanh cac cua so thoi gian (Sliding Window)
        seq_features = np.array(seq_features, dtype=np.float32)

        for i in range(0, len(seq_features) - SEQ_LEN + 1, STEP):
            window_feat = seq_features[i: i + SEQ_LEN]
            window_label = seq_labels[i + SEQ_LEN - 1]  # Nhan cua frame ket thuc

            if is_val:
                X_val.append(window_feat)
                y_val.append(window_label)
            else:
                X_train.append(window_feat)
                y_train.append(window_label)

        if (idx + 1) % 5 == 0 or (idx + 1) == total_seqs:
            elapsed = time.time() - start_time
            print(f"[{idx+1}/{total_seqs}] Da xu ly: {seq_name} ({len(frames)} frames) - Elapsed: {elapsed:.1f}s")

    X_train = np.array(X_train, dtype=np.float32)
    y_train = np.array(y_train, dtype=np.int64)
    X_val = np.array(X_val, dtype=np.float32)
    y_val = np.array(y_val, dtype=np.int64)

    print("\n" + "=" * 50)
    print("HOAN THANH TRICH XUAT DU LIEU!")
    print(f"X_train shape: {X_train.shape}, y_train shape: {y_train.shape}")
    print(f"X_val shape: {X_val.shape}, y_val shape: {y_val.shape}")
    print("Train label counts:", Counter(y_train.tolist()))
    print("Val label counts:", Counter(y_val.tolist()))

    np.save(os.path.join(OUTPUT_DIR, "X_train.npy"), X_train)
    np.save(os.path.join(OUTPUT_DIR, "y_train.npy"), y_train)
    np.save(os.path.join(OUTPUT_DIR, "X_val.npy"), X_val)
    np.save(os.path.join(OUTPUT_DIR, "y_val.npy"), y_val)

    meta = {
        "seq_len": SEQ_LEN,
        "step": STEP,
        "features_per_frame": 36,
        "train_samples": int(len(X_train)),
        "val_samples": int(len(X_val)),
        "train_sequences": train_seq_names,
        "val_sequences": val_seq_names
    }
    with open(os.path.join(OUTPUT_DIR, "meta.json"), "w", encoding="utf-8") as fp:
        json.dump(meta, fp, indent=2)

    print(f"Da luu thanh cong vao thu muc: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
