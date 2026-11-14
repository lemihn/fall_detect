"""
Extract Multiple Cameras Fall Dataset (24 scenarios x 8 elevated ceiling-edge cameras)
and convert to .npz pose cache in detect_v3/mc_cache/ using YOLOv8n-Pose.
"""
import csv
import zipfile
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np
from ultralytics import YOLO

from fall_common import FALL, IGNORE, LIE, NORMAL, YOLO_IMGSZ, pick_from_result

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
ZIP_PATH = HERE / "external_data" / "multiple_cameras_fall.zip"
EXT_DIR = HERE / "external_data" / "multiple_cameras_fall"
OUT_DIR = HERE / "mc_cache"


def find_file(base: Path, name: str):
    hits = list(base.rglob(name))
    return hits[0] if hits else None


def load_annotations(csv_path: Path):
    ann = defaultdict(list)
    with csv_path.open("r", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        for row in reader:
            chute = int(float(row["chute"]))
            cam = int(float(row["cam"]))
            start = int(float(row["start"]))
            end = int(float(row["end"]))
            label = int(float(row["label"]))
            ann[(chute, cam)].append((start, end, label))
    return ann


def main():
    if not EXT_DIR.exists() or not any(EXT_DIR.rglob("*.avi")):
        print(f"Extracting {ZIP_PATH} ...")
        with zipfile.ZipFile(ZIP_PATH, "r") as zf:
            zf.extractall(EXT_DIR)
        print("Extraction complete.")

    csv_path = find_file(EXT_DIR, "data_tuple3.csv")
    if csv_path is None:
        raise FileNotFoundError("data_tuple3.csv not found inside extracted dataset")
    ann = load_annotations(csv_path)

    weights = ROOT / "yolov8n-pose.pt"
    pose = YOLO(str(weights))
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    avi_files = sorted(EXT_DIR.rglob("*.avi"))
    print(f"Found {len(avi_files)} AVI videos. Extracting poses (imgsz={YOLO_IMGSZ})...")

    saved = skipped = 0
    for idx, avi in enumerate(avi_files, 1):
        # Expect path like .../chute01/cam1.avi
        chute_dir = avi.parent.name
        cam_stem = avi.stem
        if not chute_dir.startswith("chute") or not cam_stem.startswith("cam"):
            continue
        chute_id = int(chute_dir.replace("chute", ""))
        cam_id = int(cam_stem.replace("cam", ""))
        out_file = OUT_DIR / f"mc_chute{chute_id:02d}_cam{cam_id}.npz"
        if out_file.exists():
            saved += 1
            continue

        cap = cv2.VideoCapture(str(avi))
        if not cap.isOpened():
            skipped += 1
            continue
        W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        kpts_list, kconf_list, box_list, present_list = [], [], [], []
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            res = pose.predict(frame, imgsz=YOLO_IMGSZ, conf=0.25, verbose=False)[0]
            got = pick_from_result(res, W, H)
            if got is None:
                kpts_list.append(np.zeros((17, 2), dtype=np.float32))
                kconf_list.append(np.zeros(17, dtype=np.float32))
                box_list.append(np.zeros(4, dtype=np.float32))
                present_list.append(False)
            else:
                kp, kc, bx = got
                kpts_list.append(kp.astype(np.float32))
                kconf_list.append(kc.astype(np.float32))
                box_list.append(bx.astype(np.float32))
                present_list.append(True)
        cap.release()

        T = len(present_list)
        if T < 30:
            skipped += 1
            continue

        rows = ann.get((chute_id, cam_id), [])
        fall_rows = sorted((s, e) for s, e, lbl in rows if lbl == 1)
        nonfall_rows = sorted((s, e) for s, e, lbl in rows if lbl == 0)

        if not fall_rows:
            # chute24: chuoi ADL khong co te nga -> NORMAL
            phase = np.full(T, NORMAL, dtype=np.int64)
        else:
            # Chi gan nhan chac chan quanh cac moc thoi gian duoc chu thich, phan con lai de IGNORE (-1)
            phase = np.full(T, IGNORE, dtype=np.int64)
            # 1. Cua so truoc nga (pre-fall ~2 giay) la NORMAL
            for s, e in fall_rows:
                pre_a = max(0, s - 60)
                pre_b = max(0, s)
                if pre_b > pre_a:
                    phase[pre_a:pre_b] = NORMAL
            # 2. Cac doan (s, e, 0) khong nam ngay sau cu nga -> NORMAL
            for s, e in nonfall_rows:
                is_post_fall = any(fs <= s <= fe + 45 for fs, fe in fall_rows)
                if not is_post_fall:
                    phase[max(0, s):min(T, e + 1)] = NORMAL
            # 3. Pha roi (FALL) va pha nam ngay sau khi cham san (~1 giay = 30 frame -> LIE)
            for idx_f, (s, e) in enumerate(fall_rows):
                s_idx = max(0, s)
                e_idx = min(T - 1, e + 5)
                next_s = fall_rows[idx_f + 1][0] if idx_f + 1 < len(fall_rows) else T
                if e_idx >= s_idx:
                    phase[s_idx:min(e_idx + 1, next_s)] = FALL
                    lie_a = e_idx + 1
                    lie_b = min(T, e_idx + 35, max(lie_a, next_s - 60))
                    if lie_b > lie_a:
                        phase[lie_a:lie_b] = LIE

        np.savez_compressed(
            out_file,
            kpts=np.stack(kpts_list),
            kconf=np.stack(kconf_list),
            box=np.stack(box_list),
            present=np.array(present_list, dtype=bool),
            phase=phase,
            size=np.array([W, H], dtype=np.int64),
            chute=np.int64(chute_id),
            cam=np.int64(cam_id),
        )
        saved += 1
        if idx % 10 == 0 or idx == len(avi_files):
            print(f"  [{idx}/{len(avi_files)}] saved={saved}, skipped={skipped}", flush=True)

    print(f"Done! Saved {saved} sequences to {OUT_DIR} (skipped={skipped}).")


if __name__ == "__main__":
    main()
