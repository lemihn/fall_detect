"""
Convert GMDCSA24 dataset (Subject 1..4, 160 MP4 videos + ADL.csv / Fall.csv)
into .npz pose cache in detect_v3/gmd_cache/ using YOLOv8n-Pose.
"""
import csv
import re
from pathlib import Path

import cv2
import numpy as np
from ultralytics import YOLO

from fall_common import FALL, IGNORE, LIE, NORMAL, SIT, YOLO_IMGSZ, pick_from_result

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
GMD_DIR = HERE / "external_data" / "gmdcsa24" / "GMDCSA24-A-Dataset-for-Human-Fall-Detection-in-Videos-master"
OUT_DIR = HERE / "gmd_cache"

CLASS_PAT = re.compile(r"([A-Za-z ()]+?)\s*\[([^\[\]]*)")
RANGE_PAT = re.compile(r"([\d.]+)\s*to\s*([\d.]+)")


def parse_intervals(cls_str: str):
    out = []
    for m in CLASS_PAT.finditer(cls_str or ""):
        lbl = m.group(1).strip().lower()
        body = m.group(2)
        for rm in RANGE_PAT.finditer(body):
            t0 = float(rm.group(1))
            t1 = float(rm.group(2))
            out.append((lbl, t0, t1))
    return out


def main():
    weights = ROOT / "yolov8n-pose.pt"
    pose = YOLO(str(weights))
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    tasks = []
    for subj_dir in sorted(GMD_DIR.glob("Subject *")):
        subj_num = int(subj_dir.name.replace("Subject", "").strip())
        for kind_name, csv_name in [("adl", "ADL.csv"), ("fall", "Fall.csv")]:
            csv_path = subj_dir / csv_name
            vid_dir = subj_dir / ("ADL" if kind_name == "adl" else "Fall")
            if not csv_path.exists():
                continue
            with csv_path.open("r", encoding="utf-8-sig") as f:
                for row in csv.DictReader(f):
                    fname = (row.get("File Name") or "").strip()
                    cls_str = row.get(" Classes") or row.get("Classes") or ""
                    vpath = vid_dir / fname
                    if vpath.exists():
                        tasks.append((subj_num, kind_name, vpath, parse_intervals(cls_str)))

    print(f"Found {len(tasks)} GMDCSA24 videos across 4 subjects. Extracting poses (imgsz={YOLO_IMGSZ})...")
    saved = skipped = 0
    for idx, (subj_num, kind_name, vpath, intervals) in enumerate(tasks, 1):
        out_file = OUT_DIR / f"gmd_s{subj_num}_{kind_name}_{vpath.stem}.npz"
        if out_file.exists():
            saved += 1
            continue

        cap = cv2.VideoCapture(str(vpath))
        if not cap.isOpened():
            skipped += 1
            continue
        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        if fps <= 1.0 or fps > 65.0:
            fps = 30.0
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

        times = np.arange(T, dtype=np.float32) / float(fps)
        phase = np.full(T, IGNORE, dtype=np.int64)

        # Uu tien dung thu tu: NORMAL < SIT < LIE (nguoi dang ngoi ma doc/viet van la SIT)
        for lbl, t0, t1 in intervals:
            if "fall" in lbl or "sit" in lbl or "sleep" in lbl:
                continue
            phase[(times >= t0) & (times <= t1)] = NORMAL
        for lbl, t0, t1 in intervals:
            if "sit" in lbl:
                phase[(times >= t0) & (times <= t1)] = SIT
        for lbl, t0, t1 in intervals:
            if "sleep" in lbl:
                phase[(times >= t0) & (times <= t1)] = LIE
        for lbl, t0, t1 in intervals:
            if "fall" in lbl:
                # Pha roi dot ngot (~1.25s tu luc bat dau nga), sau do nam yen (LIE)
                t_impact = min(t1, t0 + 1.25)
                phase[(times >= t0) & (times <= t_impact)] = FALL
                if t1 > t_impact:
                    phase[(times > t_impact) & (times <= t1)] = LIE

        np.savez_compressed(
            out_file,
            kpts=np.stack(kpts_list),
            kconf=np.stack(kconf_list),
            box=np.stack(box_list),
            present=np.array(present_list, dtype=bool),
            phase=phase,
            size=np.array([W, H], dtype=np.int64),
            subject=np.int64(subj_num),
            kind=kind_name,
        )
        saved += 1
        if idx % 10 == 0 or idx == len(tasks):
            print(f"  [{idx}/{len(tasks)}] saved={saved}, skipped={skipped}", flush=True)

    print(f"Done! Saved {saved} GMDCSA24 sequences to {OUT_DIR} (skipped={skipped}).")


if __name__ == "__main__":
    main()
