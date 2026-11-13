"""
Quay du lieu cua CHINH BAN bang webcam (de mo hinh quen voi goc may / phong cua ban).
Rat nen lam: bo URFD chi co ~5 nguoi, 1-2 goc may; webcam cua ban se khac (thay nua tren, gan, nghieng...).

    python fall_pose_v2/record_samples.py --source 0

Phim:
    1  quay clip NORMAL  (di bo, di ngang qua camera, dung, cui nhat do...)    -> bam 1 lan nua / SPACE de luu
    2  quay clip SIT     (ngoi ghe / ngoi giuong)                              -> SPACE de luu
    3  quay clip LIE     (nam giuong / sofa / nam yen)                         -> SPACE de luu
    f  quay clip FALL    : dem nguoc 3s (dang di/dung) -> 'FALL NOW' -> ban nga co kiem soat
                           (NEN NGA LEN NEM / THAM DAY, CO NGUOI CHO BEN CANH). Clip tu dong luu sau ~4s.
    q  thoat
Moi clip -> fall_pose_v2/own_data/<ten>.npz ; train lai bang: python fall_pose_v2/2_train.py
Goi y so luong: moi loai ~10-20 clip, doi quan ao/vi tri/khoang cach; nhieu clip "di ngang qua" (NORMAL).
"""
import argparse
import sys
import time
from pathlib import Path

import cv2
import numpy as np
from ultralytics import YOLO

sys.path.insert(0, str(Path(__file__).resolve().parent))
from fall_common import (FALL, LIE, NORMAL, SIT, SKELETON, SRC_FPS, YOLO_IMGSZ,  # noqa: E402
                         pick_from_result)

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
LABELS = {ord("1"): ("normal", NORMAL), ord("2"): ("sit", SIT), ord("3"): ("lie", LIE)}
COUNTDOWN_S, FALL_S, AFTER_S = 3.0, 1.3, 2.7


def save_clip(out_dir, name, rec, size):
    ts = np.array([r[0] for r in rec])
    grid = np.arange(ts[0], ts[-1], 1.0 / SRC_FPS)
    idx = np.clip(np.searchsorted(ts, grid, side="right") - 1, 0, len(ts) - 1)
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"{name}.npz"
    np.savez_compressed(
        path, kpts=np.stack([rec[i][2] for i in idx]), kconf=np.stack([rec[i][3] for i in idx]),
        box=np.stack([rec[i][4] for i in idx]), present=np.array([rec[i][1] for i in idx]),
        phase=np.array([rec[i][5] for i in idx], dtype=np.int64), size=np.array(size),
        frame_nums=np.arange(len(idx)))
    print(f"  Da luu {path.name}: {len(idx)} frame ({len(idx) / SRC_FPS:.1f}s), "
          f"co nguoi {np.mean([rec[i][1] for i in idx]) * 100:.0f}%")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", default="0")
    ap.add_argument("--out", type=Path, default=HERE / "own_data")
    ap.add_argument("--weights", default=str(ROOT / "yolov8n-pose.pt"))
    args = ap.parse_args()
    src = int(args.source) if args.source.isdigit() else args.source
    weights = args.weights if Path(args.weights).exists() else "yolov8n-pose.pt"
    pose = YOLO(weights)
    cap = cv2.VideoCapture(src)
    if not cap.isOpened() and isinstance(src, int):
        cap = cv2.VideoCapture(src, cv2.CAP_DSHOW)
    if not cap.isOpened():
        raise SystemExit("Khong mo duoc camera")
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    mode, rec, t_go, name, size = None, [], None, "", None
    counter = {k: 0 for k in ("normal", "sit", "lie", "fall")}
    print(__doc__)
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        H, W = frame.shape[:2]
        size = (W, H)
        now = time.time()
        got = pick_from_result(pose.predict(frame, imgsz=YOLO_IMGSZ, conf=0.25, verbose=False)[0], W, H)

        msg, col = "San sang | 1 normal  2 sit  3 lie  f fall  q thoat", (255, 255, 255)
        if mode is not None:
            if mode == "fall":
                dt = now - t_go
                if dt < 0:
                    phase, msg, col = NORMAL, f"Chuan bi... {-dt:.1f}  (di/dung binh thuong)", (0, 255, 255)
                elif dt < FALL_S:
                    phase, msg, col = FALL, "FALL NOW!", (0, 0, 255)
                elif dt < FALL_S + AFTER_S:
                    phase, msg, col = LIE, "nam yen...", (255, 0, 160)
                else:
                    save_clip(args.out, name, rec, size)
                    counter["fall"] += 1
                    mode, rec = None, []
                    continue
            else:
                phase = LABELS[mode][1]
                msg, col = f"DANG QUAY {LABELS[mode][0].upper()}  (SPACE de luu)", (0, 255, 0)
            if got is not None:
                rec.append((now, True, got[0], got[1], got[2], phase))
            else:
                rec.append((now, False, np.zeros((17, 2), np.float32), np.zeros(17, np.float32),
                            np.zeros(4, np.float32), phase))
        if got is not None:
            kp, kc, _ = got
            for a, b in SKELETON:
                if kc[a] > 0.3 and kc[b] > 0.3:
                    cv2.line(frame, tuple(int(v) for v in kp[a]), tuple(int(v) for v in kp[b]), (0, 255, 0), 2)
        cv2.putText(frame, msg, (10, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.7, col, 2)
        cv2.putText(frame, "clips: " + " ".join(f"{k}={v}" for k, v in counter.items()), (10, H - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1)
        cv2.imshow("record_samples", frame)
        k = cv2.waitKey(1) & 0xFF
        if k == ord("q"):
            break
        if mode is None:
            if k in LABELS:
                mode, rec = k, []
                name = f"{LABELS[k][0]}_{int(time.time())}"
            elif k == ord("f"):
                mode, rec, t_go = "fall", [], time.time() + COUNTDOWN_S
                name = f"fall_{int(time.time())}"
        elif k == ord(" ") and mode != "fall":
            if len(rec) > 15:
                save_clip(args.out, name, rec, size)
                counter[LABELS[mode][0]] += 1
            mode, rec = None, []
    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
