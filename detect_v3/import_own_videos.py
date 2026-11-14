"""
Import video tu quay (.mp4, .avi, .mov, .mkv) vao detect_v3/own_data/ de train bo sung:
Cach 1 (Theo thu muc):
    Bo video vao:
      detect_v3/own_videos/normal/   (di bo, dung, cui nhat do...)
      detect_v3/own_videos/sit/      (tu dung ngoi nhanh xuong ghe, ngoi ban...)
      detect_v3/own_videos/lie/      (nam giuong / sofa)
      detect_v3/own_videos/fall/     (nga that)
    Sau do chay:
      python detect_v3/import_own_videos.py --retrain

Cach 2 (Truyen truc tiep 1 hoac nhieu file video):
    python detect_v3/import_own_videos.py --label sit --files path/to/ngoi_1.mp4 path/to/ngoi_2.mp4 --retrain
"""
import argparse
import subprocess
import sys
import time
from pathlib import Path

import cv2
import numpy as np
from ultralytics import YOLO

sys.path.insert(0, str(Path(__file__).resolve().parent))
from fall_common import FALL, LIE, NORMAL, SIT, SRC_FPS, YOLO_IMGSZ, pick_from_result  # noqa: E402

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
CLASS_MAP = {"normal": NORMAL, "sit": SIT, "lie": LIE, "fall": FALL}
VIDEO_EXTS = {".mp4", ".avi", ".mov", ".mkv", ".webm"}


def convert_video_to_npz(video_path: Path, label_name: str, out_dir: Path, pose: YOLO):
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        print(f"  [BO QUA] Khong mo duoc video: {video_path}")
        return False

    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    if fps <= 1.0 or fps > 120.0:
        fps = 30.0

    rec = []
    frame_idx = 0
    size = (640, 480)
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        H, W = frame.shape[:2]
        size = (W, H)
        t = frame_idx / fps
        frame_idx += 1
        res = pose.predict(frame, imgsz=YOLO_IMGSZ, conf=0.25, verbose=False)[0]
        got = pick_from_result(res, W, H)
        if got is not None:
            rec.append((t, True, got[0], got[1], got[2]))
        else:
            rec.append((t, False, np.zeros((17, 2), np.float32), np.zeros(17, np.float32), np.zeros(4, np.float32)))
    cap.release()

    if len(rec) < 15:
        print(f"  [BO QUA] Video qua ngan (<15 frames): {video_path.name}")
        return False

    ts = np.array([r[0] for r in rec])
    grid = np.arange(ts[0], ts[-1] + 1e-6, 1.0 / SRC_FPS)
    idx = np.clip(np.searchsorted(ts, grid, side="right") - 1, 0, len(ts) - 1)
    n_steps = len(idx)

    phase = np.full(n_steps, CLASS_MAP[label_name], dtype=np.int64)
    if label_name == "sit":
        # Neu video quay canh TU DUNG -> NGOI: 25% dau la NORMAL (dung), 75% sau la SIT (dang ngoi xuong + ngoi)
        n_stand = max(1, int(0.20 * n_steps))
        phase[:n_stand] = NORMAL
        phase[n_stand:] = SIT
    elif label_name == "fall":
        # Neu video quay canh NGA: 25% dau NORMAL, 45% giua FALL, 30% cuoi LIE
        s1 = int(0.25 * n_steps)
        s2 = int(0.75 * n_steps)
        phase[:s1] = NORMAL
        phase[s1:s2] = FALL
        phase[s2:] = LIE

    out_dir.mkdir(parents=True, exist_ok=True)
    stem = f"{label_name}_{video_path.stem}_{int(time.time() * 100) % 100000}"
    out_path = out_dir / f"{stem}.npz"
    np.savez_compressed(
        out_path,
        kpts=np.stack([rec[i][2] for i in idx]),
        kconf=np.stack([rec[i][3] for i in idx]),
        box=np.stack([rec[i][4] for i in idx]),
        present=np.array([rec[i][1] for i in idx]),
        phase=phase,
        size=np.array(size),
        frame_nums=np.arange(n_steps),
    )
    pres_pct = np.mean([rec[i][1] for i in idx]) * 100.0
    print(f"  [OK] {video_path.name} -> {out_path.name} ({n_steps} buoc = {n_steps/SRC_FPS:.1f}s, co nguoi {pres_pct:.0f}%)")
    return True


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in-dir", type=Path, default=HERE / "own_videos",
                    help="Thu muc chua cac thu muc con normal/ sit/ lie/ fall/")
    ap.add_argument("--out-dir", type=Path, default=HERE / "own_data")
    ap.add_argument("--label", choices=["normal", "sit", "lie", "fall"], default="sit",
                    help="Nhan ap dung khi truyen --files truc tiep")
    ap.add_argument("--files", nargs="*", default=[], help="Danh sach duong dan video cu the")
    ap.add_argument("--weights", default=str(ROOT / "yolov8n-pose.pt"))
    ap.add_argument("--retrain", action="store_true", help="Tu dong huan luyen lai model sau khi import")
    args = ap.parse_args()

    # Dam bao cau truc thu muc own_videos/ luon ton tai
    for sub in ("normal", "sit", "lie", "fall"):
        (args.in_dir / sub).mkdir(parents=True, exist_ok=True)

    weights = args.weights if Path(args.weights).exists() else "yolov8n-pose.pt"
    pose = YOLO(weights)

    tasks = []
    if args.files:
        for f in args.files:
            p = Path(f)
            if p.exists():
                tasks.append((p, args.label))
            else:
                print(f"[CANH BAO] Khong tim thay file: {p}")
    else:
        for sub in ("normal", "sit", "lie", "fall"):
            sub_dir = args.in_dir / sub
            for p in sorted(sub_dir.iterdir()):
                if p.is_file() and p.suffix.lower() in VIDEO_EXTS:
                    tasks.append((p, sub))

    if not tasks:
        print(f"Chua co video nao trong {args.in_dir} (normal/ sit/ lie/ fall/) hoac qua --files.")
        print("Hay copy video cua ban vao detect_v3/own_videos/sit/ roi chay lai lenh nay!")
        return

    print(f">>> Bat dau trich xuat khung xuong tu {len(tasks)} video...")
    ok_cnt = 0
    for vp, lb in tasks:
        if convert_video_to_npz(vp, lb, args.out_dir, pose):
            ok_cnt += 1
    print(f">>> Hoan tat import {ok_cnt}/{len(tasks)} video vao {args.out_dir}")

    if args.retrain and ok_cnt > 0:
        print("\n>>> Bat dau huan luyen lai model voi du lieu moi...")
        cmd = [
            sys.executable,
            str(HERE / "2_train.py"),
            "--use-fv", "--use-mc", "--use-gmd", "--hnm",
            "--out", str(HERE / "models" / "fall_net_v3_lstm.pth"),
        ]
        subprocess.run(cmd, check=True)
        # Dong bo sang fall_net.pth
        import shutil
        shutil.copy2(HERE / "models" / "fall_net_v3_lstm.pth", HERE / "models" / "fall_net.pth")
        print(">>> Da cap nhat model moi vao detect_v3/models/fall_net_v3_lstm.pth va fall_net.pth!")


if __name__ == "__main__":
    main()
