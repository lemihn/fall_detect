"""
Buoc 1: chay YOLOv8-Pose MOT LAN tren toan bo anh va luu cache (khoi chay lai khi doi nhan / doi feature).

    python fall_pose_v2/1_extract_pose_cache.py
Dau ra: fall_pose_v2/cache/<ten_chuoi>.npz  (kpts, kconf, box, present, frame_nums, size)
"""
import argparse
import re
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import cv2
from ultralytics import YOLO

sys.path.insert(0, str(Path(__file__).resolve().parent))
from fall_common import YOLO_IMGSZ, pick_from_result  # noqa: E402

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--images", type=Path, default=ROOT / "fall_detect_new" / "images" / "train")
    ap.add_argument("--out", type=Path, default=HERE / "cache")
    ap.add_argument("--weights", default=str(ROOT / "yolov8n-pose.pt"))
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--only", default=None, help="chi xu ly chuoi co ten chua chuoi nay (de test)")
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    weights = args.weights if Path(args.weights).exists() else "yolov8n-pose.pt"
    model = YOLO(weights)

    seqs = defaultdict(dict)
    pat = re.compile(r"((?:adl|fall)-\d+-cam\d)-rgb-(\d+)")
    for p in args.images.iterdir():
        m = pat.match(p.name)
        if m:
            seqs[m.group(1)][int(m.group(2))] = p
    names = sorted(seqs)
    if args.only:
        names = [n for n in names if args.only in n]
    print(f"{len(names)} chuoi, {sum(len(seqs[n]) for n in names)} anh")

    t0 = time.time()
    for si, name in enumerate(names, 1):
        out_f = args.out / f"{name}.npz"
        if out_f.exists():
            continue
        frames = seqs[name]
        f0, f1 = min(frames), max(frames)
        N = f1 - f0 + 1
        kpts = np.zeros((N, 17, 2), np.float32)
        kconf = np.zeros((N, 17), np.float32)
        box = np.zeros((N, 4), np.float32)
        present = np.zeros(N, bool)
        nums = sorted(frames)
        W = H = None
        for b in range(0, len(nums), args.batch):
            chunk = nums[b:b + args.batch]
            imgs = [cv2.imread(str(frames[n])) for n in chunk]
            if W is None:
                H, W = imgs[0].shape[:2]
            results = model.predict(imgs, imgsz=YOLO_IMGSZ, conf=0.25, verbose=False)
            for n, r in zip(chunk, results):
                got = pick_from_result(r, W, H)
                if got is not None:
                    i = n - f0
                    kpts[i], kconf[i], box[i] = got
                    present[i] = True
        np.savez_compressed(out_f, kpts=kpts, kconf=kconf, box=box, present=present,
                            frame_nums=np.arange(f0, f1 + 1), size=np.array([W, H]))
        print(f"[{si}/{len(names)}] {name}: {N} frame, co nguoi {present.mean() * 100:.0f}% "
              f"({time.time() - t0:.0f}s)", flush=True)
    print("XONG.")


if __name__ == "__main__":
    main()
