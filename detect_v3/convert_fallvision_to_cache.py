"""
Chuyen doi cac file CSV keypoints cua FallVision (5.864 chuoi) thanh cac file .npz
chuan hoa giong URFD cache (kpts, kconf, box, present, frame_nums, size).
"""
import os
import sys
import time
from pathlib import Path
import numpy as np
import pandas as pd
from concurrent.futures import ProcessPoolExecutor

ROOT = Path(r"d:\Education_Material\Addition\SMLab\DO_AN")
SRC_DIR = ROOT / "fallvision_data" / "csv_files"
OUT_DIR = ROOT / "fall_pose_v2" / "fallvision_cache"
OUT_DIR.mkdir(parents=True, exist_ok=True)

KP_NAMES = [
    'Nose', 'Left Eye', 'Right Eye', 'Left Ear', 'Right Ear',
    'Left Shoulder', 'Right Shoulder', 'Left Elbow', 'Right Elbow',
    'Left Wrist', 'Right Wrist', 'Left Hip', 'Right Hip',
    'Left Knee', 'Right Knee', 'Left Ankle', 'Right Ankle'
]
KP_MAP = {name: i for i, name in enumerate(KP_NAMES)}

def process_single_csv(csv_path_str):
    csv_path = Path(csv_path_str)
    category = csv_path.parent.name
    seq_name = f"fv_{category}_{csv_path.stem}"
    out_npz = OUT_DIR / f"{seq_name}.npz"
    if out_npz.exists():
        return True

    try:
        df = pd.read_csv(csv_path)
        if df.empty or 'Frame' not in df.columns or 'Keypoint' not in df.columns:
            return False

        frames = sorted(df['Frame'].unique())
        if len(frames) < 15:
            return False

        n_frames = len(frames)
        kpts = np.zeros((n_frames, 17, 2), dtype=np.float32)
        kconf = np.zeros((n_frames, 17), dtype=np.float32)
        box = np.zeros((n_frames, 4), dtype=np.float32)
        present = np.zeros(n_frames, dtype=bool)

        f_to_idx = {f: idx for idx, f in enumerate(frames)}

        for _, row in df.iterrows():
            f = row['Frame']
            kp_name = row['Keypoint']
            if f in f_to_idx and kp_name in KP_MAP:
                t = f_to_idx[f]
                kid = KP_MAP[kp_name]
                x = float(row.get('X', 0.0))
                y = float(row.get('Y', 0.0))
                c = float(row.get('Confidence', 0.0))
                kpts[t, kid] = [x, y]
                kconf[t, kid] = c

        # Tinh bounding box va flag present
        for t in range(n_frames):
            vis = (kconf[t] >= 0.25) & (kpts[t, :, 0] > 1) & (kpts[t, :, 1] > 1)
            if vis.sum() >= 4:
                pts = kpts[t][vis]
                x1, y1 = pts.min(0)
                x2, y2 = pts.max(0)
                w, h = max(x2 - x1, 10.0), max(y2 - y1, 10.0)
                box[t] = [x1 - 0.1 * w, y1 - 0.1 * h, x2 + 0.1 * w, y2 + 0.1 * h]
                present[t] = True

        if present.sum() < 10:
            return False

        # Uoc luong kich thuoc W, H
        W = int(max(box[:, 2].max() + 50, 640))
        H = int(max(box[:, 3].max() + 50, 480))

        # Xac dinh loai: fall hay no_fall
        is_fall = category.startswith('f_mask_')
        label_kind = "fall" if is_fall else "adl"

        np.savez_compressed(
            out_npz,
            kpts=kpts,
            kconf=kconf,
            box=box,
            present=present,
            frame_nums=np.array(frames, dtype=np.int32),
            size=np.array([W, H], dtype=np.int32),
            kind=label_kind,
            category=category
        )
        return True
    except Exception:
        return False

def main():
    csv_files = [str(p) for p in SRC_DIR.rglob("*.csv")]
    print(f"Bắt đầu chuyển đổi {len(csv_files)} file CSV sang format cache .npz...")
    t0 = time.time()
    
    # Xu ly da tien trinh de tang toc toi da
    with ProcessPoolExecutor() as executor:
        results = list(executor.map(process_single_csv, csv_files, chunksize=50))
    
    success = sum(results)
    print(f"Xong! Chuyển đổi thành công {success}/{len(csv_files)} chuỗi trong {time.time() - t0:.1f}s.")
    print(f"Lưu tại: {OUT_DIR}")

if __name__ == "__main__":
    main()
