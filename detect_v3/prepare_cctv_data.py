"""
Convert Simuletic CCTV Incident Dataset (YOLO Pose .txt labels, 1024x1024)
into static 45-frame sequences in detect_v3/own_data/ so the model learns
overhead/elevated CCTV postures (laying -> LIE, standing -> NORMAL).
"""
from pathlib import Path
import numpy as np

from fall_common import LIE, NORMAL

HERE = Path(__file__).resolve().parent
LBL_DIR = HERE / "external_data" / "cctv_incident" / "laying_dataset" / "labels"
OUT_DIR = HERE / "own_data"
W, H = 1024, 1024
T = 45  # 1.5s at 30fps


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    count_lie = count_norm = 0
    for txt_path in sorted(LBL_DIR.glob("*.txt")):
        lines = [ln.strip() for ln in txt_path.read_text(encoding="utf-8").splitlines() if ln.strip()]
        for idx, ln in enumerate(lines):
            vals = [float(x) for x in ln.split()]
            if len(vals) < 5 + 17 * 3:
                continue
            cls_id = int(vals[0])
            cx, cy, bw, bh = vals[1:5]
            x1 = (cx - bw / 2.0) * W
            y1 = (cy - bh / 2.0) * H
            x2 = (cx + bw / 2.0) * W
            y2 = (cy + bh / 2.0) * H
            box = np.array([x1, y1, x2, y2], dtype=np.float32)

            kpts_raw = np.array(vals[5:5 + 17 * 3], dtype=np.float32).reshape(17, 3)
            kp = np.zeros((17, 2), dtype=np.float32)
            kc = np.zeros(17, dtype=np.float32)
            kp[:, 0] = kpts_raw[:, 0] * W
            kp[:, 1] = kpts_raw[:, 1] * H
            # YOLO pose visibility: 2=visible, 1=occluded, 0=not labeled
            kc[:] = np.where(kpts_raw[:, 2] >= 2, 0.95, np.where(kpts_raw[:, 2] >= 1, 0.55, 0.0))

            if (kc > 0.3).sum() < 5:
                continue

            phase_val = LIE if cls_id == 0 else NORMAL
            if phase_val == LIE:
                count_lie += 1
            else:
                count_norm += 1

            kpts_seq = np.repeat(kp[None, :, :], T, axis=0)
            kconf_seq = np.repeat(kc[None, :], T, axis=0)
            box_seq = np.repeat(box[None, :], T, axis=0)
            present_seq = np.ones(T, dtype=bool)
            phase_seq = np.full(T, phase_val, dtype=np.int64)

            out_name = f"cctv_{txt_path.stem}_{idx}.npz"
            np.savez_compressed(
                OUT_DIR / out_name,
                kpts=kpts_seq,
                kconf=kconf_seq,
                box=box_seq,
                present=present_seq,
                phase=phase_seq,
                size=np.array([W, H], dtype=np.int64),
            )

    print(f"Converted CCTV Incident dataset -> {OUT_DIR}: {count_lie} LIE, {count_norm} NORMAL")


if __name__ == "__main__":
    main()
