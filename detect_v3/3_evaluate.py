"""
Buoc 3: danh gia mo hinh.
  (a) muc CUA SO : ma tran nham lan 4 lop tren cac NHOM chua tung thay
  (b) muc SU KIEN: chay lai y het luong realtime (cua so 15Hz + may trang thai canh bao):
        - ADL (di/cui/ngoi/nam): co bao dong gia khong?
        - Nga: co bao dong khong, tre bao nhieu giay?

    python fall_pose_v2/3_evaluate.py
    python fall_pose_v2/3_evaluate.py --all          # chay tren ca du lieu train (chi de tham khao)
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from fall_common import (BASE_STEP, CLASS_NAMES, FALL, SRC_FPS, AlertStateMachine,  # noqa: E402
                         load_model, window_features)
from fall_data import (FIRST_END, WindowDataset, confusion, load_sequences, make_items,  # noqa: E402
                       predict, report, split_seqs, window_indices)

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent


def sequence_probs(model, s, device):
    ends = list(range(FIRST_END, len(s.phase), 2))                  # 15Hz
    X = np.stack([window_features(s.kpts[i], s.kconf[i], s.box[i], s.present[i])
                  for i in (window_indices(e, BASE_STEP) for e in ends)])
    with torch.no_grad():
        P = torch.softmax(model(torch.from_numpy(X).to(device)), 1).cpu().numpy()
    return ends, P


def simulate(model, s, device, sm_kwargs):
    ends, P = sequence_probs(model, s, device)
    sm = AlertStateMachine(**sm_kwargs)
    onsets, states, on = [], [], False
    for e, p in zip(ends, P):
        out = sm.update(e / SRC_FPS, p, bool(s.present[e]))
        states.append(out["state"])
        if out["alarm"] and not on:
            onsets.append(e)
        on = bool(out["alarm"])
    return ends, states, onsets


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", type=Path, default=HERE / "models" / "fall_net.pth")
    ap.add_argument("--cache", type=Path, default=HERE / "cache")
    ap.add_argument("--labels", type=Path, default=ROOT / "fall_detect_new" / "labels" / "train")
    ap.add_argument("--own", type=Path, default=HERE / "own_data")
    ap.add_argument("--all", action="store_true")
    ap.add_argument("--fall-thr", type=float, default=0.75)
    ap.add_argument("--fall-confirm", type=int, default=3)
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, ck = load_model(args.model, device)
    seqs = load_sequences(args.cache, args.labels, args.own, verbose=False)
    tr, va = split_seqs(seqs, "fixed")
    ev = seqs if args.all else va
    print(f"Danh gia tren {len(ev)} chuoi ({'TAT CA - gom ca train' if args.all else 'VAL - nhom chua tung thay'})\n")

    items = make_items(ev, stride=3)
    preds, tg = predict(model, torch.utils.data.DataLoader(WindowDataset(ev, items), batch_size=128), device)
    print("=== (a) MUC CUA SO ===")
    mf1 = report(confusion(preds, tg))
    print(f"accuracy {(preds == tg).mean():.3f} | macro-F1 {mf1:.3f}\n")

    print("=== (b) MUC SU KIEN (may trang thai canh bao, giong luc chay that) ===")
    kw = dict(fall_thr=args.fall_thr, fall_confirm=args.fall_confirm)
    adl_false, fall_hit, lat, fall_pre = 0, 0, [], 0
    n_adl = n_fall = 0
    for s in ev:
        ends, states, onsets = simulate(model, s, device, kw)
        frac = {c: np.mean([x == c for x in states]) for c in CLASS_NAMES}
        shown = " ".join(f"{c}:{frac[c] * 100:3.0f}%" for c in CLASS_NAMES)
        if s.kind == "fall":
            n_fall += 1
            pre = [o for o in onsets if o < s.fall_start - 10]
            post = [o for o in onsets if o >= s.fall_start - 10]
            fall_pre += len(pre) > 0
            if post:
                fall_hit += 1
                lat.append((post[0] - s.fall_start) / SRC_FPS)
                res = f"BAO DONG tre {lat[-1]:+.2f}s"
            else:
                res = "BO SOT"
            if pre:
                res += f"  (!! bao dong som {len(pre)} lan truoc khi nga)"
            print(f"  {s.name:14} {res:38} | {shown}")
        else:
            n_adl += 1
            adl_false += len(onsets) > 0
            res = f"{len(onsets)} bao dong gia" if onsets else "khong bao dong"
            print(f"  {s.name:14} {res:38} | {shown}")
    print("\nTOM TAT")
    print(f"  Chuoi sinh hoat (khong nga): {adl_false}/{n_adl} bi bao dong gia")
    print(f"  Chuoi nga: phat hien {fall_hit}/{n_fall}"
          + (f", tre trung vi {np.median(lat):.2f}s (min {min(lat):.2f}, max {max(lat):.2f})" if lat else "")
          + f", bao dong som truoc khi nga: {fall_pre}")


if __name__ == "__main__":
    main()
