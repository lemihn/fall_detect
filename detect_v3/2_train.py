"""
Buoc 2: huan luyen mo hinh 4 lop (normal / sit / lie / fall).

    python fall_pose_v2/2_train.py                  # train + chon model tot nhat tren tap val (chia theo NHOM)
    python fall_pose_v2/2_train.py --val-mode none  # train tren toan bo du lieu (ban cuoi)
Them du lieu tu quay webcam: dat file .npz vao fall_pose_v2/own_data/ (xem record_samples.py)
"""
import argparse
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, WeightedRandomSampler

sys.path.insert(0, str(Path(__file__).resolve().parent))
from fall_common import CLASS_NAMES, FallNet, FallTransformer, save_model  # noqa: E402
from fall_data import (WindowDataset, class_counts, confusion, load_sequences,  # noqa: E402
                       make_frozen_items, make_items, predict, report, split_seqs)

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", type=Path, default=HERE / "cache")
    ap.add_argument("--labels", type=Path, default=ROOT / "fall_detect_new" / "labels" / "train")
    ap.add_argument("--own", type=Path, default=HERE / "own_data")
    ap.add_argument("--fv", type=Path, default=HERE / "fallvision_cache")
    ap.add_argument("--mc", type=Path, default=HERE / "mc_cache")
    ap.add_argument("--gmd", type=Path, default=HERE / "gmd_cache")
    ap.add_argument("--use-fv", action="store_true", help="dung dataset FallVision 5.683 chuoi")
    ap.add_argument("--use-mc", action="store_true", help="dung dataset Multiple Cameras Fall 192 chuoi")
    ap.add_argument("--use-gmd", action="store_true", help="dung dataset GMDCSA24 160 chuoi")
    ap.add_argument("--arch", choices=["lstm", "transformer"], default="lstm")
    ap.add_argument("--hidden", type=int, default=96)
    ap.add_argument("--drop", type=float, default=0.30)
    ap.add_argument("--hnm", action="store_true", help="bat Hard-Negative Mining giua qua trinh train")
    ap.add_argument("--out", type=Path, default=HERE / "models" / "fall_net.pth")
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--per-epoch", type=int, default=3000)
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--lr", type=float, default=2e-3)
    ap.add_argument("--val-mode", choices=["fixed", "none"], default="fixed")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--no-frozen", action="store_true", help="tat cua so 'dong bang' (de so sanh)")
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("device:", device)

    fv_dir = args.fv if args.use_fv else None
    mc_dir = args.mc if args.use_mc else None
    gmd_dir = args.gmd if args.use_gmd else None
    seqs = load_sequences(args.cache, args.labels, args.own, fv_dir=fv_dir, mc_dir=mc_dir, gmd_dir=gmd_dir)
    tr_seqs, va_seqs = split_seqs(seqs, args.val_mode)
    leak = {s.group for s in tr_seqs} & {s.group for s in va_seqs}
    if leak:
        raise RuntimeError(f"DATA LEAKAGE giua train/val: {leak}")
    print(f"train: {len(tr_seqs)} chuoi / {len({s.group for s in tr_seqs})} nhom | "
          f"val: {len(va_seqs)} chuoi / {len({s.group for s in va_seqs})} nhom  (khong trung nhom)")

    stride_tr = 4 if args.use_fv else 1
    # Giu stride=1 cho cac tap du lieu thuc te goc da dang (URFD, MC-Fall, GMDCSA24, own)
    # va stride=4 cho FallVision de can bang so luong cua so
    real_other = [s for s in tr_seqs if not s.name.startswith("fv_")]
    real_fv = [s for s in tr_seqs if s.name.startswith("fv_")]
    items_real = []
    idx_map = {id(s): i for i, s in enumerate(tr_seqs)}

    from fall_data import label_window, window_indices, FIRST_END, BASE_STEP, NORMAL, SIT, LIE
    for s in real_other:
        si = idx_map[id(s)]
        for end in range(FIRST_END, len(s.phase), 1):
            idx = window_indices(end, BASE_STEP)
            lab = label_window(s.phase[idx], s.present[idx])
            if lab != -1:
                items_real.append((si, end, lab, 0))
    for s in real_fv:
        si = idx_map[id(s)]
        for end in range(FIRST_END, len(s.phase), stride_tr):
            idx = window_indices(end, BASE_STEP)
            lab = label_window(s.phase[idx], s.present[idx])
            if lab != -1:
                items_real.append((si, end, lab, 0))

    items_fro = [] if args.no_frozen else make_frozen_items(tr_seqs, normal_every=8 if args.use_fv else 4)
    items_tr = items_real + items_fro
    items_va = make_items(va_seqs, stride=3)
    print("train windows (that)     :", dict(zip(CLASS_NAMES, class_counts(items_real).tolist())))
    print("train windows (dong bang):", dict(zip(CLASS_NAMES, class_counts(items_fro).tolist())))
    print("val   windows (that)     :", dict(zip(CLASS_NAMES, class_counts(items_va).tolist())))

    # Can bang lay mau theo (nguon du lieu, lop): tang trong so cho MC-Fall (cam tran), GMDCSA24, URFD
    def domain_of(s):
        if s.kind in ("mc_fall", "mc_adl"):
            return "mc"
        if s.kind in ("gmd_fall", "gmd_adl"):
            return "gmd"
        if s.kind == "own":
            return "own"
        if s.name.startswith("fv_"):
            return "fv"
        return "urfd"

    dom_cls_cnt = {}
    for it in items_tr:
        key = (domain_of(tr_seqs[it[0]]), it[2], it[3])
        dom_cls_cnt[key] = dom_cls_cnt.get(key, 0) + 1

    # Trong so can bang nhe nhang: giam bao gia ma van giu do nhay FALL cao
    dom_prior = {"mc": 1.3, "gmd": 1.3, "urfd": 0.9, "own": 0.5, "fv": 0.9}
    cls_prior = {0: 1.15, 1: 1.0, 2: 1.0, 3: 1.05}
    base_w = []
    for it in items_tr:
        d = domain_of(tr_seqs[it[0]])
        c, fro = it[2], it[3]
        key = (d, c, fro)
        fro_w = 0.30 if fro else 0.70
        base_w.append(dom_prior[d] * cls_prior[c] * fro_w / max(dom_cls_cnt[key], 1))
    base_w = np.array(base_w, dtype=np.float64)
    hnm_mult = np.ones_like(base_w)

    ds_tr = WindowDataset(tr_seqs, items_tr, augment=True, seed=args.seed)
    dl_va = DataLoader(WindowDataset(va_seqs, items_va), batch_size=128) if items_va else None

    if args.arch == "transformer":
        model = FallTransformer(drop=args.drop).to(device)
        print("Kien truc: FallTransformer (Lightweight Multi-Head Attention)")
    else:
        model = FallNet(hidden=args.hidden, drop=args.drop).to(device)
        print(f"Kien truc: FallNet (BiLSTM hidden={args.hidden} drop={args.drop})")
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-3)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs)
    crit = nn.CrossEntropyLoss(label_smoothing=0.05)

    # Tap chi so cac cua so thuc te KHONG phai FALL de quet Hard-Negative Mining
    non_fall_idx = [i for i, it in enumerate(items_tr) if it[2] != 3 and it[3] == 0]
    rng_hnm = np.random.default_rng(args.seed + 7)

    best, t0 = -1.0, time.time()
    for ep in range(1, args.epochs + 1):
        if args.hnm and ep in (12, 24) and non_fall_idx:
            model.eval()
            sub = rng_hnm.choice(non_fall_idx, size=min(10000, len(non_fall_idx)), replace=False)
            ds_probe = WindowDataset(tr_seqs, [items_tr[j] for j in sub], augment=False)
            dl_probe = DataLoader(ds_probe, batch_size=256)
            hard_cnt = 0
            hnm_mult[:] = 1.0
            with torch.no_grad():
                off = 0
                for xb, _ in dl_probe:
                    pr = torch.softmax(model(xb.to(device)), 1)[:, 3].cpu().numpy()
                    for k_i, pf in enumerate(pr):
                        if pf >= 0.35:
                            hnm_mult[sub[off + k_i]] = 2.4 if pf >= 0.55 else 1.6
                            hard_cnt += 1
                    off += len(pr)
            print(f"  [HNM ep {ep:02d}] boosted {hard_cnt}/{len(sub)} hard-negative windows", flush=True)

        sampler = WeightedRandomSampler(base_w * hnm_mult, num_samples=args.per_epoch, replacement=True)
        dl_tr = DataLoader(ds_tr, batch_size=args.batch, sampler=sampler)
        model.train()
        loss_sum = n = correct = 0
        for x, y in dl_tr:
            x, y = x.to(device), y.to(device)
            opt.zero_grad()
            out = model(x)
            loss = crit(out, y)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 2.0)
            opt.step()
            loss_sum += loss.item() * len(y)
            n += len(y)
            correct += (out.argmax(1) == y).sum().item()
        sched.step()

        msg = f"ep {ep:02d} loss {loss_sum / n:.3f} acc {correct / n:.3f}"
        if dl_va is not None:
            preds, tg = predict(model, dl_va, device)
            cm = confusion(preds, tg)
            f1s = [2 * cm[i, i] / max(cm[i].sum() + cm[:, i].sum(), 1) for i in range(len(CLASS_NAMES)) if cm[i].sum()]
            mf1 = float(np.mean(f1s))
            msg += f" | val acc {(preds == tg).mean():.3f} macroF1 {mf1:.3f}"
            if mf1 > best:
                best = mf1
                save_model(args.out, model, dict(val_macro_f1=mf1, epoch=ep))
                msg += " *"
        print(msg + f"  ({time.time() - t0:.0f}s)", flush=True)

    if dl_va is None:
        save_model(args.out, model, dict(epoch=args.epochs))
    else:
        from fall_common import load_model
        model, ck = load_model(args.out, device)
        preds, tg = predict(model, dl_va, device)
        print(f"\n== Model tot nhat (epoch {ck['epoch']}) tren tap VAL (nhom chua tung thay) ==")
        report(confusion(preds, tg))
    print("Da luu:", args.out)


if __name__ == "__main__":
    main()
