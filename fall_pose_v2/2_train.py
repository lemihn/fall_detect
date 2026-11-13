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
    ap.add_argument("--use-fv", action="store_true", help="dung dataset FallVision 5.683 chuoi")
    ap.add_argument("--arch", choices=["lstm", "transformer"], default="transformer")
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
    seqs = load_sequences(args.cache, args.labels, args.own, fv_dir=fv_dir)
    tr_seqs, va_seqs = split_seqs(seqs, args.val_mode)
    leak = {s.group for s in tr_seqs} & {s.group for s in va_seqs}
    if leak:
        raise RuntimeError(f"DATA LEAKAGE giua train/val: {leak}")
    print(f"train: {len(tr_seqs)} chuoi / {len({s.group for s in tr_seqs})} nhom | "
          f"val: {len(va_seqs)} chuoi / {len({s.group for s in va_seqs})} nhom  (khong trung nhom)")

    stride_tr = 4 if args.use_fv else 1
    items_real = make_items(tr_seqs, stride=stride_tr)
    items_fro = [] if args.no_frozen else make_frozen_items(tr_seqs, normal_every=8 if args.use_fv else 4)
    items_tr = items_real + items_fro
    items_va = make_items(va_seqs, stride=3)
    print("train windows (that)     :", dict(zip(CLASS_NAMES, class_counts(items_real).tolist())))
    print("train windows (dong bang):", dict(zip(CLASS_NAMES, class_counts(items_fro).tolist())))
    print("val   windows (that)     :", dict(zip(CLASS_NAMES, class_counts(items_va).tolist())))

    # moi lop: 60% khoi luong lay mau cho cua so that, 40% cho cua so dong bang
    n_real, n_fro = class_counts(items_real), class_counts(items_fro)
    w = []
    for it in items_tr:
        c, fro = it[2], it[3]
        if fro:
            w.append((0.4 if n_real[c] else 1.0) / max(n_fro[c], 1))
        else:
            w.append((0.6 if n_fro[c] else 1.0) / max(n_real[c], 1))
    sampler = WeightedRandomSampler(np.array(w), num_samples=args.per_epoch, replacement=True)
    dl_tr = DataLoader(WindowDataset(tr_seqs, items_tr, augment=True, seed=args.seed),
                       batch_size=args.batch, sampler=sampler)
    dl_va = DataLoader(WindowDataset(va_seqs, items_va), batch_size=128) if items_va else None

    if args.arch == "transformer":
        model = FallTransformer().to(device)
        print("Kien truc: FallTransformer (Lightweight Multi-Head Attention)")
    else:
        model = FallNet().to(device)
        print("Kien truc: FallNet (BiLSTM)")
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-3)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs)
    crit = nn.CrossEntropyLoss(label_smoothing=0.05)

    best, t0 = -1.0, time.time()
    for ep in range(1, args.epochs + 1):
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
