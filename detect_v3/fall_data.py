"""
Nhan (annotation), nap cache, tao cua so thoi gian, Dataset, chia train/val theo NHOM.

Quy tac nhan tung frame ("phase"):
  ADL  : mac dinh NORMAL; cac doan ngoi / nam duoc danh dau tay ben duoi (xem contact sheet).
  FALL : lay moc tu nhan YOLO cu (2='dang nga', 1='nam tren san') ->
           truoc moc bat dau nga           -> NORMAL
           tu luc bat dau nga .. +10 frame sau cham san -> FALL  (khoang dot ngot)
           sau do                           -> LIE
Nhan CUA SO (2 giay lich su):
  FALL  neu cua so chua >= 4 buoc thuoc pha FALL  (cua so co "su kien nga")
  con lai: nhan cua frame cuoi cua so (normal / sit / lie)
=> NAM (cham, co chu dich, hoac da nam yen) khac NGA (chuyen dot ngot dung -> san).
"""
import re
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset

from fall_common import (BASE_STEP, CLASS_NAMES, FALL, FEAT_DIM, IGNORE, LIE, NORMAL, SEQ_STEPS,
                         SIT, SRC_FPS, augment_window, window_features)

BIG = 10 ** 9
FIRST_END = int(SRC_FPS * 1.0)      # cua so dau tien ket thuc sau >= 1 giay

# Cac doan ngoi/nam trong ADL (so frame theo ten file). IGNORE = dang chuyen tu the, khong chac chan.
_BED = [(120, 160, IGNORE), (161, 195, SIT), (196, 230, IGNORE), (231, BIG, LIE)]
ADL_ANNOTATIONS = {
    "adl-07-cam0": [(112, 127, IGNORE), (128, BIG, SIT)],     # ngoi ghe
    "adl-08-cam0": [(105, 128, IGNORE), (129, BIG, SIT)],     # ngoi ghe
    "adl-09-cam0": [(88, 130, IGNORE), (131, BIG, SIT)],      # ngoi mep giuong
    "adl-10-cam0": _BED,                                      # ngoi giuong -> nam giuong
    "adl-11-cam0": _BED,
}
# adl-01..06, 12..17: di / cui / ngoi xom nhat do -> toan bo NORMAL (hard negatives chong bao dong gia)

VAL_ADL = {"adl-04-cam0", "adl-08-cam0", "adl-11-cam0", "adl-14-cam0"}
VAL_FALL_EVENTS = {"fall-09", "fall-13", "fall-18", "fall-22", "fall-26", "fall-29"}
VAL_MC_CHUTES = {"mc-chute04", "mc-chute09", "mc-chute15", "mc-chute21", "mc-chute24"}
VAL_GMD_SUBJECTS = {"gmd-s4"}


@dataclass
class SeqData:
    name: str
    group: str
    kind: str                 # adl | fall | own
    kpts: np.ndarray
    kconf: np.ndarray
    box: np.ndarray
    present: np.ndarray
    phase: np.ndarray
    W: int
    H: int
    fall_start: int = -1      # index frame bat dau nga (de do do tre), -1 neu khong co


def load_frame_labels(lbl_dir):
    pat = re.compile(r"(fall-\d+-cam\d)-rgb-(\d+)")
    out = defaultdict(dict)
    for p in Path(lbl_dir).glob("*.txt"):
        m = pat.match(p.name)
        if not m:
            continue
        txt = p.read_text().split()
        if txt:
            out[m.group(1)][int(m.group(2))] = int(txt[0])
    return out


def fall_timeline(frame_lbl):
    f2 = sorted(f for f, c in frame_lbl.items() if c == 2)
    f1 = sorted(f for f, c in frame_lbl.items() if c == 1)
    if not f1 and not f2:
        return None
    start = f2[0] if f2 else f1[0] - 12
    lie = f1[0] if f1 else f2[-1] + 5
    if lie <= start:
        lie = start + 10
    return start, lie


def load_sequences(cache_dir, lbl_dir, own_dir=None, fv_dir=None, mc_dir=None, gmd_dir=None, include_cam1=False, verbose=True):
    """
    Nap chuoi tu:
      1. URFD cache (goc quay ngang cam0)
      2. Own data (bao gom du lieu tu quay hoac CCTV tinh)
      3. FallVision cache (5.683 chuoi da dang kieu phong, ngoi ghe, nam giuong, nga giuong/ghe/dung)
      4. Multiple Cameras Fall cache (192 chuoi tu 8 goc camera tren cao)
      5. GMDCSA24 cache (160 chuoi tu nhieu goc phong & che khuat 1 phan)
    """
    labels = load_frame_labels(lbl_dir)
    seqs = []
    for f in sorted(Path(cache_dir).glob("*.npz")):
        name = f.stem
        if (not include_cam1) and "-cam1" in name:
            continue
        z = np.load(f)
        nums = z["frame_nums"]
        W, H = (int(v) for v in z["size"])
        fs = -1
        if name.startswith("adl"):
            phase = np.full(len(nums), NORMAL, dtype=np.int64)
            for a, b, c in ADL_ANNOTATIONS.get(name, []):
                phase[(nums >= a) & (nums <= b)] = c
            group, kind = name, "adl"
        else:
            tl = fall_timeline(labels.get(name, {}))
            if tl is None:
                if verbose:
                    print(f"  bo qua {name}: khong co nhan")
                continue
            start, lie = tl
            phase = np.where(nums < start, NORMAL, np.where(nums <= lie + 10, FALL, LIE)).astype(np.int64)
            group = "fall-" + re.match(r"fall-(\d+)", name).group(1)
            kind = "fall"
            fs = int(start - nums[0])
        seqs.append(SeqData(name, group, kind, z["kpts"], z["kconf"], z["box"], z["present"],
                            phase, W, H, fs))
    if own_dir and Path(own_dir).exists():
        for f in sorted(Path(own_dir).glob("*.npz")):
            z = np.load(f)
            W, H = (int(v) for v in z["size"])
            ph = z["phase"].astype(np.int64)
            fs = int(np.argmax(ph == FALL)) if (ph == FALL).any() else -1
            seqs.append(SeqData("own-" + f.stem, "own-" + f.stem, "own", z["kpts"], z["kconf"],
                                z["box"], z["present"], ph, W, H, fs))

    if fv_dir and Path(fv_dir).exists():
        for f in sorted(Path(fv_dir).glob("*.npz")):
            z = np.load(f)
            W, H = (int(v) for v in z["size"])
            kind = str(z["kind"])
            cat = str(z["category"])
            nums = z["frame_nums"]
            n_f = len(nums)
            if kind == "adl":
                # Phan biet ro hanh vi sinh hoat:
                if "_b_" in cat:      # giuong (bed) -> nua dau ngoi mep giuong, nua sau nam giuong
                    mid = n_f // 2
                    phase = np.where(np.arange(n_f) < mid, SIT, LIE).astype(np.int64)
                elif "_c_" in cat:    # ghe (chair) -> ngoi (sit)
                    phase = np.full(n_f, SIT, dtype=np.int64)
                else:                 # dung/di lai (standing) -> normal
                    phase = np.full(n_f, NORMAL, dtype=np.int64)
                fs = -1
            else:
                # Chuoi nga trong FallVision: khoang 30-60% la pha bat dau nga -> nga xong thi nam
                f_start = int(n_f * 0.35)
                f_land = min(int(n_f * 0.65), n_f - 1)
                pre_phase = SIT if ("_b_" in cat or "_c_" in cat) else NORMAL
                phase = np.where(np.arange(n_f) < f_start, pre_phase,
                                 np.where(np.arange(n_f) <= f_land, FALL, LIE)).astype(np.int64)
                fs = f_start
            seqs.append(SeqData(f.stem, f.stem, kind, z["kpts"], z["kconf"], z["box"], z["present"],
                                phase, W, H, fs))

    if mc_dir and Path(mc_dir).exists():
        for f in sorted(Path(mc_dir).glob("*.npz")):
            z = np.load(f)
            W, H = (int(v) for v in z["size"])
            ph = z["phase"].astype(np.int64)
            chute_id = int(z["chute"])
            group = f"mc-chute{chute_id:02d}"
            fs = int(np.argmax(ph == FALL)) if (ph == FALL).any() else -1
            kind = "mc_fall" if fs >= 0 else "mc_adl"
            seqs.append(SeqData(f.stem, group, kind, z["kpts"], z["kconf"], z["box"], z["present"],
                                ph, W, H, fs))

    if gmd_dir and Path(gmd_dir).exists():
        for f in sorted(Path(gmd_dir).glob("*.npz")):
            z = np.load(f)
            W, H = (int(v) for v in z["size"])
            ph = z["phase"].astype(np.int64)
            subj = int(z["subject"])
            group = f"gmd-s{subj}"
            fs = int(np.argmax(ph == FALL)) if (ph == FALL).any() else -1
            kind = "gmd_fall" if fs >= 0 else "gmd_adl"
            seqs.append(SeqData(f.stem, group, kind, z["kpts"], z["kconf"], z["box"], z["present"],
                                ph, W, H, fs))
    return seqs


# ----------------------------------------------------------------------------
def window_indices(end, step):
    # dau chuoi: lap lai frame dau (cho phep du doan khi moi co ~1 giay lich su)
    return np.maximum(end - np.round(np.arange(SEQ_STEPS - 1, -1, -1) * step).astype(int), 0)


def label_window(phase_win, present_win):
    last = phase_win[-1]
    if last == IGNORE or not present_win[-1] or present_win.mean() < 0.75:
        return IGNORE
    if last == LIE and int((phase_win[-12:] == LIE).sum()) >= 10:
        return LIE
    n_fall = int((phase_win == FALL).sum())
    if n_fall >= 4:
        return FALL
    if last == FALL:                       # moi bat dau nga, chua du bang chung -> bo qua
        return IGNORE
    return int(last)


def make_items(seqs, stride=1):
    items = []
    first_end = FIRST_END
    for i, s in enumerate(seqs):
        for end in range(first_end, len(s.phase), stride):
            idx = window_indices(end, BASE_STEP)
            lab = label_window(s.phase[idx], s.present[idx])
            if lab != IGNORE:
                items.append((i, end, lab, 0))
    return items


def make_frozen_items(seqs, normal_every=4):
    """
    Cua so "dong bang": lap 1 frame tinh (nguoi dung yen) trong ca 2 giay.
    Bo sung rat nhieu mau NGOI / NAM yen (du lieu that rat it), de mo hinh hoc "the tinh" tu hinh dang,
    con "NGA" chi hoc tu chuyen dong. Khong bao gio tao nhan FALL tu cua so dong bang.
    """
    items = []
    for i, s in enumerate(seqs):
        for f in range(len(s.phase)):
            if not s.present[f]:
                continue
            ph = int(s.phase[f])
            if ph in (SIT, LIE) or (ph == NORMAL and f % normal_every == 0):
                items.append((i, f, ph, 1))
    return items


def split_seqs(seqs, mode="fixed"):
    if mode == "none":
        return list(seqs), []
    tr, va = [], []
    for s in seqs:
        is_val = (
            (s.kind == "adl" and s.name in VAL_ADL)
            or (s.kind == "fall" and s.group in VAL_FALL_EVENTS)
            or (s.kind in ("mc_fall", "mc_adl") and s.group in VAL_MC_CHUTES)
            or (s.kind in ("gmd_fall", "gmd_adl") and s.group in VAL_GMD_SUBJECTS)
        )
        (va if is_val else tr).append(s)
    return tr, va


def class_counts(items, n=4):
    c = np.zeros(n, dtype=int)
    for it in items:
        c[it[2]] += 1
    return c


class WindowDataset(Dataset):
    def __init__(self, seqs, items, augment=False, seed=0):
        self.seqs, self.items, self.augment = seqs, items, augment
        self.rng = np.random.default_rng(seed)

    def __len__(self):
        return len(self.items)

    def __getitem__(self, k):
        si, end, lab, frozen = self.items[k]
        s = self.seqs[si]
        idx = np.full(SEQ_STEPS, end) if frozen else window_indices(end, BASE_STEP)
        if self.augment and not frozen:
            idx2 = window_indices(end, BASE_STEP * self.rng.uniform(0.75, 1.3))   # doi toc do hanh dong
            if label_window(s.phase[idx2], s.present[idx2]) == lab:
                idx = idx2
        kp, kc, bx, pr = s.kpts[idx], s.kconf[idx], s.box[idx], s.present[idx]
        if self.augment:
            kp, kc, bx, pr = augment_window(kp, kc, bx, pr, s.W, s.H, self.rng)
        x = window_features(kp, kc, bx, pr)
        assert x.shape == (SEQ_STEPS, FEAT_DIM)
        return torch.from_numpy(x), lab


# ----------------------------------------------------------------------------
# DANH GIA
# ----------------------------------------------------------------------------
def predict(model, loader, device):
    model.eval()
    ps, ts = [], []
    with torch.no_grad():
        for x, y in loader:
            ps.append(torch.argmax(model(x.to(device)), 1).cpu().numpy())
            ts.append(y.numpy())
    return np.concatenate(ps), np.concatenate(ts)


def confusion(preds, targets, n=len(CLASS_NAMES)):
    cm = np.zeros((n, n), dtype=int)
    for t, p in zip(targets, preds):
        cm[t, p] += 1
    return cm


def report(cm):
    """In ma tran nham lan (hang = nhan that, cot = du doan) + precision/recall/F1."""
    f1s = []
    print(f"{'that \\ doan':>12}" + "".join(f"{c:>9}" for c in CLASS_NAMES) + "   |  prec   rec    f1     n")
    for i, name in enumerate(CLASS_NAMES):
        tp = cm[i, i]
        prec = tp / max(cm[:, i].sum(), 1)
        rec = tp / max(cm[i].sum(), 1)
        f1 = 2 * prec * rec / max(prec + rec, 1e-9)
        if cm[i].sum() > 0:
            f1s.append(f1)
        print(f"{name:>12}" + "".join(f"{v:>9}" for v in cm[i]) +
              f"   | {prec:5.2f} {rec:5.2f} {f1:5.2f} {cm[i].sum():>5}")
    return float(np.mean(f1s)) if f1s else 0.0
