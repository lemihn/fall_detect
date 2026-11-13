"""
Code DUNG CHUNG cho train / evaluate / realtime (tranh viec 3 file tu viet lai feature roi lech nhau).

4 lop hanh dong:
    normal : di, dung, cui nhat do, ngoi xom  (khong canh bao)
    sit    : ngoi ghe / giuong
    lie    : nam (nam giuong, hoac nam yen tren san)
    fall   : cu nga DOT NGOT (dung -> san trong thoi gian ngan)

Dau vao mo hinh: chuoi SEQ_STEPS buoc, lay mau o GRID_HZ (theo THOI GIAN, khong phu thuoc FPS camera).
"""
from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

# ----------------------------------------------------------------------------
# HANG SO
# ----------------------------------------------------------------------------
CLASS_NAMES = ["normal", "sit", "lie", "fall"]
NORMAL, SIT, LIE, FALL = 0, 1, 2, 3
IGNORE = -1

SRC_FPS = 30.0            # URFD quay 30 fps
GRID_HZ = 15.0            # tan so lay mau cua mo hinh
SEQ_STEPS = 30            # 30 buoc @15Hz = 2 giay lich su
BASE_STEP = SRC_FPS / GRID_HZ

YOLO_IMGSZ = 480          # DUNG CHUNG luc extract va luc chay that
KPT_CONF_THR = 0.30
MIN_VALID_KPTS = 5
MIN_BOX_AREA_RATIO = 0.01

FLIP_PAIRS = [(1, 2), (3, 4), (5, 6), (7, 8), (9, 10), (11, 12), (13, 14), (15, 16)]
LOWER_BODY = list(range(11, 17))
FEAT_DIM = 66

SKELETON = [(5, 6), (5, 7), (7, 9), (6, 8), (8, 10), (5, 11), (6, 12), (11, 12),
            (11, 13), (13, 15), (12, 14), (14, 16), (0, 1), (0, 2), (1, 3), (2, 4)]


# ----------------------------------------------------------------------------
# CHON NGUOI TRONG KHUNG HINH (dung chung)
# ----------------------------------------------------------------------------
def select_person(kxy, kconf, boxes, frame_w, frame_h):
    """
    kxy (n,17,2), kconf (n,17), boxes (n,4). Tra ve index nguoi chinh hoac None.
    Uu tien nguoi co box ro net (kconf cao), dien tich lon va nam trong vung trung tam khung hinh
    (tranh bat nham nguoi ngoi canh ban / ghe o sat mep trai-phai).
    """
    n = len(boxes)
    if n == 0:
        return None
    best, best_score = None, -1.0
    min_area = MIN_BOX_AREA_RATIO * frame_w * frame_h
    for i in range(n):
        x1, y1, x2, y2 = boxes[i]
        area = max(0.0, x2 - x1) * max(0.0, y2 - y1)
        if area < min_area:
            continue
        good = int(((kconf[i] >= KPT_CONF_THR) & (kxy[i][:, 0] > 1) & (kxy[i][:, 1] > 1)).sum())
        if good < MIN_VALID_KPTS:
            continue
        cx = 0.5 * (x1 + x2)
        edge_w = 0.20 if (cx < 0.15 * frame_w or cx > 0.85 * frame_w) else 1.0
        mean_kc = float(np.clip(kconf[i].mean(), 0.1, 1.0))
        score = area * (mean_kc ** 2) * edge_w
        if score > best_score:
            best, best_score = i, score
    return best


def pick_from_result(result, frame_w, frame_h):
    """Tu ket qua ultralytics -> (kpts(17,2), kconf(17), box(4)) hoac None."""
    if result.keypoints is None or result.boxes is None or len(result.boxes) == 0:
        return None
    kxy = result.keypoints.xy.cpu().numpy()
    boxes = result.boxes.xyxy.cpu().numpy()
    kc = (result.keypoints.conf.cpu().numpy() if result.keypoints.conf is not None
          else np.ones(kxy.shape[:2], dtype=np.float32))
    n = min(len(kxy), len(boxes))
    i = select_person(kxy[:n], kc[:n], boxes[:n], frame_w, frame_h)
    if i is None:
        return None
    return kxy[i].astype(np.float32), kc[i].astype(np.float32), boxes[i].astype(np.float32)


# ----------------------------------------------------------------------------
# FEATURE (66 chieu / buoc)
# ----------------------------------------------------------------------------
def _ffill_index(present):
    T = len(present)
    idx = np.arange(T)
    last = np.maximum.accumulate(np.where(present, idx, -1))
    if present.any():
        first = int(idx[present][0])
        last = np.where(last < 0, first, last)
    else:
        last = np.zeros(T, dtype=int)
    return last


def _mid(kp, vis, a, b):
    va, vb = vis[:, a], vis[:, b]
    m = np.where((va & vb)[:, None], (kp[:, a] + kp[:, b]) / 2.0,
                 np.where(va[:, None], kp[:, a], kp[:, b]))
    return m, (va | vb)


def window_features(kpts, kconf, box, present):
    """
    kpts (T,17,2) kconf (T,17) box (T,4) present (T,) -> (T, FEAT_DIM) float32.
    Gom:  34 toa do khop chuan hoa theo box | 17 co hien thi | 1 co co nguoi |
          14 dac trung hinh hoc + van toc (so voi buoc dau cua cua so)
    """
    present = np.asarray(present, dtype=bool)
    T = kpts.shape[0]
    li = _ffill_index(present)
    kp = kpts[li].astype(np.float32)
    kc = np.where(present[:, None], kconf[li], 0.0).astype(np.float32)
    bx = box[li].astype(np.float32)
    vis = kc >= KPT_CONF_THR

    x1, y1, x2, y2 = bx[:, 0], bx[:, 1], bx[:, 2], bx[:, 3]
    bw = np.maximum(x2 - x1, 1.0)
    bh = np.maximum(y2 - y1, 1.0)
    cx = (x1 + x2) / 2.0
    cy = (y1 + y2) / 2.0
    long_side = np.maximum(bw, bh)
    sref = max(float(np.median(long_side[present])) if present.any() else 1.0, 1.0)

    rel = (kp - np.stack([cx, cy], 1)[:, None, :]) / long_side[:, None, None]
    rel = np.where(vis[:, :, None], rel, 0.0)
    shape = rel.reshape(T, 34)

    # truc co the: hong->vai, neu thieu hong (webcam chi thay nua tren) thi dau->vai
    sh, sh_ok = _mid(kp, vis, 5, 6)
    hp, hp_ok = _mid(kp, vis, 11, 12)
    hv = vis[:, :5]
    cnt = hv.sum(1)
    head = (kp[:, :5] * hv[:, :, None]).sum(1) / np.maximum(cnt, 1)[:, None]
    head_ok = cnt > 0
    ok_t = sh_ok & hp_ok
    ok_h = head_ok & sh_ok
    v = np.where(ok_t[:, None], sh - hp, np.where(ok_h[:, None], head - sh, 0.0))
    ok = (ok_t | ok_h)
    L = np.linalg.norm(v, axis=1)
    ok = ok & (L > 4.0)
    Ls = np.maximum(L, 1e-6)
    sin_t = np.where(ok, np.abs(v[:, 0]) / Ls, 0.0)
    cos_t = np.where(ok, -v[:, 1] / Ls, 0.0)          # dung thang = +1, nam ngang = 0

    cxr = (cx - cx[0]) / sref
    y1r = (y1 - y1[0]) / sref
    y2r = (y2 - y2[0]) / sref
    bwr = bw / sref
    bhr = bh / sref
    logar = np.clip(np.log(bw / bh), -2.0, 2.0)

    def d(a):
        return np.concatenate([[0.0], np.diff(a)]) * 5.0

    geo = np.stack([cxr, y1r, y2r, bwr, bhr, logar, sin_t, cos_t, ok.astype(np.float32),
                    d(y1r), d(y2r), d(bhr), d(cos_t), d(logar)], axis=1)
    feats = np.concatenate([shape, vis.astype(np.float32),
                            present[:, None].astype(np.float32), geo], axis=1)
    feats = np.clip(feats, -5.0, 5.0).astype(np.float32)
    assert feats.shape[1] == FEAT_DIM, feats.shape
    return feats


# ----------------------------------------------------------------------------
# AUGMENT (tren du lieu tho, truoc khi tinh feature)
# ----------------------------------------------------------------------------
def _boxes_from_kpts(kp, vis, fallback, margin=0.08):
    out = fallback.copy()
    for t in range(len(kp)):
        if vis[t].sum() >= 3:
            pts = kp[t][vis[t]]
            lo, hi = pts.min(0), pts.max(0)
            w, h = hi - lo
            out[t] = [lo[0] - margin * w, lo[1] - margin * h, hi[0] + margin * w, hi[1] + margin * h]
    return out


def augment_window(kp, kc, bx, present, frame_w, frame_h, rng):
    kp, kc, bx, present = kp.copy(), kc.copy(), bx.copy(), present.copy()

    if rng.random() < 0.5:                                   # lat ngang
        kp[..., 0] = frame_w - kp[..., 0]
        for a, b in FLIP_PAIRS:
            kp[:, [a, b]] = kp[:, [b, a]]
            kc[:, [a, b]] = kc[:, [b, a]]
        bx = np.stack([frame_w - bx[:, 2], bx[:, 1], frame_w - bx[:, 0], bx[:, 3]], 1)

    if rng.random() < 0.6:                                   # camera hoi nghieng
        ang = math.radians(float(np.clip(rng.normal(0, 5), -12, 12)))
        c, s = math.cos(ang), math.sin(ang)
        R = np.array([[c, -s], [s, c]], dtype=np.float32)
        pivot = np.array([(bx[0, 0] + bx[0, 2]) / 2, (bx[0, 1] + bx[0, 3]) / 2], dtype=np.float32)
        kp = (kp - pivot) @ R.T + pivot
        cs = np.stack([bx[:, [0, 1]], bx[:, [2, 1]], bx[:, [2, 3]], bx[:, [0, 3]]], 1)
        cs = (cs - pivot) @ R.T + pivot
        bx = np.concatenate([cs.min(1), cs.max(1)], 1).astype(np.float32)

    long_side = np.maximum(bx[:, 2] - bx[:, 0], bx[:, 3] - bx[:, 1]).clip(1.0)
    kp = kp + rng.normal(0, 1.0, kp.shape).astype(np.float32) * (0.006 * long_side)[:, None, None]

    if rng.random() < 0.30:                                  # chi thay nua tren (webcam laptop)
        kc[:, LOWER_BODY] = 0.0
        bx = _boxes_from_kpts(kp, kc >= KPT_CONF_THR, bx)

    if rng.random() < 0.15:                                  # YOLO truot vai frame
        for _ in range(int(rng.integers(1, 4))):
            present[int(rng.integers(0, len(present)))] = False
    return kp, kc, bx, present


# ----------------------------------------------------------------------------
# MO HINH
# ----------------------------------------------------------------------------
class FallNet(nn.Module):
    def __init__(self, in_dim=FEAT_DIM, hidden=96, layers=2, n_cls=len(CLASS_NAMES), drop=0.3):
        super().__init__()
        self.inp = nn.Sequential(nn.Linear(in_dim, hidden), nn.ReLU())
        self.lstm = nn.LSTM(hidden, hidden, layers, batch_first=True, dropout=drop if layers > 1 else 0.0)
        self.head = nn.Sequential(nn.Dropout(drop), nn.Linear(hidden * 2, 64), nn.ReLU(),
                                  nn.Dropout(drop), nn.Linear(64, n_cls))

    def forward(self, x):                       # x: [B, T, F]
        h, _ = self.lstm(self.inp(x))
        z = torch.cat([h[:, -1, :], h.max(dim=1).values], dim=1)
        return self.head(z)                     # [B, n_cls]


class FallTransformer(nn.Module):
    """
    Lightweight Temporal Transformer Encoder (theo kien truc Kaggle FallVision).
    Dung Multi-Head Attention de bat khoanh khac roi/nga tren chuoi thoi gian.
    """
    def __init__(self, in_dim=FEAT_DIM, d_model=64, nhead=4, num_layers=3, n_cls=len(CLASS_NAMES), drop=0.2):
        super().__init__()
        self.proj = nn.Linear(in_dim, d_model)
        self.pos_emb = nn.Parameter(torch.randn(1, SEQ_STEPS, d_model) * 0.02)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=128,
            dropout=drop, activation="relu", batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.head = nn.Sequential(
            nn.Dropout(drop),
            nn.Linear(d_model * 2, 64),
            nn.ReLU(),
            nn.Dropout(drop),
            nn.Linear(64, n_cls)
        )

    def forward(self, x):                       # x: [B, T, F]
        z = self.proj(x) + self.pos_emb[:, :x.size(1), :]
        out = self.transformer(z)               # [B, T, d_model]
        pooled = torch.cat([out[:, -1, :], out.mean(dim=1)], dim=1)
        return self.head(pooled)


def save_model(path, model, extra=None):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    arch = "transformer" if isinstance(model, FallTransformer) else "lstm"
    meta = dict(state_dict=model.state_dict(), feat_dim=FEAT_DIM, arch=arch,
                classes=CLASS_NAMES, grid_hz=GRID_HZ, seq_steps=SEQ_STEPS, imgsz=YOLO_IMGSZ)
    if arch == "lstm":
        meta.update(hidden=model.lstm.hidden_size, layers=model.lstm.num_layers)
    else:
        meta.update(d_model=model.proj.out_features, num_layers=len(model.transformer.layers))
    if extra:
        meta.update(extra)
    torch.save(meta, path)


def load_model(path, device):
    ck = torch.load(path, map_location=device, weights_only=False)
    arch = ck.get("arch", "lstm")
    if arch == "transformer":
        m = FallTransformer(ck["feat_dim"], ck.get("d_model", 64), num_layers=ck.get("num_layers", 3),
                            n_cls=len(ck["classes"])).to(device)
    else:
        m = FallNet(ck["feat_dim"], ck["hidden"], ck["layers"], len(ck["classes"])).to(device)
    m.load_state_dict(ck["state_dict"])
    m.eval()
    return m, ck


# ----------------------------------------------------------------------------
# MAY TRANG THAI CANH BAO (thay cho luat tay cu)
# ----------------------------------------------------------------------------
class AlertStateMachine:
    """
    Nhan xac suat 4 lop moi buoc (~15Hz) -> trang thai hien thi + muc canh bao.
      - 'fall' chi bat khi p_fall (da lam muot) >= fall_thr lien tuc fall_confirm buoc
      - Sau khi bat: GIU canh bao (nguoi dang nam tren san). Chi nha khi nguoi
        dung/ngoi lai on dinh recover_s giay (va da giu toi thieu min_hold_s).
      - Nam / ngoi / di binh thuong KHONG BAO DONG.
      - Nam bat dong sau ngã > immobile_s  -> alarm = 2 (khan cap)
    """

    def __init__(self, ema=0.55, fall_thr=0.75, fall_confirm=3, posture_confirm=4,
                 min_hold_s=3.0, recover_s=2.0, immobile_s=8.0, absent_reset_s=1.5):
        self.ema = ema
        self.fall_thr = fall_thr
        self.fall_confirm = fall_confirm
        self.posture_confirm = posture_confirm
        self.min_hold_s = min_hold_s
        self.recover_s = recover_s
        self.immobile_s = immobile_s
        self.absent_reset_s = absent_reset_s
        self.reset()

    def reset(self):
        self.p = None
        self.fall_run = 0
        self.posture = NORMAL
        self.cand = NORMAL
        self.cand_run = 0
        self.latched_at = None
        self.upright_since = None
        self.lie_since = None
        self.absent_since = None
        self.last_t = None

    def update(self, t, probs, present):
        if not present:
            if self.absent_since is None:
                self.absent_since = t
            if t - self.absent_since > self.absent_reset_s:
                latched, latched_at = self.latched_at is not None, self.latched_at
                self.reset()
                if latched and (t - latched_at) < 10.0:     # nga ra khoi khung hinh: giu 10s
                    self.latched_at = latched_at
            return self._output(t)
        self.absent_since = None

        probs = np.asarray(probs, dtype=np.float32)
        self.p = probs if self.p is None else self.ema * probs + (1 - self.ema) * self.p
        p = self.p

        self.fall_run = self.fall_run + 1 if p[FALL] >= self.fall_thr else 0
        if self.latched_at is None and self.fall_run >= self.fall_confirm:
            self.latched_at = t
            self.upright_since = None
            self.lie_since = t

        cand = int(np.argmax(p[[NORMAL, SIT, LIE]]))          # 0/1/2 trung voi NORMAL/SIT/LIE
        if cand == self.posture:
            self.cand_run = 0
        elif cand == self.cand:
            self.cand_run += 1
            if self.cand_run >= self.posture_confirm:
                self.posture, self.cand_run = cand, 0
        else:
            self.cand, self.cand_run = cand, 1

        if self.latched_at is not None:
            if self.posture in (NORMAL, SIT) and p[FALL] < 0.3:
                if self.upright_since is None:
                    self.upright_since = t
                if (t - self.upright_since) >= self.recover_s and (t - self.latched_at) >= self.min_hold_s:
                    self.latched_at = None
                    self.upright_since = None
                    self.lie_since = None
            else:
                self.upright_since = None
        return self._output(t)

    def _output(self, t):
        if self.latched_at is not None:
            level = 2 if (t - self.latched_at) >= self.immobile_s else 1
            return dict(state="fall", alarm=level, posture=self.posture,
                        since=t - self.latched_at)
        return dict(state=CLASS_NAMES[self.posture], alarm=0, posture=self.posture, since=0.0)
