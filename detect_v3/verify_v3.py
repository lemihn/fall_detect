"""
Kiem chung toan dien detect_v3 sau khi huan luyen:
  1. Danh gia metric tren tap Validation (URFD + Multiple Cameras Fall cam1..cam8 + GMDCSA24 Subject 4).
  2. Chay Pipeline thuc te tren video that (.avi / .mp4) qua ca 8 goc camera tran & Subject 4 chua tung thay.
  3. Kiem tra Multi-Person (2-3 nguoi dong thoi trong 1 khung hinh) voi ByteTrack + trang thai DOC LAP tung nguoi.
"""
import argparse
import importlib.util
import sys
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np
import torch
from torch.utils.data import DataLoader

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(HERE))

from fall_common import CLASS_NAMES, FALL, load_model  # noqa: E402
from fall_data import WindowDataset, confusion, load_sequences, make_items, predict, report, split_seqs  # noqa: E402

spec = importlib.util.spec_from_file_location("rt_v3", HERE / "4_realtime.py")
rt_v3 = importlib.util.module_from_spec(spec)
spec.loader.exec_module(rt_v3)


def eval_validation_metrics(model_path: Path, device):
    print("\n" + "=" * 70, flush=True)
    print("1. DANH GIA METRIC TREN TAP VALIDATION (ZERO DATA LEAKAGE)", flush=True)
    print("=" * 70, flush=True)
    model, ck = load_model(model_path, device)
    print(f"Model: {model_path.name} | arch={ck.get('arch')} | val_macro_f1_ckpt={ck.get('val_macro_f1', 0):.4f}", flush=True)

    seqs = load_sequences(
        HERE / "cache",
        ROOT / "fall_detect_new" / "labels" / "train",
        HERE / "own_data",
        fv_dir=HERE / "fallvision_cache",
        mc_dir=HERE / "mc_cache",
        gmd_dir=HERE / "gmd_cache",
        verbose=False,
    )
    _, va_seqs = split_seqs(seqs, "fixed")
    urfd_va = [s for s in va_seqs if s.kind in ("adl", "fall")]
    mc_va = [s for s in va_seqs if s.kind in ("mc_adl", "mc_fall")]
    gmd_va = [s for s in va_seqs if s.kind in ("gmd_adl", "gmd_fall")]

    for title, sub_seqs in [
        ("URFD Validation (Goc ngang)", urfd_va),
        ("Multiple Cameras Fall Val (8 goc cam tren cao)", mc_va),
        ("GMDCSA24 Subject 4 Val (Chua tung thay - da goc/che khuat)", gmd_va),
    ]:
        if not sub_seqs:
            continue
        items = make_items(sub_seqs, stride=2)
        dl = DataLoader(WindowDataset(sub_seqs, items), batch_size=128)
        preds, tg = predict(model, dl, device)
        cm = confusion(preds, tg)
        print(f"\n--- {title} ({len(sub_seqs)} chuoi, {len(items)} cua so) ---", flush=True)
        mf1 = report(cm)
        acc = float((preds == tg).mean())
        print(f"Accuracy={acc:.4f} | Macro-F1={mf1:.4f}", flush=True)

    if mc_va:
        print("\n--- Phan tich theo tung goc camera (cam1 -> cam8) ---", flush=True)
        by_cam = defaultdict(list)
        for s in mc_va:
            cam = s.name.split("_")[-1]
            by_cam[cam].append(s)
        for cam in sorted(by_cam.keys()):
            items = make_items(by_cam[cam], stride=2)
            dl = DataLoader(WindowDataset(by_cam[cam], items), batch_size=128)
            preds, tg = predict(model, dl, device)
            cm = confusion(preds, tg)
            fall_rec = cm[FALL, FALL] / max(cm[FALL].sum(), 1)
            fall_prec = cm[FALL, FALL] / max(cm[:, FALL].sum(), 1)
            acc = float((preds == tg).mean())
            print(f"  {cam}: acc={acc:.3f} | FALL recall={fall_rec:.3f} ({cm[FALL, FALL]}/{cm[FALL].sum()}) | FALL prec={fall_prec:.3f}", flush=True)


def run_video_pipeline(model_path: Path, weights: str, device, vpath: Path, fall_thr: float, fall_confirm: int, max_frames: int = 0):
    pipe = rt_v3.Pipeline(model_path, weights, device, dict(fall_thr=fall_thr, fall_confirm=fall_confirm))
    cap = cv2.VideoCapture(str(vpath))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    if fps <= 1.0 or fps > 65.0:
        fps = 30.0
    i = 0
    states = []
    max_persons = 0
    while True:
        if max_frames and i >= max_frames:
            break
        ok, frame = cap.read()
        if not ok:
            break
        t = i / fps
        pipe.process(frame, t)
        max_persons = max(max_persons, len(pipe.active_ids))
        states.append(pipe.out["state"])
        i += 1
    cap.release()
    fall_frames = sum(1 for s in states if s == "fall")
    return len(states), fall_frames, max_persons


def eval_real_videos(model_path: Path, device, fall_thr: float, fall_confirm: int):
    print("\n" + "=" * 70, flush=True)
    print(f"2. KIEM THU PIPELINE THUC TE TREN VIDEO [thr={fall_thr}, confirm={fall_confirm}]", flush=True)
    print("=" * 70, flush=True)
    weights = str(ROOT / "yolov8n-pose.pt")
    mc_root = HERE / "external_data" / "multiple_cameras_fall"
    gmd_s4 = HERE / "external_data" / "gmdcsa24" / "GMDCSA24-A-Dataset-for-Human-Fall-Detection-in-Videos-master" / "Subject 4"

    # A. Multiple Cameras Fall chute04 (nga that tren 8 goc cam)
    fall_avis = sorted(mc_root.rglob("chute04/cam*.avi"))
    fall_detected = 0
    print("\n[A] Nga that tren 8 goc camera tran (chute04/cam1..cam8):", flush=True)
    for avi in fall_avis:
        n_f, f_f, mp = run_video_pipeline(model_path, weights, device, avi, fall_thr, fall_confirm)
        ok = f_f > 0
        fall_detected += int(ok)
        print(f"  {avi.parent.name}/{avi.name}: {'DETECTED FALL' if ok else 'MISSED'} (fall_frames={f_f}/{n_f}, max_persons={mp})", flush=True)

    # B. Multiple Cameras Fall chute24 (ADL 8 goc cam - lay 600 frames = 20s dau moi cam)
    adl_avis = sorted(mc_root.rglob("chute24/cam*.avi"))
    adl_false = 0
    print("\n[B] Sinh hoat binh thuong tren 8 goc cam tran (chute24/cam1..cam8, 20s moi cam):", flush=True)
    for avi in adl_avis:
        n_f, f_f, mp = run_video_pipeline(model_path, weights, device, avi, fall_thr, fall_confirm, max_frames=600)
        fa = f_f > 0
        adl_false += int(fa)
        print(f"  {avi.parent.name}/{avi.name}: {'FALSE ALARM!' if fa else 'OK (NO FALL)'} (fall_frames={f_f}/{n_f})", flush=True)

    # C. GMDCSA24 Subject 4 (nguoi & phong hoan toan chua tung thay)
    gmd_falls = sorted((gmd_s4 / "Fall").glob("*.mp4"))[:6]
    gmd_adls = [gmd_s4 / "ADL" / fn for fn in ["01.mp4", "05.mp4", "08.mp4", "11.mp4", "15.mp4", "18.mp4"] if (gmd_s4 / "ADL" / fn).exists()]
    gmd_det = gmd_fa = 0
    print("\n[C] GMDCSA24 Subject 4 (chua tung thay - Fall & Hard-negative ADL):", flush=True)
    for vp in gmd_falls:
        n_f, f_f, _ = run_video_pipeline(model_path, weights, device, vp, fall_thr, fall_confirm)
        ok = f_f > 0
        gmd_det += int(ok)
        print(f"  Fall/{vp.name}: {'DETECTED FALL' if ok else 'MISSED'} (fall_frames={f_f}/{n_f})", flush=True)
    for vp in gmd_adls:
        n_f, f_f, _ = run_video_pipeline(model_path, weights, device, vp, fall_thr, fall_confirm)
        fa = f_f > 0
        gmd_fa += int(fa)
        print(f"  ADL/{vp.name}: {'FALSE ALARM!' if fa else 'OK (NO FALL)'} (fall_frames={f_f}/{n_f})", flush=True)

    # D. Video chuan 03.mp4
    v03 = HERE / "03.mp4"
    if v03.exists():
        n_f, f_f, _ = run_video_pipeline(model_path, weights, device, v03, fall_thr, fall_confirm)
        print(f"\n[D] Video chuan 03.mp4: {'DETECTED FALL' if f_f > 0 else 'MISSED'} (fall_frames={f_f}/{n_f})", flush=True)

    print(f"\n=> Tong ket video thuc te:", flush=True)
    print(f"   - 8 goc cam tran Fall: {fall_detected}/{len(fall_avis)} | False Alarm: {adl_false}/{len(adl_avis)}", flush=True)
    print(f"   - Subject 4 unseen Fall: {gmd_det}/{len(gmd_falls)} | Hard-ADL False Alarm: {gmd_fa}/{len(gmd_adls)}", flush=True)


def eval_multiperson(model_path: Path, device, fall_thr: float, fall_confirm: int):
    print("\n" + "=" * 70, flush=True)
    print("3. KIEM THU DA NGUOI DONG THOI (2-3 NGUOI) & TRANG THAI DOC LAP TUNG NGUOI", flush=True)
    print("=" * 70, flush=True)
    weights = str(ROOT / "yolov8n-pose.pt")
    gmd_s4 = HERE / "external_data" / "gmdcsa24" / "GMDCSA24-A-Dataset-for-Human-Fall-Detection-in-Videos-master" / "Subject 4"
    v_fall = HERE / "03.mp4"
    v_adl1 = gmd_s4 / "ADL" / "08.mp4"   # nguoi di bo binh thuong
    v_adl2 = gmd_s4 / "ADL" / "11.mp4"   # nguoi ngoi ban doc/viet

    caps = [cv2.VideoCapture(str(v)) for v in (v_fall, v_adl1, v_adl2)]
    pipe = rt_v3.Pipeline(model_path, weights, device, dict(fall_thr=fall_thr, fall_confirm=fall_confirm))
    fps = 30.0
    n_frames = min([int(c.get(cv2.CAP_PROP_FRAME_COUNT)) for c in caps] + [210])

    max_simultaneous = 0
    three_tracked_frames = 0
    independent_pass_frames = 0
    sample_logs = []

    for i in range(n_frames):
        frames = []
        for c in caps:
            ok, f = c.read()
            if not ok:
                break
            frames.append(cv2.resize(f, (480, 360)))
        if len(frames) < 3:
            break
        combined = np.hstack(frames)  # 1440 x 360: 3 nguoi dong thoi (Nga | Di bo | Ngoi)
        t = i / fps
        pipe.process(combined, t)

        n_ppl = len(pipe.active_ids)
        max_simultaneous = max(max_simultaneous, n_ppl)
        if n_ppl >= 3:
            three_tracked_frames += 1

        states_per_id = {tid: pipe.tracks[tid].out["state"] for tid in pipe.active_ids if pipe.tracks[tid].warm}
        has_fall = any(s == "fall" for s in states_per_id.values())
        n_non_fall = sum(1 for s in states_per_id.values() if s != "fall")
        if has_fall and n_non_fall >= 2:
            independent_pass_frames += 1
            if len(sample_logs) < 5 or i % 25 == 0:
                sample_logs.append((t, dict(states_per_id)))

    for c in caps:
        c.release()

    print(f"Da chay {n_frames} frames video ghep 3 nguoi dong thoi (03.mp4 Nga | S4 ADL Đi | S4 ADL Ngồi):", flush=True)
    print(f"  - So nguoi theo doi dong thoi toi da : {max_simultaneous}", flush=True)
    print(f"  - So frame theo doi du >= 3 nguoi     : {three_tracked_frames}/{n_frames}", flush=True)
    print(f"  - So frame trang thai DOC LAP chuan (1 nguoi FALL, >=2 nguoi KHONG FALL): {independent_pass_frames}", flush=True)
    for t_log, st_map in sample_logs[:5]:
        print(f"    t={t_log:5.2f}s -> {st_map}", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", type=Path, default=HERE / "models" / "fall_net_v3_lstm.pth")
    ap.add_argument("--fall-thr", type=float, default=0.69)
    ap.add_argument("--fall-confirm", type=int, default=4)
    ap.add_argument("--skip-videos", action="store_true")
    args = ap.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    eval_validation_metrics(args.model, device)
    if not args.skip_videos:
        eval_real_videos(args.model, device, args.fall_thr, args.fall_confirm)
        eval_multiperson(args.model, device, args.fall_thr, args.fall_confirm)


if __name__ == "__main__":
    main()
