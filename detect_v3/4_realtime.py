"""
Buoc 4 (v3 - Multi-Person & Multi-Angle):
Chay THUC TE (webcam / video) -> phat hien & giam sat DOC LAP TAT CA moi nguoi trong khung hinh:
    normal | sit | lie | fall  (+ gui ESP32, web dashboard)

    python detect_v3/4_realtime.py --source 0 --esp32-ip 192.168.1.108
    python detect_v3/4_realtime.py --source video.mp4
    python detect_v3/4_realtime.py --selftest video.mp4
"""
import argparse
import json
import os
import queue
import sys
import threading
import time
from collections import deque
from pathlib import Path

import cv2
import numpy as np
import requests
import torch
from flask import Flask, Response, jsonify
from ultralytics import YOLO

sys.path.insert(0, str(Path(__file__).resolve().parent))
from fall_common import (CLASS_NAMES, GRID_HZ, INFER_CONF, INFER_IMGSZ, SEQ_STEPS, SKELETON,  # noqa: E402
                         AlertStateMachine, PoseTracker, _bbox_iomin, _bbox_iou, extract_all_persons, load_model,
                         window_features)

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
WARMUP_S = 1.0          # toi thieu 1 giay lich su de tranh bao gia do lap frame dau video
STALE_S = 0.35          # mau cu hon nguong nay coi nhu "khong co nguoi"
TRACK_TTL_S = 2.5       # xoa track neu nguoi binh thuong roi khoi khung hinh qua 2.5s
FALL_TTL_S = 25.0       # giu track nguoi dang nga/nam bat dong tren san toi da 25s (du cho dem >10s)
DEFAULT_TELEGRAM_TOKEN = "9047524853:AAEC9YVCLMYpnlEuQjgiVJ_ci568drEvWmo"
TELEGRAM_CHATS_FILE = HERE / "telegram_chats.json"
COLORS = {"normal": (0, 200, 0), "sit": (255, 160, 0), "lie": (255, 0, 160), "fall": (0, 0, 255)}


# ----------------------------------------------------------------------------
class PersonTrack:
    """Trang thai doc lap cho tung nguoi (theo track_id)."""

    def __init__(self, track_id, sm_kwargs):
        self.id = track_id
        self.sm = AlertStateMachine(**sm_kwargs)
        self.buf = deque()                  # (t, present, kp, kc, bx)
        self.last_tick = -1e9
        self.last_seen = -1e9
        self.warm = False
        self.probs = np.array([1, 0, 0, 0], dtype=np.float32)
        self.out = dict(state="normal", alarm=0, posture=0, since=0.0)
        self.last_det = None                # (kp, kc, bx)
        self.seen_count = 0
        self.hq_count = 0
        self.vhq_count = 0
        self.max_disp = 0.0
        self.confirmed = False
        self.fall_wall_time = None          # Thoi diem bat dau nga (HH:MM:SS - DD/MM/YYYY)
        self.telegram_sent = False          # Da gui canh bao Telegram cho dot nga nay chua

    def append_obs(self, t, present, kp, kc, bx, bc=1.0, frame_area=1.0):
        vis = (kc >= 0.30) & (kp[:, 0] > 1.0) & (kp[:, 1] > 1.0)
        n_vis = int(vis.sum())
        n_head = int(vis[:5].sum())
        mkc = float(kc.mean()) if present else 0.0
        xkc = float(kc.max()) if present else 0.0
        bw, bh = max(0.0, float(bx[2] - bx[0])), max(0.0, float(bx[3] - bx[1]))
        area_ratio = (bw * bh) / max(float(frame_area), 1.0)

        is_hq = bool(present and bc >= 0.30 and n_vis >= 11 and n_head >= 1 and mkc >= 0.62 and xkc >= 0.94)
        is_vhq = bool(
            present and (
                (bc >= 0.76 and area_ratio >= 0.022 and n_vis >= 11 and n_head >= 1 and mkc >= 0.64 and xkc >= 0.980)
                or (bc >= 0.58 and area_ratio >= 0.060 and n_vis >= 11 and n_head >= 1 and mkc >= 0.50 and xkc >= 0.935)
            )
        )

        self.buf.append((t, present, kp, kc, bx, is_hq, is_vhq))
        while self.buf and t - self.buf[0][0] > 3.0:
            self.buf.popleft()
        if present:
            self.last_seen = t
            self.last_det = (kp, kc, bx)
            self.seen_count += 1
            if not self.confirmed:
                c = np.array([0.5 * (bx[0] + bx[2]), 0.5 * (bx[1] + bx[3])], dtype=np.float32)
                cur_disp = 0.0
                for b_t, b_pr, _, _, b_bx, _, _ in self.buf:
                    if b_pr and (t - b_t) <= 2.0 and _bbox_iou(bx, b_bx) < 0.12 and _bbox_iomin(bx, b_bx) < 0.18:
                        c0 = np.array([0.5 * (b_bx[0] + b_bx[2]), 0.5 * (b_bx[1] + b_bx[3])], dtype=np.float32)
                        disp = float(np.linalg.norm(c - c0))
                        if disp > cur_disp:
                            cur_disp = disp
                self.max_disp = cur_disp
                self.hq_count = sum(1 for b in self.buf if b[5])
                self.vhq_count = sum(1 for b in self.buf if b[6])
                if self.vhq_count >= 3 or (self.hq_count >= 4 and self.max_disp >= 48.0):
                    self.confirmed = True
        self.warm = len(self.buf) > 0 and (t - self.buf[0][0]) >= WARMUP_S

    def build_window(self, t):
        times = np.array([b[0] for b in self.buf])
        tt = t - (SEQ_STEPS - 1 - np.arange(SEQ_STEPS)) / GRID_HZ
        idx = np.clip(np.searchsorted(times, tt, side="right") - 1, 0, len(times) - 1)
        stale = (tt - times[idx]) > STALE_S
        kp = np.stack([self.buf[i][2] for i in idx])
        kc = np.stack([self.buf[i][3] for i in idx])
        bx = np.stack([self.buf[i][4] for i in idx])
        pr = np.array([self.buf[i][1] for i in idx]) & ~stale
        return window_features(kp, kc, bx, pr)


class Pipeline:
    """Multi-person Pose + PoseTracker (IoU + Torso Centroid) -> bo dem thoi gian doc lap tung nguoi -> batch inference."""

    def __init__(self, model_path, weights, device, sm_kwargs, telegram_delay=10.0):
        self.device = device
        self.weights = weights
        self.sm_kwargs = sm_kwargs
        self.telegram_delay = float(telegram_delay)
        self.pose = YOLO(weights)
        self.model, self.ck = load_model(model_path, device)
        self.tracker = PoseTracker(ttl_s=TRACK_TTL_S, sticky_ttl_s=FALL_TTL_S)
        self.tracks = {}
        self.active_ids = []
        self.out = dict(state="normal", alarm=0, posture=0, since=0.0,
                        fallen_ids=[], persons=[], telegram_sent=False, telegram_triggers=[])
        self.probs = np.array([1, 0, 0, 0], dtype=np.float32)

    def reset(self):
        self.tracker.reset()
        self.tracks.clear()
        self.active_ids = []
        self.out = dict(state="normal", alarm=0, posture=0, since=0.0,
                        fallen_ids=[], persons=[], telegram_sent=False, telegram_triggers=[])
        self.probs = np.array([1, 0, 0, 0], dtype=np.float32)

    def process(self, frame, t):
        H, W = frame.shape[:2]
        frame_area = min(float(W), 1.8 * float(H)) * float(H)
        res = self.pose.predict(
            frame, imgsz=INFER_IMGSZ, conf=INFER_CONF, iou=0.5, verbose=False
        )[0]
        dets = extract_all_persons(res, W, H)
        persons = self.tracker.update(dets, t, W, H)
        seen_ids = set()

        for p_item in persons:
            tid, kp, kc, bx = p_item[0], p_item[1], p_item[2], p_item[3]
            bc = float(p_item[4]) if len(p_item) > 4 else 1.0
            seen_ids.add(tid)
            if tid not in self.tracks:
                self.tracks[tid] = PersonTrack(tid, self.sm_kwargs)
            self.tracks[tid].append_obs(t, True, kp, kc, bx, bc=bc, frame_area=frame_area)
            if self.tracks[tid].confirmed:
                self.tracker.set_sticky(tid, self.tracker.tracks[tid].get("sticky", False), confirmed=True)

        # Cap nhat cho cac track tam khuat ngan han va xoa track da roi khung hinh lau hoac bi track moi de len
        z17, z4 = np.zeros((17, 2), np.float32), np.zeros(4, np.float32)
        stale_ids = []
        for tid, tr in self.tracks.items():
            is_sticky = tr.confirmed and (tr.out["state"] in ("fall", "lie") or tr.sm.latched_at is not None)
            self.tracker.set_sticky(tid, is_sticky, confirmed=tr.confirmed)
            ttl = FALL_TTL_S if is_sticky else (TRACK_TTL_S if tr.confirmed else 1.2)
            if tid not in seen_ids:
                # Neu mot track cu dang tam mat dau ma co 1 track confirmed moi dang nhin thay de len cung vi tri -> xoa track cu
                overlapped = False
                if tr.last_det is not None and (t - tr.last_seen) > 0.30:
                    bx_old = tr.last_det[2]
                    c_old = np.array([0.5 * (bx_old[0] + bx_old[2]), 0.5 * (bx_old[1] + bx_old[3])])
                    d_old = max(float(np.hypot(bx_old[2] - bx_old[0], bx_old[3] - bx_old[1])), 1.0)
                    for sid in seen_ids:
                        strk = self.tracks[sid]
                        if strk.last_det is None or not strk.confirmed:
                            continue
                        bx_new = strk.last_det[2]
                        c_new = np.array([0.5 * (bx_new[0] + bx_new[2]), 0.5 * (bx_new[1] + bx_new[3])])
                        d_new = max(float(np.hypot(bx_new[2] - bx_new[0], bx_new[3] - bx_new[1])), 1.0)
                        if _bbox_iou(bx_old, bx_new) >= 0.15 or _bbox_iomin(bx_old, bx_new) >= 0.35 or float(np.linalg.norm(c_old - c_new)) <= 0.65 * max(d_old, d_new):
                            overlapped = True
                            break
                if overlapped or (t - tr.last_seen > ttl):
                    stale_ids.append(tid)
                else:
                    tr.append_obs(t, False, z17, np.zeros(17, np.float32), z4, bc=0.0, frame_area=frame_area)
        for tid in stale_ids:
            del self.tracks[tid]
            self.tracker.tracks.pop(tid, None)

        # Batch inference cho tat ca track da duoc xac nhan (confirmed), warmup va den nhip 15Hz
        ready_tracks, batch_x = [], []
        for tid, tr in self.tracks.items():
            if tr.confirmed and tr.warm and (t - tr.last_tick) >= 1.0 / GRID_HZ - 1e-6:
                tr.last_tick = t
                ready_tracks.append(tr)
                batch_x.append(tr.build_window(t))

        if ready_tracks:
            X = torch.from_numpy(np.stack(batch_x)).to(self.device)
            with torch.no_grad():
                probs_batch = torch.softmax(self.model(X), 1).cpu().numpy()
            for tr, p in zip(ready_tracks, probs_batch):
                tr.probs = p
                bx = tr.last_det[2] if tr.last_det is not None else (0, 0, 1, 2)
                bh = max(float(bx[3] - bx[1]), 1.0)
                box_ar = float(bx[2] - bx[0]) / bh
                bh_ratio = bh / max(float(H), 1.0)
                at_border = (bx[0] <= 10.0 or bx[1] <= 10.0 or bx[2] >= W - 10.0 or bx[3] >= H - 10.0)
                newly_seen = (tr.id in seen_ids) or ((t - tr.last_seen) <= 1.2 / GRID_HZ)
                present_recent = newly_seen if at_border else ((t - tr.last_seen) <= STALE_S)
                tr.out = tr.sm.update(t, p, present_recent, box_ar=box_ar, newly_seen=present_recent, bh_ratio=bh_ratio)
                is_sticky = tr.out["state"] in ("fall", "lie") or tr.sm.latched_at is not None
                self.tracker.set_sticky(tr.id, is_sticky, confirmed=tr.confirmed)

        # Chi hien thi cac track confirmed dang hien dien (hoac dang nam/nga tren san)
        self.active_ids = []
        for tid in sorted(self.tracks.keys()):
            tr = self.tracks[tid]
            if not tr.confirmed:
                continue
            dt_unseen = t - tr.last_seen
            if dt_unseen <= 0.60 or (tr.out["state"] in ("fall", "lie") and dt_unseen <= 18.0):
                self.active_ids.append(tid)

        # Tong hop trang thai toan phong CHỈ TRÊN CÁC TRACK ĐANG HIỆN DIỆN (active_ids)
        fallen_ids = []
        max_alarm = 0
        max_since = 0.0
        any_telegram_sent = False
        telegram_triggers = []
        person_summaries = []
        overall_state = "normal"
        priority = {"normal": 0, "sit": 1, "lie": 2, "fall": 3}

        best_tr = None
        best_key = (-1, -1.0)

        for tid in self.active_ids:
            tr = self.tracks[tid]
            st = tr.out["state"]
            since_s = float(tr.out.get("since", 0.0))

            # Quan ly bo dem thoi gian nga -> Telegram cho tung nguoi doc lap
            if st == "fall":
                if tr.fall_wall_time is None:
                    tr.fall_wall_time = time.strftime("%H:%M:%S - %d/%m/%Y")
                    tr.telegram_sent = False
                if since_s >= self.telegram_delay and not tr.telegram_sent:
                    tr.telegram_sent = True
                    telegram_triggers.append({
                        "id": tid,
                        "since": round(since_s, 1),
                        "fall_time": tr.fall_wall_time,
                    })
                if tr.telegram_sent:
                    any_telegram_sent = True
            else:
                # Neu nguoi do da thay doi trang thai (dung/ngoi day) truoc 10s -> huy bo dem
                tr.fall_wall_time = None
                tr.telegram_sent = False

            if priority[st] > priority[overall_state]:
                overall_state = st
            if st == "fall":
                fallen_ids.append(tid)
            if tr.out["alarm"] > max_alarm:
                max_alarm = tr.out["alarm"]
            if since_s > max_since:
                max_since = since_s
            cand_key = (priority[st], float(tr.probs[3] + 0.5 * tr.probs[2]))
            if cand_key > best_key:
                best_key = cand_key
                best_tr = tr
            person_summaries.append({
                "id": tid,
                "state": st,
                "alarm": tr.out["alarm"],
                "since": round(since_s, 1),
                "telegram_sent": tr.telegram_sent,
                "fall_time": tr.fall_wall_time,
                "warm": tr.warm,
                "probs": [round(float(v), 3) for v in tr.probs],
            })

        any_warm = any(self.tracks[tid].warm for tid in self.active_ids)
        if best_tr is not None:
            self.probs = best_tr.probs
        else:
            self.probs = np.array([1, 0, 0, 0], dtype=np.float32)

        self.out = dict(
            state=overall_state,
            alarm=max_alarm,
            posture=priority[overall_state],
            since=max_since,
            fallen_ids=fallen_ids,
            persons=person_summaries,
            telegram_sent=any_telegram_sent,
            telegram_triggers=telegram_triggers,
        )
        return len(self.active_ids) > 0, any_warm

    def draw(self, frame, present, warm):
        # Ve khung xuong + box + trang thai DOC LAP cho TUNG nguoi that dang hien dien (active_ids)
        for tid in self.active_ids:
            tr = self.tracks[tid]
            if tr.last_det is None:
                continue
            kp, kc, bx = tr.last_det
            st = tr.out["state"] if tr.warm else "warmup"
            col = COLORS.get(st, (180, 180, 180))
            for a, b in SKELETON:
                if kc[a] > 0.3 and kc[b] > 0.3:
                    cv2.line(frame, tuple(int(v) for v in kp[a]), tuple(int(v) for v in kp[b]), col, 2)
            for i in range(17):
                if kc[i] > 0.3:
                    cv2.circle(frame, tuple(int(v) for v in kp[i]), 3, (255, 255, 255), -1)
            x1, y1, x2, y2 = [int(v) for v in bx]
            thick = 3 if st == "fall" else 2
            cv2.rectangle(frame, (x1, y1), (x2, y2), col, thick)
            tag = f"ID {tid}: {st.upper()}"
            if st == "fall":
                tag += f" {tr.out['since']:.1f}s"
            cv2.putText(frame, tag, (x1, max(18, y1 - 8)), cv2.FONT_HERSHEY_SIMPLEX, 0.55, col, 2)

        # Thanh trang thai tong o tren cung
        st_all = self.out["state"]
        col_all = COLORS[st_all]
        n_ppl = len(self.active_ids)
        if not self.active_ids:
            label = "NO PERSON"
        elif not warm:
            label = f"WARMING UP ({n_ppl} nguoi)"
        else:
            label = f"{st_all.upper()} | {n_ppl} NGUOI"
        if self.out["fallen_ids"]:
            ids_str = ",".join(str(i) for i in self.out["fallen_ids"])
            since_s = self.out["since"]
            if self.out.get("telegram_sent"):
                label = f"FALL ID[{ids_str}] ({since_s:.1f}s - DA GUI TELEGRAM)"
            else:
                label = f"FALL ID[{ids_str}] ({since_s:.1f}s / {self.telegram_delay:.0f}s -> Telegram)"

        cv2.rectangle(frame, (0, 0), (frame.shape[1], 64), (0, 0, 0), -1)
        cv2.putText(frame, label, (12, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.72, col_all, 2)
        txt = " ".join(f"{n}:{p:.2f}" for n, p in zip(CLASS_NAMES, self.probs))
        cv2.putText(frame, txt, (12, 54), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        if st_all == "fall":
            cv2.rectangle(frame, (0, 0), (frame.shape[1] - 1, frame.shape[0] - 1), (0, 0, 255), 6)


# ----------------------------------------------------------------------------
class Esp32Sender:
    """Gui trang thai sang ESP32 o thread rieng, KHONG bao gio chan vong lap AI."""

    def __init__(self, ip):
        self.ip, self.q, self.last = ip, queue.Queue(maxsize=1), (None, 0.0)
        if ip:
            threading.Thread(target=self._worker, daemon=True).start()

    def send(self, state):
        if not self.ip:
            return
        s = "fall" if state == "fall" else ("warning" if state in ("sit", "lie") else "normal")
        now = time.time()
        if s == self.last[0] and now - self.last[1] < 1.0:
            return
        self.last = (s, now)
        try:
            self.q.get_nowait()
        except queue.Empty:
            pass
        self.q.put_nowait(s)

    def _worker(self):
        while True:
            s = self.q.get()
            try:
                requests.get(f"http://{self.ip}/set?state={s}", timeout=1.0)
            except Exception:
                pass


# ----------------------------------------------------------------------------
class TelegramSender:
    """
    Gui tin nhan + anh chup hien truong qua Telegram Bot sau khi nguoi nga >= 10 giay
    ma khong co dau hieu thay doi trang thai.
    Tu dong phat hien chat_id khi nguoi than nhan '/start' vao bot (@PhatHienNguoiNga_bot).
    """

    def __init__(self, token, chat_ids=None, chats_file=TELEGRAM_CHATS_FILE):
        self.token = (token or "").strip()
        self.chats_file = Path(chats_file)
        self.lock = threading.Lock()
        self.chat_ids = set()
        self.last_update_id = 0
        self.q = queue.Queue(maxsize=16)

        if chat_ids:
            for cid in str(chat_ids).split(","):
                cid = cid.strip()
                if cid:
                    self.chat_ids.add(str(cid))
        self._load_saved_chats()

        if self.token:
            threading.Thread(target=self._poll_updates_loop, daemon=True).start()
            threading.Thread(target=self._send_worker, daemon=True).start()
            if self.chat_ids:
                print(f">>> Telegram Bot SAN SANG (da luu {len(self.chat_ids)} chat_id: {', '.join(self.chat_ids)})")
            else:
                print(">>> Telegram Bot SAN SANG! Hay mo Telegram, tim @PhatHienNguoiNga_bot va bam Start (hoac nhan 1 tin bat ky) de tu dong dang ky nhan canh bao.")

    def _load_saved_chats(self):
        if self.chats_file.exists():
            try:
                data = json.loads(self.chats_file.read_text(encoding="utf-8"))
                for cid in data.get("chat_ids", []):
                    self.chat_ids.add(str(cid))
            except Exception:
                pass

    def _save_chats(self):
        try:
            self.chats_file.write_text(
                json.dumps({"chat_ids": sorted(self.chat_ids)}, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
        except Exception:
            pass

    def poll_once(self):
        if not self.token:
            return
        try:
            url = f"https://api.telegram.org/bot{self.token}/getUpdates"
            params = {"timeout": 2}
            if self.last_update_id:
                params["offset"] = self.last_update_id + 1
            r = requests.get(url, params=params, timeout=5.0).json()
            if not r.get("ok"):
                return
            for upd in r.get("result", []):
                self.last_update_id = max(self.last_update_id, int(upd.get("update_id", 0)))
                msg = upd.get("message") or upd.get("channel_post") or {}
                chat = msg.get("chat") or {}
                cid = chat.get("id")
                if cid is not None:
                    cid_str = str(cid)
                    with self.lock:
                        is_new = cid_str not in self.chat_ids
                        self.chat_ids.add(cid_str)
                        if is_new:
                            self._save_chats()
                    text = (msg.get("text") or "").strip()
                    if is_new or text.startswith("/start"):
                        name = chat.get("first_name") or chat.get("title") or "ban"
                        welcome = (
                            f"✅ Xin chào {name}! Đã kết nối thành công với Hệ thống Phát hiện Té ngã AI v3.\n"
                            f"🆔 Chat ID của bạn: {cid_str}\n"
                            f"⏱️ Khi phát hiện có người ngã quá 10 giây không dậy, hệ thống sẽ tự động gửi ảnh hiện trường và thời gian ngã về đây."
                        )
                        requests.post(
                            f"https://api.telegram.org/bot{self.token}/sendMessage",
                            data={"chat_id": cid_str, "text": welcome},
                            timeout=5.0,
                        )
                        print(f">>> [Telegram] Da dang ky nguoi nhan moi: {name} (chat_id={cid_str})")
        except Exception:
            pass

    def _poll_updates_loop(self):
        while True:
            self.poll_once()
            time.sleep(3.0)

    def send_fall_alert(self, person_id, duration_s, fall_time_str, camera_src, jpg_bytes):
        if not self.token:
            return
        caption = (
            f"🚨 CẢNH BÁO KHẨN CẤP: PHÁT HIỆN NGƯỜI TÉ NGÃ!\n"
            f"👤 Đối tượng: ID {person_id}\n"
            f"🕒 Thời điểm bắt đầu ngã: {fall_time_str}\n"
            f"⏱️ Thời gian nằm bất động: {duration_s:.1f} giây (không có dấu hiệu đứng dậy)\n"
            f"📷 Nguồn camera: {camera_src}\n"
            f"⚠️ Loa và đèn tại chỗ đã bật nhưng người ngã chưa tỉnh/dậy. Vui lòng kiểm tra ngay!"
        )
        try:
            self.q.put_nowait((caption, jpg_bytes))
        except queue.Full:
            pass

    def _send_worker(self):
        while True:
            caption, jpg_bytes = self.q.get()
            self.poll_once()
            with self.lock:
                targets = list(self.chat_ids)
            if not targets:
                print(">>> [Telegram CANH BAO] Chua co chat_id nao! Hay mo Telegram nhan '/start' cho @PhatHienNguoiNga_bot")
                continue
            for cid in targets:
                try:
                    if jpg_bytes:
                        resp = requests.post(
                            f"https://api.telegram.org/bot{self.token}/sendPhoto",
                            data={"chat_id": cid, "caption": caption},
                            files={"photo": ("fall_alert.jpg", jpg_bytes, "image/jpeg")},
                            timeout=10.0,
                        )
                    else:
                        resp = requests.post(
                            f"https://api.telegram.org/bot{self.token}/sendMessage",
                            data={"chat_id": cid, "text": caption},
                            timeout=10.0,
                        )
                    if resp.ok:
                        print(f">>> [Telegram] Da gui CANH BAO + ANH NGA toi chat_id={cid}")
                    else:
                        print(f">>> [Telegram LOI] {resp.text}")
                except Exception as e:
                    print(f">>> [Telegram LOI KET NOI] {e}")


# ----------------------------------------------------------------------------
class Shared:
    def __init__(self):
        self.lock = threading.Lock()
        self.cond = threading.Condition(self.lock)
        self.jpeg = None
        self.status = dict(state="normal", alarm=0, fps=0.0, probs=[1, 0, 0, 0],
                           camera="-", persons=[], fallen_ids=[], since=0.0, telegram_sent=False)
        self.reset_req = False
        self.switch_req = None


def open_capture(src):
    cap = cv2.VideoCapture(src)
    if (not cap.isOpened()) and isinstance(src, int):
        cap = cv2.VideoCapture(src, cv2.CAP_DSHOW)
    if cap.isOpened() and isinstance(src, int):
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    return cap


def ai_thread(args, shared, device):
    pipe = Pipeline(args.model, args.weights, device,
                    dict(fall_thr=args.fall_thr, fall_confirm=args.fall_confirm),
                    telegram_delay=args.telegram_delay)
    esp = Esp32Sender(args.esp32_ip)
    tele = TelegramSender(args.telegram_token, args.telegram_chat_id)
    src = args.source
    cap = open_capture(src)
    print(f">>> Camera/video {src}: {'OK' if cap.isOpened() else 'KHONG MO DUOC'}")
    is_file = isinstance(src, str)
    vid_fps = (cap.get(cv2.CAP_PROP_FPS) or 30.0) if is_file else 30.0
    if is_file and (vid_fps <= 1.0 or vid_fps > 65.0):
        vid_fps = 30.0
    frame_idx = 0
    prev = time.time()
    fps = 0.0
    while True:
        with shared.lock:
            reset, sw = shared.reset_req, shared.switch_req
            shared.reset_req, shared.switch_req = False, None
        if reset:
            pipe.reset()
        if sw is not None and sw != src:
            cap.release()
            cap, src, is_file = open_capture(sw), sw, isinstance(sw, str)
            vid_fps = (cap.get(cv2.CAP_PROP_FPS) or 30.0) if is_file else 30.0
            if is_file and (vid_fps <= 1.0 or vid_fps > 65.0):
                vid_fps = 30.0
            frame_idx = 0
            pipe.reset()
        ok, frame = cap.read() if cap.isOpened() else (False, None)
        if not ok:
            if is_file:
                cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                frame_idx = 0
                pipe.reset()
            time.sleep(0.05)
            continue
        now = time.time()
        t_sim = (frame_idx / vid_fps) if is_file else now
        frame_idx += 1
        present, warm = pipe.process(frame, t_sim)
        fps = 0.9 * fps + 0.1 / max(now - prev, 1e-3)
        prev = now
        pipe.draw(frame, present, warm)

        # 1. Phat tin hieu NGAY LAP TUC sang ESP32 de bat loa va den canh bao tai cho
        esp.send(pipe.out["state"])

        ok, jpg = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 82])
        jpg_bytes = jpg.tobytes() if ok else None

        # 2. Neu nguoi nga nam bat dong >= telegram_delay (mac dinh 10s) ma khong thay doi trang thai -> gui Telegram kem anh + thoi gian nga
        for trig in pipe.out.get("telegram_triggers", []):
            tele.send_fall_alert(
                person_id=trig["id"],
                duration_s=trig["since"],
                fall_time_str=trig["fall_time"],
                camera_src=str(src),
                jpg_bytes=jpg_bytes,
            )

        if args.show:
            cv2.imshow("Fall Detection v3 Multi-Person (Nhan Q de thoat)", frame)
            if (cv2.waitKey(1) & 0xFF) in (ord("q"), ord("Q"), 27):
                os._exit(0)

        with shared.cond:
            shared.jpeg = jpg_bytes
            shared.status = dict(
                state=pipe.out["state"],
                alarm=pipe.out["alarm"],
                since=round(float(pipe.out["since"]), 1),
                telegram_sent=bool(pipe.out.get("telegram_sent", False)),
                fps=round(fps, 1),
                probs=[round(float(p), 3) for p in pipe.probs],
                camera=str(src),
                persons=pipe.out["persons"],
                fallen_ids=pipe.out["fallen_ids"],
            )
            shared.cond.notify_all()


PAGE = """<!doctype html><meta charset=utf-8><title>Fall detection v3</title>
<body style="background:#111827;color:#f3f4f6;font-family:sans-serif;text-align:center">
<h2>Giám sát té ngã v3 (Đa người & Đa góc) — ESP32 + Telegram Alert (10s)</h2>
<div id=banner style="margin:8px auto;padding:10px;max-width:720px;border-radius:8px;font-size:20px">...</div>
<div id=plist style="margin:6px auto;max-width:720px;font-size:14px;color:#cbd5e1"></div>
<img src="/video_feed" style="max-width:96%;border-radius:10px"><br><br>
<button onclick="fetch('/reset_alert')">Reset cảnh báo</button>
<button onclick="fetch('/switch_camera/0')">Camera 0</button>
<button onclick="fetch('/switch_camera/1')">Camera 1</button>
<script>
const C={normal:'#166534',sit:'#9a6700',lie:'#86198f',fall:'#b91c1c'};
setInterval(()=>fetch('/status').then(r=>r.json()).then(s=>{
 const b=document.getElementById('banner');b.style.background=C[s.state];
 let extra='';
 if(s.fallen_ids&&s.fallen_ids.length){
  extra=' [ID '+s.fallen_ids.join(',')+' — '+s.since+'s'+(s.telegram_sent?' — ĐÃ GỬI TELEGRAM':' / 10s chờ Telegram')+']';
 }
 b.innerText=s.state.toUpperCase()+extra+'   ('+s.fps+' fps)';
 document.getElementById('plist').innerText='Theo dõi '+s.persons.length+' người: '+s.persons.map(p=>'ID'+p.id+':'+p.state.toUpperCase()+(p.state==='fall'?'('+p.since+'s)':'')).join(' | ');
}),400);
</script></body>"""


def make_app(shared):
    app = Flask(__name__)

    @app.route("/")
    def index():
        return PAGE

    @app.route("/status")
    def status():
        with shared.lock:
            return jsonify(shared.status)

    @app.route("/reset_alert")
    def reset_alert():
        with shared.lock:
            shared.reset_req = True
        return jsonify(ok=True)

    @app.route("/switch_camera/<int:i>")
    def switch_camera(i):
        with shared.lock:
            shared.switch_req = i
        return jsonify(ok=True, camera=i)

    @app.route("/video_feed")
    def video_feed():
        def gen():
            last = None
            while True:
                with shared.cond:
                    shared.cond.wait_for(lambda: shared.jpeg is not last, timeout=2.0)
                    jpg = shared.jpeg
                if jpg is None or jpg is last:
                    continue
                last = jpg
                yield b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" + jpg + b"\r\n"
        return Response(gen(), mimetype="multipart/x-mixed-replace; boundary=frame")

    return app


# ----------------------------------------------------------------------------
def selftest(args, device):
    """Chay dong bo tren video, thoi gian gia lap theo fps video -> in dong thoi gian trang thai + su kien Telegram."""
    pipe = Pipeline(args.model, args.weights, device,
                    dict(fall_thr=args.fall_thr, fall_confirm=args.fall_confirm),
                    telegram_delay=args.telegram_delay)
    tele = TelegramSender(args.telegram_token, args.telegram_chat_id) if args.send_telegram_on_selftest else None
    cap = cv2.VideoCapture(args.selftest)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    if fps <= 1.0 or fps > 65.0:
        fps = 30.0
    i, seg, tele_events = 0, [], []
    t0 = time.time()
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        t = i / fps
        present, warm = pipe.process(frame, t)
        st = pipe.out["state"] if warm else "warmup"
        if not seg or seg[-1][2] != st:
            seg.append([t, t, st])
        seg[-1][1] = t
        for trig in pipe.out.get("telegram_triggers", []):
            tele_events.append((t, trig))
            if tele is not None:
                pipe.draw(frame, present, warm)
                ok_jpg, jpg = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 82])
                tele.send_fall_alert(
                    person_id=trig["id"],
                    duration_s=trig["since"],
                    fall_time_str=trig["fall_time"],
                    camera_src=str(args.selftest),
                    jpg_bytes=jpg.tobytes() if ok_jpg else None,
                )
        i += 1
    cap.release()
    print(f"{i} frame, {time.time() - t0:.1f}s xu ly")
    for a, b, st in seg:
        print(f"  {a:6.2f}s - {b:6.2f}s : {st}")
    if tele_events:
        for t_ev, trig in tele_events:
            print(f"  >>> [TELEGRAM ALERT TRIGGERED at video t={t_ev:.2f}s] ID={trig['id']}, "
                  f"nam bat dong={trig['since']}s (>= {args.telegram_delay}s), thoi diem nga={trig['fall_time']}")
        if tele is not None:
            time.sleep(2.5)
    else:
        print(f"  (Khong co dot nga nao keo dai >= {args.telegram_delay}s de gui Telegram)")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", default="0", help="so (webcam) hoac duong dan video / URL")
    ap.add_argument("--model", type=Path, default=HERE / "models" / "fall_net_v3_lstm.pth")
    ap.add_argument("--weights", default=str(ROOT / "yolov8n-pose.pt"))
    ap.add_argument("--esp32-ip", default=os.environ.get("ESP32_IP", ""))
    ap.add_argument("--telegram-token", default=os.environ.get("TELEGRAM_TOKEN", DEFAULT_TELEGRAM_TOKEN),
                    help="HTTP API Token cua Telegram Bot")
    ap.add_argument("--telegram-chat-id", default=os.environ.get("TELEGRAM_CHAT_ID", ""),
                    help="Chat ID nguoi nhan (de trong se tu dong lay khi nguoi dung nhan /start cho Bot)")
    ap.add_argument("--telegram-delay", type=float, default=10.0,
                    help="So giay dem nguoc sau khi nga khong day truoc khi gui Telegram (mac dinh 10s)")
    ap.add_argument("--send-telegram-on-selftest", action="store_true",
                    help="Cho phep gui tin nhan Telegram that khi chay --selftest")
    ap.add_argument("--host", default="0.0.0.0")
    ap.add_argument("--port", type=int, default=5000)
    ap.add_argument("--fall-thr", type=float, default=0.69)
    ap.add_argument("--fall-confirm", type=int, default=4)
    ap.add_argument("--show", action="store_true", help="hien cua so video OpenCV truc tiep")
    ap.add_argument("--selftest", default=None)
    args = ap.parse_args()
    if not Path(args.weights).exists():
        args.weights = "yolov8n-pose.pt"
    args.source = int(args.source) if str(args.source).isdigit() else args.source
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if args.selftest:
        return selftest(args, device)

    shared = Shared()
    threading.Thread(target=ai_thread, args=(args, shared, device), daemon=True).start()
    print(f">>> Mo trinh duyet: http://127.0.0.1:{args.port}")
    make_app(shared).run(host=args.host, port=args.port, debug=False, threaded=True)


if __name__ == "__main__":
    main()
