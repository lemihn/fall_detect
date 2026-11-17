# Hệ Thống Phát Hiện Té Ngã Đa Người & Đa Góc (`detect_v3`)

Hệ thống nhận diện tư thế và phát hiện té ngã theo thời gian thực phân loại **4 trạng thái** (`normal`, `sit`, `lie`, `fall`) cho **nhiều người độc lập cùng lúc**, kết hợp **YOLOv8n-Pose**, mạng **BiLSTM (`FallNet`)**, **Web Dashboard**, cảnh báo phần cứng tức thì qua **ESP32** (còi, đèn LED, màn hình TFT) và gửi **ảnh hiện trường + thời gian ngã qua Telegram** sau 10 giây nếu người ngã chưa đứng dậy.

---

## 1. Cấu trúc Repository

```text
fall_detect/
├── detect_v3/                        # Hệ thống v3 đã sắp xếp gọn theo 5 thư mục
│   ├── data_prep/                    # Mục chuẩn bị dữ liệu (trích xuất pose cache .npz)
│   │   ├── 1_extract_pose_cache.py
│   │   ├── download_fallvision.py
│   │   ├── convert_fallvision_to_cache.py
│   │   ├── prepare_multicam_data.py
│   │   ├── prepare_gmdcsa24_data.py
│   │   ├── prepare_cctv_data.py
│   │   ├── import_own_videos.py
│   │   └── record_samples.py
│   ├── model/                        # Mục model (thuật toán, train, đánh giá, chạy thực tế)
│   │   ├── fall_common.py            # Bộ lọc khớp xương, PoseTracker, 66 đặc trưng, FallNet & AlertStateMachine
│   │   ├── fall_data.py              # Nạp đa nguồn dữ liệu, cửa sổ 2s & chia tập Zero Data Leakage
│   │   ├── 2_train.py                # Train mô hình (BiLSTM / Transformer + HNM)
│   │   ├── 3_evaluate.py             # Đánh giá Window-level & Event-level
│   │   ├── 4_realtime.py             # Chạy thực tế (Camera/Video + Web Dashboard + ESP32 + Telegram 10s)
│   │   ├── verify_v3.py              # Kiểm chứng đa góc & đa người đồng thời
│   │   ├── telegram_chats.json
│   │   └── weights/
│   │       ├── fall_net_v3_lstm.pth
│   │       └── fall_net.pth
│   ├── data/                         # Dữ liệu thô & cache .npz
│   ├── test/                         # Video test đầu vào (.mp4)
│   └── result/                       # Video & ảnh kết quả sau khi chạy test
├── TEST_GIAO_DIEN/
│   └── TEST_GIAO_DIEN.ino            # Firmware ESP32 (WebServer :80, còi báo động, LED, TFT ST7735, SIM)
└── scripts/
    └── check_env.py                  # Kiểm tra PyTorch/CUDA, OpenCV và kết nối Camera
```

---

## 2. Cài đặt môi trường

```bash
git clone https://github.com/lemihn/fall_detect.git
cd fall_detect

python3 -m venv .venv
# Linux / Raspberry Pi:
source .venv/bin/activate
# Windows PowerShell:
# .\.venv\Scripts\Activate.ps1

pip install --upgrade pip
pip install torch ultralytics opencv-python numpy flask requests
```

Kiểm tra nhanh môi trường & camera:
```bash
python scripts/check_env.py
```

---

## 3. Chạy thực tế trên PC / Raspberry Pi

> Lần chạy đầu cần có Internet để Ultralytics tự tải `yolov8n-pose.pt` (hoặc đặt sẵn `yolov8n-pose.pt` tại thư mục gốc `fall_detect/`).

```bash
# Chạy với camera mặc định (index 0) + kết nối ESP32 + Web Dashboard (:5000)
python3 detect_v3/model/4_realtime.py --source 0 --esp32-ip 192.168.1.108

# Trên PC muốn bật thêm cửa sổ OpenCV trực tiếp:
python detect_v3/model/4_realtime.py --source 0 --esp32-ip 192.168.1.108 --show

# Chạy thử trên file video trong mục test/ và lưu video kết quả vào mục result/:
python detect_v3/model/4_realtime.py --selftest detect_v3/test/03.mp4 --save-result
```

### Luồng cảnh báo 2 tầng
1. **Ngay lập tức (`0s`):** Gửi HTTP `GET /set?state=fall` sang ESP32 để bật còi và đèn đỏ tại chỗ.
2. **Sau `10s` (`--telegram-delay 10`):** Nếu người ngã vẫn nằm bất động chưa đứng/ngồi dậy, hệ thống tự động chụp **ảnh gốc từ camera** kèm **thời điểm ngã** gửi về ứng dụng Telegram qua Bot.

---

## 4. Tự thêm video quay thực tế & Huấn luyện lại (Tùy chọn)

1. Bỏ các video `.mp4` tự quay vào các thư mục con tương ứng trong `detect_v3/data/own_videos/{normal,sit,lie,fall}/`.
2. Chạy lệnh import và huấn luyện lại:
```bash
python detect_v3/data_prep/import_own_videos.py --retrain
```
3. Kiểm chứng toàn diện mô hình:
```bash
python detect_v3/model/verify_v3.py
```
