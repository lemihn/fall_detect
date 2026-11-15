# Hệ Thống Phát Hiện Té Ngã Đa Người & Đa Góc (`detect_v3`)

Hệ thống nhận diện tư thế và phát hiện té ngã theo thời gian thực phân loại **4 trạng thái** (`normal`, `sit`, `lie`, `fall`) cho **nhiều người độc lập cùng lúc**, kết hợp **YOLOv8n-Pose**, mạng **BiLSTM (`FallNet`)**, **Web Dashboard**, cảnh báo phần cứng tức thì qua **ESP32** (còi, đèn LED, màn hình TFT) và gửi **ảnh hiện trường + thời gian ngã qua Telegram** sau 10 giây nếu người ngã chưa đứng dậy.

---

## 1. Cấu trúc Repository

```text
fall_detect/
├── detect_v3/                        # Pipeline nhận diện, huấn luyện & chạy thực tế v3
│   ├── fall_common.py                # Bộ lọc khớp xương, PoseTracker, đặc trưng 66 chiều, FallNet & AlertStateMachine
│   ├── fall_data.py                  # Nạp đa nguồn dữ liệu, tạo cửa sổ 2s & chia tập không rò rỉ (Zero Leakage)
│   ├── 1_extract_pose_cache.py       # Bước 1: Trích xuất khung xương YOLOv8-Pose ra .npz
│   ├── 2_train.py                    # Bước 2: Huấn luyện mô hình 4 lớp (BiLSTM + Hard-Negative Mining)
│   ├── 3_evaluate.py                 # Bước 3: Đánh giá theo cửa sổ & mức sự kiện
│   ├── 4_realtime.py                 # Bước 4: Chạy thực tế (Camera/Video + Web Dashboard + ESP32 + Telegram)
│   ├── verify_v3.py                  # Kiểm chứng đa góc camera & đa người đồng thời
│   ├── record_samples.py             # Thu thập thêm mẫu qua Webcam
│   ├── import_own_videos.py          # Import video tự quay (.mp4/.avi) để huấn luyện bổ sung
│   ├── download_fallvision.py        # Tải keypoints CSV bộ FallVision
│   ├── convert_fallvision_to_cache.py# Chuyển đổi FallVision CSV sang .npz
│   ├── prepare_multicam_data.py      # Chuẩn bị dữ liệu Multiple Cameras Fall (8 góc trần)
│   ├── prepare_gmdcsa24_data.py      # Chuẩn bị dữ liệu GMDCSA24 (đa góc & che khuất)
│   ├── prepare_cctv_data.py          # Chuẩn bị dữ liệu Simuletic CCTV góc cao
│   ├── telegram_chats.json           # Lưu danh sách chat_id nhận cảnh báo Telegram
│   └── models/
│       ├── fall_net_v3_lstm.pth      # Trọng số mô hình BiLSTM dùng cho 4_realtime.py & verify_v3.py
│       └── fall_net.pth              # Trọng số mô hình BiLSTM chuẩn (mặc định cho 2_train.py / 3_evaluate.py)
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
python3 detect_v3/4_realtime.py --source 0 --esp32-ip 192.168.1.108

# Trên PC muốn bật thêm cửa sổ OpenCV trực tiếp:
python detect_v3/4_realtime.py --source 0 --esp32-ip 192.168.1.108 --show

# Chạy thử trên file video có sẵn (giới hạn kích thước hiển thị 640px):
python detect_v3/4_realtime.py --source path/to/video.mp4 --max-size 640 --show
```

### Luồng cảnh báo 2 tầng
1. **Ngay lập tức (`0s`):** Gửi HTTP `GET /set?state=fall` sang ESP32 để bật còi và đèn đỏ tại chỗ.
2. **Sau `10s` (`--telegram-delay 10`):** Nếu người ngã vẫn nằm bất động chưa đứng/ngồi dậy, hệ thống tự động chụp **ảnh gốc từ camera** kèm **thời điểm ngã** gửi về ứng dụng Telegram qua Bot.

---

## 4. Tự thêm video quay thực tế & Huấn luyện lại (Tùy chọn)

1. Bỏ các video `.mp4` tự quay vào các thư mục con tương ứng trong `detect_v3/own_videos/{normal,sit,lie,fall}/`.
2. Chạy lệnh import và huấn luyện lại:
```bash
python detect_v3/import_own_videos.py --retrain
```
3. Kiểm chứng toàn diện mô hình:
```bash
python detect_v3/verify_v3.py
```
