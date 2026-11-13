# Hệ Thống Phát Hiện Té Ngã 4 Trạng Thái (Fall Detection Pipeline)

Hệ thống nhận diện tư thế và phát hiện té ngã thời gian thực phân loại **4 trạng thái** (`normal`, `sit`, `lie`, `fall`) kết hợp **YOLOv8-Pose**, mạng chuỗi thời gian **BiLSTM / Light Transformer**, tối ưu hóa **NCNN** cho Raspberry Pi và tích hợp cảnh báo phần cứng qua **ESP32** (màn hình TFT ST7735 1.8", còi báo động PWM 2 tông và đèn LED đỏ).

---

## 1. Hướng dẫn Clone dự án về máy

Repository này được cấu hình ở chế độ **Public (Read-Only đối với người ngoài)** — bạn có thể tự do tải (`git clone` / `git pull`) toàn bộ mã nguồn và trọng số mô hình về máy để sử dụng mà không lo ảnh hưởng đến nhánh chính (`main`).

Mở **Terminal / PowerShell / Command Prompt** tại thư mục bạn muốn lưu dự án và chạy:

```bash
git clone https://github.com/lemihn/fall_detect.git
cd fall_detect
```

Khi muốn cập nhật phiên bản code mới nhất từ repository:

```bash
git pull origin main
```

---

## 2. Cài đặt môi trường (Python Virtual Environment)

Khuyến nghị sử dụng **Python 3.10 – 3.12** và tạo môi trường ảo (`venv`) riêng biệt cho dự án.

### Trên Windows (PowerShell):

```powershell
# 1. Tạo môi trường ảo .venv
python -m venv .venv

# 2. Kích hoạt môi trường ảo
.\.venv\Scripts\Activate.ps1

# 3. Nâng cấp pip và cài đặt các thư viện cần thiết
python -m pip install --upgrade pip
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
pip install ultralytics opencv-python numpy flask requests ncnn
```

> **Lưu ý**: Nếu máy không có GPU NVIDIA (chỉ chạy CPU hoặc trên Raspberry Pi), bạn có thể cài PyTorch bản CPU bằng lệnh:
> ```bash
> pip install torch torchvision
> pip install ultralytics opencv-python numpy flask requests ncnn
> ```

### Trên Linux / Raspberry Pi / macOS:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install torch torchvision ultralytics opencv-python numpy flask requests ncnn
```

### Kiểm tra môi trường và Camera:

Sau khi cài đặt xong, chạy script kiểm tra nhanh PyTorch/CUDA và kết nối Webcam:

```bash
python scripts/check_env.py
```

---

## 3. Cấu trúc thư mục chính

```text
fall_detect/
├── fall_pose_v2/                     # Pipeline nhận diện & huấn luyện mô hình 4 trạng thái
│   ├── fall_common.py                # Kiến trúc mạng (FallNet BiLSTM, FallTransformer), trích xuất 89 đặc trưng & AlertStateMachine
│   ├── fall_data.py                  # Xử lý dữ liệu chuỗi thời gian & chia tập chống rò rỉ dữ liệu (Group Split)
│   ├── 1_extract_pose_cache.py       # Bước 1: Trích xuất khung xương YOLOv8-Pose ra cache (.npz)
│   ├── 2_train.py                    # Bước 2: Huấn luyện mô hình 4 lớp (normal / sit / lie / fall)
│   ├── 3_evaluate.py                 # Bước 3: Đánh giá mô hình ở mức cửa sổ (Window) & mức sự kiện (Event)
│   ├── 4_realtime.py                 # Bước 4: Chạy thực tế qua Webcam/Video + Web Dashboard + gửi cảnh báo ESP32
│   ├── record_samples.py             # Công cụ tự thu thập thêm mẫu dữ liệu từ webcam
│   └── models/
│       └── fall_net.pth              # Trọng số mô hình phân loại chuỗi tư thế đã huấn luyện sẵn
├── yolov8n-pose_ncnn_model/          # Mô hình YOLOv8n-Pose đã xuất sang định dạng NCNN (tối ưu cho Raspberry Pi)
├── TEST_GIAO_DIEN/
│   └── TEST_GIAO_DIEN.ino            # Firmware Arduino/ESP32 điều khiển màn hình TFT ST7735, còi báo động & LED
├── scripts/
│   └── check_env.py                  # Script kiểm tra môi trường PyTorch, GPU CUDA và Camera
└── Bao_Cao_Qua_Trinh_Fall_Detection.ipynb  # Báo cáo chi tiết toàn bộ quy trình nghiên cứu, thực nghiệm & tối ưu
```

---

## 4. Hướng dẫn sử dụng nhanh

Trọng số mô hình đã được đóng gói sẵn tại `fall_pose_v2/models/fall_net.pth` (và model NCNN tại `yolov8n-pose_ncnn_model/`), bạn có thể chạy nhận diện ngay sau khi clone về mà không bắt buộc phải huấn luyện lại.

### 4.1. Chạy nhận diện thời gian thực (Webcam + Web Dashboard)

```bash
# Chạy với webcam mặc định (source 0) và mở giao diện Web tại http://127.0.0.1:5000
python fall_pose_v2/4_realtime.py --source 0

# Hoặc hiển thị trực tiếp cửa sổ OpenCV kèm Web Dashboard
python fall_pose_v2/4_realtime.py --source 0 --show
```

Sau khi chạy, mở trình duyệt truy cập **`http://127.0.0.1:5000`** để xem luồng video trực tiếp, xác suất 4 trạng thái (`normal`, `sit`, `lie`, `fall`) và trạng thái báo động.

### 4.2. Chạy thử trên file Video có sẵn

```bash
# Chạy kèm Web Dashboard với đầu vào là file video
python fall_pose_v2/4_realtime.py --source duong_dan_video.mp4

# Chế độ selftest (chạy nhanh không cần GUI, in ra dòng thời gian các trạng thái phát hiện được)
python fall_pose_v2/4_realtime.py --selftest duong_dan_video.mp4
```

### 4.3. Kết nối với mạch cảnh báo ESP32 qua Wi-Fi

1. Mở file `TEST_GIAO_DIEN/TEST_GIAO_DIEN.ino` bằng **Arduino IDE**, chỉnh lại thông tin Wi-Fi (`ssid`, `password`) và nạp vào mạch **ESP32**.
2. Xem địa chỉ IP của ESP32 trên Serial Monitor (hoặc trên màn hình TFT ST7735).
3. Chạy chương trình nhận diện kèm tham số `--esp32-ip`:

```bash
python fall_pose_v2/4_realtime.py --source 0 --esp32-ip 192.168.1.25
```

---

## 5. Huấn luyện và đánh giá lại mô hình (Tùy chọn)

Nếu bạn có tập dữ liệu mới và muốn chạy lại toàn bộ pipeline huấn luyện:

```bash
# Bước 1: Trích xuất đặc trưng khung xương từ ảnh/video sang cache
python fall_pose_v2/1_extract_pose_cache.py

# Bước 2: Huấn luyện mô hình (tự động lưu model tốt nhất vào fall_pose_v2/models/fall_net.pth)
python fall_pose_v2/2_train.py

# Bước 3: Đánh giá độ chính xác (Confusion Matrix & kiểm tra báo động giả / độ trễ phát hiện)
python fall_pose_v2/3_evaluate.py
```

Để xem giải thích chi tiết về toán học, cách trích xuất 89 đặc trưng hình học khung xương, thiết kế máy trạng thái chống báo động giả (`AlertStateMachine`) và kết quả thực nghiệm, vui lòng mở file notebook **`Bao_Cao_Qua_Trinh_Fall_Detection.ipynb`**.

---

## 6. Quy định đóng góp (Dành cho đồng nghiệp)

- Nhánh `main` đã được bật **Branch Protection (Read-Only đối với bên ngoài)** để đảm bảo tính ổn định của mã nguồn.
- Nếu bạn muốn đề xuất chỉnh sửa hoặc thêm tính năng:
  1. Nhấn nút **Fork** trên góc phải giao diện GitHub để tạo bản sao về tài khoản của bạn.
  2. Tạo nhánh mới (`git checkout -b feature/ten-tinh-nang`), thực hiện thay đổi và commit.
  3. Tạo **Pull Request** gửi về nhánh `main` của repository này để chủ sở hữu duyệt và gộp code.
