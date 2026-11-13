---
description: Quy trình kiểm tra mã nguồn hệ thống thời gian thực Python (AI Stream) và mã vi điều khiển ESP32, rà soát nghẽn mạng và rò rỉ bộ nhớ.
---

# Workflow: Rà soát Mã nguồn Hệ thống Thời gian thực & IoT (Review IoT Code)

Workflow này hỗ trợ Agent kiểm tra các lỗi tiềm ẩn trong việc tích hợp giữa Trí tuệ Nhân tạo (Python) và Thiết bị Nhúng (ESP32).

## Các bước thực hiện:

### Bước 1: Rà soát đường dẫn cứng (Hardcoded Paths Audit)
1. Quét toàn bộ mã nguồn Python (`.py`) và C++ (`.ino`).
2. Phát hiện các đường dẫn tĩnh cố định (như `E:\...` hay `C:\...`).
3. Chuyển đổi sang `os.path` hoặc cấu hình tập trung trong file `config.yaml` / `.env`.

### Bước 2: Kiểm tra tắc nghẽn luồng xử lý Video (Blocking Calls in AI Loop)
1. Kiểm tra vòng lặp `ai_loop()`:
   - Có lệnh `requests.get()` hoặc `requests.post()` nào đang chạy đồng bộ (blocking) không?
   - Nếu có, tái cấu trúc để đưa việc gửi HTTP sang `ThreadPoolExecutor` hoặc hàng đợi ngầm (Queue).
2. Kiểm tra hàm streaming video Flask (`generate_frames()`):
   - Có cơ chế đồng bộ frame mới không? (Tránh vòng lặp `while True` encode liên tục gây 100% CPU).

### Bước 3: Rà soát an toàn đa luồng (Thread-Safety Check)
1. Kiểm tra các biến toàn cục chia sẻ giữa các luồng:
   - Biến hình ảnh `latest_frame`.
   - Biến trạng thái `current_state`, `current_confidence`.
2. Đảm bảo mọi thao tác đọc/ghi vào biến chia sẻ đều nằm trong khối `with frame_lock:`.

### Bước 4: Kiểm tra mã nguồn ESP32 Firmware
1. Kiểm tra quản lý bộ nhớ: Mảng nhật ký `detectionLogs` có kiểm soát giới hạn tối đa `MAX_LOGS` để tránh tràn RAM không?
2. Kiểm tra cơ chế tự phục hồi kết nối: Khi mất sóng WiFi, ESP32 có tự động thử kết nối lại không?
3. Xác thực các lệnh điều khiển ngoại vi (OLED I2C, còi buzzer, module SIM).
