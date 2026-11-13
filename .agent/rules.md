# AI Coding Agent Rules - Computer Vision & IoT Fall Detection System

## 1. Tech Stack & Environment
- **Ngôn ngữ:** Python 3.10+, C++ (Arduino / ESP32 Core).
- **Deep Learning:** PyTorch (`torch`, `torch.nn`), Ultralytics YOLOv8 (Pose / Detection).
- **Thị giác máy tính & Xử lý số liệu:** OpenCV (`cv2`), NumPy, SciPy, Matplotlib, Seaborn.
- **Giao thức & Nhúng:** Flask / FastAPI, Requests, WebSockets, MQTT, ESP32 WebServer, I2C (OLED SSD1306).

---

## 2. Tiêu chuẩn Dữ liệu & Chống Rò rỉ Dữ liệu (Data Leakage Prevention)
- **QUY TẮC CỐT LÕI (Zero Data Leakage):** Tuyệt đối KHÔNG chia tập Train/Val ngẫu nhiên theo từng frame (`random.shuffle` trên ảnh) đối với dữ liệu video liên tục.
  - Phải chia tập dữ liệu theo **Tên chuỗi Video / Cảnh quay / Người thực hiện (Group/Scene Split)**.
  - Tất cả các frame thuộc cùng một video/cảnh phải nằm trọn vẹn trong Train hoặc Val, không được phân tán cả hai.
  - **Cùng một sự kiện quay bằng nhiều camera (vd `fall-09-cam0` và `fall-09-cam1`) phải thuộc cùng một nhóm** — chia theo `fall-09`, không chia theo từng camera. (Lỗi này đã từng xảy ra ở bản v1 và làm số liệu val bị thổi phồng.)
- **Định nghĩa nhãn hành động (bản v2, 4 lớp):** `normal` (đi/đứng/cúi nhặt đồ), `sit`, `lie` (nằm), `fall` (chuyển ĐỘT NGỘT từ đứng xuống sàn). Phân biệt `lie` và `fall` bằng **chuyển động theo thời gian**, không bằng hình dạng một frame.
- **Không dùng luật tay (ngưỡng tỉ lệ khung / góc nghiêng / vận tốc) chồng lên mô hình học** trong luồng realtime: chúng gây báo động giả khi người đi ngang qua. Chỉ dùng máy trạng thái làm mượt/xác nhận (`AlertStateMachine`).
- **Đánh giá phải có mức sự kiện:** ngoài confusion matrix theo cửa sổ, luôn chạy `3_evaluate.py` để đếm số chuỗi sinh hoạt bị báo động giả và độ trễ phát hiện ngã.
- **Đồng bộ Nhãn chuỗi thời gian (Temporal Alignment):** Khi cắt cửa sổ trượt (Sliding Window), nhãn của mẫu thời gian phải phản ánh chính xác hành vi tại thời điểm kết thúc cửa sổ (End-of-window label) hoặc trạng thái sự kiện chính.
- **Chuẩn hóa Tọa độ Khớp xương (Keypoint Normalization):**
  - Luôn chuẩn hóa tọa độ $(x, y)$ của các điểm khớp tương đối theo bounding box hoặc điểm trọng tâm (Hông/Vai) và tỷ lệ cơ thể: $x_{norm} = \frac{x - x_{center}}{\text{scale}}$, $y_{norm} = \frac{y - y_{center}}{\text{scale}}$.
  - Tuyệt đối không dùng tọa độ pixel màn hình tuyệt đối đưa vào mô hình học vì sẽ phụ thuộc vào vị trí người đứng gần/xa camera.
- **Kiểm tra Mất cân bằng Lớp & Hard Negatives:** Luôn kiểm tra tỷ lệ giữa các lớp `Normal`, `Fall`, `Warning`. Bổ sung các mẫu dễ nhầm lẫn (nằm ngủ, cúi nhặt đồ, ngồi bệt) để triệt tiêu báo động giả.

---

## 3. Tiêu chuẩn Xây dựng & Tối ưu Mô hình Deep Learning
- **Quản lý Thiết bị phần cứng (Hardware Agnostic):**
  - Luôn viết code tự động chọn thiết bị: `device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')`.
  - Đảm bảo toàn bộ tensor và mô hình đều được gán `.to(device)` đồng nhất để tránh lỗi `RuntimeError: Expected all tensors to be on the same device`.
- **In ấn & Xác thực Kích thước Tensor (Shape Debugging):**
  - Khi thiết kế lớp LSTM/GRU/Linear, luôn ghi chú và kiểm tra shape đầu vào - đầu ra: `[Batch, Seq_Len, Features] -> [Batch, Hidden] -> [Batch, Classes]`.
- **Đánh giá Đa chiều (Comprehensive Evaluation):**
  - Không chỉ đánh giá qua Accuracy. Luôn vẽ và kiểm tra **Confusion Matrix**, **Precision**, **Recall**, **F1-Score** cho từng lớp (đặc biệt là lớp `Fall` và `Warning`).
  - Đánh giá độ trễ thực thi (Inference Latency): Đo thời gian chạy thực tế trên CPU (FPS $\ge 25-30$) để đảm bảo khả năng chạy thời gian thực.
- **Xuất bản & Tối ưu Mô hình (Model Export):**
  - Khi mô hình đạt hiệu năng tốt, lưu file trọng số `.pth` kèm thông tin metadata (`input_dim`, `seq_len`, `classes`).
  - Hỗ trợ xuất sang định dạng ONNX (`torch.onnx.export`) để tăng tốc độ suy luận.

---

## 4. Tiêu chuẩn Lập trình Thời gian thực & Kết nối IoT (Python & ESP32)
- **Bất đồng bộ trong Giao tiếp Mạng (Non-blocking Network I/O):**
  - Tuyệt đối KHÔNG gọi các lệnh HTTP đồng bộ chặn luồng (`requests.get()` với timeout lâu) trực tiếp trong vòng lặp AI (`ai_loop`).
  - Bắt buộc dùng `ThreadPoolExecutor`, `threading.Thread` hoặc hàng đợi `queue.Queue` để gửi dữ liệu sang ESP32 nhằm tránh làm tụt FPS của Camera.
- **An toàn Đa luồng (Thread-Safety):**
  - Mọi biến dùng chung giữa luồng xử lý AI và luồng Flask Stream (ví dụ `latest_frame`, `current_state`) bắt buộc phải được bảo vệ bằng `threading.Lock()`.
- **Cơ chế Chống Báo động giả theo Thời gian (Temporal Debouncing):**
  - Không kích hoạt cảnh báo đỏ chỉ với 1 frame đơn lẻ. Cần có bộ đệm xác nhận (ví dụ: phát hiện liên tục trong $N$ frames hoặc duy trì tư thế ngã $> 5-10$ giây).
- **Cấu hình Động (No Hardcoded Paths):**
  - Tuyệt đối không để đường dẫn ổ đĩa cố định (như `E:\...`). Sử dụng `os.path` hoặc `pathlib.Path` tương đối với thư mục dự án, hoặc đọc từ file cấu hình `config.yaml` / `.env`.

---

## 5. Quy chuẩn Phối hợp với Người dùng (Collaboration Guidelines)
- Khi phát hiện lỗi logic hoặc nguy cơ tràn bộ nhớ/drop frame, phải chỉ rõ nguyên nhân gốc rễ (Root Cause) và đề xuất giải pháp tối ưu thay vì chỉ sửa tạm thời.
- Viết code có chú thích rõ ràng bằng tiếng Việt hoặc tiếng Anh chuẩn, có khối kiểm tra lỗi (`try...except`) với thông báo lỗi chi tiết.
- Luôn kiểm tra khả năng tương thích trên môi trường Windows (PowerShell, đường dẫn dấu `\` và `/`).
