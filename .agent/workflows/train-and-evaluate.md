---
description: Quy trình huấn luyện mô hình học sâu chuỗi thời gian (LSTM/GRU), theo dõi loss/accuracy, vẽ ma trận nhầm lẫn (Confusion Matrix) và đo lường FPS.
---

# Workflow: Huấn luyện & Đánh giá Mô hình (Train and Evaluate)

Workflow này hướng dẫn Agent cùng Người dùng thiết lập, chạy thử nghiệm và đánh giá chất lượng mô hình phân loại chuỗi thời gian để phát hiện té ngã.

## Các bước thực hiện:

### Bước 1: Chuẩn bị Môi trường & Tensor Loader
1. Kiểm tra thiết bị (`torch.cuda.is_available()`). Ưu tiên GPU nếu có, tự động chuyển về CPU nếu không có.
2. Load các tệp dữ liệu đã trích xuất (`X_train.npy`, `y_train.npy`, `X_val.npy`, `y_val.npy`).
3. Xác thực shape của dữ liệu:
   - `X_train`: `[N_samples, Seq_Len, Features]` (ví dụ: `[N, 20, 36]`).
   - `y_train`: `[N_samples]`.
4. Đóng gói vào `DataLoader` với `batch_size` phù hợp (thường là 32 hoặc 64) và `shuffle=True` cho tập Train.

### Bước 2: Thiết lập Kiến trúc & Siêu tham số (Hyperparameters)
1. Kiến trúc khuyến nghị:
   - **Tầng đầu vào:** Nhận feature 36 chiều (17 khớp xương chuẩn hóa + góc nghiêng cột sống + aspect ratio).
   - **Tầng cốt lõi:** 2 tầng `LSTM` hoặc `BiLSTM` (hidden size 64 hoặc 128) + `Dropout(0.3)`.
   - **Tầng đầu ra:** `Linear(hidden_size, num_classes)`.
2. Hàm mất mát (Loss): `nn.CrossEntropyLoss()` kết hợp `weight=class_weights` nếu dữ liệu mất cân bằng.
3. Bộ tối ưu: `Adam` hoặc `AdamW` với learning rate khởi điểm `0.001` và `ReduceLROnPlateau`.

### Bước 3: Vòng lặp Huấn luyện & Lưu trọng số tối ưu
1. Huấn luyện qua 30 - 50 epochs.
2. Tại mỗi epoch, ghi nhận `train_loss`, `train_acc`, `val_loss`, `val_acc`.
3. Áp dụng cơ chế **Early Stopping**: Lưu checkpoint `best_model.pth` khi `val_loss` đạt giá trị nhỏ nhất hoặc `val_f1` cao nhất.

### Bước 4: Đánh giá Chi tiết & Báo cáo Hiệu năng
1. Tính toán **Classification Report** (Precision, Recall, F1-Score cho từng class `Normal`, `Fall`, `Warning`).
2. Xuất **Confusion Matrix (Ma trận nhầm lẫn)** để kiểm tra:
   - Có bao nhiêu trường hợp `Fall` bị nhầm thành `Normal` (False Negative - cực kỳ nguy hiểm trong y tế).
   - Có bao nhiêu trường hợp `Normal` bị báo động nhầm thành `Fall` (False Positive).
3. **Đo độ trễ thực tế (Inference Latency):**
   - Chạy thử 100 lần suy luận trên 1 mẫu tensor để tính toán thời gian phản hồi trung bình (ms) và số FPS có thể đạt được trên phần cứng mục tiêu.
