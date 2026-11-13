---
description: Kiểm tra chất lượng dữ liệu huấn luyện, phát hiện Data Leakage giữa các frame video và đánh giá độ cân bằng nhãn.
---

# Workflow: Kiểm tra Dữ liệu & Chống Rò rỉ Dữ liệu (Audit Data)

Workflow này giúp Agent và Người dùng rà soát toàn diện tập dữ liệu trước khi huấn luyện mô hình thị giác máy tính và nhận diện hành động.

## Các bước thực hiện:

### Bước 1: Rà soát cấu trúc thư mục và số lượng mẫu
1. Quét danh sách file trong tập dữ liệu (ảnh, video, nhãn).
2. Phân loại theo tiền tố chuỗi video (ví dụ: `adl-*`, `fall-*`, camera ID).
3. Đếm số lượng frame thực tế của từng chuỗi và kiểm tra tính liên tục của frame number (ví dụ: `001, 002, 003,...`).

### Bước 2: Kiểm tra nguy cơ Data Leakage (Rò rỉ dữ liệu)
1. **Quy tắc vàng:** Kiểm tra xem tập Train và Val có bị chia ngẫu nhiên theo frame hay không (`random.shuffle` trên danh sách ảnh).
2. Đảm bảo phân chia tập Train / Val hoàn toàn theo **Chuỗi Video / Cảnh quay (Sequence-level Split)**:
   - Các video chuỗi A thuộc Train thì KHÔNG ĐƯỢC có frame nào xuất hiện ở Val.
   - Tập Val phải đại diện cho các đối tượng hoặc góc quay chưa từng thấy trong Train.

### Bước 3: Phân tích phân bố nhãn (Class Distribution)
1. Thống kê tỷ lệ các lớp:
   - `0: Normal` (Đi lại, sinh hoạt, ngồi, nằm an toàn).
   - `1: Fall` (Té ngã trên sàn).
   - `2: Warning` (Mất thăng bằng, chới với, bắt đầu ngã).
2. Tính tỷ lệ mất cân bằng (Imbalance Ratio). Nếu số lượng mẫu ngã quá ít, đề xuất kỹ thuật Data Augmentation trên chuỗi khớp xương hoặc áp dụng Class Weights / Focal Loss.

### Bước 4: Kiểm tra tính hợp lệ của Khung xương (Pose Integrity Check)
1. Chạy thử mô hình Pose Estimation trên một mẫu đại diện (khoảng 50-100 frame).
2. Kiểm tra tỷ lệ khung hình không phát hiện được người (Missing Keypoints).
3. Đảm bảo hàm chuẩn hóa tọa độ (Relative Normalization) không sinh ra giá trị `NaN` hoặc `Inf`.
