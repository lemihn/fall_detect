---
name: fall-model-optimizer
description: Chuyên môn tối ưu hóa mô hình phát hiện té ngã, trích xuất chuỗi đặc trưng khung xương (YOLOv8-Pose), tinh chỉnh siêu tham số mạng LSTM/GRU, và thiết lập cơ chế chống báo động giả cho người già.
---

# Kỹ năng Tối ưu hóa Mô hình Phát hiện Té ngã (Fall Model Optimizer)

Kỹ năng này hướng dẫn các kỹ thuật chuyên sâu để xây dựng một mô hình phát hiện té ngã đạt độ chính xác cao, độ trễ thấp và loại bỏ báo động giả trong thực tế.

---

## 1. Trích xuất & Chuẩn hóa Đặc trưng Khung xương (Feature Extraction)

### 1.1. Cấu trúc 17 Điểm Khớp (COCO Keypoints)
Mô hình `yolov8n-pose` cung cấp 17 khớp xương:
- `0`: Mũi (Nose)
- `1-4`: Mắt và Tai (Eyes & Ears)
- `5-6`: Khớp Vai trái/phải (Shoulders)
- `7-10`: Khuỷu tay và Cổ tay (Elbows & Wrists)
- `11-12`: Khớp Hông trái/phải (Hips)
- `13-14`: Khớp Đầu gối trái/phải (Knees)
- `15-16`: Mắt cá chân trái/phải (Ankles)

### 1.2. Công thức Chuẩn hóa Bất biến Không gian (Invariant Normalization)
Để mô hình không bị ảnh hưởng khi người đứng gần hay xa camera:
1. Xác định tâm đối tượng: $x_{center} = \frac{x_1 + x_2}{2}$, $y_{center} = \frac{y_1 + y_2}{2}$ từ bounding box.
2. Xác định tỷ lệ chuẩn: $\text{scale} = \max(w_{box}, h_{box}, 1.0)$.
3. Tọa độ chuẩn hóa cho từng khớp $i$:
   $$x'_i = \frac{x_i - x_{center}}{\text{scale}}, \quad y'_i = \frac{y_i - y_{center}}{\text{scale}}$$
4. Bổ sung 2 đặc trưng hình học nâng cao:
   - **Tỷ lệ khung hình:** $\text{aspect\_ratio} = \frac{w_{box}}{h_{box}}$
   - **Góc nghiêng cột sống:** Tính góc giữa vector trung điểm 2 vai và trung điểm 2 hông so với phương thẳng đứng:
     $$\theta_{torso} = \arctan\left(\frac{|\Delta x|}{|\Delta y| + 10^{-4}}\right)$$
   - Đứng thẳng: $\theta \approx 0^\circ$ ($0$ rad). Nằm ngang: $\theta \approx 90^\circ$ ($\approx 1.57$ rad).

---

## 2. Kiến trúc Mạng Nơ-ron Chuỗi Thời gian (PyTorch LSTM / GRU)

### 2.1. Cấu trúc Mô hình Đề xuất
```python
import torch
import torch.nn as nn

class FallDetectionLSTM(nn.Module):
    def __init__(self, input_size=36, hidden_size=64, num_layers=2, num_classes=3, dropout=0.3):
        super(FallDetectionLSTM, self).__init__()
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
            bidirectional=False # Dùng 1 chiều để suy luận real-time từng frame
        )
        self.fc = nn.Sequential(
            nn.Linear(hidden_size, 32),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(32, num_classes)
        )
        
    def forward(self, x):
        # x: [Batch, Seq_Len, Features]
        lstm_out, _ = self.lstm(x)
        # Lay trang thai an cua frame cuoi cung
        last_out = lstm_out[:, -1, :]
        out = self.fc(last_out)
        return out
```

### 2.2. Chiến lược Tối ưu hóa Huấn luyện
1. **Trọng số lớp (Class Weights):** Khi tập dữ liệu có ít mẫu ngã hơn mẫu sinh hoạt:
   $$W_c = \frac{N_{total}}{N_{classes} \times N_c}$$
   Truyền vào `nn.CrossEntropyLoss(weight=weights)`.
2. **Kích thước cửa sổ (Window Size):** 20 frames là tối ưu nhất cho bài toán té ngã (vừa đủ nhận diện pha mất thăng bằng và ngã, độ trễ phản hồi chỉ ~0.66s).

---

## 3. Chiến lược Khử Báo động giả (False-Positive Suppression)

1. **Bộ đệm xác nhận (Confirmation Buffer):** Chỉ xác nhận trạng thái `Fall` khi xác suất dự đoán của mô hình $> 0.85$ liên tục trong ít nhất 8–10 frame.
2. **Thời gian bất động sau khi ngã (Inactivity Check):** Sau khi phát hiện ngã, theo dõi thêm 3–5 giây. Nếu tọa độ các khớp không thay đổi đáng kể (nằm yên dưới sàn), kích hoạt báo động cấp độ cao nhất. Nếu người đó tự đứng dậy ngay, tự động hạ mức cảnh báo về an toàn.
