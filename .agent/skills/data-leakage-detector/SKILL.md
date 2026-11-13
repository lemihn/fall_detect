---
name: data-leakage-detector
description: Chuyên môn quét và phân tích tập dữ liệu video thị giác máy tính, phát hiện rò rỉ dữ liệu (Data Leakage) giữa tập huấn luyện (Train) và tập kiểm thử (Val/Test).
---

# Kỹ năng Phát hiện Rò rỉ Dữ liệu Video (Data Leakage Detector)

Kỹ năng này cung cấp các nguyên tắc và công cụ kiểm tra để ngăn chặn hiện tượng mô hình đạt độ chính xác ảo (như 99.5% trong phòng lab) nhưng thất bại hoàn toàn khi triển khai thực tế.

---

## 1. Bản chất của Rò rỉ Dữ liệu trong Thị giác Máy tính Video

### Nguyên nhân gây ra:
Trong dữ liệu video liên tục (như UR Fall Detection, Le2i):
- Hai khung hình kế tiếp nhau ($t$ và $t+1$) chỉ cách nhau 33 miligiây. Về mặt thị giác, chúng gần như là **hai bức ảnh song sinh** (cùng người, cùng quần áo, cùng góc camera, cùng ánh sáng và đồ vật trong phòng).
- Nếu dùng lệnh `random.shuffle()` trên toàn bộ tập ảnh rồi chia 80% Train và 20% Val:
  - Khung hình thứ 100 rơi vào Train.
  - Khung hình thứ 101 rơi vào Val.
  - Mạng nơ-ron chỉ cần "ghi nhớ" hình nền và khuôn mặt là dự đoán chính xác 100%, nhưng không hề học được bản chất của hành động té ngã!

---

## 2. Quy trình Kiểm toán Dữ liệu Chuẩn (Auditing Checklist)

Khi kiểm tra một tập dữ liệu hoặc mã nguồn tiền xử lý:

1. **Kiểm tra phương thức phân chia:**
   - [ ] Kiểm tra xem code có chia ngẫu nhiên ảnh đơn lẻ không? *(Cảnh báo đỏ nếu có)*.
   - [ ] Có danh sách phân chia tường minh theo `video_id` hoặc `sequence_name` không?
2. **Kiểm tra độ giao thoa thực thể (Overlap Check):**
   - [ ] Tập Train và Val có chung người diễn xuất không?
   - [ ] Tập Train và Val có chung góc quay/căn phòng không? (Lý tưởng nhất là Leave-One-Room-Out hoặc Leave-One-Subject-Out).
3. **Kiểm tra thứ tự thời gian:**
   - [ ] Trong mỗi chuỗi video, các khung hình có được sắp xếp tuần tự theo đúng chỉ số thời gian (`frame_001, frame_002, ...`) trước khi cắt cửa sổ trượt (sliding window) hay không?

---

## 3. Mã kiểm tra Tự động Mẫu (Verification Code Snippet)

```python
def verify_zero_leakage(train_sequences, val_sequences):
    """Xác nhận không có chuỗi nào trùng lặp giữa Train và Val"""
    overlap = set(train_sequences).intersection(set(val_sequences))
    if len(overlap) > 0:
        raise ValueError(f"PHÁT HIỆN DATA LEAKAGE! Các chuỗi bị trùng: {overlap}")
    print("XÁC NHẬN: Không có rò rỉ dữ liệu giữa Train và Val (Zero Leakage).")
```
