# Cấu trúc Thư mục `detect_v3/`

Thư mục `detect_v3/` đã được tổ chức lại gọn gàng thành 5 mục theo đúng từng công đoạn:

1. **`data_prep/` — Mục chuẩn bị dữ liệu**
   - `1_extract_pose_cache.py`, `download_fallvision.py`, `convert_fallvision_to_cache.py`, `prepare_multicam_data.py`, `prepare_gmdcsa24_data.py`, `prepare_cctv_data.py`, `import_own_videos.py`, `record_samples.py`
2. **`data/` — Mục lưu trữ dữ liệu thô & cache `.npz`**
   - `cache/`, `fallvision_cache/`, `mc_cache/`, `gmd_cache/`, `own_data/`, `own_videos/`, `external_data/`
3. **`model/` — Mục mô hình (thuật toán, train, đánh giá, chạy thực tế)**
   - `fall_common.py`, `fall_data.py`
   - `2_train.py`: Train mô hình (`python detect_v3/model/2_train.py --use-fv --use-mc --use-gmd --hnm`)
   - `3_evaluate.py`: Đánh giá mô hình (`python detect_v3/model/3_evaluate.py`)
   - `4_realtime.py`: Chạy thực tế (`python detect_v3/model/4_realtime.py --source detect_v3/test/03.mp4 --show`)
   - `verify_v3.py`: Kiểm chứng đa góc & đa người
   - `weights/`: Chứa các trọng số `.pth` (`fall_net_v3_lstm.pth`, `fall_net.pth`, ...)
4. **`test/` — Mục chứa các video test đầu vào**
   - Chứa `03.mp4`, `1.mp4`, `21.mp4`, `cam1.mp4`, `cam5.mp4`, `test.mp4`, `test_1.mp4`, `test_2.mp4`, `test_3.mp4`, `test_cam1.mp4`, `test_toi_1.mp4`
5. **`result/` — Mục chứa video & ảnh kết quả sau khi chạy test**
   - Khi chạy test video với cờ `--save-result`:
     ```bash
     python detect_v3/model/4_realtime.py --selftest detect_v3/test/03.mp4 --save-result
     ```
     Video đầu ra có vẽ khung xương + trạng thái (`03_result.mp4`) sẽ tự động được lưu vào `detect_v3/result/`.
