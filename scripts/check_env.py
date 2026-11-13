import sys


def check_pytorch_and_cuda() -> None:
    print("=" * 50)
    print("1. KIỂM TRA PYTORCH & CUDA (GPU RTX)")
    print("=" * 50)
    try:
        import torch
    except ImportError:
        print("[LỖI] Chưa cài đặt PyTorch (torch).")
        return

    print(f"- Phiên bản PyTorch : {torch.__version__}")
    cuda_available = torch.cuda.is_available()
    print(f"- Trạng thái CUDA   : {'Khả dụng (Available)' if cuda_available else 'Không khả dụng (Not Available)'}")

    if cuda_available:
        print(f"- Phiên bản CUDA    : {torch.version.cuda}")
        device_count = torch.cuda.device_count()
        print(f"- Số lượng GPU      : {device_count}")
        for i in range(device_count):
            device_name = torch.cuda.get_device_name(i)
            props = torch.cuda.get_device_properties(i)
            vram_gb = props.total_memory / (1024 ** 3)
            print(f"  + GPU [{i}]         : {device_name} ({vram_gb:.2f} GB VRAM)")

        try:
            x = torch.rand(3, 3, device="cuda")
            print(f"- Kiểm tra Tensor   : Thành công (đã cấp phát tensor trên {x.device})")
        except Exception as e:
            print(f"- Kiểm tra Tensor   : Thất bại ({e})")
    else:
        print("[CẢNH BÁO] PyTorch đang chạy trên CPU. Hãy kiểm tra lại driver NVIDIA hoặc bản cài đặt PyTorch kèm CUDA.")


def check_opencv_camera(camera_index: int = 0) -> None:
    print("\n" + "=" * 50)
    print("2. KIỂM TRA OPENCV & CAMERA")
    print("=" * 50)
    try:
        import cv2
    except ImportError:
        print("[LỖI] Chưa cài đặt OpenCV (opencv-python).")
        return

    print(f"- Phiên bản OpenCV  : {cv2.__version__}")
    print(f"- Đang thử mở camera (index = {camera_index})...")

    # Sử dụng CAP_DSHOW trên Windows để mở camera nhanh và ổn định hơn, fallback về mặc định nếu cần
    cap = cv2.VideoCapture(camera_index, cv2.CAP_DSHOW) if sys.platform == "win32" else cv2.VideoCapture(camera_index)
    if not cap.isOpened():
        cap = cv2.VideoCapture(camera_index)

    if not cap.isOpened():
        print(f"[LỖI] Không thể mở được camera ở index {camera_index}.")
        return

    ret, frame = cap.read()
    if ret and frame is not None:
        height, width = frame.shape[:2]
        print(f"- Trạng thái Camera : Hoạt động tốt (Mở và đọc frame thành công)")
        print(f"- Độ phân giải Frame: {width}x{height}")
    else:
        print(f"[CẢNH BÁO] Đã mở được camera index {camera_index} nhưng không đọc được khung hình (frame).")

    cap.release()


def main() -> None:
    print(f"Python: {sys.version.split()[0]} ({sys.executable})")
    check_pytorch_and_cuda()
    check_opencv_camera(0)
    print("\n" + "=" * 50)
    print("HOÀN TẤT KIỂM TRA MÔI TRƯỜNG")
    print("=" * 50)


if __name__ == "__main__":
    main()
