# Tutorial: cài đặt và chạy nhận diện

## 1. Yêu cầu

- Python 3.10 trở lên.
- Project có sẵn `models/weights/best.pt`.
- Webcam nếu muốn chạy camera trực tiếp.

## 2. Tạo môi trường và cài thư viện

Mở PowerShell tại thư mục project:

```powershell
cd C:\Users\ADMIN\checkout-product-detection
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -e ".[dev]"
```

Nếu PowerShell không cho kích hoạt môi trường, có thể dùng Git Bash thay vì thay đổi chính sách bảo mật:

```bash
source .venv/Scripts/activate
```

Gói cài đặt sẽ lấy Ultralytics, PyTorch, OpenCV và PyYAML theo dependency của project. Lần cài đầu có thể mất thời gian và cần mạng.

## 3. Chạy webcam

```powershell
python -m checkout_vision --source 0
```

Cửa sổ hiển thị frame có bounding box, tên lớp và confidence. Nhấn `Q` hoặc `Esc` để dừng. Video đã gắn nhãn được lưu trong `outputs/` nếu `save_output: true`.

Nếu camera mặc định không mở, thử chỉ số khác, ví dụ `--source 1`. Đảm bảo ứng dụng khác không đang giữ camera.

## 4. Chạy file video

```powershell
python -m checkout_vision --source "C:/Videos/checkout.mp4"
```

Ứng dụng xử lý tuần tự từng frame, hiển thị kết quả và lưu video MP4 trong `outputs/`. Nhấn `Q` hoặc `Esc` để dừng sớm.

## 5. Chạy một ảnh

```powershell
python -m checkout_vision --source "C:/Images/products.jpg"
```

Ảnh kết quả được lưu với hậu tố `_detected`, ví dụ `outputs/products_detected.jpg`. Để không mở cửa sổ preview:

```powershell
python -m checkout_vision --source "C:/Images/products.jpg" --no-display
```

## 6. Tùy chỉnh nhanh

```powershell
python -m checkout_vision --source 0 --confidence 0.35 --iou 0.45 --image-size 640 --device 0
```

- `--confidence`: ngưỡng confidence từ 0 đến 1; tăng ngưỡng sẽ loại nhiều dự đoán yếu hơn.
- `--iou`: ngưỡng NMS IoU từ 0 đến 1.
- `--image-size`: kích thước ảnh đưa vào model.
- `--device 0`: GPU CUDA thứ nhất; dùng `--device cpu` để chạy CPU.
- `--no-display`: chỉ xử lý/lưu, không tạo cửa sổ xem.
- `--no-save`: không ghi ảnh/video kết quả.

Tham số mặc định nằm trong `configs/inference.yaml`. Có thể thay `model` bằng đường dẫn checkpoint khác hoặc đặt `source` thành đường dẫn file. Đường dẫn tương đối trong file cấu hình được tính từ thư mục gốc project.

## 7. Kiểm thử

```powershell
python -m unittest discover -s tests/unit -v
```

## 8. Huấn luyện lại

Inference không cần chạy notebook. Nếu muốn huấn luyện lại, mở `notebooks/train_model.ipynb` và đặt `ROBOFLOW_API_KEY` trong secret/environment của notebook; không ghi API key trực tiếp vào file. Notebook gốc gọi Roboflow `version(1)`, còn metadata của dataset local ghi version 2. Hãy xác nhận phiên bản nguồn phù hợp trước khi train hoặc đánh giá.

## Xử lý lỗi thường gặp

- **Không tìm thấy checkpoint**: kiểm tra `models/weights/best.pt` hoặc giá trị `model` trong `configs/inference.yaml`.
- **Thiếu package**: chạy `python -m pip install -e ".[dev]"` trong đúng virtual environment.
- **Không có cửa sổ preview**: thêm `--no-display`; kết quả vẫn được lưu nếu bật `save_output`.
- **Không mở được camera**: thử `--source 1` hoặc kiểm tra quyền truy cập camera.
- **Không dùng được GPU**: thử `--device cpu`; để dùng GPU cần PyTorch/CUDA tương thích với máy.
