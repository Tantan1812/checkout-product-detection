# Checkout Product Detection

Project nhận diện sản phẩm tại quầy thanh toán bằng YOLO26. Luồng inference đã triển khai cho ảnh, video và camera: thu nhận frame → YOLO26 phát hiện → overlay bounding box, tên lớp và confidence → hiển thị và lưu kết quả.

## Tài liệu

- [Giới thiệu — Tiếng Việt](docs/introduction.vi.md) · [Introduction — English](docs/introduction.en.md)
- [Hướng dẫn — Tiếng Việt](docs/tutorial.vi.md) · [Tutorial — English](docs/tutorial.en.md)
- [README in English](README.en.md)

## Nội dung đã tích hợp

- Dataset Roboflow phiên bản 2: 3,946 ảnh/nhãn, 25 lớp, tại `data/processed/product_dataset/`.
- Checkpoint huấn luyện: `models/weights/best.pt`.
- Notebook train gốc được lưu tại `notebooks/train_model.ipynb`; credential Roboflow dùng biến môi trường và output cũ đã được xóa.
- Cấu hình dataset và inference nằm trong `configs/`.

> Bạn xác nhận checkpoint và dataset tương ứng cùng bộ dữ liệu. Metadata hiện còn lệch: notebook gốc gọi Roboflow `version(1)`, còn `data.yaml` của dataset local ghi version 2. Inference dùng trực tiếp `best.pt`; nếu huấn luyện lại, hãy xác nhận nguồn/version Roboflow cần dùng.

## Cấu trúc

```text
checkout-product-detection/
├── configs/                 # Cấu hình dataset và inference
├── data/                    # Dữ liệu; không commit dữ liệu thật lên Git
├── docs/                    # Tài liệu thiết kế
├── models/weights/          # Trọng số YOLO đã huấn luyện
├── outputs/                 # Kết quả chạy và log
├── scripts/                 # Lệnh train/evaluate/predict
├── src/checkout_vision/     # Mã nguồn ứng dụng
│   ├── camera/              # Thu nhận luồng camera
│   ├── detection/           # Nạp model và hậu xử lý detection
│   ├── ui/                  # Giao diện hiển thị
│   ├── utils/               # Tiện ích và vẽ kết quả
│   └── video/               # Xử lý video theo từng frame
└── tests/                   # Unit và integration tests
```

## Bắt đầu

Yêu cầu Python 3.10+.

```bash
python -m venv .venv
# Windows PowerShell:
.venv\Scripts\Activate.ps1
# Git Bash:
source .venv/Scripts/activate
python -m pip install -e ".[dev]"
```

Cấu hình dataset nằm ở `configs/dataset.yaml`; checkpoint, nguồn vào và ngưỡng mặc định ở `configs/inference.yaml`. Mã inference đã triển khai; huấn luyện tiếp tục dùng notebook `notebooks/train_model.ipynb`.

## Chạy nhận diện

```bash
# Webcam mặc định, xem trực tiếp và lưu video vào outputs/
python -m checkout_vision --source 0

# Video: nhấn Q để dừng
python -m checkout_vision --source "path/to/checkout.mp4"

# Ảnh; tắt cửa sổ xem nếu chạy không có giao diện đồ họa
python -m checkout_vision --source "path/to/product.jpg" --no-display
```

Có thể chỉnh model, confidence, IoU, kích thước ảnh, thiết bị và chế độ lưu bằng các tùy chọn CLI hoặc `configs/inference.yaml`. Mặc định model dùng GPU CUDA 0; thêm `--device cpu` nếu cần chạy CPU.

## Dataset YOLO

```text
data/processed/product_dataset/
├── train/{images,labels}/
├── valid/{images,labels}/
└── test/{images,labels}/
```

Mỗi ảnh có file nhãn `.txt` cùng tên theo định dạng YOLO: `class_id x_center y_center width height` (tọa độ chuẩn hóa 0–1). Cấu hình đường dẫn và class names trong `configs/dataset.yaml`.

## Nguyên tắc Git

Không commit dữ liệu ảnh/video thật, trọng số lớn, virtual environment hoặc kết quả sinh ra. Các thư mục có `.gitkeep` chỉ để giữ cấu trúc ban đầu. Xem `.gitignore` trước khi thêm dữ liệu.
