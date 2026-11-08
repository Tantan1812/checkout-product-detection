# Giới thiệu

## Mục tiêu

Project nhận diện các nhóm sản phẩm bán lẻ tại quầy thanh toán bằng YOLO26. Ứng dụng nhận đầu vào là ảnh, video hoặc camera, phát hiện đối tượng trong từng ảnh/frame, rồi hiển thị bounding box cùng tên lớp và độ tin cậy.

## Luồng xử lý

```text
Ảnh / video / webcam
        ↓
Đọc frame bằng OpenCV
        ↓
YOLO26 + models/weights/best.pt
        ↓
Bounding box + class ID + confidence
        ↓
Vẽ nhãn và hộp giới hạn lên frame
        ↓
Hiển thị trực tiếp và/hoặc lưu trong outputs/
```

Model tự tiền xử lý frame theo cấu hình `image_size`. Với video và webcam, vòng lặp xử lý liên tiếp từng frame; nhấn `Q` hoặc `Esc` để dừng. Nhãn lớp được lấy từ metadata của checkpoint để giữ đúng thứ tự lớp đã huấn luyện.

## Dataset và checkpoint

- Dataset local: 3.946 ảnh có nhãn YOLO, gồm train 2.762, valid 789 và test 395.
- Dataset có 25 lớp; danh sách và thứ tự lớp nằm trong `configs/dataset.yaml`.
- Checkpoint dùng cho inference: `models/weights/best.pt`.
- Dataset Roboflow có license CC BY 4.0; giữ file attribution trong `data/processed/product_dataset/README.roboflow.txt` khi chia sẻ.

Project owner xác nhận checkpoint và dataset local tương ứng cùng bộ dữ liệu. Metadata vẫn khác cách ghi version: notebook train gốc tải Roboflow `version(1)`, còn `source_data.yaml` ghi version 2. Việc chạy inference với `best.pt` không cần tải dataset; trước khi huấn luyện lại, hãy xác nhận version nguồn cần dùng.

## Các thành phần chính

- `src/checkout_vision/cli.py`: nhận tham số dòng lệnh và nạp cấu hình.
- `src/checkout_vision/config.py`: đọc `configs/inference.yaml`, giải quyết đường dẫn từ thư mục gốc project.
- `src/checkout_vision/detection/detector.py`: gọi Ultralytics YOLO và chuẩn hóa kết quả detection.
- `src/checkout_vision/pipeline.py`: phân luồng ảnh với video/camera.
- `src/checkout_vision/video/video_processor.py`: đọc, xử lý, hiển thị và lưu video từng frame.
- `configs/inference.yaml`: checkpoint, nguồn vào, ngưỡng, thiết bị và output.
- `notebooks/train_model.ipynb`: notebook huấn luyện gốc; không cần chạy để dùng model hiện có.

## Giới hạn hiện tại

Model phân loại theo 25 nhóm lớp của dataset, không nhận diện barcode, giá tiền hay SKU riêng lẻ và chưa kết nối hệ thống POS. Tốc độ thực tế phụ thuộc GPU/CPU, độ phân giải và cấu hình model.
