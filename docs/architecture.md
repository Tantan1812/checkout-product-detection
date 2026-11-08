# Luồng inference đã triển khai

1. **Nạp cấu hình**: `configs/inference.yaml` cung cấp checkpoint `best.pt`, camera mặc định, ngưỡng confidence/IoU, image size, thiết bị và thư mục output. Đường dẫn tương đối được giải quyết từ thư mục gốc project.
2. **Thu nhận đầu vào**: `pipeline.py` phân loại ảnh theo phần mở rộng; video và camera được xử lý bằng `video/video_processor.py`.
3. **Tiền xử lý và phát hiện**: `ProductDetector` chuyển từng ảnh/frame cho Ultralytics YOLO26, áp dụng `confidence`, `iou`, `image_size`.
4. **Chuẩn hóa kết quả**: mỗi detection chứa bounding box `xyxy`, class ID, tên lớp và confidence. Tên lớp lấy từ metadata checkpoint.
5. **Vẽ kết quả**: `Results.plot()` vẽ hộp giới hạn, tên lớp và confidence lên ảnh/frame.
6. **Hiển thị/lưu**: ảnh được lưu thành `<tên>_detected.<đuôi>`; video/camera được lưu MP4 trong `outputs/`. Cửa sổ video cập nhật từng frame; nhấn `Q` hoặc `Esc` để dừng.

Chạy `python -m checkout_vision --source 0` để mở camera mặc định, hoặc truyền đường dẫn ảnh/video. Chạy `--no-display` khi môi trường không có cửa sổ đồ họa. Huấn luyện model vẫn thực hiện qua `notebooks/train_model.ipynb`.
