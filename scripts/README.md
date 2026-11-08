# Scripts

Ứng dụng inference chạy qua CLI của package:

```bash
python -m checkout_vision --source 0
python -m checkout_vision --source "path/to/video.mp4"
python -m checkout_vision --source "path/to/product.jpg" --no-display
```

`src/checkout_vision/cli.py` nạp cấu hình, model và chuyển nguồn vào cho pipeline. Huấn luyện model thực hiện bằng notebook `notebooks/train_model.ipynb`; `configs/dataset.yaml` là cấu hình dataset chuẩn hóa cho Ultralytics.
