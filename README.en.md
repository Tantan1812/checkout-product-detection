# Checkout Product Detection

YOLO26-based retail product-category detection for checkout images, video, and live camera streams. The application loads `models/weights/best.pt`, draws bounding boxes, class names, and confidence scores, and can save annotated results under `outputs/`.

## Included assets

- 3,946 labeled images across train/valid/test and 25 classes.
- The supplied checkpoint at `models/weights/best.pt`.
- A sanitized training notebook at `notebooks/train_model.ipynb`.
- A normalized dataset config at `configs/dataset.yaml` and inference config at `configs/inference.yaml`.

The project owner confirmed that the supplied checkpoint and local dataset correspond to the same data. The original notebook downloads Roboflow `version(1)`, while the local export metadata records version 2. Inference uses the supplied checkpoint directly; confirm the source version before retraining.

## Documentation

- [Introduction — English](docs/introduction.en.md) · [Giới thiệu — Tiếng Việt](docs/introduction.vi.md)
- [Tutorial — English](docs/tutorial.en.md) · [Hướng dẫn — Tiếng Việt](docs/tutorial.vi.md)

## Quick start on Windows

```powershell
cd C:\Users\ADMIN\checkout-product-detection
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -e ".[dev]"
python -m checkout_vision --source 0
```

Press `Q` or `Esc` to stop webcam/video preview. Use a file path for image/video input:

```powershell
python -m checkout_vision --source "C:/Videos/checkout.mp4"
python -m checkout_vision --source "C:/Images/product.jpg" --no-display
```

For options, setup details, troubleshooting, and retraining notes, see the tutorials above. The current detector predicts the dataset's 25 categories; it does not identify barcodes, prices, or individual SKUs.
