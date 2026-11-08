# Tutorial: Install and Run Inference

## 1. Requirements

- Python 3.10 or later.
- The project includes `models/weights/best.pt`.
- A webcam is required for live camera inference.

## 2. Create an environment and install dependencies

Open PowerShell in the project directory:

```powershell
cd C:\Users\ADMIN\checkout-product-detection
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -e ".[dev]"
```

If PowerShell does not allow environment activation, use Git Bash rather than changing security policy:

```bash
source .venv/Scripts/activate
```

Installation pulls Ultralytics, PyTorch, OpenCV, and PyYAML according to the project dependencies. The first installation may take time and requires an internet connection.

## 3. Run the webcam

```powershell
python -m checkout_vision --source 0
```

A preview window shows frames with bounding boxes, class names, and confidence scores. Press `Q` or `Esc` to stop. Annotated video is saved under `outputs/` when `save_output: true`.

If the default camera does not open, try another index such as `--source 1`. Make sure another application is not using the camera.

## 4. Run a video file

```powershell
python -m checkout_vision --source "C:/Videos/checkout.mp4"
```

The application processes frames sequentially, previews detections, and saves an MP4 under `outputs/`. Press `Q` or `Esc` to stop early.

## 5. Run a single image

```powershell
python -m checkout_vision --source "C:/Images/products.jpg"
```

The annotated image is saved with the `_detected` suffix, for example `outputs/products_detected.jpg`. To skip the preview window:

```powershell
python -m checkout_vision --source "C:/Images/products.jpg" --no-display
```

## 6. Quick options

```powershell
python -m checkout_vision --source 0 --confidence 0.35 --iou 0.45 --image-size 640 --device 0
```

- `--confidence`: threshold from 0 to 1; a higher threshold filters more low-confidence predictions.
- `--iou`: NMS IoU threshold from 0 to 1.
- `--image-size`: model input image size.
- `--device 0`: first CUDA GPU; use `--device cpu` for CPU inference.
- `--no-display`: process/save without opening a preview window.
- `--no-save`: do not write annotated images/videos.

Defaults are in `configs/inference.yaml`. Set `model` to another checkpoint or `source` to an input path. Relative paths in the config are resolved from the project root.

## 7. Run tests

```powershell
python -m unittest discover -s tests/unit -v
```

## 8. Retraining

Inference does not require running the notebook. To retrain, open `notebooks/train_model.ipynb` and provide `ROBOFLOW_API_KEY` through the notebook's secret/environment mechanism; do not hard-code the key in the file. The original notebook calls Roboflow `version(1)`, while the local dataset metadata records version 2. Confirm the authoritative source version before training or evaluation.

## Troubleshooting

- **Checkpoint not found**: check `models/weights/best.pt` or `model` in `configs/inference.yaml`.
- **Missing packages**: run `python -m pip install -e ".[dev]"` in the active virtual environment.
- **No preview window**: add `--no-display`; results are still saved when `save_output` is enabled.
- **Camera will not open**: try `--source 1` or check camera access permissions.
- **GPU unavailable**: try `--device cpu`; GPU inference requires a PyTorch/CUDA combination compatible with the machine.
