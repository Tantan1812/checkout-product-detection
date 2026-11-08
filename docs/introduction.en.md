# Introduction

## Purpose

This project detects retail product categories at a checkout counter with YOLO26. It accepts an image, video, or camera stream, detects objects in each image/frame, and displays bounding boxes with class names and confidence scores.

## Processing flow

```text
Image / video / webcam
        ↓
Read frames with OpenCV
        ↓
YOLO26 + models/weights/best.pt
        ↓
Bounding box + class ID + confidence
        ↓
Draw labels and boxes on the frame
        ↓
Preview live and/or save to outputs/
```

The model preprocesses each frame using the configured `image_size`. Video and webcam inputs are processed frame by frame; press `Q` or `Esc` to stop. Class names come from checkpoint metadata so the trained class ordering is preserved.

## Dataset and checkpoint

- Local dataset: 3,946 YOLO-labeled images: 2,762 train, 789 valid, and 395 test images.
- The dataset has 25 classes; names and ordering are in `configs/dataset.yaml`.
- Inference checkpoint: `models/weights/best.pt`.
- The Roboflow dataset is licensed under CC BY 4.0. Retain the attribution file at `data/processed/product_dataset/README.roboflow.txt` when redistributing it.

The project owner confirmed that the supplied checkpoint and local dataset correspond to the same data. Their metadata uses different version notation: the original training notebook downloads Roboflow `version(1)`, while `source_data.yaml` records version 2. Inference with `best.pt` does not require downloading the dataset; confirm the authoritative source version before retraining.

## Main components

- `src/checkout_vision/cli.py`: parses command-line options and loads configuration.
- `src/checkout_vision/config.py`: reads `configs/inference.yaml` and resolves paths from the project root.
- `src/checkout_vision/detection/detector.py`: invokes Ultralytics YOLO and normalizes detections.
- `src/checkout_vision/pipeline.py`: routes image inputs versus video/camera inputs.
- `src/checkout_vision/video/video_processor.py`: reads, processes, previews, and saves video frame by frame.
- `configs/inference.yaml`: checkpoint, input source, thresholds, device, and output settings.
- `notebooks/train_model.ipynb`: original training notebook; it is not required to run the supplied checkpoint.

## Current limitations

The model predicts the dataset's 25 product categories; it does not read barcodes, prices, or individual SKUs and is not connected to a POS system. Actual speed depends on CPU/GPU, image size, and model settings.
