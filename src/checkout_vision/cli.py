"""Command-line entry point for image, video, and webcam inference."""

import argparse
from pathlib import Path
from typing import Sequence

from .config import load_config
from .detection.detector import ProductDetector
from .pipeline import parse_source, process_source


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG = PROJECT_ROOT / "configs" / "inference.yaml"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="checkout-vision",
        description="Detect checkout products in an image, video, or live camera stream.",
    )
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG, help="Inference YAML config")
    parser.add_argument("--source", help="Image/video path or camera index such as 0")
    parser.add_argument("--model", type=Path, help="Override the model checkpoint path")
    parser.add_argument("--confidence", type=float, help="Override confidence threshold (0–1)")
    parser.add_argument("--iou", type=float, help="Override NMS IoU threshold (0–1)")
    parser.add_argument("--image-size", type=int, help="Override inference image size")
    parser.add_argument("--device", help="Override device, for example 0 or cpu")
    parser.add_argument("--no-display", action="store_true", help="Do not open a preview window")
    parser.add_argument("--no-save", action="store_true", help="Do not save annotated output")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        config = load_config(args.config)
        model_path = args.model or config.model
        if args.model is not None and not args.model.is_absolute():
            model_path = (Path.cwd() / args.model).resolve()
        source = config.source if args.source is None else parse_source(args.source)
        confidence = config.confidence if args.confidence is None else args.confidence
        iou = config.iou if args.iou is None else args.iou
        image_size = config.image_size if args.image_size is None else args.image_size
        device = config.device if args.device is None else _parse_device(args.device)
        show = config.show and not args.no_display
        save_output = config.save_output and not args.no_save

        detector = ProductDetector(
            model_path=model_path,
            confidence=confidence,
            iou=iou,
            image_size=image_size,
            device=device,
        )
        summary = process_source(
            source,
            detector,
            show=show,
            save_output=save_output,
            output_dir=config.output_dir,
        )
    except (FileNotFoundError, RuntimeError, ValueError, OSError) as exc:
        parser.exit(2, f"Error: {exc}\n")

    print(
        f"Processed {summary['frames_processed']} frame(s); "
        f"{summary['detections']} detection(s)."
    )
    if summary["output_path"] is not None:
        print(f"Annotated output: {summary['output_path']}")
    return 0


def _parse_device(value: str) -> int | str:
    text = value.strip()
    return int(text) if text.isdigit() else text
