"""Route image/video/camera inputs through the detector."""

from pathlib import Path
from typing import Any


IMAGE_EXTENSIONS = {".bmp", ".jpeg", ".jpg", ".png", ".tif", ".tiff", ".webp"}


def parse_source(value: str | int | Path) -> int | str:
    """Turn a numeric CLI source into a camera index; keep paths unchanged."""
    if isinstance(value, bool):
        raise ValueError("source cannot be a boolean")
    if isinstance(value, int):
        if value < 0:
            raise ValueError("camera index cannot be negative")
        return value
    text = str(value).strip()
    if not text:
        raise ValueError("source cannot be empty")
    if text.isdigit():
        return int(text)
    return text


def is_image_source(source: int | str | Path) -> bool:
    if isinstance(source, int):
        return False
    return Path(str(source)).suffix.lower() in IMAGE_EXTENSIONS


def process_source(
    source: int | str | Path,
    detector: Any,
    *,
    show: bool = True,
    save_output: bool = True,
    output_dir: str | Path = "outputs",
) -> dict[str, Any]:
    """Run one image or stream a video/camera, returning a compact run summary."""
    source = parse_source(source)
    if is_image_source(source):
        return process_image(
            source,
            detector,
            show=show,
            save_output=save_output,
            output_dir=output_dir,
        )

    from .video.video_processor import process_video

    return process_video(
        source,
        detector,
        show=show,
        save_output=save_output,
        output_dir=output_dir,
    )


def process_image(
    source: str | Path,
    detector: Any,
    *,
    show: bool = True,
    save_output: bool = True,
    output_dir: str | Path = "outputs",
) -> dict[str, Any]:
    try:
        import cv2
    except ImportError as exc:
        raise RuntimeError(
            "OpenCV is not installed. Install the project dependencies with "
            "`python -m pip install -e .`."
        ) from exc

    source_path = Path(source)
    image = cv2.imread(str(source_path))
    if image is None:
        raise FileNotFoundError(f"Could not read image: {source_path}")

    detections, annotated = detector.predict(image)
    output_path: Path | None = None
    if save_output:
        output_root = Path(output_dir)
        output_root.mkdir(parents=True, exist_ok=True)
        output_path = output_root / f"{source_path.stem}_detected{source_path.suffix.lower()}"
        if not cv2.imwrite(str(output_path), annotated):
            raise OSError(f"Could not write annotated image: {output_path}")

    if show:
        try:
            cv2.imshow("Checkout product detection", annotated)
            cv2.waitKey(0)
        finally:
            cv2.destroyAllWindows()

    return {
        "frames_processed": 1,
        "detections": len(detections),
        "output_path": output_path,
    }
