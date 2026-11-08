"""OpenCV loop for files and live camera streams."""

from datetime import datetime
from pathlib import Path
from typing import Any


def process_video(
    source: int | str | Path,
    detector: Any,
    *,
    show: bool = True,
    save_output: bool = True,
    output_dir: str | Path = "outputs",
    window_name: str = "Checkout product detection",
) -> dict[str, Any]:
    """Detect products in every frame until the source ends or the user presses Q."""
    try:
        import cv2
    except ImportError as exc:
        raise RuntimeError(
            "OpenCV is not installed. Install the project dependencies with "
            "`python -m pip install -e .`."
        ) from exc

    capture_source: int | str = source if isinstance(source, int) else str(source)
    capture = cv2.VideoCapture(capture_source)
    if not capture.isOpened():
        capture.release()
        raise FileNotFoundError(f"Could not open video or camera source: {source}")

    writer = None
    output_path: Path | None = None
    frames_processed = 0
    detections_total = 0
    output_root = Path(output_dir)
    fps = float(capture.get(cv2.CAP_PROP_FPS))
    if fps <= 0:
        fps = 30.0

    if save_output:
        output_root.mkdir(parents=True, exist_ok=True)
        output_path = output_root / _output_name(source)

    try:
        while True:
            ok, frame = capture.read()
            if not ok:
                break

            detections, annotated = detector.predict(frame)
            frames_processed += 1
            detections_total += len(detections)

            if save_output and writer is None:
                height, width = annotated.shape[:2]
                fourcc = cv2.VideoWriter_fourcc(*"mp4v")
                writer = cv2.VideoWriter(str(output_path), fourcc, fps, (width, height))
                if not writer.isOpened():
                    raise RuntimeError(f"Could not create output video: {output_path}")
            if writer is not None:
                writer.write(annotated)

            if show:
                cv2.imshow(window_name, annotated)
                key = cv2.waitKey(1) & 0xFF
                if key in (ord("q"), 27):
                    break
    finally:
        capture.release()
        if writer is not None:
            writer.release()
        if show:
            cv2.destroyAllWindows()

    if frames_processed == 0:
        raise RuntimeError(f"No frames could be read from source: {source}")

    return {
        "frames_processed": frames_processed,
        "detections": detections_total,
        "output_path": output_path,
    }


def _output_name(source: int | str | Path) -> str:
    if isinstance(source, int):
        stem = f"camera_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    else:
        stem = Path(str(source)).stem or "video"
        stem = f"{stem}_detected"
    return f"{stem}.mp4"
