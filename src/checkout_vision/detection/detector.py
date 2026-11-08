"""YOLO26 model adapter and detection result normalization."""

from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class Detection:
    """One detected product, with an xyxy box and model confidence."""

    box: tuple[float, float, float, float]
    class_id: int
    label: str
    confidence: float


class ProductDetector:
    """Wrap an Ultralytics YOLO model and return portable detection records."""

    def __init__(
        self,
        model_path: str | Path,
        confidence: float = 0.25,
        iou: float = 0.45,
        image_size: int = 640,
        device: int | str | None = None,
        model: Any | None = None,
    ) -> None:
        if not 0.0 <= confidence <= 1.0:
            raise ValueError("confidence must be between 0 and 1")
        if not 0.0 <= iou <= 1.0:
            raise ValueError("iou must be between 0 and 1")
        if image_size < 1:
            raise ValueError("image_size must be positive")

        self.model_path = Path(model_path)
        self.confidence = confidence
        self.iou = iou
        self.image_size = image_size
        self.device = device

        if model is None:
            if not self.model_path.is_file():
                raise FileNotFoundError(f"Model checkpoint not found: {self.model_path}")
            try:
                from ultralytics import YOLO
            except ImportError as exc:
                raise RuntimeError(
                    "Ultralytics is not installed. Install the project dependencies with "
                    "`python -m pip install -e .`."
                ) from exc
            model = YOLO(str(self.model_path))
        self.model = model

    def predict(self, frame: Any) -> tuple[list[Detection], Any]:
        """Run detection on one OpenCV frame and return boxes plus annotated frame."""
        results = self.model.predict(
            source=frame,
            conf=self.confidence,
            iou=self.iou,
            imgsz=self.image_size,
            device=self.device,
            verbose=False,
        )
        if not results:
            return [], frame

        result = results[0]
        boxes = getattr(result, "boxes", None)
        detections: list[Detection] = []
        if boxes is not None:
            xyxy = self._to_list(boxes.xyxy)
            class_ids = self._to_list(boxes.cls)
            confidences = self._to_list(boxes.conf)
            if not (len(xyxy) == len(class_ids) == len(confidences)):
                raise ValueError("YOLO returned mismatched box, class, and confidence counts")

            names = getattr(result, "names", None) or getattr(self.model, "names", {})
            for coords, raw_class_id, raw_confidence in zip(xyxy, class_ids, confidences):
                if len(coords) != 4:
                    raise ValueError("Each YOLO bounding box must contain four coordinates")
                class_id = int(raw_class_id)
                detections.append(
                    Detection(
                        box=tuple(float(value) for value in coords),
                        class_id=class_id,
                        label=self._class_name(names, class_id),
                        confidence=float(raw_confidence),
                    )
                )

        annotated = result.plot(labels=True, boxes=True, conf=True)
        return detections, annotated

    @staticmethod
    def _to_list(values: Any) -> list[Any]:
        if hasattr(values, "detach"):
            values = values.detach()
        if hasattr(values, "cpu"):
            values = values.cpu()
        if hasattr(values, "tolist"):
            values = values.tolist()
        return list(values)

    @staticmethod
    def _class_name(names: Any, class_id: int) -> str:
        if isinstance(names, dict):
            return str(names.get(class_id, names.get(str(class_id), class_id)))
        if isinstance(names, (list, tuple)) and 0 <= class_id < len(names):
            return str(names[class_id])
        return str(class_id)
