"""Load inference settings and resolve project-relative paths."""

from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class InferenceConfig:
    model: Path
    source: int | str
    confidence: float
    iou: float
    image_size: int
    device: int | str | None
    show: bool
    save_output: bool
    output_dir: Path


def load_config(path: str | Path) -> InferenceConfig:
    config_path = Path(path).expanduser().resolve()
    if not config_path.is_file():
        raise FileNotFoundError(f"Inference config not found: {config_path}")
    try:
        import yaml
    except ImportError as exc:
        raise RuntimeError(
            "PyYAML is not installed. Install the project dependencies with "
            "`python -m pip install -e .`."
        ) from exc

    raw: Any = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ValueError("Inference config must contain a YAML mapping")
    project_root = config_path.parent.parent

    model = _resolve_path(raw.get("model"), project_root, "model")
    output_dir = _resolve_path(raw.get("output_dir", "outputs"), project_root, "output_dir")
    source = _resolve_source(raw.get("source", 0), project_root)

    confidence = float(raw.get("confidence", 0.25))
    iou = float(raw.get("iou", 0.45))
    image_size = int(raw.get("image_size", 640))
    if not 0.0 <= confidence <= 1.0:
        raise ValueError("confidence must be between 0 and 1")
    if not 0.0 <= iou <= 1.0:
        raise ValueError("iou must be between 0 and 1")
    if image_size < 1:
        raise ValueError("image_size must be positive")

    return InferenceConfig(
        model=model,
        source=source,
        confidence=confidence,
        iou=iou,
        image_size=image_size,
        device=raw.get("device"),
        show=bool(raw.get("show", True)),
        save_output=bool(raw.get("save_output", True)),
        output_dir=output_dir,
    )


def _resolve_path(value: Any, root: Path, key: str) -> Path:
    if not isinstance(value, (str, Path)) or not str(value).strip():
        raise ValueError(f"{key} must be a non-empty path")
    path = Path(value).expanduser()
    return path.resolve() if path.is_absolute() else (root / path).resolve()


def _resolve_source(value: Any, root: Path) -> int | str:
    if isinstance(value, bool):
        raise ValueError("source cannot be a boolean")
    if isinstance(value, int):
        if value < 0:
            raise ValueError("camera index cannot be negative")
        return value
    if not isinstance(value, str) or not value.strip():
        raise ValueError("source must be a camera index or a non-empty path")
    value = value.strip()
    if value.isdigit():
        return int(value)
    path = Path(value).expanduser()
    return str(path.resolve() if path.is_absolute() else (root / path).resolve())
