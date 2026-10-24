"""YOLO model loading. COCO-pretrained weights only know 'person' and 'sports ball'."""
from __future__ import annotations

from pathlib import Path

from config import settings


class ModelError(Exception):
    """Raised when YOLO weights or the ultralytics package are unavailable."""


def pick_device():
    try:
        import torch
        return 0 if torch.cuda.is_available() else "cpu"
    except Exception:
        return "cpu"


def model_path(name: str) -> Path:
    # Only allow plain file names inside models/ (no path traversal).
    return settings.MODELS_DIR / Path(name).name


def load_model(name: str = settings.DEFAULT_MODEL):
    try:
        from ultralytics import YOLO
    except ImportError as e:
        raise ModelError("The 'ultralytics' package is not installed. Run: pip install -r requirements.txt") from e

    path = model_path(name)
    if not path.exists():
        try:
            from ultralytics.utils.downloads import attempt_download_asset
            settings.MODELS_DIR.mkdir(parents=True, exist_ok=True)
            attempt_download_asset(str(path))
        except Exception as e:  # network down, blocked, etc.
            raise ModelError(
                f"Model weights '{path.name}' not found in the models/ folder and the automatic download failed "
                f"({type(e).__name__}). Download it manually from "
                f"https://github.com/ultralytics/assets/releases and place it in: models/") from e
    if not path.exists():
        raise ModelError(f"Model weights '{path.name}' are missing. Put the .pt file in the models/ folder.")
    try:
        return YOLO(str(path))
    except Exception as e:
        raise ModelError(f"Could not load model '{path.name}': {e}") from e
