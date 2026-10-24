"""Manual pitch calibration: image points <-> real pitch coordinates (metres) via homography.

Valid only for the frame(s) where the camera is static (a panning broadcast camera needs a
per-frame calibration, which is Phase 5 work).
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import cv2
import numpy as np

from config import settings

L, W = settings.PITCH_LENGTH_M, settings.PITCH_WIDTH_M
_PA_Y0, _PA_Y1 = (W - 40.32) / 2, (W + 40.32) / 2

# Landmark presets: name -> list of (label, (x_m, y_m)). x runs along the pitch length, y across.
PRESETS: Dict[str, List] = {
    "Left penalty area corners": [("Goal line / top box corner", (0, _PA_Y0)), ("Box front / top corner", (16.5, _PA_Y0)),
                                  ("Box front / bottom corner", (16.5, _PA_Y1)), ("Goal line / bottom box corner", (0, _PA_Y1))],
    "Right penalty area corners": [("Goal line / top box corner", (L, _PA_Y0)), ("Box front / top corner", (L - 16.5, _PA_Y0)),
                                   ("Box front / bottom corner", (L - 16.5, _PA_Y1)), ("Goal line / bottom box corner", (L, _PA_Y1))],
    "Halfway line & centre circle": [("Halfway line / top touchline", (L / 2, 0)), ("Centre circle top", (L / 2, W / 2 - 9.15)),
                                     ("Centre circle bottom", (L / 2, W / 2 + 9.15)), ("Halfway line / bottom touchline", (L / 2, W))],
}


class CalibrationError(Exception):
    pass


def compute_homography(image_pts: Sequence[Sequence[float]], pitch_pts: Sequence[Sequence[float]]) -> np.ndarray:
    ip = np.asarray(image_pts, dtype=np.float32)
    pp = np.asarray(pitch_pts, dtype=np.float32)
    if ip.shape != (4, 2) or pp.shape != (4, 2):
        raise CalibrationError("Exactly 4 image points and 4 pitch points are required.")
    for pts, name in ((ip, "image"), (pp, "pitch")):
        if len(np.unique(pts, axis=0)) < 4:
            raise CalibrationError(f"Duplicate {name} points.")
        # quadrilateral area must be non-trivial (rejects collinear points)
        x, y = pts[:, 0], pts[:, 1]
        area = 0.5 * abs(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1)))
        if area < 1e-3:
            raise CalibrationError(f"The 4 {name} points are collinear / degenerate.")
    H = cv2.getPerspectiveTransform(ip, pp)
    if not np.all(np.isfinite(H)):
        raise CalibrationError("Could not compute a valid homography.")
    return H


def apply_homography(H: np.ndarray, pts: np.ndarray) -> np.ndarray:
    pts = np.asarray(pts, dtype=np.float32).reshape(-1, 1, 2)
    if len(pts) == 0:
        return np.zeros((0, 2), dtype=np.float32)
    return cv2.perspectiveTransform(pts, np.asarray(H, dtype=np.float64)).reshape(-1, 2)


def _path(job_id: int) -> Path:
    return settings.job_dir(job_id) / "calibration.json"


def save_calibration(job_id: int, frame_idx: int, image_pts, pitch_pts, H: np.ndarray) -> None:
    _path(job_id).write_text(json.dumps({"frame_idx": frame_idx, "image_pts": np.asarray(image_pts).tolist(),
                                         "pitch_pts": np.asarray(pitch_pts).tolist(), "H": np.asarray(H).tolist()}))


def load_calibration(job_id: int) -> Optional[dict]:
    p = _path(job_id)
    if not p.exists():
        return None
    d = json.loads(p.read_text())
    d["H"] = np.asarray(d["H"], dtype=np.float64)
    return d


def clear_calibration(job_id: int) -> None:
    _path(job_id).unlink(missing_ok=True)
