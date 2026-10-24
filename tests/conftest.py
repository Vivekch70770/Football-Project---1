import sys
from pathlib import Path

import cv2
import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


@pytest.fixture
def sample_video(tmp_path):
    """A small synthetic 25 fps video (moving white square on green)."""
    p = tmp_path / "sample.mp4"
    w = cv2.VideoWriter(str(p), cv2.VideoWriter_fourcc(*"mp4v"), 25, (320, 180))
    for i in range(75):
        f = np.full((180, 320, 3), (40, 120, 40), np.uint8)
        cv2.rectangle(f, (10 + i * 3, 80), (30 + i * 3, 100), (255, 255, 255), -1)
        w.write(f)
    w.release()
    return p
