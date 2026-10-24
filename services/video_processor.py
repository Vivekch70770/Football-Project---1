"""Video I/O helpers: validation, metadata, frame iteration and FFmpeg access."""
from __future__ import annotations

import math
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator, Optional, Tuple

import cv2
import numpy as np

from config import settings


class VideoError(Exception):
    """Raised for missing, unsupported or corrupt video files / FFmpeg failures."""


@dataclass(frozen=True)
class VideoMeta:
    path: str
    fps: float
    frame_count: int
    width: int
    height: int

    @property
    def duration_s(self) -> float:
        return self.frame_count / self.fps if self.fps > 0 else 0.0


def frame_to_time(frame_idx: int, fps: float) -> float:
    if fps <= 0:
        raise ValueError("fps must be positive")
    return frame_idx / fps


def time_to_frame(time_s: float, fps: float) -> int:
    if fps <= 0:
        raise ValueError("fps must be positive")
    return max(0, int(round(time_s * fps)))


def validate_video_path(path) -> Path:
    p = Path(path)
    if not p.is_file():
        raise VideoError(f"Video file not found: {p.name}")
    if p.suffix.lower() not in settings.ALLOWED_EXTENSIONS:
        allowed = ", ".join(sorted(settings.ALLOWED_EXTENSIONS))
        raise VideoError(f"Unsupported file type '{p.suffix}'. Allowed: {allowed}")
    if p.stat().st_size == 0:
        raise VideoError("The video file is empty.")
    return p


def read_metadata(path) -> VideoMeta:
    p = validate_video_path(path)
    cap = cv2.VideoCapture(str(p))
    try:
        if not cap.isOpened():
            raise VideoError("OpenCV could not open this video (corrupt file or unsupported codec). "
                             "Try re-encoding it to H.264 MP4 (see README, 'Troubleshooting').")
        fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
        n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
        ok, frame = cap.read()
        if not ok or frame is None:
            raise VideoError("The first frame could not be decoded - the video is corrupt or uses an unsupported codec.")
        if w <= 0 or h <= 0:
            h, w = frame.shape[:2]
        if not math.isfinite(fps) or fps <= 0 or fps > 240:
            fps = 25.0
        if n <= 0:
            raise VideoError("Could not determine the number of frames in this video.")
        return VideoMeta(str(p), fps, n, w, h)
    finally:
        cap.release()


def iter_frames(path, start_frame: int = 0, end_frame: Optional[int] = None,
                stride: int = 1) -> Iterator[Tuple[int, np.ndarray]]:
    """Yield (frame_index, BGR frame) one at a time - never loads the whole video."""
    if stride < 1:
        raise ValueError("stride must be >= 1")
    cap = cv2.VideoCapture(str(path))
    try:
        if not cap.isOpened():
            raise VideoError("Could not open the video for reading.")
        if start_frame > 0:
            cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
        idx = start_frame
        while end_frame is None or idx < end_frame:
            if (idx - start_frame) % stride == 0:
                ok, frame = cap.read()
                if not ok:
                    break
                yield idx, frame
            else:
                if not cap.grab():
                    break
            idx += 1
    finally:
        cap.release()


def read_frame(path, frame_idx: int) -> Optional[np.ndarray]:
    cap = cv2.VideoCapture(str(path))
    try:
        cap.set(cv2.CAP_PROP_POS_FRAMES, max(0, frame_idx))
        ok, frame = cap.read()
        return frame if ok else None
    finally:
        cap.release()


# --------------------------------------------------------------------- FFmpeg
def ffmpeg_exe() -> str:
    try:
        import imageio_ffmpeg
        return imageio_ffmpeg.get_ffmpeg_exe()
    except Exception:
        exe = shutil.which("ffmpeg")
        if exe:
            return exe
    raise VideoError("FFmpeg not found. Run: pip install imageio-ffmpeg (or install FFmpeg and add it to PATH).")


def run_ffmpeg(args: list) -> None:
    cmd = [ffmpeg_exe(), "-y", "-hide_banner", "-loglevel", "error", *map(str, args)]
    proc = subprocess.run(cmd, capture_output=True, text=True)
    if proc.returncode != 0:
        tail = "\n".join(proc.stderr.strip().splitlines()[-6:])
        raise VideoError(f"FFmpeg failed: {tail}")


def has_audio(path) -> bool:
    proc = subprocess.run([ffmpeg_exe(), "-hide_banner", "-i", str(path)], capture_output=True, text=True)
    return "Audio:" in proc.stderr


def transcode_h264(src, dst, crf: int = 23, preset: str = "veryfast") -> None:
    """Re-encode to browser-playable H.264 (OpenCV's mp4v output does not play in browsers)."""
    run_ffmpeg(["-i", src, "-c:v", "libx264", "-preset", preset, "-crf", crf,
                "-pix_fmt", "yuv420p", "-movflags", "+faststart", "-an", dst])
