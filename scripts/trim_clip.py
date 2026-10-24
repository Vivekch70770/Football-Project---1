"""Cut a short test clip out of a long match video (no re-upload limits, much faster to analyse).

Usage:  python scripts/trim_clip.py "C:/path/match.mp4" --start 120 --duration 45
Output: data/samples/<name>_clip.mp4  (H.264, 1280px wide, audio kept if present)
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from config import settings  # noqa: E402
from services.video_processor import VideoError, run_ffmpeg  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("video")
ap.add_argument("--start", type=float, default=0)
ap.add_argument("--duration", type=float, default=45)
ap.add_argument("--width", type=int, default=1280)
a = ap.parse_args()

src = Path(a.video)
settings.SAMPLES_DIR.mkdir(parents=True, exist_ok=True)
dst = settings.SAMPLES_DIR / f"{src.stem}_clip.mp4"
try:
    run_ffmpeg(["-ss", a.start, "-t", a.duration, "-i", src, "-vf", f"scale={a.width}:-2", "-c:v", "libx264",
                "-preset", "veryfast", "-crf", "23", "-pix_fmt", "yuv420p", "-c:a", "aac", dst])
    print(f"Saved {dst}")
except VideoError as e:
    print(e)
    sys.exit(1)
