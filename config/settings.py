"""Central configuration for VARVision AI."""
from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
UPLOAD_DIR = DATA_DIR / "uploads"
SAMPLES_DIR = DATA_DIR / "samples"
OUTPUT_DIR = ROOT / "outputs"
MODELS_DIR = ROOT / "models"
DB_PATH = DATA_DIR / "varvision.db"

ALLOWED_EXTENSIONS = {".mp4", ".mov", ".mkv", ".avi", ".m4v"}
MAX_UPLOAD_MB = 500

# Ultralytics model weights. COCO-pretrained models know "person" (0) and
# "sports ball" (32) only - NOT referee / goalkeeper / goal / pitch lines.
DEFAULT_MODEL = "yolov8n.pt"
MODEL_CHOICES = ["yolov8n.pt", "yolov8s.pt", "yolov8m.pt"]
TRACKER_CHOICES = ["bytetrack", "botsort"]
PERSON_CLASS = 0
BALL_CLASS = 32

# Standard FIFA pitch used for the top-down view (metres).
PITCH_LENGTH_M = 105.0
PITCH_WIDTH_M = 68.0

TEAM_LABELS = {0: "Team A", 1: "Team B", 2: "Referee/GK/Other", -1: "Unassigned"}

EVENT_TYPES = ["goal", "shot", "save", "tackle", "foul", "corner", "other",
               "ball_speed_spike", "possible_duel"]
EVENT_STATUSES = ["pending", "confirmed", "rejected"]


def ensure_dirs() -> None:
    for d in (DATA_DIR, UPLOAD_DIR, SAMPLES_DIR, OUTPUT_DIR, MODELS_DIR):
        d.mkdir(parents=True, exist_ok=True)


def job_dir(job_id: int) -> Path:
    d = OUTPUT_DIR / f"job_{job_id}"
    d.mkdir(parents=True, exist_ok=True)
    return d
