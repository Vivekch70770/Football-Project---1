"""Download YOLO weights into models/.   Usage: python scripts/download_model.py [yolov8n.pt]"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from services.detection import ModelError, load_model  # noqa: E402

name = sys.argv[1] if len(sys.argv) > 1 else "yolov8n.pt"
try:
    load_model(name)
    print(f"OK - {name} is ready in models/")
except ModelError as e:
    print(f"FAILED: {e}")
    sys.exit(1)
