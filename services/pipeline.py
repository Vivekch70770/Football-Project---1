"""End-to-end analysis job: track -> save data -> render annotated video."""
from __future__ import annotations

import json
import math
import traceback
from pathlib import Path
from typing import Callable, Dict, Optional

import cv2
import numpy as np
import pandas as pd

from config import settings
from services import database as db
from services.detection import ModelError, load_model
from services.tracking import estimate_teams, run_tracking
from services.video_processor import VideoError, iter_frames, read_metadata, transcode_h264

ProgressCB = Optional[Callable[[float, str], None]]

TEAM_BGR = {0: (0, 170, 255), 1: (255, 120, 40), 2: (180, 180, 180), -1: (255, 255, 255)}
BALL_BGR = (0, 255, 255)


# ------------------------------------------------------------------ drawing
def annotate_frame(frame: np.ndarray, frame_df: pd.DataFrame, teams: Dict[int, int],
                   ball_trail=None, show_conf: bool = True) -> np.ndarray:
    out = frame.copy()
    for r in frame_df.itertuples():
        x1, y1, x2, y2 = int(r.x1), int(r.y1), int(r.x2), int(r.y2)
        if r.cls == "ball":
            c = (int((x1 + x2) / 2), int((y1 + y2) / 2))
            cv2.circle(out, c, max(4, int((x2 - x1) / 2) + 2), BALL_BGR, 2)
            continue
        col = TEAM_BGR.get(teams.get(int(r.track_id), -1), TEAM_BGR[-1])
        cv2.rectangle(out, (x1, y1), (x2, y2), col, 2)
        label = f"#{int(r.track_id)}" + (f" {r.conf:.2f}" if show_conf else "")
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.45, 1)
        cv2.rectangle(out, (x1, y1 - th - 6), (x1 + tw + 4, y1), col, -1)
        cv2.putText(out, label, (x1 + 2, y1 - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 0), 1, cv2.LINE_AA)
        cv2.ellipse(out, (int(r.foot_x), int(r.foot_y)), (int((x2 - x1) / 2), 5), 0, 0, 360, col, 1)
    if ball_trail is not None and len(ball_trail) > 1:
        cv2.polylines(out, [np.array(ball_trail, np.int32).reshape(-1, 1, 2)], False, BALL_BGR, 1, cv2.LINE_AA)
    return out


def render_annotated(video_path, df: pd.DataFrame, teams: Dict[int, int], out_path: Path,
                     params: dict, fps: float, progress_cb: ProgressCB = None) -> None:
    start_f, end_f, stride = params["start_frame"], params["end_frame"], params["stride"]
    total = max(1, math.ceil((end_f - start_f) / stride))
    by_frame = {k: g for k, g in df.groupby("frame_idx")}
    raw = out_path.with_suffix(".raw.mp4")
    writer = None
    trail: list = []
    try:
        for i, (idx, frame) in enumerate(iter_frames(video_path, start_f, end_f, stride)):
            if writer is None:
                h, w = frame.shape[:2]
                writer = cv2.VideoWriter(str(raw), cv2.VideoWriter_fourcc(*"mp4v"), max(1.0, fps / stride), (w, h))
                if not writer.isOpened():
                    raise VideoError("Could not create the output video writer.")
            fdf = by_frame.get(idx)
            if fdf is not None:
                ball = fdf[fdf["cls"] == "ball"]
                if len(ball):
                    trail.append((int(ball.iloc[0].foot_x), int(ball.iloc[0].foot_y)))
                    trail = trail[-25:]
                frame = annotate_frame(frame, fdf, teams, trail)
            writer.write(frame)
            if progress_cb and (i % 10 == 0 or i == total - 1):
                progress_cb(min(1.0, (i + 1) / total), f"Rendering annotated video: frame {i + 1}/{total}")
    finally:
        if writer is not None:
            writer.release()
    if not raw.exists():
        raise VideoError("No frames were rendered.")
    transcode_h264(raw, out_path)
    raw.unlink(missing_ok=True)


# ------------------------------------------------------------------ storage
def tracks_path(job_id: int) -> Path:
    return settings.job_dir(job_id) / "tracks.csv"


def annotated_path(job_id: int) -> Path:
    return settings.job_dir(job_id) / "annotated.mp4"


def load_tracks(job_id: int) -> pd.DataFrame:
    p = tracks_path(job_id)
    return pd.read_csv(p) if p.exists() else pd.DataFrame()


def load_teams(job_id: int) -> Dict[int, int]:
    p = settings.job_dir(job_id) / "teams.json"
    return {int(k): int(v) for k, v in json.loads(p.read_text()).items()} if p.exists() else {}


def save_teams(job_id: int, teams: Dict[int, int]) -> None:
    (settings.job_dir(job_id) / "teams.json").write_text(json.dumps({str(k): v for k, v in teams.items()}))


def rerender(job_id: int, progress_cb: ProgressCB = None) -> None:
    job = db.get_job(job_id)
    render_annotated(job["video_path"], load_tracks(job_id), load_teams(job_id), annotated_path(job_id),
                     job["params"], job["fps"], progress_cb)


# ------------------------------------------------------------------ main job
def run_job(video_path, display_name: str, user_params: dict, progress_cb: ProgressCB = None) -> int:
    meta = read_metadata(video_path)  # raises VideoError for bad files before a job is created
    stride = int(user_params["stride"])
    start_f = int(user_params.get("start_s", 0) * meta.fps)
    dur = user_params.get("duration_s") or 0
    end_f = meta.frame_count if not dur else min(meta.frame_count, start_f + int(dur * meta.fps))
    if start_f >= meta.frame_count:
        raise VideoError("The start time is beyond the end of the video.")
    params = {**user_params, "start_frame": start_f, "end_frame": end_f, "stride": stride}
    job_id = db.create_job(display_name, str(video_path), meta, params)
    try:
        def sub(lo, hi):
            return (lambda f, msg: progress_cb(lo + (hi - lo) * f, msg)) if progress_cb else None

        if progress_cb:
            progress_cb(0.0, "Loading YOLO model (first run downloads weights)...")
        model = load_model(params["model"])
        df = run_tracking(model, video_path, meta, params, sub(0.0, 0.75))
        df.to_csv(tracks_path(job_id), index=False)
        teams, _ = estimate_teams(df)
        save_teams(job_id, teams)
        render_annotated(video_path, df, teams, annotated_path(job_id), params, meta.fps, sub(0.75, 1.0))
        db.update_job(job_id, status="done")
    except MemoryError:
        db.update_job(job_id, status="failed", error="Out of memory. Lower the inference size, increase frame "
                      "stride, or process a shorter time window.")
    except (VideoError, ModelError) as e:
        db.update_job(job_id, status="failed", error=str(e))
    except Exception as e:  # noqa: BLE001 - surface any unexpected failure to the UI, log details to disk
        (settings.job_dir(job_id) / "error.log").write_text(traceback.format_exc())
        db.update_job(job_id, status="failed", error=f"Unexpected error: {type(e).__name__}: {e} "
                      f"(details in outputs/job_{job_id}/error.log)")
    return job_id
