"""Detection + multi-object tracking loop, torso colour extraction and team estimation."""
from __future__ import annotations

import math
from typing import Callable, Dict, List, Optional, Tuple

import cv2
import numpy as np
import pandas as pd

from config import settings
from services.detection import pick_device
from services.video_processor import VideoMeta, iter_frames

TRACK_COLUMNS = ["frame_idx", "time_s", "track_id", "cls", "conf",
                 "x1", "y1", "x2", "y2", "foot_x", "foot_y", "cr", "cg", "cb"]

ProgressCB = Optional[Callable[[float, str], None]]


def _np(x) -> np.ndarray:
    return x.cpu().numpy() if hasattr(x, "cpu") else np.asarray(x)


def torso_color(frame: np.ndarray, x1: float, y1: float, x2: float, y2: float) -> Optional[Tuple[int, int, int]]:
    """Median RGB of the torso region with green (grass) pixels removed."""
    fh, fw = frame.shape[:2]
    h, w = y2 - y1, x2 - x1
    ty1, ty2 = int(max(0, y1 + 0.2 * h)), int(min(fh, y1 + 0.5 * h))
    tx1, tx2 = int(max(0, x1 + 0.25 * w)), int(min(fw, x2 - 0.25 * w))
    crop = frame[ty1:ty2, tx1:tx2]
    if crop.shape[0] < 3 or crop.shape[1] < 3:
        return None
    hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
    grass = (hsv[..., 0] >= 35) & (hsv[..., 0] <= 85) & (hsv[..., 1] >= 60)
    pix = crop[~grass]
    if len(pix) < 0.2 * crop.shape[0] * crop.shape[1]:
        pix = crop.reshape(-1, 3)
    b, g, r = np.median(pix, axis=0)
    return int(r), int(g), int(b)


def extract_rows(result, frame: np.ndarray, frame_idx: int, time_s: float,
                 player_conf: float, ball_conf: float) -> List[dict]:
    """Convert one ultralytics result into rows. Keeps only the best ball per frame."""
    boxes = getattr(result, "boxes", None)
    if boxes is None or len(boxes) == 0:
        return []
    xyxy = _np(boxes.xyxy)
    conf = _np(boxes.conf)
    cls = _np(boxes.cls).astype(int)
    ids = _np(boxes.id).astype(int) if getattr(boxes, "id", None) is not None else None

    rows: List[dict] = []
    best_ball = None
    for i in range(len(xyxy)):
        x1, y1, x2, y2 = (float(v) for v in xyxy[i])
        c = float(conf[i])
        if cls[i] == settings.PERSON_CLASS:
            if c < player_conf or ids is None:
                continue
            col = torso_color(frame, x1, y1, x2, y2)
            rows.append(dict(frame_idx=frame_idx, time_s=time_s, track_id=int(ids[i]), cls="player", conf=c,
                             x1=x1, y1=y1, x2=x2, y2=y2, foot_x=(x1 + x2) / 2, foot_y=y2,
                             cr=col[0] if col else np.nan, cg=col[1] if col else np.nan,
                             cb=col[2] if col else np.nan))
        elif cls[i] == settings.BALL_CLASS and c >= ball_conf:
            if best_ball is None or c > best_ball["conf"]:
                best_ball = dict(frame_idx=frame_idx, time_s=time_s,
                                 track_id=int(ids[i]) if ids is not None else -1, cls="ball", conf=c,
                                 x1=x1, y1=y1, x2=x2, y2=y2, foot_x=(x1 + x2) / 2, foot_y=(y1 + y2) / 2,
                                 cr=np.nan, cg=np.nan, cb=np.nan)
    if best_ball:
        rows.append(best_ball)
    return rows


def run_tracking(model, video_path, meta: VideoMeta, params: dict, progress_cb: ProgressCB = None) -> pd.DataFrame:
    start_f, end_f, stride = params["start_frame"], params["end_frame"], params["stride"]
    total = max(1, math.ceil((end_f - start_f) / stride))
    device = pick_device()
    conf_min = min(params["player_conf"], params["ball_conf"])
    model.predictor = None  # reset tracker state so IDs start fresh for this video

    rows: List[dict] = []
    for i, (idx, frame) in enumerate(iter_frames(video_path, start_f, end_f, stride)):
        results = model.track(frame, persist=True, tracker=f"{params['tracker']}.yaml", conf=conf_min,
                              imgsz=params["imgsz"], classes=[settings.PERSON_CLASS, settings.BALL_CLASS],
                              device=device, verbose=False)
        rows += extract_rows(results[0], frame, idx, idx / meta.fps, params["player_conf"], params["ball_conf"])
        if progress_cb and (i % 5 == 0 or i == total - 1):
            progress_cb(min(1.0, (i + 1) / total), f"Detecting & tracking: frame {i + 1}/{total}")
    return pd.DataFrame(rows, columns=TRACK_COLUMNS)


def estimate_teams(df: pd.DataFrame, min_frames: int = 15) -> Tuple[Dict[int, int], Dict[int, Tuple[int, int, int]]]:
    """Cluster per-track median torso colour into two teams (K-means in Lab space).

    Tracks seen for fewer than `min_frames` -> -1 (unassigned). Tracks whose colour is far from both
    team centres (often referee / goalkeeper) -> 2. This is a HEURISTIC; correct it manually in the UI.
    """
    from sklearn.cluster import KMeans

    players = df[(df["cls"] == "player")].dropna(subset=["cr", "cg", "cb"])
    if players.empty:
        return {}, {}
    g = players.groupby("track_id").agg(frames=("frame_idx", "count"), cr=("cr", "median"),
                                        cg=("cg", "median"), cb=("cb", "median"))
    teams: Dict[int, int] = {int(t): -1 for t in df.loc[df["cls"] == "player", "track_id"].unique()}
    valid = g[g["frames"] >= min_frames]
    if len(valid) < 2:
        return teams, {}
    rgb = valid[["cr", "cg", "cb"]].to_numpy(dtype=np.uint8).reshape(1, -1, 3)
    lab = cv2.cvtColor(rgb, cv2.COLOR_RGB2LAB).reshape(-1, 3).astype(float)
    km = KMeans(n_clusters=2, n_init=10, random_state=0).fit(lab)
    labels = km.labels_.copy()
    if (labels == 1).sum() > (labels == 0).sum():  # team 0 = the larger cluster
        labels = 1 - labels
        centers = km.cluster_centers_[::-1]
    else:
        centers = km.cluster_centers_
    dists = np.linalg.norm(lab - centers[labels], axis=1)
    cutoff = max(25.0, 2.5 * float(np.median(dists)))
    colors: Dict[int, Tuple[int, int, int]] = {}
    for team in (0, 1):
        sel = valid[labels == team]
        if len(sel):
            colors[team] = tuple(int(v) for v in sel[["cr", "cg", "cb"]].median())
    for tid, lab_i, d in zip(valid.index, labels, dists):
        teams[int(tid)] = 2 if (d > cutoff and len(valid) >= 6) else int(lab_i)
    return teams, colors
