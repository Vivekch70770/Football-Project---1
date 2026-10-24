"""Modular CANDIDATE event detection (heuristic).

Plain YOLO boxes cannot classify goals, saves, fouls, etc. These detectors only propose moments
worth a human look. `confidence` is None because no calibrated probability exists. Add trained
action-spotting models as extra detector functions returning the same dict format (Phase 5).
"""
from __future__ import annotations

from typing import Callable, Dict, List

import numpy as np
import pandas as pd

PRE_POST = {"goal": (10, 8), "shot": (6, 4), "save": (6, 4), "default": (5, 4)}


def clip_window(event_type: str, time_s: float, duration_s: float) -> tuple:
    pre, post = PRE_POST.get(event_type, PRE_POST["default"])
    return max(0.0, time_s - pre), min(duration_s, time_s + post)


def _event(kind: str, t: float, duration_s: float, track_ids="", note="") -> dict:
    s, e = clip_window(kind, t, duration_s)
    return dict(type=kind, time_s=float(t), confidence=None, start_s=s, end_s=e,
                track_ids=str(track_ids), source="auto", status="pending", note=note)


def ball_series(df: pd.DataFrame, stride: int, fps: float, max_gap_s: float = 0.5) -> pd.DataFrame:
    b = df[df["cls"] == "ball"].drop_duplicates("frame_idx").sort_values("frame_idx")
    if len(b) < 3:
        return pd.DataFrame(columns=["frame_idx", "time_s", "cx", "cy"])
    frames = np.arange(int(b["frame_idx"].min()), int(b["frame_idx"].max()) + 1, stride)
    s = b.set_index("frame_idx")[["foot_x", "foot_y"]].reindex(frames)
    limit = max(1, int(max_gap_s * fps / stride))
    s = s.interpolate(limit=limit, limit_area="inside")
    s = s.rolling(3, center=True, min_periods=1).median()
    return pd.DataFrame({"frame_idx": frames, "time_s": frames / fps, "cx": s["foot_x"].to_numpy(),
                         "cy": s["foot_y"].to_numpy()})


def detect_ball_speed_spikes(df: pd.DataFrame, fps: float, stride: int, frame_width: int, duration_s: float,
                             speed_thresh: float = 0.5, max_speed: float = 3.0, min_run: int = 2,
                             min_gap_s: float = 4.0) -> List[dict]:
    """Fast ball movement (frame widths / second) sustained for `min_run` samples."""
    s = ball_series(df, stride, fps)
    if len(s) < 4:
        return []
    dt = np.diff(s["time_s"].to_numpy())
    v = np.hypot(np.diff(s["cx"].to_numpy()), np.diff(s["cy"].to_numpy())) / dt / frame_width
    mask = np.nan_to_num(v, nan=0) > speed_thresh
    mask &= np.nan_to_num(v, nan=np.inf) < max_speed  # physically implausible jumps = detector glitch
    events, i, last_t = [], 0, -1e9
    while i < len(mask):
        if mask[i]:
            j = i
            while j < len(mask) and mask[j]:
                j += 1
            if j - i >= min_run:
                k = i + int(np.nanargmax(v[i:j]))
                t = float(s["time_s"].iloc[k])
                if t - last_t >= min_gap_s:
                    events.append(_event("ball_speed_spike", t, duration_s,
                                         note=f"Fast ball movement ({np.nanmax(v[i:j]):.2f} frame-widths/s). "
                                              "Possible shot/long pass - review."))
                    last_t = t
            i = j
        else:
            i += 1
    return events


def detect_player_duels(df: pd.DataFrame, teams: Dict[int, int], duration_s: float, dist_factor: float = 0.5,
                        ball_factor: float = 1.5, min_hits: int = 3, max_gap_s: float = 0.5,
                        min_gap_s: float = 4.0) -> List[dict]:
    """Opposing-team players very close together next to the ball for several samples."""
    p = df[df["cls"] == "player"].copy()
    p["team"] = p["track_id"].map(teams).fillna(-1).astype(int)
    p = p[p["team"].isin([0, 1])]
    ball = df[df["cls"] == "ball"].drop_duplicates("frame_idx").set_index("frame_idx")
    hits = []
    for fidx, g in p.groupby("frame_idx"):
        if fidx not in ball.index:
            continue
        a, b = g[g["team"] == 0], g[g["team"] == 1]
        if a.empty or b.empty:
            continue
        pa, pb = a[["foot_x", "foot_y"]].to_numpy(), b[["foot_x", "foot_y"]].to_numpy()
        ha, hb = (a["y2"] - a["y1"]).to_numpy(), (b["y2"] - b["y1"]).to_numpy()
        d = np.linalg.norm(pa[:, None, :] - pb[None, :, :], axis=2)
        h = np.minimum(ha[:, None], hb[None, :])
        ratio = d / np.maximum(h, 1)
        ia, ib = np.unravel_index(np.argmin(ratio), ratio.shape)
        if ratio[ia, ib] > dist_factor:
            continue
        mid = (pa[ia] + pb[ib]) / 2
        bx = ball.loc[fidx]
        if np.hypot(*(mid - np.array([bx["foot_x"], bx["foot_y"]]))) > ball_factor * h[ia, ib]:
            continue
        hits.append((float(g["time_s"].iloc[0]), int(a.iloc[ia]["track_id"]), int(b.iloc[ib]["track_id"])))
    events, run, last_t = [], [], -1e9

    def flush():
        nonlocal last_t
        if len(run) >= min_hits:
            t = float(np.median([r[0] for r in run]))
            if t - last_t >= min_gap_s:
                ids = sorted({x for r in run for x in r[1:]})
                events.append(_event("possible_duel", t, duration_s, ",".join(map(str, ids)),
                                     "Opposing players close together near the ball. Possible tackle/foul - review."))
                last_t = t

    for h_ in hits:
        if run and h_[0] - run[-1][0] > max_gap_s:
            flush()
            run = []
        run.append(h_)
    flush()
    return events


DETECTORS: Dict[str, Callable] = {"ball_speed_spike": detect_ball_speed_spikes, "possible_duel": detect_player_duels}


def detect_candidates(df: pd.DataFrame, teams: Dict[int, int], fps: float, stride: int, frame_width: int,
                      duration_s: float, speed_thresh: float = 0.5) -> List[dict]:
    ev = detect_ball_speed_spikes(df, fps, stride, frame_width, duration_s, speed_thresh=speed_thresh)
    ev += detect_player_duels(df, teams, duration_s)
    return sorted(ev, key=lambda e: e["time_s"])
