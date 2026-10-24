"""Tactical statistics and plots. Everything here is ESTIMATED from detections.

Real-world metres are only produced when a pitch calibration (homography) exists.
"""
from __future__ import annotations

from typing import Dict, Iterable, Optional

import cv2
import numpy as np
import pandas as pd
from matplotlib.patches import Arc, Circle, Rectangle
from scipy.ndimage import gaussian_filter

from config import settings
from services.pitch_calibration import apply_homography

L, W = settings.PITCH_LENGTH_M, settings.PITCH_WIDTH_M
MAX_HUMAN_SPEED_MS = 12.0  # steps faster than this are treated as tracking glitches


def players_with_team(df: pd.DataFrame, teams: Dict[int, int]) -> pd.DataFrame:
    p = df[df["cls"] == "player"].copy()
    p["team"] = p["track_id"].map(teams).fillna(-1).astype(int)
    return p


def add_pitch_coords(players: pd.DataFrame, H: np.ndarray) -> pd.DataFrame:
    """Project the ground-contact point (bottom-centre of the box) onto the pitch plane."""
    out = players.copy()
    if out.empty:
        out["px"], out["py"] = [], []
        return out
    xy = apply_homography(H, out[["foot_x", "foot_y"]].to_numpy())
    out["px"], out["py"] = xy[:, 0], xy[:, 1]
    return out[(out["px"].between(-5, L + 5)) & (out["py"].between(-5, W + 5))]


# -------------------------------------------------------------- statistics
def distance_stats(pp: pd.DataFrame, min_points: int = 10) -> pd.DataFrame:
    """Per-track distance (m) and top speed (m/s). Requires pitch coords. ESTIMATE."""
    rows = []
    for tid, g in pp.sort_values("time_s").groupby("track_id"):
        if len(g) < min_points:
            continue
        x = g["px"].rolling(5, center=True, min_periods=1).median().to_numpy()
        y = g["py"].rolling(5, center=True, min_periods=1).median().to_numpy()
        t = g["time_s"].to_numpy()
        dt = np.diff(t)
        step = np.hypot(np.diff(x), np.diff(y))
        ok = (dt > 0) & (dt <= 1.0)
        speed = np.where(ok, step / np.where(dt > 0, dt, 1), np.nan)
        valid = ok & (speed <= MAX_HUMAN_SPEED_MS)
        rows.append(dict(track_id=int(tid), team=int(g["team"].iloc[0]), samples=len(g),
                         distance_m=float(step[valid].sum()),
                         top_speed_ms=float(np.nanmax(speed[valid])) if valid.any() else np.nan,
                         tracked_s=float(t[-1] - t[0])))
    return pd.DataFrame(rows, columns=["track_id", "team", "samples", "distance_m", "top_speed_ms", "tracked_s"])


def team_shape(pp: pd.DataFrame) -> pd.DataFrame:
    """Average position per track + per-team centroid / width / depth (pitch metres)."""
    avg = pp.groupby("track_id").agg(px=("px", "mean"), py=("py", "mean"), team=("team", "first"),
                                     samples=("px", "count")).reset_index()
    return avg


def team_shape_summary(avg: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for team, g in avg[avg["team"].isin([0, 1])].groupby("team"):
        if len(g) < 3:
            continue
        rows.append(dict(team=settings.TEAM_LABELS[int(team)], players=len(g), centroid_x_m=g.px.mean(),
                         centroid_y_m=g.py.mean(), width_m=g.py.max() - g.py.min(), depth_m=g.px.max() - g.px.min()))
    return pd.DataFrame(rows)


def estimate_possession(df: pd.DataFrame, teams: Dict[int, int], radius_factor: float = 0.6) -> Dict[str, float]:
    """Share of ball-visible frames where the nearest team player is within `radius_factor`
    bounding-box heights of the ball (image space). Rough ESTIMATE - ignores aerial balls."""
    p = players_with_team(df, teams)
    p = p[p["team"].isin([0, 1])]
    ball = df[df["cls"] == "ball"].drop_duplicates("frame_idx").set_index("frame_idx")
    counts = {0: 0, 1: 0}
    frames_with_ball = 0
    for fidx, g in p.groupby("frame_idx"):
        if fidx not in ball.index:
            continue
        frames_with_ball += 1
        b = ball.loc[fidx]
        d = np.hypot(g["foot_x"] - b["foot_x"], g["foot_y"] - b["foot_y"]) / (g["y2"] - g["y1"]).clip(lower=1)
        i = d.idxmin()
        if d[i] <= radius_factor:
            counts[int(g.loc[i, "team"])] += 1
    tot = counts[0] + counts[1]
    return {"frames_with_ball": frames_with_ball, "frames_with_possession": tot,
            "team_a_pct": 100 * counts[0] / tot if tot else float("nan"),
            "team_b_pct": 100 * counts[1] / tot if tot else float("nan")}


def ball_stats(df: pd.DataFrame, total_frames_processed: int) -> Dict[str, float]:
    b = df[df["cls"] == "ball"].drop_duplicates("frame_idx")
    return {"ball_detection_rate_pct": 100 * len(b) / total_frames_processed if total_frames_processed else 0.0}


# -------------------------------------------------------------- pitch plots
def draw_pitch(ax, line="#e8f5ec", bg="#14532d") -> None:
    ax.add_patch(Rectangle((-3, -3), L + 6, W + 6, fc=bg, ec="none", zorder=0))
    kw = dict(fill=False, ec=line, lw=1.4)
    ax.add_patch(Rectangle((0, 0), L, W, **kw))
    ax.plot([L / 2, L / 2], [0, W], color=line, lw=1.4)
    ax.add_patch(Circle((L / 2, W / 2), 9.15, **kw))
    for gx, sgn in ((0, 1), (L, -1)):
        ax.add_patch(Rectangle((gx if sgn > 0 else L - 16.5, (W - 40.32) / 2), 16.5, 40.32, **kw))
        ax.add_patch(Rectangle((gx if sgn > 0 else L - 5.5, (W - 18.32) / 2), 5.5, 18.32, **kw))
        ax.add_patch(Circle((gx + sgn * 11, W / 2), 0.3, color=line))
    ax.set_xlim(-3, L + 3)
    ax.set_ylim(W + 3, -3)  # image-like orientation: y grows downward
    ax.set_aspect("equal")
    ax.axis("off")


def plot_heatmap(ax, xs, ys, sigma: float = 2.0) -> None:
    draw_pitch(ax)
    if len(xs) < 3:
        return
    h, _, _ = np.histogram2d(xs, ys, bins=[int(L), int(W)], range=[[0, L], [0, W]])
    h = gaussian_filter(h, sigma)
    h = np.ma.masked_less(h, h.max() * 0.04)
    ax.imshow(h.T, extent=[0, L, W, 0], cmap="hot", alpha=0.7, interpolation="bilinear", zorder=3, aspect="equal")


def plot_trails(ax, pp: pd.DataFrame, track_ids: Iterable[int], colors: Optional[Dict[int, str]] = None) -> None:
    draw_pitch(ax)
    for tid in track_ids:
        g = pp[pp["track_id"] == tid].sort_values("time_s")
        if len(g) < 2:
            continue
        c = (colors or {}).get(int(g["team"].iloc[0]), "#ffffff")
        ax.plot(g["px"], g["py"], "-", color=c, lw=1.2, alpha=0.8)
        ax.scatter(g["px"].iloc[-1], g["py"].iloc[-1], s=30, color=c, zorder=4)
        ax.text(g["px"].iloc[-1] + 0.8, g["py"].iloc[-1], f"#{tid}", color="white", fontsize=7)


def plot_average_positions(ax, avg: pd.DataFrame, colors: Dict[int, str]) -> None:
    draw_pitch(ax)
    for team, g in avg.groupby("team"):
        c = colors.get(int(team), "#cccccc")
        ax.scatter(g["px"], g["py"], s=140, color=c, ec="white", zorder=4)
        for r in g.itertuples():
            ax.text(r.px, r.py, str(int(r.track_id)), color="black", fontsize=6, ha="center", va="center", zorder=5)


# -------------------------------------------------------------- image-space views (no calibration)
def heatmap_on_frame(frame_bgr: np.ndarray, xs, ys, sigma_frac: float = 0.02) -> np.ndarray:
    """Overlay a movement density map in IMAGE space. Not a pitch/metric heatmap."""
    h, w = frame_bgr.shape[:2]
    acc = np.zeros((h, w), np.float32)
    xi = np.clip(np.asarray(xs, dtype=int), 0, w - 1)
    yi = np.clip(np.asarray(ys, dtype=int), 0, h - 1)
    np.add.at(acc, (yi, xi), 1)
    acc = cv2.GaussianBlur(acc, (0, 0), max(3, sigma_frac * w))
    if acc.max() <= 0:
        return frame_bgr
    acc = (255 * acc / acc.max()).astype(np.uint8)
    col = cv2.applyColorMap(acc, cv2.COLORMAP_JET)
    mask = (acc > 12)[..., None]
    return np.where(mask, cv2.addWeighted(frame_bgr, 0.45, col, 0.55, 0), frame_bgr)


def trails_on_frame(frame_bgr: np.ndarray, pdf: pd.DataFrame, track_ids: Iterable[int]) -> np.ndarray:
    out = frame_bgr.copy()
    rng = np.random.default_rng(1)
    for tid in track_ids:
        g = pdf[pdf["track_id"] == tid].sort_values("time_s")
        if len(g) < 2:
            continue
        pts = g[["foot_x", "foot_y"]].to_numpy().astype(np.int32).reshape(-1, 1, 2)
        color = tuple(int(v) for v in rng.integers(80, 255, 3))
        cv2.polylines(out, [pts], False, color, 2, cv2.LINE_AA)
        cv2.putText(out, f"#{tid}", tuple(pts[-1, 0]), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1, cv2.LINE_AA)
    return out
