import numpy as np
import pandas as pd
import pytest

from services import pitch_calibration as pc
from services import tactical_analysis as ta


def test_homography_roundtrip():
    img = [(100, 100), (500, 100), (600, 400), (50, 400)]
    pitch = [(0, 0), (40, 0), (40, 30), (0, 30)]
    H = pc.compute_homography(img, pitch)
    out = pc.apply_homography(H, np.array(img))
    assert np.allclose(out, pitch, atol=1e-3)
    mid = pc.apply_homography(H, np.array([[325, 250]]))
    assert 0 < mid[0, 0] < 40 and 0 < mid[0, 1] < 30


def test_homography_rejects_bad_input():
    with pytest.raises(pc.CalibrationError):
        pc.compute_homography([(0, 0), (1, 1), (2, 2), (3, 3)], [(0, 0), (1, 0), (1, 1), (0, 1)])
    with pytest.raises(pc.CalibrationError):
        pc.compute_homography([(0, 0), (1, 0), (1, 1)], [(0, 0), (1, 0), (1, 1)])
    with pytest.raises(pc.CalibrationError):
        pc.compute_homography([(0, 0), (0, 0), (1, 1), (0, 1)], [(0, 0), (1, 0), (1, 1), (0, 1)])


def _runner(speed=5.0, n=50, fps=25.0, jump_at=None):
    t = np.arange(n) / fps
    x = 10 + speed * t
    if jump_at:
        x[jump_at:] += 60  # tracking glitch: should be ignored
    return pd.DataFrame(dict(track_id=1, team=0, time_s=t, px=x, py=30.0))


def test_distance_stats_constant_speed():
    s = ta.distance_stats(_runner())
    assert s.iloc[0]["distance_m"] == pytest.approx(5.0 * 49 / 25, rel=0.05)
    assert s.iloc[0]["top_speed_ms"] == pytest.approx(5.0, rel=0.1)


def test_distance_stats_ignores_glitch_jump():
    s = ta.distance_stats(_runner(jump_at=25))
    assert s.iloc[0]["distance_m"] < 20  # a 60 m teleport would add 60 if not filtered


def test_distance_stats_short_tracks_skipped():
    assert ta.distance_stats(_runner(n=5)).empty


def test_possession_estimate():
    rows = []
    for f in range(10):
        rows.append(dict(frame_idx=f, track_id=1, cls="player", x1=0, y1=0, x2=20, y2=100, foot_x=10, foot_y=100))
        rows.append(dict(frame_idx=f, track_id=2, cls="player", x1=300, y1=0, x2=320, y2=100, foot_x=310, foot_y=100))
        rows.append(dict(frame_idx=f, track_id=-1, cls="ball", x1=14, y1=90, x2=20, y2=98, foot_x=17, foot_y=94))
    p = ta.estimate_possession(pd.DataFrame(rows), {1: 0, 2: 1})
    assert p["team_a_pct"] == 100 and p["team_b_pct"] == 0
    assert p["frames_with_ball"] == 10


def test_team_shape_summary():
    avg = pd.DataFrame(dict(track_id=range(4), px=[10, 20, 30, 40], py=[5, 15, 25, 35], team=0, samples=10))
    s = ta.team_shape_summary(avg)
    assert s.iloc[0]["width_m"] == 30 and s.iloc[0]["depth_m"] == 30


def test_heatmap_on_frame_shape():
    f = np.zeros((100, 200, 3), np.uint8)
    out = ta.heatmap_on_frame(f, [50, 60, 60], [40, 40, 45])
    assert out.shape == f.shape
