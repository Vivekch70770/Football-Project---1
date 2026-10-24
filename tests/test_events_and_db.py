import numpy as np
import pandas as pd

from config import settings
from services import database as db
from services import event_detection as ed
from services.video_processor import VideoMeta


def _ball_df(fast_from=50, fast_to=60):
    rows, x = [], 100.0
    for f in range(0, 150):
        x += 60 if fast_from <= f < fast_to else 1   # 60 px/frame * 25 fps = 1500 px/s ~ 1.5 widths/s
        rows.append(dict(frame_idx=f, time_s=f / 25, track_id=-1, cls="ball", conf=.5, x1=x, y1=100, x2=x + 6, y2=106,
                         foot_x=x + 3, foot_y=103))
    return pd.DataFrame(rows)


def test_ball_speed_spike_detected():
    ev = ed.detect_ball_speed_spikes(_ball_df(), fps=25, stride=1, frame_width=1000, duration_s=10)
    assert len(ev) == 1 and ev[0]["type"] == "ball_speed_spike"
    assert 2.0 <= ev[0]["time_s"] <= 2.5
    assert ev[0]["confidence"] is None and ev[0]["status"] == "pending"
    assert ev[0]["start_s"] < ev[0]["time_s"] < ev[0]["end_s"]


def test_no_events_without_fast_motion_or_ball():
    assert ed.detect_ball_speed_spikes(_ball_df(0, 0), 25, 1, 1000, 10) == []
    assert ed.detect_ball_speed_spikes(pd.DataFrame(columns=_ball_df().columns), 25, 1, 1000, 10) == []


def test_glitchy_ball_jump_is_ignored():
    df = _ball_df(0, 0)
    df.loc[df.frame_idx >= 70, ["foot_x", "x1", "x2"]] += 5000  # single teleport
    assert ed.detect_ball_speed_spikes(df, 25, 1, 1000, 10) == []


def test_duel_detection_requires_ball_and_proximity():
    rows = []
    for f in range(10):
        rows.append(dict(frame_idx=f, time_s=f / 25, track_id=1, cls="player", x1=100, y1=0, x2=120, y2=100, foot_x=110, foot_y=100))
        rows.append(dict(frame_idx=f, time_s=f / 25, track_id=2, cls="player", x1=115, y1=0, x2=135, y2=100, foot_x=125, foot_y=100))
        rows.append(dict(frame_idx=f, time_s=f / 25, track_id=-1, cls="ball", x1=114, y1=95, x2=120, y2=101, foot_x=117, foot_y=98))
    df = pd.DataFrame(rows)
    ev = ed.detect_player_duels(df, {1: 0, 2: 1}, duration_s=10)
    assert len(ev) == 1 and ev[0]["track_ids"] == "1,2"
    assert ed.detect_player_duels(df[df.cls != "ball"], {1: 0, 2: 1}, 10) == []
    assert ed.detect_player_duels(df, {1: 0, 2: 0}, 10) == []  # same team


def test_clip_window_clamped():
    s, e = ed.clip_window("goal", 3.0, 20.0)
    assert s == 0.0 and e == 11.0


def test_database_roundtrip(tmp_path, monkeypatch):
    monkeypatch.setattr(settings, "DB_PATH", tmp_path / "t.db")
    monkeypatch.setattr(settings, "DATA_DIR", tmp_path)
    db.init_db()
    meta = VideoMeta("x.mp4", 25.0, 100, 320, 180)
    jid = db.create_job("demo", "x.mp4", meta, {"stride": 2})
    assert db.get_job(jid)["params"] == {"stride": 2} and db.get_job(jid)["status"] == "running"
    db.update_job(jid, status="done")
    assert db.list_jobs()[0]["status"] == "done"
    db.add_events(jid, [dict(type="goal", time_s=5.0, start_s=0, end_s=9, source="manual", status="confirmed"),
                        dict(type="shot", time_s=2.0, start_s=0, end_s=6, source="auto")])
    assert [e["time_s"] for e in db.get_events(jid)] == [2.0, 5.0]
    db.clear_auto_pending_events(jid)
    assert [e["type"] for e in db.get_events(jid)] == ["goal"]
    assert db.get_job(9999) is None
