import numpy as np
import pandas as pd

from services.tracking import TRACK_COLUMNS, estimate_teams, extract_rows, torso_color


class T:  # fake torch tensor
    def __init__(self, a): self.a = np.asarray(a)
    def cpu(self): return self
    def numpy(self): return self.a


class Boxes:
    def __init__(self, xyxy, conf, cls, ids):
        self.xyxy, self.conf, self.cls, self.id = T(xyxy), T(conf), T(cls), (None if ids is None else T(ids))
    def __len__(self): return len(self.xyxy.a)


class Result:
    def __init__(self, boxes): self.boxes = boxes


def test_extract_rows_players_and_best_ball():
    frame = np.full((200, 300, 3), 128, np.uint8)
    r = Result(Boxes([[10, 10, 50, 120], [100, 50, 110, 60], [200, 50, 207, 57]],
                     [0.9, 0.3, 0.6], [0, 32, 32], [7, 3, 4]))
    rows = extract_rows(r, frame, 12, 0.48, player_conf=0.3, ball_conf=0.1)
    kinds = [x["cls"] for x in rows]
    assert kinds.count("player") == 1 and kinds.count("ball") == 1
    ball = [x for x in rows if x["cls"] == "ball"][0]
    assert ball["conf"] == 0.6  # best ball kept
    p = rows[0]
    assert p["track_id"] == 7 and p["frame_idx"] == 12 and p["time_s"] == 0.48
    assert p["foot_x"] == 30 and p["foot_y"] == 120


def test_extract_rows_empty_and_unidentified():
    frame = np.zeros((50, 50, 3), np.uint8)
    assert extract_rows(Result(None), frame, 0, 0, 0.3, 0.1) == []
    r = Result(Boxes([[0, 0, 10, 30]], [0.9], [0], None))  # no tracker id -> player skipped
    assert extract_rows(r, frame, 0, 0, 0.3, 0.1) == []


def test_torso_color_ignores_grass():
    frame = np.zeros((100, 60, 3), np.uint8)
    frame[:] = (0, 160, 0)                 # grass (BGR)
    frame[20:50, 15:45] = (0, 0, 255)      # red shirt (BGR)
    r, g, b = torso_color(frame, 5, 10, 55, 90)
    assert r > 200 and g < 60 and b < 60


def _tracks(colors):
    rows = []
    for tid, (r, g, b) in colors.items():
        for f in range(20):
            rows.append(dict(frame_idx=f, time_s=f / 25, track_id=tid, cls="player", conf=.9, x1=0, y1=0,
                             x2=10, y2=30, foot_x=5, foot_y=30, cr=r, cg=g, cb=b))
    return pd.DataFrame(rows, columns=TRACK_COLUMNS)


def test_estimate_teams_two_clusters():
    colors = {1: (220, 20, 20), 2: (210, 30, 25), 3: (225, 25, 30), 4: (20, 20, 220), 5: (25, 30, 210)}
    teams, cols = estimate_teams(_tracks(colors))
    assert teams[1] == teams[2] == teams[3]
    assert teams[4] == teams[5] and teams[1] != teams[4]
    assert set(cols) == {0, 1}


def test_estimate_teams_not_enough_data():
    assert estimate_teams(pd.DataFrame(columns=TRACK_COLUMNS)) == ({}, {})
    teams, _ = estimate_teams(_tracks({1: (200, 0, 0)}))
    assert teams == {1: -1}
