"""VARVision AI - Streamlit dashboard.  Run with:  streamlit run app.py"""
from __future__ import annotations

import math
import time
import uuid
from pathlib import Path

import cv2
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st
from PIL import Image

from components import ui
from config import settings
from services import database as db
from services import event_detection as ed
from services import highlight_generator as hg
from services import pipeline
from services import pitch_calibration as pc
from services import tactical_analysis as ta
from services.video_processor import VideoError, read_frame, read_metadata

st.set_page_config(page_title="VARVision AI", page_icon="⚽", layout="wide")
settings.ensure_dirs()
db.init_db()
ui.inject_css()

LABEL_TO_CODE = {v: k for k, v in settings.TEAM_LABELS.items()}
TEAM_HEX = {0: "#f59e0b", 1: "#3b82f6", 2: "#9ca3af", -1: "#e5e7eb"}


# =========================================================================== helpers
def _team_label(code) -> str:
    return settings.TEAM_LABELS.get(int(code), "Unassigned")


def _job_meta(job):
    p = job["params"]
    fps, stride = job["fps"], p["stride"]
    frames = list(range(p["start_frame"], p["end_frame"], stride))
    return fps, stride, frames


def _progress_ui():
    bar, txt = st.progress(0.0), st.empty()

    def cb(frac: float, msg: str):
        bar.progress(min(1.0, max(0.0, frac)))
        txt.caption(msg)
    return cb, bar, txt


# =========================================================================== page: new analysis
def page_new_analysis():
    st.header("New analysis")
    st.caption("Upload a short clip (30-120 s is ideal on a CPU laptop). Only analyse footage you have permission to use.")
    src = st.radio("Video source", ["Upload a file", "Pick a file from data/samples/"], horizontal=True)
    video_path = None
    if src.startswith("Upload"):
        up = st.file_uploader("Match video", type=[e.strip(".") for e in sorted(settings.ALLOWED_EXTENSIONS)])
        if up is None:
            ui.empty_state("No video yet", "Drop an MP4/MOV/MKV/AVI file above, or copy a clip into data/samples/.")
        elif up.size > settings.MAX_UPLOAD_MB * 1024 * 1024:
            st.error(f"File is larger than {settings.MAX_UPLOAD_MB} MB. Trim it first (see README: scripts/trim_clip.py).")
        else:
            sig = f"{up.name}-{up.size}"
            if st.session_state.get("upload_sig") != sig:
                dest = settings.UPLOAD_DIR / f"{uuid.uuid4().hex}{Path(up.name).suffix.lower()}"
                dest.write_bytes(up.getbuffer())
                st.session_state.update(upload_sig=sig, upload_path=str(dest), upload_name=up.name)
            video_path = st.session_state["upload_path"]
    else:
        files = sorted(p for p in settings.SAMPLES_DIR.glob("*") if p.suffix.lower() in settings.ALLOWED_EXTENSIONS)
        if not files:
            ui.empty_state("data/samples/ is empty", "Copy a short football clip into the data/samples folder, then refresh.")
        else:
            choice = st.selectbox("File", [f.name for f in files])
            video_path = str(settings.SAMPLES_DIR / choice)
            st.session_state["upload_name"] = choice
    if not video_path:
        return

    try:
        meta = read_metadata(video_path)
    except VideoError as e:
        st.error(f"Cannot use this video: {e}")
        return
    ui.stat_cards([("Resolution", f"{meta.width}x{meta.height}"), ("FPS", f"{meta.fps:.1f}"),
                   ("Frames", f"{meta.frame_count:,}"), ("Duration", f"{meta.duration_s:.1f} s")])
    if Path(video_path).suffix.lower() in {".mp4", ".mov", ".m4v"}:
        with st.expander("Preview original video"):
            st.video(video_path)

    st.subheader("Settings")
    c1, c2, c3 = st.columns(3)
    model = c1.selectbox("YOLO model", settings.MODEL_CHOICES, help="n = fastest (CPU friendly), m = most accurate")
    tracker = c1.selectbox("Tracker", settings.TRACKER_CHOICES, help="ByteTrack is faster; BoT-SORT handles camera motion better")
    imgsz = c2.select_slider("Inference size", [480, 640, 800, 960, 1280], value=960,
                             help="Larger finds the small ball more often but is slower")
    stride = c2.select_slider("Process every Nth frame", [1, 2, 3, 4, 5, 6], value=2,
                              help="2-3 is a good CPU trade-off; tracking still works")
    player_conf = c3.slider("Player confidence", 0.1, 0.9, 0.35, 0.05)
    ball_conf = c3.slider("Ball confidence", 0.05, 0.8, 0.15, 0.05, help="The ball is tiny; low thresholds find more but add false positives")
    d1, d2 = st.columns(2)
    start_s = d1.number_input("Start at (seconds)", 0.0, float(meta.duration_s), 0.0, 1.0)
    dur = d2.number_input("Duration to analyse (seconds, 0 = until end)", 0.0, float(meta.duration_s), min(60.0, float(meta.duration_s)), 5.0)

    est_frames = int((dur or meta.duration_s - start_s) * meta.fps / stride)
    st.caption(f"≈ {est_frames:,} frames to process. On an i5 CPU with yolov8n expect roughly 3-8 frames/s at 640-960 px.")

    if st.button("▶ Run analysis", type="primary"):
        cb, bar, txt = _progress_ui()
        params = dict(model=model, tracker=tracker, imgsz=int(imgsz), stride=int(stride), player_conf=player_conf,
                      ball_conf=ball_conf, start_s=float(start_s), duration_s=float(dur))
        t0 = time.time()
        try:
            job_id = pipeline.run_job(video_path, st.session_state.get("upload_name", Path(video_path).name), params, cb)
        except VideoError as e:
            st.error(str(e))
            return
        job = db.get_job(job_id)
        if job["status"] == "done":
            df = pipeline.load_tracks(job_id)
            st.success(f"Analysis finished in {time.time() - t0:.0f} s.")
            if df.empty:
                st.warning("No players or ball were detected. Try a larger inference size, lower confidence, or a clearer clip.")
            st.session_state.update(job_id=job_id, _goto="Match dashboard")
            st.rerun()
        else:
            bar.empty(); txt.empty()
            st.error(f"Analysis failed: {job['error']}")


# =========================================================================== dashboard
def page_dashboard():
    jobs = db.list_jobs()
    if not jobs:
        ui.empty_state("No analyses yet", "Go to 'New analysis' in the sidebar and process a video first.")
        return
    ids = [j["id"] for j in jobs]
    default = ids.index(st.session_state["job_id"]) if st.session_state.get("job_id") in ids else 0
    names = {j["id"]: f"#{j['id']} · {j['name'][:28]} · {j['status']}" for j in jobs}
    job_id = st.sidebar.selectbox("Analysis job", ids, index=default, format_func=names.get)
    st.session_state["job_id"] = job_id
    job = db.get_job(job_id)
    if job["status"] == "failed":
        st.error(f"This analysis failed: {job['error']}")
        return
    if job["status"] != "done":
        st.warning("This job never finished (the app was closed or crashed). Start a new analysis.")
        return

    df, teams = pipeline.load_tracks(job_id), pipeline.load_teams(job_id)
    fps, stride, frames = _job_meta(job)
    players = df[df["cls"] == "player"] if not df.empty else df
    n_tracks = int((players.groupby("track_id").size() >= 15).sum()) if not df.empty else 0
    ball_rate = ta.ball_stats(df, len(frames))["ball_detection_rate_pct"] if not df.empty else 0.0
    st.header(job["name"])
    ui.stat_cards([("Frames processed", f"{len(frames):,}"), ("Analysed span", f"{len(frames) * stride / fps:.0f} s"),
                   ("Player tracks (≥15 frames)", n_tracks, "Track IDs can fragment when players are occluded"),
                   ("Ball detected in", f"{ball_rate:.0f}% of frames", "COCO model: the ball is often missed"),
                   ("Model", job["params"]["model"])])
    if df.empty:
        st.warning("No detections were saved for this job.")
        return

    tabs = st.tabs(["Player Tracking", "VAR Review", "Tactical Analysis", "Match Events", "Highlights"])
    with tabs[0]:
        tab_tracking(job, df, teams)
    with tabs[1]:
        tab_var(job, df, teams)
    with tabs[2]:
        tab_tactical(job, df, teams)
    with tabs[3]:
        tab_events(job, df, teams)
    with tabs[4]:
        tab_highlights(job)


# --------------------------------------------------------------------------- tab: tracking
def tab_tracking(job, df, teams):
    jid = job["id"]
    ui.data_legend()
    ann = pipeline.annotated_path(jid)
    if ann.exists():
        st.video(str(ann))
        c1, c2, _ = st.columns([1, 1, 3])
        c1.download_button("⬇ Annotated video", ann.read_bytes(), file_name=f"varvision_job{jid}_annotated.mp4", mime="video/mp4")
        c2.download_button("⬇ Tracking data (CSV)", pipeline.tracks_path(jid).read_bytes(),
                           file_name=f"varvision_job{jid}_tracks.csv", mime="text/csv")
    else:
        ui.empty_state("Annotated video missing", "Re-run the analysis.")
    st.caption("Box colours: amber = Team A, blue = Team B, grey = referee/GK/other, white = unassigned. "
               "Team colours come from a torso-colour clustering heuristic; the model itself cannot tell referees "
               "or goalkeepers apart (COCO only has 'person').")

    st.subheader("Team assignment (manual correction)")
    pl = df[df["cls"] == "player"]
    summ = pl.groupby("track_id").agg(frames=("frame_idx", "count"), mean_conf=("conf", "mean"),
                                      first_s=("time_s", "min"), last_s=("time_s", "max")).reset_index()
    summ = summ[summ["frames"] >= 5].sort_values("frames", ascending=False)
    summ["team"] = summ["track_id"].map(lambda t: _team_label(teams.get(int(t), -1)))
    edited = st.data_editor(summ, hide_index=True, width="stretch", key=f"teams_{jid}",
                            disabled=["track_id", "frames", "mean_conf", "first_s", "last_s"],
                            column_config={"team": st.column_config.SelectboxColumn("team", options=list(LABEL_TO_CODE),
                                                                                    required=True),
                                           "mean_conf": st.column_config.NumberColumn(format="%.2f"),
                                           "first_s": st.column_config.NumberColumn(format="%.1f"),
                                           "last_s": st.column_config.NumberColumn(format="%.1f")})
    if st.button("Apply team changes & re-render video"):
        new = dict(teams)
        for r in edited.itertuples():
            new[int(r.track_id)] = LABEL_TO_CODE[r.team]
        pipeline.save_teams(jid, new)
        cb, bar, txt = _progress_ui()
        try:
            pipeline.rerender(jid, cb)
        except VideoError as e:
            st.error(str(e))
            return
        st.rerun()


# --------------------------------------------------------------------------- tab: VAR review
def tab_var(job, df, teams):
    jid = job["id"]
    fps, stride, frames = _job_meta(job)
    st.warning("**Experimental review aid - not official VAR.** Nothing here decides goals, fouls or offside. "
               "Offside-line overlay is **not implemented yet** (needs pitch calibration per frame - Phase 4).")
    ui.data_legend()
    events = db.get_events(jid)
    key = f"var_i_{jid}"
    st.session_state.setdefault(key, 0)
    ev_labels = {f"{e['type']} @ {e['time_s']:.1f}s ({e['status']})": e for e in events}

    def jump():
        e = ev_labels.get(st.session_state.get(f"var_ev_{jid}"))
        if e:
            st.session_state[key] = int(np.clip(round((e["time_s"] * fps - frames[0]) / stride), 0, len(frames) - 1))

    def step(n):
        st.session_state[key] = int(np.clip(st.session_state[key] + n, 0, len(frames) - 1))

    st.selectbox("Jump to a candidate event", ["—"] + list(ev_labels), key=f"var_ev_{jid}", on_change=jump)
    if len(frames) < 2:
        st.info("Only one frame was processed.")
        return
    b = st.columns(5)
    b[0].button("⏮ -10", on_click=step, args=(-10,))
    b[1].button("◀ -1", on_click=step, args=(-1,))
    b[2].button("▶ +1", on_click=step, args=(1,))
    b[3].button("+10 ⏭", on_click=step, args=(10,))
    speed = b[4].selectbox("Slow-mo speed", [0.25, 0.5, 1.0], index=1, label_visibility="collapsed")
    st.slider("Processed frame", 0, len(frames) - 1, key=key)
    fidx = frames[st.session_state[key]]

    holder = st.empty()

    def show(fi):
        frame = read_frame(job["video_path"], fi)
        if frame is None:
            holder.error("Could not read this frame from the source video.")
            return
        fdf = df[df["frame_idx"] == fi]
        img = pipeline.annotate_frame(frame, fdf, teams)
        holder.image(cv2.cvtColor(img, cv2.COLOR_BGR2RGB), caption=f"Frame {fi} · t = {fi / fps:.2f} s", width="stretch")
        return fdf

    fdf = show(fidx)
    if st.button("▶ Play slow-motion from here (max 60 frames)"):
        i0 = st.session_state[key]
        for i in range(i0, min(len(frames), i0 + 60)):
            show(frames[i])
            time.sleep(max(0.02, (stride / fps) / speed))
    if fdf is not None and len(fdf):
        t = fdf[fdf["cls"] == "player"][["track_id", "conf", "x1", "y1", "x2", "y2", "foot_x", "foot_y"]].copy()
        t["team"] = t["track_id"].map(lambda x: _team_label(teams.get(int(x), -1)))
        st.markdown(ui.badge("measured", "Boxes & conf: measured") + ui.badge("estimated", "foot_x/foot_y: estimated ground contact (box bottom-centre)")
                    + ui.badge("uncertain", "team: heuristic"), unsafe_allow_html=True)
        st.dataframe(t.round(1), hide_index=True, width="stretch")
    with st.expander("What is still missing for an offside line?"):
        st.markdown("1. Per-frame camera calibration (pitch lines → homography); the Tactical tab has a single-frame version for static cameras.  \n"
                    "2. Reliable detection of the last defender and the attacker at the ball-contact frame.  \n"
                    "3. Limb/body-part keypoints (pose model) - bounding boxes cannot say which body part is ahead of the line.  \n"
                    "Until these exist, the app will not draw a result rather than present a misleading one.")


# --------------------------------------------------------------------------- tab: tactical
def _calibration_ui(job, fps, frames):
    jid = job["id"]
    cal = pc.load_calibration(jid)
    if cal:
        st.success(f"Calibration saved (frame {cal['frame_idx']}). Pitch coordinates are available.")
        if st.button("Remove calibration"):
            pc.clear_calibration(jid)
            st.rerun()
        return
    st.info("Pitch calibration maps image pixels to metres. Click 4 known pitch landmarks on a frame. "
            "**Only valid for static-camera footage** (a panning broadcast camera breaks a single-frame calibration).")
    try:
        from streamlit_image_coordinates import streamlit_image_coordinates
    except ImportError:
        st.error("Install the click component: pip install streamlit-image-coordinates")
        return
    names = list(pc.PRESETS) + ["Custom (enter metres)"]
    preset = st.selectbox("Landmark set", names, key=f"cal_preset_{jid}")
    t = st.slider("Calibration frame (s)", float(frames[0] / fps), float(frames[-1] / fps), float(frames[0] / fps), 0.5, key=f"cal_t_{jid}")
    fidx = frames[int(np.argmin([abs(f / fps - t) for f in frames]))]
    pk, lk = f"cal_pts_{jid}", f"cal_last_{jid}"
    pts = st.session_state.setdefault(pk, [])
    if preset.startswith("Custom"):
        pitch_pts = []
        cols = st.columns(4)
        for i in range(4):
            x = cols[i].number_input(f"P{i + 1} x (m)", 0.0, settings.PITCH_LENGTH_M, [0.0, 16.5, 16.5, 0.0][i], key=f"cx{i}_{jid}")
            y = cols[i].number_input(f"P{i + 1} y (m)", 0.0, settings.PITCH_WIDTH_M, [13.84, 13.84, 54.16, 54.16][i], key=f"cy{i}_{jid}")
            pitch_pts.append((x, y))
        labels = [f"Point {i + 1}" for i in range(4)]
    else:
        labels = [l for l, _ in pc.PRESETS[preset]]
        pitch_pts = [p for _, p in pc.PRESETS[preset]]
    frame = read_frame(job["video_path"], fidx)
    if frame is None:
        st.error("Could not read the calibration frame.")
        return
    vis = frame.copy()
    for i, (px, py) in enumerate(pts):
        cv2.circle(vis, (int(px), int(py)), 8, (0, 255, 0), -1)
        cv2.putText(vis, str(i + 1), (int(px) + 10, int(py) - 10), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 255), 2)
    if len(pts) < 4:
        st.markdown(f"**Click point {len(pts) + 1} of 4:** {labels[len(pts)]} → pitch ({pitch_pts[len(pts)][0]:.1f} m, {pitch_pts[len(pts)][1]:.1f} m)")
    click = streamlit_image_coordinates(Image.fromarray(cv2.cvtColor(vis, cv2.COLOR_BGR2RGB)), key=f"cal_img_{jid}", width=900)
    if click and click != st.session_state.get(lk) and len(pts) < 4:
        st.session_state[lk] = click
        s = frame.shape[1] / click["width"]
        pts.append((click["x"] * s, click["y"] * s))
        st.rerun()
    c1, c2 = st.columns(2)
    if c1.button("Undo last point") and pts:
        pts.pop()
        st.rerun()
    if c2.button("Compute calibration", disabled=len(pts) != 4, type="primary"):
        try:
            H = pc.compute_homography(pts, pitch_pts)
        except pc.CalibrationError as e:
            st.error(f"Calibration failed: {e}")
            return
        pc.save_calibration(jid, fidx, pts, pitch_pts, H)
        st.session_state[pk] = []
        st.rerun()


def tab_tactical(job, df, teams):
    jid = job["id"]
    fps, stride, frames = _job_meta(job)
    ui.data_legend()
    st.caption("Every number below is an **estimate** derived from detections. Distances/positions in metres need pitch calibration.")
    t_lo, t_hi = frames[0] / fps, frames[-1] / fps
    if t_hi - t_lo > 1:
        w0, w1 = st.slider("Analysis window (s)", float(t_lo), float(t_hi), (float(t_lo), float(t_hi)), 0.5, key=f"win_{jid}")
    else:
        w0, w1 = t_lo, t_hi
    d = df[(df["time_s"] >= w0) & (df["time_s"] <= w1)]
    with st.expander("Pitch calibration", expanded=pc.load_calibration(jid) is None):
        _calibration_ui(job, fps, frames)
    cal = pc.load_calibration(jid)

    poss = ta.estimate_possession(d, teams)
    ui.stat_cards([("Possession A (est.)", "n/a" if math.isnan(poss["team_a_pct"]) else f"{poss['team_a_pct']:.0f}%",
                    "Share of ball-visible frames where a Team A player is nearest the ball"),
                   ("Possession B (est.)", "n/a" if math.isnan(poss["team_b_pct"]) else f"{poss['team_b_pct']:.0f}%"),
                   ("Frames with ball", poss["frames_with_ball"]), ("Frames with possession", poss["frames_with_possession"])])
    pl = ta.players_with_team(d, teams)
    if pl.empty:
        ui.empty_state("No players in this window", "Widen the analysis window.")
        return
    counts = pl.groupby("track_id").size().sort_values(ascending=False)
    default_ids = [int(i) for i in counts.index[:5]]
    ids = st.multiselect("Players for trails / heatmap (track IDs)", [int(i) for i in counts.index], default=default_ids, key=f"ids_{jid}")
    team_pick = st.radio("Heatmap subject", ["Selected players", "Team A", "Team B"], horizontal=True, key=f"hm_{jid}")

    def subject(frame):
        if team_pick == "Team A":
            return frame[frame["team"] == 0]
        if team_pick == "Team B":
            return frame[frame["team"] == 1]
        return frame[frame["track_id"].isin(ids)]

    if cal:
        pp = ta.add_pitch_coords(pl, cal["H"])
        v1, v2, v3, v4 = st.tabs(["Heatmap", "Movement trails", "Team shape / formation snapshot", "Distance & speed"])
        with v1:
            s = subject(pp)
            fig, ax = plt.subplots(figsize=(9, 6)); fig.patch.set_facecolor("#0b1210")
            ta.plot_heatmap(ax, s["px"].to_numpy(), s["py"].to_numpy())
            st.pyplot(fig); plt.close(fig)
        with v2:
            fig, ax = plt.subplots(figsize=(9, 6)); fig.patch.set_facecolor("#0b1210")
            ta.plot_trails(ax, pp, ids, TEAM_HEX)
            st.pyplot(fig); plt.close(fig)
        with v3:
            avg = ta.team_shape(pp)
            fig, ax = plt.subplots(figsize=(9, 6)); fig.patch.set_facecolor("#0b1210")
            ta.plot_average_positions(ax, avg, TEAM_HEX)
            st.pyplot(fig); plt.close(fig)
            summ = ta.team_shape_summary(avg)
            if len(summ):
                st.dataframe(summ.round(1), hide_index=True, width="stretch")
            st.caption("Average position over the window per track ID. A formation label (4-4-2 …) is intentionally not guessed: "
                       "it needs reliable player identities and a moving calibration.")
        with v4:
            ds = ta.distance_stats(pp)
            if ds.empty:
                st.info("Not enough continuous track data.")
            else:
                ds["team"] = ds["team"].map(_team_label)
                st.dataframe(ds.round(2), hide_index=True, width="stretch")
                st.caption("Estimated: smoothed ground positions; steps faster than 12 m/s are discarded as tracking glitches.")
    else:
        st.info("No calibration yet → showing **image-space** views only (pixels, not metres). Calibrate above for top-down pitch views and distances.")
        bg = read_frame(job["video_path"], frames[int(np.argmin([abs(f / fps - w0) for f in frames]))])
        if bg is None:
            return
        v1, v2 = st.tabs(["Heatmap on frame", "Movement trails on frame"])
        with v1:
            s = subject(pl)
            st.image(cv2.cvtColor(ta.heatmap_on_frame(bg, s["foot_x"], s["foot_y"]), cv2.COLOR_BGR2RGB), width="stretch")
        with v2:
            st.image(cv2.cvtColor(ta.trails_on_frame(bg, pl, ids), cv2.COLOR_BGR2RGB), width="stretch")
        st.caption("With a panning camera, image-space heatmaps blend camera motion with player motion.")


# --------------------------------------------------------------------------- tab: events
def _normalise_event_rows(edited: pd.DataFrame, duration_s: float):
    rows = []
    for r in edited.to_dict("records"):
        if pd.isna(r.get("time_s")):
            continue
        t = float(r["time_s"])
        etype = r.get("type") if isinstance(r.get("type"), str) else "other"
        s0, e0 = ed.clip_window(etype, t, duration_s)
        s = float(r["start_s"]) if not pd.isna(r.get("start_s")) else s0
        e = float(r["end_s"]) if not pd.isna(r.get("end_s")) else e0
        rows.append(dict(type=etype, time_s=t, confidence=None if pd.isna(r.get("confidence")) else float(r["confidence"]),
                         start_s=min(s, t), end_s=max(e, t), track_ids=r.get("track_ids") if isinstance(r.get("track_ids"), str) else "",
                         source=r.get("source") if isinstance(r.get("source"), str) else "manual",
                         status=r.get("status") if r.get("status") in settings.EVENT_STATUSES else "pending",
                         note=r.get("note") if isinstance(r.get("note"), str) else ""))
    return rows


def tab_events(job, df, teams):
    jid = job["id"]
    fps, stride, frames = _job_meta(job)
    duration_s = job["frame_count"] / fps
    st.warning("Candidates come from simple heuristics (fast ball movement, close duels near the ball). They are **suggestions to review**, "
               "not recognised goals/fouls. Confidence is left empty because no calibrated probability exists. "
               "Real event recognition needs a trained action-spotting model (Phase 5).")
    c1, c2 = st.columns([2, 1])
    thr = c1.slider("Ball-speed threshold (frame-widths per second)", 0.2, 1.5, 0.5, 0.05)
    if c2.button("🔍 Find candidate events", type="primary"):
        db.clear_auto_pending_events(jid)
        found = ed.detect_candidates(df, teams, fps, stride, job["width"], duration_s, thr)
        db.add_events(jid, found)
        st.success(f"{len(found)} candidate(s) added." if found else "No candidates found. Try a lower threshold or a clip with a detected ball.")
        st.rerun()

    with st.form(f"add_ev_{jid}", clear_on_submit=True):
        a, b, c = st.columns([2, 2, 1])
        etype = a.selectbox("Add manually: type", settings.EVENT_TYPES[:7])
        tsec = b.number_input("Time (s in full video)", 0.0, float(duration_s), float(frames[0] / fps), 0.5)
        if c.form_submit_button("Add event"):
            s, e = ed.clip_window(etype, tsec, duration_s)
            db.add_events(jid, [dict(type=etype, time_s=tsec, start_s=s, end_s=e, source="manual", status="confirmed")])
            st.rerun()

    evs = db.get_events(jid)
    if not evs:
        ui.empty_state("No events yet", "Run candidate detection or add an event manually.")
        return
    edf = pd.DataFrame(evs)[db.EVENT_COLUMNS]
    edited = st.data_editor(
        edf, hide_index=True, width="stretch", num_rows="dynamic", key=f"ev_edit_{jid}",
        disabled=["id", "source", "track_ids", "confidence"],
        column_config={"type": st.column_config.SelectboxColumn(options=settings.EVENT_TYPES, required=True),
                       "status": st.column_config.SelectboxColumn(options=settings.EVENT_STATUSES, required=True),
                       "time_s": st.column_config.NumberColumn("time (s)", format="%.1f"),
                       "start_s": st.column_config.NumberColumn("clip start (s)", format="%.1f"),
                       "end_s": st.column_config.NumberColumn("clip end (s)", format="%.1f")})
    s1, s2, _ = st.columns([1, 1, 3])
    if s1.button("💾 Save review changes"):
        db.save_event_rows(jid, _normalise_event_rows(edited, duration_s))
        st.success("Saved.")
        st.rerun()
    s2.download_button("⬇ Events CSV", edf.to_csv(index=False), file_name=f"varvision_job{jid}_events.csv", mime="text/csv")

    st.subheader("Preview an event (annotated video)")
    pick = st.selectbox("Event", [e["id"] for e in evs], format_func=lambda i: next(f"{e['type']} @ {e['time_s']:.1f}s [{e['status']}]" for e in evs if e["id"] == i))
    e = next(e for e in evs if e["id"] == pick)
    off = job["params"]["start_frame"] / fps
    ann = pipeline.annotated_path(jid)
    if ann.exists() and e["end_s"] - off > 0:
        st.video(str(ann), start_time=int(max(0, e["start_s"] - off)), end_time=int(math.ceil(e["end_s"] - off)))
    else:
        st.info("This event is outside the analysed window.")


# --------------------------------------------------------------------------- tab: highlights
def tab_highlights(job):
    jid = job["id"]
    fps, _, frames = _job_meta(job)
    duration_s = job["frame_count"] / fps
    out_dir = settings.job_dir(jid) / "highlights"
    evs = db.get_events(jid)
    include_pending = st.checkbox("Also include events that are still 'pending' review", value=False)
    chosen = [e for e in evs if e["status"] == "confirmed" or (include_pending and e["status"] == "pending")]
    if not chosen:
        ui.empty_state("No events selected", "Confirm events in the 'Match Events' tab (or tick the box above) to build highlights.")
    else:
        st.write(f"**{len(chosen)}** event(s) selected. Overlapping clips are merged automatically.")
    c1, c2, c3 = st.columns(3)
    own = c1.checkbox("Use each event's own clip window", value=True)
    pre = c1.slider("Seconds before event", 0.0, 20.0, 6.0, 0.5, disabled=own)
    post = c1.slider("Seconds after event", 0.0, 20.0, 4.0, 0.5, disabled=own)
    fmt = c2.radio("Format", ["Vertical 9:16 (Reels)", "Original aspect"])
    width = c2.selectbox("Output width", [540, 720, 1080], index=1, help="Vertical 720 = 720x1280, 1080 = 1080x1920")
    captions = c2.checkbox("Captions (event label + timestamp)", value=True)
    crf = c3.slider("Quality (CRF, lower = better)", 18, 32, 23)
    preset = c3.selectbox("Encoding speed", ["ultrafast", "veryfast", "medium", "slow"], index=1)
    source = c3.radio("Source", ["Original video", "Annotated video (analysed window only)"])

    if st.button("🎬 Generate highlights", type="primary", disabled=not chosen):
        use_ann = source.startswith("Annotated")
        src = str(pipeline.annotated_path(jid)) if use_ann else job["video_path"]
        offset = job["params"]["start_frame"] / fps if use_ann else 0.0
        win_lo, win_hi = job["params"]["start_frame"] / fps, job["params"]["end_frame"] / fps
        sel = [e for e in chosen if not use_ann or win_lo <= e["time_s"] <= win_hi]
        if not sel:
            st.error("No selected events fall inside the annotated window.")
            return
        if out_dir.exists():
            for f in out_dir.glob("*.mp4"):
                f.unlink()
        cb, bar, txt = _progress_ui()
        try:
            hg.generate_highlights(src, sel, duration_s, (job["width"], job["height"]), out_dir,
                                   hg.ExportSettings(vertical=fmt.startswith("Vertical"), width=int(width), crf=int(crf),
                                                     preset=preset, captions=captions),
                                   None if own else pre, None if own else post, offset, cb)
        except (VideoError, hg.HighlightError, ValueError) as ex:
            bar.empty(); txt.empty()
            st.error(f"Export failed: {ex}")
            return
        bar.empty(); txt.empty()
        st.rerun()

    comp = out_dir / "highlights_compilation.mp4"
    if comp.exists():
        st.subheader("Preview")
        st.video(str(comp))
        st.download_button("⬇ Download compilation", comp.read_bytes(), file_name=f"varvision_job{jid}_highlights.mp4", mime="video/mp4")
        with st.expander("Individual clips"):
            for f in sorted(out_dir.glob("clip_*.mp4")):
                st.video(str(f))
                st.download_button(f"⬇ {f.name}", f.read_bytes(), file_name=f.name, mime="video/mp4", key=f"dl_{f.name}")
        st.caption("Files are saved locally only. Nothing is uploaded or posted - publish manually if you want to share.")


# =========================================================================== help page
def page_help():
    st.header("Help & feature status")
    st.markdown("""
| Feature | Status |
|---|---|
| Upload, validation, progress, error handling | ✅ Working |
| YOLO person + ball detection, ByteTrack / BoT-SORT tracking | ✅ Working (COCO model: weak on the small ball) |
| Annotated video + tracking CSV export | ✅ Working |
| Team split by jersey colour + manual correction | ✅ Heuristic, correctable |
| Referee / goalkeeper detection | ⚠️ Not detected by the model; only manual labelling |
| Frame-by-frame review, slow-motion | ✅ Working |
| Pitch calibration (4-point homography, static camera) | ✅ Working, single frame |
| Heatmaps, trails, average positions, distance / speed | ✅ Estimated (metres need calibration) |
| Possession estimate | ⚠️ Rough heuristic |
| Candidate events (ball-speed spikes, duels) + review table | ✅ Working heuristics |
| Goal / save / foul / corner *recognition* | ❌ Needs a trained model |
| Highlight clips, 9:16 reels, compilation, captions | ✅ Working (FFmpeg) |
| Offside line overlay | ❌ Not implemented (Phase 4) |
| Passing network | ❌ Not implemented (needs reliable ball + possession) |
| Audio-cue events | ❌ Not implemented |
""")
    st.caption("See README.md for installation, troubleshooting and the roadmap. This is a review-assistance prototype, "
               "not an official VAR or refereeing system.")


# =========================================================================== router
with st.sidebar:
    ui.brand()
if "_goto" in st.session_state:
    st.session_state["nav"] = st.session_state.pop("_goto")
page = st.sidebar.radio("Navigate", ["New analysis", "Match dashboard", "Help & status"], key="nav")
if page == "New analysis":
    page_new_analysis()
elif page == "Match dashboard":
    page_dashboard()
else:
    page_help()
