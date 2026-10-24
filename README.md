# VARVision AI ⚽

AI-assisted football match analysis: player & ball tracking, tactical views, candidate-event review,
highlight / Reels export, and a frame-by-frame review tool. Runs locally on a Windows laptop (CPU is fine).

> **Review-assistance prototype.** It does not reproduce official VAR / semi-automated offside technology and never
> declares goals, fouls or offside decisions. Every estimated value is labelled as such in the UI.

## 1. What genuinely works (v0.1) vs. what does not

| Area | Status |
|---|---|
| Upload / pick clip, validation, progress bar, friendly errors | ✅ |
| YOLOv8 detection (person + ball) with ByteTrack / BoT-SORT tracking | ✅ |
| Annotated MP4 (boxes, IDs, confidence, ball trail) + `tracks.csv` with timestamps | ✅ |
| Team split by torso colour (K-means) + **manual correction** and re-render | ✅ heuristic |
| Frame-by-frame review, slow-motion playback, jump to events | ✅ |
| 4-point pitch calibration (homography) → top-down pitch coordinates | ✅ static camera only |
| Heatmaps, trails, average positions, team shape, distance/speed (estimated) | ✅ |
| Image-space heatmaps/trails without calibration | ✅ (pixels, not metres) |
| Possession estimate | ⚠️ rough heuristic |
| Candidate events (fast ball movement, duels near ball) + confirm/reject/edit table | ✅ heuristic |
| Highlight clips, combined reel, 9:16 vertical with blurred background, captions, CRF/preset/resolution | ✅ |
| SQLite storage of jobs & events | ✅ |
| Referee / goalkeeper **detection** | ❌ COCO model has only `person` |
| Goal / save / foul / corner **recognition** | ❌ needs trained model |
| Offside line overlay | ❌ Phase 4 (needs per-frame calibration + pose/keypoints) |
| Passing network, audio-cue events, FastAPI/React | ❌ future |

**Model limits (important).** The pretrained YOLO weights are trained on COCO (80 classes). Football-relevant ones are
`person` (0) and `sports ball` (32). The ball is tiny and often missed on broadcast footage; referees, goalkeepers,
goals and pitch lines are not separate classes. A custom model trained on a football dataset (e.g. Roboflow Universe
"football players detection", SoccerNet) would fix this — see Phase 5.

## 2. Install on Windows + VS Code

Prerequisites: **Python 3.10–3.12** (https://www.python.org/downloads/, tick *Add Python to PATH*) and **VS Code**
with the *Python* extension. FFmpeg does **not** need to be installed — `imageio-ffmpeg` bundles it.

Open the `varvision-ai` folder in VS Code (`File → Open Folder`), then open a terminal (`Ctrl+``) and run:

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
```

If PowerShell blocks activation: `Set-ExecutionPolicy -Scope CurrentUser RemoteSigned`, reopen the terminal.
Select the interpreter: `Ctrl+Shift+P` → *Python: Select Interpreter* → `.venv`.

**CPU-only PyTorch (smaller download, recommended without an NVIDIA GPU):**
```powershell
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
pip install -r requirements.txt
```

### Model weights
The first run downloads `yolov8n.pt` (~6 MB) into `models/` automatically. To do it up front:
```powershell
python scripts/download_model.py yolov8n.pt
```
If you are offline, download `yolov8n.pt` from https://github.com/ultralytics/assets/releases and drop it into `models/`.

## 3. Run

```powershell
streamlit run app.py
```
or press **F5** in VS Code and choose *VARVision AI (Streamlit)*. The app opens at http://localhost:8501.

Run the tests: `pytest -v` (or the *Run tests* launch config).

## 4. Sample workflow

1. Get a **short** clip you have the right to analyse (your own recording, a club-provided clip, or a CC-licensed video).
   Trim it: `python scripts/trim_clip.py "C:\path\match.mp4" --start 120 --duration 45` → `data/samples/match_clip.mp4`.
2. **New analysis** → choose the file → keep defaults (yolov8n, stride 2, size 960) → **Run analysis**.
3. **Player Tracking** tab: watch/download the annotated video; fix wrong team labels in the table → *Apply & re-render*.
4. **Tactical Analysis**: try image-space heatmap/trails. For a static camera (e.g. tactical/fixed cam), expand
   *Pitch calibration*, click four known landmarks (e.g. penalty-area corners) → compute → top-down heatmap, trails, shape, distances.
5. **Match Events** → *Find candidate events*, review each (status = confirmed / rejected), adjust clip windows, or add events manually → *Save*.
6. **Highlights** → choose Vertical 9:16, generate, preview, download. Nothing is ever posted automatically.
7. **VAR Review** → step frame-by-frame or play slow-motion around any event.

**CPU speed tips:** yolov8n, stride 3, inference size 640 for a first pass; analyse 30–60 s windows.
Expect roughly 3–8 processed frames/s on an i5.

## 5. Project structure

```
app.py                      Streamlit dashboard (tabs, review tables)
config/settings.py          paths, model choices, constants
services/video_processor.py metadata, safe frame iteration, FFmpeg helpers
services/detection.py       YOLO weight loading / download
services/tracking.py        detection+tracking loop, torso colours, team clustering
services/pipeline.py        job orchestration, annotated rendering, storage
services/pitch_calibration.py  4-point homography
services/tactical_analysis.py  heatmaps, trails, distance, possession (estimates)
services/event_detection.py    modular candidate detectors (add trained models here)
services/highlight_generator.py clip maths, 9:16 export, compilation
services/database.py        SQLite (jobs, events)
components/ui.py            theme + cards
scripts/                    download_model.py, trim_clip.py
tests/                      pytest suite
data/ outputs/ models/      local only (git-ignored)
```

Large files (weights, videos, outputs, database) are git-ignored. Uploads are saved under random file names;
no absolute paths are shown in the UI.

## 6. Troubleshooting

| Problem | Fix |
|---|---|
| `ModuleNotFoundError` | Venv not active/selected. Re-run activation and `pip install -r requirements.txt`. |
| *"OpenCV could not open this video"* | Re-encode: `python scripts/trim_clip.py file.mkv --duration 60` (outputs H.264 MP4). HEVC/H.265 often fails in OpenCV. |
| Annotated video won't play in the browser | It is transcoded to H.264 automatically; if FFmpeg failed, run `pip install -U imageio-ffmpeg`. |
| Model download fails | Download the `.pt` manually into `models/` (see above). |
| Very slow | yolov8n, stride 3–5, size 640, shorter window. Close other apps. |
| Ball rarely detected | Raise inference size to 1280, lower ball confidence to 0.1 — or train a custom ball detector (Phase 5). |
| Out of memory | Lower inference size, raise stride, shorter window. |
| Many track IDs for one player | Normal with occlusions/camera cuts; use BoT-SORT, or merge by correcting team labels. Cuts in broadcast footage break tracking. |
| Calibration looks wrong | Only valid for a static camera; click corners precisely and use the "Undo last point" button. |
| Upload > 500 MB | Trim first with `scripts/trim_clip.py`. |

## 7. Roadmap

- **Phase 2+**: per-frame camera calibration (pitch-line keypoint model) so panning footage gets metric stats; passing network.
- **Phase 3+**: audio-peak detector (crowd/commentary), SoccerNet action-spotting model plugged into `services/event_detection.py`.
- **Phase 4**: offside overlay — needs calibration + pose keypoints for body parts; show uncertainty and refuse when evidence is weak.
- **Phase 5**: fine-tune YOLO on football data (classes: player, goalkeeper, referee, ball), FastAPI backend, React UI.

Fine-tuning sketch: `yolo detect train data=football.yaml model=yolov8s.pt epochs=50 imgsz=960` with a Roboflow/SoccerNet
dataset, then put the resulting `best.pt` in `models/` and add its name to `MODEL_CHOICES` in `config/settings.py`
(update the class ids in `config/settings.py` too).
