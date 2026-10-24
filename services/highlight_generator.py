"""Clip boundary maths, de-duplication and FFmpeg export (incl. 9:16 vertical reels)."""
from __future__ import annotations

import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, List, Optional, Sequence, Tuple

from PIL import Image, ImageDraw, ImageFont

from services.video_processor import VideoError, has_audio, run_ffmpeg

ProgressCB = Optional[Callable[[float, str], None]]


class HighlightError(Exception):
    pass


@dataclass
class ExportSettings:
    vertical: bool = True
    width: int = 1080          # output width (vertical -> 1080x1920 for width=1080)
    crf: int = 23
    preset: str = "veryfast"
    captions: bool = True


def compute_clip_bounds(event_time_s: float, pre_s: float, post_s: float, video_duration_s: float) -> Tuple[float, float]:
    if event_time_s < 0 or video_duration_s <= 0:
        raise ValueError("event time must be >= 0 and video duration > 0")
    if pre_s < 0 or post_s < 0:
        raise ValueError("pre/post seconds must be >= 0")
    start = max(0.0, event_time_s - pre_s)
    end = min(video_duration_s, event_time_s + post_s)
    if end <= start:
        raise ValueError("clip would be empty (event outside the video)")
    return start, end


def merge_overlapping(clips: Sequence[dict], min_gap_s: float = 0.0) -> List[dict]:
    """clips: dicts with start_s, end_s, label. Overlapping/adjacent clips are merged, labels joined."""
    out: List[dict] = []
    for c in sorted(clips, key=lambda c: c["start_s"]):
        if out and c["start_s"] <= out[-1]["end_s"] + min_gap_s:
            out[-1]["end_s"] = max(out[-1]["end_s"], c["end_s"])
            if c["label"] not in out[-1]["label"].split(" + "):
                out[-1]["label"] += f" + {c['label']}"
        else:
            out.append(dict(c))
    return out


def _font(size: int):
    try:
        return ImageFont.load_default(size=size)
    except TypeError:  # very old Pillow
        return ImageFont.load_default()


def build_caption_png(path: Path, width: int, height: int, top: str = "", bottom: str = "") -> None:
    img = Image.new("RGBA", (width, height), (0, 0, 0, 0))
    d = ImageDraw.Draw(img)
    f = _font(max(18, width // 18))
    for text, y in ((top, int(height * 0.06)), (bottom, int(height * 0.90))):
        if not text:
            continue
        l, t, r, b = d.textbbox((0, 0), text, font=f)
        x = (width - (r - l)) // 2
        d.rounded_rectangle((x - 20, y - 12, x + (r - l) + 20, y + (b - t) + 16), 14, fill=(0, 0, 0, 150))
        d.text((x, y), text, font=f, fill=(255, 255, 255, 255))
    img.save(path)


def _even(n: int) -> int:
    return n if n % 2 == 0 else n + 1


def export_clip(src, dst, start_s: float, end_s: float, settings: ExportSettings, src_size: Tuple[int, int],
                caption_top: str = "", caption_bottom: str = "") -> Path:
    if end_s <= start_s:
        raise HighlightError("Clip end must be after clip start.")
    if not Path(src).exists():
        raise HighlightError("Source video not found.")
    dur = end_s - start_s
    w = _even(settings.width)
    if settings.vertical:
        h = _even(int(round(w * 16 / 9)))
        vf = (f"[0:v]split=2[a][b];[a]scale={w}:{h}:force_original_aspect_ratio=increase,crop={w}:{h},"
              f"boxblur=25:5[bg];[b]scale={w}:-2[fg];[bg][fg]overlay=(W-w)/2:(H-h)/2[base]")
    else:
        h = _even(int(round(w * src_size[1] / src_size[0])))
        vf = f"[0:v]scale={w}:{h}[base]"
    args = ["-ss", f"{start_s:.3f}", "-t", f"{dur:.3f}", "-i", src]
    with tempfile.TemporaryDirectory() as tmp:
        if settings.captions and (caption_top or caption_bottom):
            png = Path(tmp) / "cap.png"
            build_caption_png(png, w, h, caption_top, caption_bottom)
            args += ["-loop", "1", "-t", f"{dur:.3f}", "-i", str(png)]
            vf += ";[base][1:v]overlay=0:0:shortest=1,format=yuv420p[v]"
        else:
            vf += ";[base]format=yuv420p[v]"
        args += ["-filter_complex", vf, "-map", "[v]", "-map", "0:a?", "-t", f"{dur:.3f}",
                 "-c:v", "libx264", "-preset", settings.preset, "-crf", settings.crf,
                 "-c:a", "aac", "-b:a", "128k", "-movflags", "+faststart", str(dst)]
        run_ffmpeg(args)
    return Path(dst)


def concat_clips(clips: Sequence[Path], dst: Path) -> Path:
    if not clips:
        raise HighlightError("No clips to combine.")
    audio_flags = {has_audio(c) for c in clips}
    with tempfile.TemporaryDirectory() as tmp:
        lst = Path(tmp) / "list.txt"
        lst.write_text("".join(f"file '{Path(c).resolve().as_posix()}'\n" for c in clips), encoding="utf-8")
        extra = [] if len(audio_flags) == 1 else ["-an"]  # mixed audio/no-audio clips: drop audio
        run_ffmpeg(["-f", "concat", "-safe", "0", "-i", str(lst), "-c", "copy", *extra, str(dst)])
    return Path(dst)


def generate_highlights(src, events: Sequence[dict], video_duration_s: float, src_size: Tuple[int, int],
                        out_dir: Path, settings: ExportSettings, pre_s: Optional[float] = None,
                        post_s: Optional[float] = None, time_offset_s: float = 0.0,
                        progress_cb: ProgressCB = None) -> Tuple[List[Path], Optional[Path]]:
    """Export one clip per (merged) event plus a combined compilation.

    If pre_s/post_s are None, each event's own start_s/end_s are used.
    time_offset_s is subtracted from times (use when src is the annotated video of a processed window).
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    raw = []
    for e in events:
        if pre_s is not None and post_s is not None:
            s, t = compute_clip_bounds(e["time_s"], pre_s, post_s, video_duration_s)
        else:
            s, t = max(0.0, e["start_s"]), min(video_duration_s, e["end_s"])
        raw.append(dict(start_s=s, end_s=t, label=str(e["type"]).replace("_", " ").title(), time_s=e["time_s"]))
    clips = merge_overlapping(raw)
    paths: List[Path] = []
    for i, c in enumerate(clips, 1):
        if progress_cb:
            progress_cb((i - 1) / max(1, len(clips)), f"Exporting clip {i}/{len(clips)}")
        s, t = c["start_s"] - time_offset_s, c["end_s"] - time_offset_s
        if t <= 0:
            continue
        mm, ss = divmod(int(c["time_s"]), 60)
        paths.append(export_clip(src, out_dir / f"clip_{i:02d}.mp4", max(0, s), t, settings, src_size,
                                 caption_top=c["label"].upper(), caption_bottom=f"{mm:02d}:{ss:02d}"))
    combined = None
    if len(paths) >= 1:
        if progress_cb:
            progress_cb(0.95, "Building compilation")
        combined = concat_clips(paths, out_dir / "highlights_compilation.mp4")
    if progress_cb:
        progress_cb(1.0, "Done")
    return paths, combined
