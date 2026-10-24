import pytest

from services import highlight_generator as hg
from services import video_processor as vp


def test_clip_bounds_clamping():
    assert hg.compute_clip_bounds(10, 5, 3, 60) == (5, 13)
    assert hg.compute_clip_bounds(2, 5, 3, 60) == (0, 5)
    assert hg.compute_clip_bounds(58, 5, 10, 60) == (53, 60)


def test_clip_bounds_invalid():
    with pytest.raises(ValueError):
        hg.compute_clip_bounds(-1, 5, 3, 60)
    with pytest.raises(ValueError):
        hg.compute_clip_bounds(10, -1, 3, 60)
    with pytest.raises(ValueError):
        hg.compute_clip_bounds(100, 2, 2, 60)  # event beyond video end


def test_merge_overlapping():
    clips = [dict(start_s=0, end_s=10, label="Goal"), dict(start_s=8, end_s=14, label="Shot"),
             dict(start_s=30, end_s=40, label="Save"), dict(start_s=9, end_s=11, label="Goal")]
    m = hg.merge_overlapping(clips)
    assert len(m) == 2
    assert (m[0]["start_s"], m[0]["end_s"]) == (0, 14) and m[0]["label"] == "Goal + Shot"
    assert m[1]["label"] == "Save"


def test_export_vertical_clip_no_audio(sample_video, tmp_path):
    out = tmp_path / "clip.mp4"
    s = hg.ExportSettings(vertical=True, width=360, crf=28, preset="ultrafast")
    hg.export_clip(sample_video, out, 0.5, 2.0, s, (320, 180), "GOAL", "00:01")
    m = vp.read_metadata(out)
    assert (m.width, m.height) == (360, 640)
    assert m.duration_s == pytest.approx(1.5, abs=0.3)
    assert not vp.has_audio(out)


def test_export_original_aspect_without_captions(sample_video, tmp_path):
    out = tmp_path / "o.mp4"
    hg.export_clip(sample_video, out, 0, 1, hg.ExportSettings(vertical=False, width=320, captions=False,
                                                              preset="ultrafast"), (320, 180))
    assert (vp.read_metadata(out).width, vp.read_metadata(out).height) == (320, 180)


def test_export_errors(sample_video, tmp_path):
    with pytest.raises(hg.HighlightError):
        hg.export_clip(sample_video, tmp_path / "x.mp4", 2, 1, hg.ExportSettings(), (320, 180))
    with pytest.raises(hg.HighlightError):
        hg.export_clip(tmp_path / "missing.mp4", tmp_path / "x.mp4", 0, 1, hg.ExportSettings(), (320, 180))


def test_generate_highlights_with_compilation(sample_video, tmp_path):
    events = [dict(type="goal", time_s=1.0, start_s=0.0, end_s=2.0),
              dict(type="shot", time_s=1.5, start_s=0.5, end_s=2.5)]   # overlaps -> merged into 1 clip
    s = hg.ExportSettings(width=360, crf=30, preset="ultrafast")
    clips, combined = hg.generate_highlights(sample_video, events, 3.0, (320, 180), tmp_path / "h", s)
    assert len(clips) == 1 and combined.exists()
    clips2, combined2 = hg.generate_highlights(sample_video, events, 3.0, (320, 180), tmp_path / "h2", s,
                                               pre_s=0.2, post_s=0.2)
    assert len(clips2) == 2 and combined2.exists()  # 0.2s padding: no overlap, so two clips
