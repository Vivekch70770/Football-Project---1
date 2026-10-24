import pytest

from services import video_processor as vp


def test_metadata(sample_video):
    m = vp.read_metadata(sample_video)
    assert (m.width, m.height) == (320, 180)
    assert m.fps == pytest.approx(25, abs=0.5)
    assert m.frame_count == 75
    assert m.duration_s == pytest.approx(3.0, abs=0.1)


def test_timestamp_conversions():
    assert vp.frame_to_time(50, 25.0) == 2.0
    assert vp.time_to_frame(2.0, 25.0) == 50
    assert vp.time_to_frame(-1, 25.0) == 0
    with pytest.raises(ValueError):
        vp.frame_to_time(1, 0)


def test_missing_file(tmp_path):
    with pytest.raises(vp.VideoError, match="not found"):
        vp.read_metadata(tmp_path / "nope.mp4")


def test_unsupported_extension(tmp_path):
    f = tmp_path / "a.txt"
    f.write_text("hi")
    with pytest.raises(vp.VideoError, match="Unsupported"):
        vp.read_metadata(f)


def test_corrupt_video(tmp_path):
    f = tmp_path / "bad.mp4"
    f.write_bytes(b"not a video at all" * 100)
    with pytest.raises(vp.VideoError):
        vp.read_metadata(f)


def test_empty_file(tmp_path):
    f = tmp_path / "empty.mp4"
    f.write_bytes(b"")
    with pytest.raises(vp.VideoError, match="empty"):
        vp.read_metadata(f)


def test_iter_frames_stride_and_range(sample_video):
    idx = [i for i, _ in vp.iter_frames(sample_video, start_frame=10, end_frame=40, stride=5)]
    assert idx == [10, 15, 20, 25, 30, 35]
    with pytest.raises(ValueError):
        list(vp.iter_frames(sample_video, stride=0))
