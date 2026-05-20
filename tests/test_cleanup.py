from __future__ import annotations

from pathlib import Path

from emotion_radar import cleanup


def _touch(p: Path, content: bytes = b"x") -> None:
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_bytes(content)


def test_cleanup_removes_videos_and_frames_but_not_contact_sheets(tmp_path: Path):
    data = tmp_path / "data"
    tmp_videos = data / "tmp" / "videos"
    tmp_frames = data / "tmp" / "frames"
    contact_sheets = data / "contact_sheets"

    _touch(tmp_videos / "vid1.mp4")
    _touch(tmp_videos / "vid2.mp4")
    _touch(tmp_frames / "vid1" / "t0.00.jpg")
    _touch(tmp_frames / "vid1" / "t0.50.jpg")
    _touch(tmp_frames / "vid2" / "t1.00.jpg")
    sheet1 = contact_sheets / "vid1.jpg"
    sheet2 = contact_sheets / "vid2.jpg"
    _touch(sheet1)
    _touch(sheet2)

    summary = cleanup.cleanup_temp(tmp_videos, tmp_frames)

    assert summary.videos_removed == 2
    assert summary.frame_dirs_removed == 2
    assert list(tmp_videos.iterdir()) == []
    assert list(tmp_frames.iterdir()) == []
    # Contact sheets must survive.
    assert sheet1.exists()
    assert sheet2.exists()


def test_cleanup_handles_missing_dirs(tmp_path: Path):
    summary = cleanup.cleanup_temp(tmp_path / "no_videos", tmp_path / "no_frames")
    assert summary.videos_removed == 0
    assert summary.frame_dirs_removed == 0


def test_remove_video_file_idempotent(tmp_path: Path):
    p = tmp_path / "vid.mp4"
    p.write_bytes(b"x")
    assert cleanup.remove_video_file(p) is True
    assert cleanup.remove_video_file(p) is False


def test_remove_frame_dir_idempotent(tmp_path: Path):
    d = tmp_path / "frames"
    d.mkdir()
    (d / "t0.jpg").write_bytes(b"x")
    assert cleanup.remove_frame_dir(d) is True
    assert cleanup.remove_frame_dir(d) is False
