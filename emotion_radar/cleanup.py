"""Temp file cleanup.

`cleanup_temp` deletes everything under data/tmp/. Contact sheets and the
SQLite database live elsewhere and are never touched by this module."""

from __future__ import annotations

import shutil
from dataclasses import dataclass
from pathlib import Path


@dataclass
class CleanupSummary:
    videos_removed: int
    frame_dirs_removed: int


def remove_video_file(video_path: Path) -> bool:
    if video_path.exists() and video_path.is_file():
        video_path.unlink()
        return True
    return False


def remove_frame_dir(frame_dir: Path) -> bool:
    if frame_dir.exists() and frame_dir.is_dir():
        shutil.rmtree(frame_dir)
        return True
    return False


def cleanup_temp(tmp_videos_dir: Path, tmp_frames_dir: Path) -> CleanupSummary:
    """Remove every file under tmp_videos_dir and every subdir under
    tmp_frames_dir. The directories themselves are left in place."""
    videos_removed = 0
    if tmp_videos_dir.exists():
        for entry in tmp_videos_dir.iterdir():
            if entry.is_file():
                entry.unlink()
                videos_removed += 1
            elif entry.is_dir():
                shutil.rmtree(entry)
                videos_removed += 1

    frame_dirs_removed = 0
    if tmp_frames_dir.exists():
        for entry in tmp_frames_dir.iterdir():
            if entry.is_dir():
                shutil.rmtree(entry)
                frame_dirs_removed += 1
            elif entry.is_file():
                entry.unlink()
                frame_dirs_removed += 1

    return CleanupSummary(
        videos_removed=videos_removed,
        frame_dirs_removed=frame_dirs_removed,
    )
