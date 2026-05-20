"""Video download, frame extraction, and contact-sheet composition.

Pipeline:
  download_video(url, out_dir, video_id) -> Path
  extract_frames(video_path, out_dir, timestamps) -> list[Path]
  build_contact_sheet(frame_paths, timestamps, out_path) -> Path

Frames come out of ffmpeg as JPEGs. The contact sheet is composed in PIL
so we can label timestamps and JPEG-compress to a sane size.
"""

from __future__ import annotations

import shutil
import subprocess
from pathlib import Path
from typing import Iterable

import requests
from PIL import Image, ImageDraw, ImageFont

from .config import (
    CONTACT_SHEET_COLS,
    CONTACT_SHEET_JPEG_QUALITY,
    CONTACT_SHEET_THUMB_WIDTH,
    FRAME_TIMESTAMPS_SEC,
)


class VideoError(RuntimeError):
    pass


def download_video(url: str, out_dir: Path, video_id: str, timeout: int = 120) -> Path:
    """Stream the MP4 to disk. Apify's hosted MP4 URLs are HTTPS direct
    downloads, no special headers required."""
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{video_id}.mp4"
    with requests.get(url, stream=True, timeout=timeout) as r:
        if not r.ok:
            raise VideoError(f"Failed to download video {url}: HTTP {r.status_code}")
        with out_path.open("wb") as fh:
            for chunk in r.iter_content(chunk_size=1 << 16):
                if chunk:
                    fh.write(chunk)
    if out_path.stat().st_size == 0:
        raise VideoError(f"Downloaded video is empty: {url}")
    return out_path


def _require_ffmpeg() -> str:
    path = shutil.which("ffmpeg")
    if not path:
        raise VideoError(
            "ffmpeg not found on PATH. Install ffmpeg (apt-get install ffmpeg) "
            "and re-run."
        )
    return path


def extract_frames(
    video_path: Path,
    out_dir: Path,
    timestamps: Iterable[float] = FRAME_TIMESTAMPS_SEC,
) -> list[Path]:
    """Extract one JPEG per timestamp using ffmpeg -ss seeking. Returns
    paths in input-timestamp order. Missing/failed frames are skipped
    silently so a partly-corrupt video can still produce a contact sheet."""
    ffmpeg = _require_ffmpeg()
    out_dir.mkdir(parents=True, exist_ok=True)
    out_paths: list[Path] = []
    for ts in timestamps:
        frame_path = out_dir / f"t{ts:0.2f}.jpg"
        cmd = [
            ffmpeg,
            "-y",
            "-loglevel", "error",
            "-ss", f"{ts:.3f}",
            "-i", str(video_path),
            "-frames:v", "1",
            "-q:v", "3",
            str(frame_path),
        ]
        proc = subprocess.run(cmd, capture_output=True, text=True)
        if proc.returncode == 0 and frame_path.exists() and frame_path.stat().st_size > 0:
            out_paths.append(frame_path)
    if not out_paths:
        raise VideoError(f"ffmpeg produced no frames from {video_path}.")
    return out_paths


def _load_font(size: int) -> ImageFont.ImageFont:
    # Best-effort font lookup. Default bitmap font is fine if no TTF available.
    for candidate in (
        "DejaVuSans-Bold.ttf",
        "Arial.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
    ):
        try:
            return ImageFont.truetype(candidate, size)
        except OSError:
            continue
    return ImageFont.load_default()


def build_contact_sheet(
    frame_paths: list[Path],
    timestamps: list[float],
    out_path: Path,
    thumb_width: int = CONTACT_SHEET_THUMB_WIDTH,
    cols: int = CONTACT_SHEET_COLS,
    jpeg_quality: int = CONTACT_SHEET_JPEG_QUALITY,
) -> Path:
    """Compose frames into a labeled grid JPEG.

    Each cell shows the frame scaled to `thumb_width` with its source
    timestamp burned into the top-left corner. Grid is column-major in
    insertion order, wrapping every `cols` cells."""
    if not frame_paths:
        raise VideoError("No frames to compose into a contact sheet.")
    if len(timestamps) < len(frame_paths):
        # Pad with empty labels rather than crashing.
        timestamps = list(timestamps) + [0.0] * (len(frame_paths) - len(timestamps))

    thumbs: list[tuple[Image.Image, float]] = []
    target_h = 0
    for path, ts in zip(frame_paths, timestamps):
        with Image.open(path) as im:
            im = im.convert("RGB")
            ratio = thumb_width / im.width
            new_size = (thumb_width, max(1, int(im.height * ratio)))
            thumb = im.resize(new_size, Image.LANCZOS)
        target_h = max(target_h, thumb.height)
        thumbs.append((thumb, ts))

    # Normalize all thumbs to same height to keep the grid clean.
    normed: list[tuple[Image.Image, float]] = []
    for thumb, ts in thumbs:
        if thumb.height != target_h:
            ratio = target_h / thumb.height
            new_w = max(1, int(thumb.width * ratio))
            thumb = thumb.resize((new_w, target_h), Image.LANCZOS)
        normed.append((thumb, ts))

    pad = 6
    cell_w = max(t.width for t, _ in normed)
    cell_h = target_h
    rows = (len(normed) + cols - 1) // cols
    sheet_w = cols * cell_w + (cols + 1) * pad
    sheet_h = rows * cell_h + (rows + 1) * pad

    sheet = Image.new("RGB", (sheet_w, sheet_h), color=(18, 18, 18))
    draw = ImageDraw.Draw(sheet)
    font = _load_font(size=max(12, thumb_width // 18))

    for idx, (thumb, ts) in enumerate(normed):
        col = idx % cols
        row = idx // cols
        x = pad + col * (cell_w + pad)
        y = pad + row * (cell_h + pad)
        sheet.paste(thumb, (x, y))
        label = f"{ts:.1f}s"
        # Drop-shadow label for legibility on any frame background.
        tx, ty = x + 6, y + 4
        draw.text((tx + 1, ty + 1), label, font=font, fill=(0, 0, 0))
        draw.text((tx, ty), label, font=font, fill=(255, 255, 255))

    out_path.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(out_path, format="JPEG", quality=jpeg_quality, optimize=True)
    return out_path
