"""Configuration: env loading, paths, safety constants.

APIFY_TOKEN resolution order:
  1. process environment
  2. /root/.hermes/.env (VPS canonical path)
  3. ./.env (local dev fallback)
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

HERMES_ENV_PATH = Path("/root/.hermes/.env")
LOCAL_ENV_PATH = Path(".env")

DEFAULT_DATA_DIR = Path("data")
DEFAULT_DB_FILENAME = "emotion_radar.db"

# Apify cost safety: hard cap on URLs per command unless --confirm-large.
DEFAULT_MAX_URLS = 3

# Frame extraction window (seconds from start of video).
FRAME_TIMESTAMPS_SEC = (0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 5.0)

# Contact sheet defaults.
CONTACT_SHEET_THUMB_WIDTH = 320
CONTACT_SHEET_COLS = 4
CONTACT_SHEET_JPEG_QUALITY = 78

# Apify actor and run input defaults.
APIFY_ACTOR_ID = "clockworks~tiktok-video-scraper"
APIFY_RUN_INPUT_DEFAULTS = {
    "scrapeRelatedVideos": False,
    "resultsPerPage": 1,
    "shouldDownloadVideos": True,
    "shouldDownloadCovers": True,
    "downloadSubtitlesOptions": "NEVER_DOWNLOAD_SUBTITLES",
    "shouldDownloadSlideshowImages": False,
}


def _parse_env_file(path: Path) -> dict[str, str]:
    """Tiny .env parser. Supports KEY=VALUE, ignores blanks and #-comments.
    Strips matching single/double quotes from values. No interpolation."""
    out: dict[str, str] = {}
    if not path.is_file():
        return out
    try:
        text = path.read_text(encoding="utf-8")
    except OSError:
        return out
    for raw in text.splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        if "=" not in line:
            continue
        key, _, value = line.partition("=")
        key = key.strip()
        value = value.strip()
        if len(value) >= 2 and value[0] == value[-1] and value[0] in ("'", '"'):
            value = value[1:-1]
        if key:
            out[key] = value
    return out


def load_env(extra_paths: Iterable[Path] | None = None) -> dict[str, str]:
    """Merge env-file values with process env. Process env wins.
    Caller can pass extra_paths (e.g. test fixtures); they are checked
    *before* the standard fallbacks but still lose to os.environ."""
    merged: dict[str, str] = {}
    paths: list[Path] = list(extra_paths or [])
    paths.extend([HERMES_ENV_PATH, LOCAL_ENV_PATH])
    # Later entries fill in gaps left by earlier ones.
    for p in paths:
        for k, v in _parse_env_file(p).items():
            merged.setdefault(k, v)
    # Process env always wins.
    for k, v in os.environ.items():
        merged[k] = v
    return merged


def get_apify_token(env: dict[str, str] | None = None) -> str:
    env = env if env is not None else load_env()
    token = env.get("APIFY_TOKEN", "").strip()
    if not token:
        raise RuntimeError(
            "APIFY_TOKEN not found. Set it in the environment or write it to "
            "/root/.hermes/.env (or ./.env for local dev)."
        )
    return token


@dataclass(frozen=True)
class Paths:
    data_dir: Path
    db_path: Path
    tmp_videos_dir: Path
    tmp_frames_dir: Path
    contact_sheets_dir: Path

    def ensure(self) -> None:
        for p in (
            self.data_dir,
            self.tmp_videos_dir,
            self.tmp_frames_dir,
            self.contact_sheets_dir,
        ):
            p.mkdir(parents=True, exist_ok=True)


def resolve_paths(
    data_dir: Path | str | None = None,
    db_path: Path | str | None = None,
    env: dict[str, str] | None = None,
) -> Paths:
    env = env if env is not None else load_env()
    data_dir_p = Path(
        data_dir
        or env.get("EMOTION_RADAR_DATA_DIR")
        or DEFAULT_DATA_DIR
    )
    db_path_p = Path(
        db_path
        or env.get("EMOTION_RADAR_DB")
        or (data_dir_p / DEFAULT_DB_FILENAME)
    )
    return Paths(
        data_dir=data_dir_p,
        db_path=db_path_p,
        tmp_videos_dir=data_dir_p / "tmp" / "videos",
        tmp_frames_dir=data_dir_p / "tmp" / "frames",
        contact_sheets_dir=data_dir_p / "contact_sheets",
    )


def enforce_url_cap(
    urls: list[str],
    confirm_large: bool,
    max_urls: int = DEFAULT_MAX_URLS,
) -> None:
    """Raise if too many URLs and the user has not explicitly confirmed."""
    if len(urls) > max_urls and not confirm_large:
        raise ValueError(
            f"Refusing to analyze {len(urls)} URLs in one command. "
            f"Default cap is {max_urls} (Apify cost safety). "
            f"Pass --confirm-large to override."
        )
