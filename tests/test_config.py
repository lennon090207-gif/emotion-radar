from __future__ import annotations

from pathlib import Path

import pytest

from emotion_radar import config


def test_parse_env_file_basic(tmp_path: Path):
    env_file = tmp_path / ".env"
    env_file.write_text(
        "# comment line\n"
        "\n"
        "APIFY_TOKEN=abc123\n"
        "FOO=\"bar baz\"\n"
        "QUOTED_SINGLE='hello'\n"
        "EMPTY=\n",
        encoding="utf-8",
    )
    parsed = config._parse_env_file(env_file)
    assert parsed["APIFY_TOKEN"] == "abc123"
    assert parsed["FOO"] == "bar baz"
    assert parsed["QUOTED_SINGLE"] == "hello"
    assert parsed["EMPTY"] == ""


def test_load_env_prefers_process_env(tmp_path: Path, monkeypatch):
    env_file = tmp_path / ".env"
    env_file.write_text("APIFY_TOKEN=from_file\n", encoding="utf-8")
    monkeypatch.setenv("APIFY_TOKEN", "from_process")
    merged = config.load_env(extra_paths=[env_file])
    assert merged["APIFY_TOKEN"] == "from_process"


def test_load_env_falls_back_to_file(tmp_path: Path, monkeypatch):
    monkeypatch.delenv("APIFY_TOKEN", raising=False)
    env_file = tmp_path / ".env"
    env_file.write_text("APIFY_TOKEN=from_file\n", encoding="utf-8")
    merged = config.load_env(extra_paths=[env_file])
    assert merged["APIFY_TOKEN"] == "from_file"


def test_get_apify_token_raises_when_missing(monkeypatch):
    monkeypatch.delenv("APIFY_TOKEN", raising=False)
    # Force load_env to see no token.
    with pytest.raises(RuntimeError):
        config.get_apify_token(env={})


def test_enforce_url_cap_under_limit():
    config.enforce_url_cap(["a", "b", "c"], confirm_large=False)


def test_enforce_url_cap_over_limit_requires_confirm():
    with pytest.raises(ValueError):
        config.enforce_url_cap(["a", "b", "c", "d"], confirm_large=False)


def test_enforce_url_cap_over_limit_with_confirm():
    config.enforce_url_cap(["a", "b", "c", "d", "e"], confirm_large=True)


def test_resolve_paths_defaults(tmp_path: Path, monkeypatch):
    monkeypatch.delenv("EMOTION_RADAR_DB", raising=False)
    monkeypatch.delenv("EMOTION_RADAR_DATA_DIR", raising=False)
    paths = config.resolve_paths(data_dir=tmp_path, env={})
    assert paths.data_dir == tmp_path
    assert paths.db_path == tmp_path / "emotion_radar.db"
    assert paths.tmp_videos_dir == tmp_path / "tmp" / "videos"
    assert paths.tmp_frames_dir == tmp_path / "tmp" / "frames"
    assert paths.contact_sheets_dir == tmp_path / "contact_sheets"


def test_resolve_paths_ensure(tmp_path: Path):
    paths = config.resolve_paths(data_dir=tmp_path, env={})
    paths.ensure()
    assert paths.tmp_videos_dir.is_dir()
    assert paths.tmp_frames_dir.is_dir()
    assert paths.contact_sheets_dir.is_dir()
