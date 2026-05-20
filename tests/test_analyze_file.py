"""analyze-file (Phase 7) CLI integration tests. All network mocked.

Apify is never called for local files; we deliberately patch
ApifyClient to AssertionError so any leak into the local-file path
fails loudly."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from click.testing import CliRunner

from emotion_radar import cli as cli_mod
from emotion_radar.cli import cli
from emotion_radar.db import get_report, list_reports

# Re-use the heavyweight provider/Pass mocks from the analyze-link tests.
from tests.test_cli_analyze_link import (  # type: ignore[no-redef]
    DEFAULT_PASS3,
    PASS1_OLIVER_GOOD,
    PASS2_OLIVER_GOOD,
    _patch_providers,
)


# ---- shared fixtures -------------------------------------------------------

@pytest.fixture
def mock_local_infrastructure(monkeypatch, tmp_path: Path):
    """Mock the local-file half of the pipeline: extract_frames and
    build_contact_sheet write fake bytes. Apify must NOT be touched —
    we patch ApifyClient to explode if any code path instantiates it."""

    class _ApifyMustNotBeCalled:
        def __init__(self, *a, **kw):
            raise AssertionError(
                "analyze-file / analyze-folder must NOT instantiate ApifyClient"
            )

    monkeypatch.setattr(cli_mod, "ApifyClient", _ApifyMustNotBeCalled)

    def _fake_extract(video_path, out_dir, timestamps):
        out_dir = Path(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        paths_out = []
        for ts in timestamps:
            p = out_dir / f"t{ts:0.2f}.jpg"
            p.write_bytes(b"\xff\xd8\xff\xe0frame")
            paths_out.append(p)
        return paths_out

    def _fake_sheet(frame_paths, timestamps, out_path, **kw):
        out_path = Path(out_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_bytes(b"\xff\xd8\xff\xe0contact-sheet-bytes\xff\xd9")
        return out_path

    monkeypatch.setattr(cli_mod, "extract_frames", _fake_extract)
    monkeypatch.setattr(cli_mod, "build_contact_sheet", _fake_sheet)

    return {}


def _invoke(tmp_path: Path, *args: str):
    runner = CliRunner()
    return runner.invoke(
        cli, ["--data-dir", str(tmp_path), *args], catch_exceptions=False,
    )


def _make_fake_video(folder: Path, name: str, payload: bytes = b"fake mp4") -> Path:
    folder.mkdir(parents=True, exist_ok=True)
    p = folder / name
    p.write_bytes(payload)
    return p


# ---- analyze-file: no Apify, three-pass by default ------------------------

def test_analyze_file_does_not_touch_apify(mock_local_infrastructure, monkeypatch, tmp_path: Path):
    """If any code path tries to instantiate ApifyClient, the mock
    explodes. This test passes iff analyze-file stays local."""
    video = _make_fake_video(tmp_path / "videos", "seed_clip.mp4")
    _patch_providers(
        monkeypatch,
        pass1_text=json.dumps(PASS1_OLIVER_GOOD),
        pass2_text=json.dumps(PASS2_OLIVER_GOOD),
    )
    result = _invoke(tmp_path, "analyze-file", str(video), "--skip-evaluation")
    assert result.exit_code == 0, result.output
    assert "seed clip" in result.output
    assert "no Apify call" in result.output


def test_analyze_file_uses_three_pass_by_default(mock_local_infrastructure, monkeypatch, tmp_path: Path):
    video = _make_fake_video(tmp_path / "videos", "seed_clip.mp4")
    _patch_providers(
        monkeypatch,
        pass1_text=json.dumps(PASS1_OLIVER_GOOD),
        pass2_text=json.dumps(PASS2_OLIVER_GOOD),
    )
    result = _invoke(tmp_path, "analyze-file", str(video), "--skip-evaluation")
    assert result.exit_code == 0, result.output
    assert "Pass 1 (visual event)" in result.output
    assert "Pass 2 (hook strategy)" in result.output
    assert "Pass 3 (specificity)" in result.output
    assert "SPECIFIC HOOK SCENES" in result.output

    rows = list_reports(tmp_path / "emotion_radar.db", limit=10)
    assert len(rows) == 1
    row = rows[0]
    assert row["platform"] == "seed_clip"
    assert row["raw_analysis"]["analysis_mode"] == "three_pass"


def test_analyze_file_no_vision_skips_provider(mock_local_infrastructure, monkeypatch, tmp_path: Path):
    video = _make_fake_video(tmp_path / "videos", "seed_clip.mp4")

    def _explode(env, role):
        raise AssertionError("provider must not be built in --no-vision mode")

    monkeypatch.setattr(cli_mod, "build_provider_for_role", _explode)
    result = _invoke(tmp_path, "analyze-file", str(video), "--no-vision")
    assert result.exit_code == 0, result.output
    assert "--no-vision: skipping" in result.output


def test_analyze_file_no_specificity_falls_back_to_two_pass(
    mock_local_infrastructure, monkeypatch, tmp_path: Path,
):
    video = _make_fake_video(tmp_path / "videos", "seed_clip.mp4")
    _patch_providers(
        monkeypatch,
        pass1_text=json.dumps(PASS1_OLIVER_GOOD),
        pass2_text=json.dumps(PASS2_OLIVER_GOOD),
    )
    result = _invoke(
        tmp_path, "analyze-file", str(video),
        "--no-specificity", "--skip-evaluation",
    )
    assert result.exit_code == 0, result.output
    assert "Pass 3 (specificity)" not in result.output
    rows = list_reports(tmp_path / "emotion_radar.db", limit=10)
    assert rows[0]["raw_analysis"]["analysis_mode"] == "two_pass"


def test_analyze_file_dry_run_prints_all_three_prompts(
    mock_local_infrastructure, monkeypatch, tmp_path: Path,
):
    video = _make_fake_video(tmp_path / "videos", "seed_clip.mp4")

    def _explode(env, role):
        raise AssertionError("provider must not be built in --dry-run-vision")

    monkeypatch.setattr(cli_mod, "build_provider_for_role", _explode)
    result = _invoke(tmp_path, "analyze-file", str(video), "--dry-run-vision")
    assert result.exit_code == 0, result.output
    assert "=== PASS 1 SYSTEM" in result.output
    assert "=== PASS 2 SYSTEM" in result.output
    assert "=== PASS 3 SYSTEM" in result.output


# ---- analyze-file: extension validation -----------------------------------

@pytest.mark.parametrize("ext", [".mp4", ".mov", ".m4v", ".webm"])
def test_analyze_file_accepts_all_supported_extensions(
    mock_local_infrastructure, monkeypatch, tmp_path: Path, ext: str,
):
    video = _make_fake_video(tmp_path / "videos", f"seed_clip{ext}")
    _patch_providers(
        monkeypatch,
        pass1_text=json.dumps(PASS1_OLIVER_GOOD),
        pass2_text=json.dumps(PASS2_OLIVER_GOOD),
    )
    result = _invoke(tmp_path, "analyze-file", str(video), "--skip-evaluation")
    assert result.exit_code == 0, result.output


def test_analyze_file_rejects_unsupported_extension(
    mock_local_infrastructure, monkeypatch, tmp_path: Path,
):
    video = _make_fake_video(tmp_path / "videos", "seed_clip.avi")
    result = _invoke(tmp_path, "analyze-file", str(video))
    # ClickException → non-zero exit, no row inserted.
    assert result.exit_code != 0
    assert "Unsupported video extension" in result.output
    assert ".avi" in result.output
    rows = list_reports(tmp_path / "emotion_radar.db", limit=10)
    assert rows == []


def test_analyze_file_rejects_missing_path(
    mock_local_infrastructure, monkeypatch, tmp_path: Path,
):
    result = _invoke(tmp_path, "analyze-file", str(tmp_path / "does_not_exist.mp4"))
    assert result.exit_code != 0  # click.Path(exists=True) rejects it


# ---- analyze-file: source_metadata persistence ----------------------------

def test_source_metadata_set_after_ingestion(
    mock_local_infrastructure, monkeypatch, tmp_path: Path,
):
    video = _make_fake_video(tmp_path / "videos", "lobster_bag_drop.mp4")
    _patch_providers(
        monkeypatch,
        pass1_text=json.dumps(PASS1_OLIVER_GOOD),
        pass2_text=json.dumps(PASS2_OLIVER_GOOD),
    )
    result = _invoke(tmp_path, "analyze-file", str(video), "--skip-evaluation")
    assert result.exit_code == 0, result.output

    rows = list_reports(tmp_path / "emotion_radar.db", limit=10)
    row = rows[0]
    raw = row["raw_analysis"]
    assert isinstance(raw, dict)
    meta = raw.get("source_metadata")
    assert isinstance(meta, dict)
    assert meta["source_type"] == "drive_seed_clip"
    assert meta["source_filename"] == "lobster_bag_drop.mp4"
    assert meta["known_viral"] is True
    assert meta["analytics_available"] is False
    assert meta["original_local_path"] == str(video)


def test_source_metadata_survives_three_pass_overwrite(
    mock_local_infrastructure, monkeypatch, tmp_path: Path,
):
    """The three-pass merge rebuilds raw_analysis from scratch. We must
    not lose source_metadata when that happens — Phase 7's whole point
    is being able to tell seed clips apart afterwards."""
    video = _make_fake_video(tmp_path / "videos", "lobster_bag_drop.mp4")
    _patch_providers(
        monkeypatch,
        pass1_text=json.dumps(PASS1_OLIVER_GOOD),
        pass2_text=json.dumps(PASS2_OLIVER_GOOD),
    )
    result = _invoke(tmp_path, "analyze-file", str(video), "--skip-evaluation")
    assert result.exit_code == 0, result.output

    rows = list_reports(tmp_path / "emotion_radar.db", limit=10)
    raw = rows[0]["raw_analysis"]
    # Both the three-pass content AND the seed metadata are present.
    assert raw["analysis_mode"] == "three_pass"
    assert "visual_event_pass" in raw
    assert "hook_strategy_pass" in raw
    assert "specificity_pass" in raw
    meta = raw["source_metadata"]
    assert meta["source_type"] == "drive_seed_clip"
    assert meta["source_filename"] == "lobster_bag_drop.mp4"


def test_source_metadata_survives_two_pass_overwrite(
    mock_local_infrastructure, monkeypatch, tmp_path: Path,
):
    video = _make_fake_video(tmp_path / "videos", "lobster_bag_drop.mp4")
    _patch_providers(
        monkeypatch,
        pass1_text=json.dumps(PASS1_OLIVER_GOOD),
        pass2_text=json.dumps(PASS2_OLIVER_GOOD),
    )
    result = _invoke(
        tmp_path, "analyze-file", str(video),
        "--no-specificity", "--skip-evaluation",
    )
    assert result.exit_code == 0, result.output
    row = list_reports(tmp_path / "emotion_radar.db", limit=10)[0]
    raw = row["raw_analysis"]
    assert raw["analysis_mode"] == "two_pass"
    assert raw["source_metadata"]["source_type"] == "drive_seed_clip"


# ---- analyze-file: error handling -----------------------------------------

def test_frame_extraction_failure_lands_on_report_error(
    monkeypatch, tmp_path: Path,
):
    """If ffmpeg / extract_frames fails for a corrupt seed clip, the
    row is still inserted with error set — we should not silently
    drop the file."""
    from emotion_radar.video import VideoError

    class _ApifyMustNotBeCalled:
        def __init__(self, *a, **kw):
            raise AssertionError("Apify must not be called")

    monkeypatch.setattr(cli_mod, "ApifyClient", _ApifyMustNotBeCalled)

    def _fake_extract(video_path, out_dir, timestamps):
        raise VideoError("ffmpeg produced no frames (mocked)")

    monkeypatch.setattr(cli_mod, "extract_frames", _fake_extract)

    video = _make_fake_video(tmp_path / "videos", "corrupt_clip.mp4")
    result = _invoke(tmp_path, "analyze-file", str(video), "--skip-evaluation")
    assert result.exit_code == 0, result.output  # error printed, not crashed
    assert "Pipeline error" in result.output
    rows = list_reports(tmp_path / "emotion_radar.db", limit=10)
    assert len(rows) == 1
    assert rows[0]["error"]
    assert "ffmpeg produced no frames" in rows[0]["error"]
    # source_metadata still attached so we can identify the file later.
    assert rows[0]["raw_analysis"]["source_metadata"]["source_filename"] == "corrupt_clip.mp4"


# ---- slug helper ----------------------------------------------------------

# ---- Phase 7.1: relative path bug ------------------------------------------

def test_analyze_file_relative_path_no_value_error(
    mock_local_infrastructure, monkeypatch, tmp_path: Path,
):
    """Pre-fix this raised:
        ValueError: relative path can't be expressed as a file URI
    because _local_seed_report_stub called Path.as_uri() on a relative
    path. Resolving to absolute first fixes it."""
    video_dir = tmp_path / "videos"
    video_dir.mkdir()
    (video_dir / "rel.mp4").write_bytes(b"fake")

    _patch_providers(
        monkeypatch,
        pass1_text=json.dumps(PASS1_OLIVER_GOOD),
        pass2_text=json.dumps(PASS2_OLIVER_GOOD),
    )

    monkeypatch.chdir(tmp_path)
    result = _invoke(tmp_path, "analyze-file", "videos/rel.mp4", "--skip-evaluation")
    assert result.exit_code == 0, result.output
    rows = list_reports(tmp_path / "emotion_radar.db", limit=10)
    assert len(rows) == 1
    # submitted_url is a proper file:// URI with absolute path embedded.
    sub = rows[0]["submitted_url"]
    assert sub.startswith("file://")
    assert "rel.mp4" in sub
    # original_local_path is the absolute resolved form.
    meta = rows[0]["raw_analysis"]["source_metadata"]
    assert Path(meta["original_local_path"]).is_absolute()


# ---- Phase 7.1: --raw-output-on-parse-error --------------------------------

def test_raw_output_flag_saves_debug_file_on_pass1_parse_failure(
    mock_local_infrastructure, monkeypatch, tmp_path: Path,
):
    """Pass 1 returns garbage AND repair also fails. The CLI's
    --raw-output-on-parse-error flag should drop the original raw
    output to data/debug/model_outputs/ and surface a pass-labeled
    error."""
    video = _make_fake_video(tmp_path / "videos", "seed_clip.mp4")

    bad_pass1 = "This is definitely not JSON, the model went off the rails."

    class _MP:
        name = "mock"
        def __init__(self, label, image, text):
            self.model = label; self._image = image; self._text = text
        def analyze_image(self, *a, **kw):
            return self._image
        def analyze_text(self, *a, **kw):
            return self._text

    def _fake_build(env, role):
        if role == "vision_event":
            return _MP("v", bad_pass1, "")
        # strategy provider's analyze_text is the repair target; return
        # nonsense so repair ALSO fails and the original error surfaces.
        return _MP("s", "", "still nonsense")

    monkeypatch.setattr(cli_mod, "build_provider_for_role", _fake_build)

    result = _invoke(
        tmp_path, "analyze-file", str(video),
        "--skip-evaluation", "--raw-output-on-parse-error",
    )
    assert result.exit_code != 0  # propagates as ClickException
    # Pass-labeled error message.
    assert "Pass 1 visual event JSON parse failed" in result.output
    # Path hint in the error message.
    debug_root = tmp_path / "debug" / "model_outputs"
    assert "data/debug/model_outputs" in result.output.replace("\\", "/") or str(debug_root) in result.output
    # The debug file actually exists with the raw output.
    matches = list(debug_root.glob("*_pass1_visual_event.txt"))
    assert len(matches) == 1
    assert matches[0].read_text(encoding="utf-8") == bad_pass1


def test_raw_output_flag_off_does_not_save_debug_file(
    mock_local_infrastructure, monkeypatch, tmp_path: Path,
):
    """Without the flag, parse failures still raise (with a pass label)
    but no debug file is written."""
    video = _make_fake_video(tmp_path / "videos", "seed_clip.mp4")

    class _MP:
        name = "mock"
        def __init__(self, label, image, text):
            self.model = label; self._image = image; self._text = text
        def analyze_image(self, *a, **kw): return self._image
        def analyze_text(self, *a, **kw): return self._text

    def _fake_build(env, role):
        if role == "vision_event":
            return _MP("v", "garbage pass 1", "")
        return _MP("s", "", "still nonsense")

    monkeypatch.setattr(cli_mod, "build_provider_for_role", _fake_build)

    result = _invoke(tmp_path, "analyze-file", str(video), "--skip-evaluation")
    assert result.exit_code != 0
    assert "Pass 1 visual event JSON parse failed" in result.output
    # No debug directory should exist.
    debug_root = tmp_path / "debug" / "model_outputs"
    assert not debug_root.exists()


def test_pass2_parse_failure_labeled(
    mock_local_infrastructure, monkeypatch, tmp_path: Path,
):
    """Pass 1 succeeds, Pass 2 returns garbage → error says Pass 2."""
    video = _make_fake_video(tmp_path / "videos", "seed_clip.mp4")

    class _MP:
        name = "mock"
        def __init__(self, label, image, texts):
            self.model = label; self._image = image
            self._texts = list(texts); self._i = 0
        def analyze_image(self, *a, **kw): return self._image
        def analyze_text(self, *a, **kw):
            if self._i < len(self._texts):
                r = self._texts[self._i]; self._i += 1; return r
            return self._texts[-1]

    def _fake_build(env, role):
        if role == "vision_event":
            return _MP("v", json.dumps(PASS1_OLIVER_GOOD), [""])
        # Pass 2 returns garbage; repair also returns garbage.
        return _MP("s", "", ["pass2 is not json", "still not json"])

    monkeypatch.setattr(cli_mod, "build_provider_for_role", _fake_build)

    result = _invoke(
        tmp_path, "analyze-file", str(video),
        "--no-specificity", "--skip-evaluation",
    )
    assert result.exit_code != 0
    assert "Pass 2 hook strategy JSON parse failed" in result.output


def test_pass3_parse_failure_labeled(
    mock_local_infrastructure, monkeypatch, tmp_path: Path,
):
    """Pass 1 + Pass 2 succeed, Pass 3 fails → error says Pass 3."""
    video = _make_fake_video(tmp_path / "videos", "seed_clip.mp4")

    class _MP:
        name = "mock"
        def __init__(self, label, image, texts):
            self.model = label; self._image = image
            self._texts = list(texts); self._i = 0
        def analyze_image(self, *a, **kw): return self._image
        def analyze_text(self, *a, **kw):
            if self._i < len(self._texts):
                r = self._texts[self._i]; self._i += 1; return r
            return self._texts[-1]

    def _fake_build(env, role):
        if role == "vision_event":
            return _MP("v", json.dumps(PASS1_OLIVER_GOOD), [""])
        # Pass 2 fine; Pass 3 garbage; repair fails.
        return _MP("s", "", [
            json.dumps(PASS2_OLIVER_GOOD),
            "pass 3 garbage",
            "still garbage",
        ])

    monkeypatch.setattr(cli_mod, "build_provider_for_role", _fake_build)

    result = _invoke(tmp_path, "analyze-file", str(video), "--skip-evaluation")
    assert result.exit_code != 0
    assert "Pass 3 specificity JSON parse failed" in result.output


def test_slugify_filename_collapses_specials():
    assert cli_mod._slugify_filename("My Cool Clip (final)") == "My_Cool_Clip_final"
    assert cli_mod._slugify_filename("video.123") == "video_123"
    assert cli_mod._slugify_filename("") == "untitled"
    assert cli_mod._slugify_filename("____") == "untitled"
    # Length cap.
    long = "x" * 200
    assert len(cli_mod._slugify_filename(long)) == 80
