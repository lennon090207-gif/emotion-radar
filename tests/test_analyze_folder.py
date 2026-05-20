"""analyze-folder (Phase 7) CLI integration tests. All mocked."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from click.testing import CliRunner

from emotion_radar import cli as cli_mod
from emotion_radar.cli import cli
from emotion_radar.db import get_report, list_reports

from tests.test_cli_analyze_link import (  # type: ignore[no-redef]
    DEFAULT_PASS3,
    PASS1_OLIVER_GOOD,
    PASS2_OLIVER_GOOD,
    _patch_providers,
)
from tests.test_analyze_file import (  # type: ignore[no-redef]
    mock_local_infrastructure,  # fixture re-export
    _make_fake_video,
)


def _invoke(tmp_path: Path, *args: str):
    runner = CliRunner()
    return runner.invoke(
        cli, ["--data-dir", str(tmp_path), *args], catch_exceptions=False,
    )


# ---- _find_video_files unit tests -----------------------------------------

def test_find_video_files_filters_to_supported_extensions(tmp_path: Path):
    folder = tmp_path / "drive"
    folder.mkdir()
    (folder / "a.mp4").write_bytes(b"x")
    (folder / "b.mov").write_bytes(b"x")
    (folder / "c.m4v").write_bytes(b"x")
    (folder / "d.webm").write_bytes(b"x")
    (folder / "e.avi").write_bytes(b"x")     # not supported
    (folder / "f.txt").write_bytes(b"x")     # not supported
    (folder / "g.MP4").write_bytes(b"x")     # case-insensitive match
    matched = cli_mod._find_video_files(folder)
    names = {p.name for p in matched}
    assert names == {"a.mp4", "b.mov", "c.m4v", "d.webm", "g.MP4"}


def test_find_video_files_sorts_alphabetically(tmp_path: Path):
    folder = tmp_path / "drive"
    folder.mkdir()
    for name in ("zeta.mp4", "alpha.mp4", "middle.mp4"):
        (folder / name).write_bytes(b"x")
    matched = cli_mod._find_video_files(folder)
    assert [p.name for p in matched] == ["alpha.mp4", "middle.mp4", "zeta.mp4"]


def test_find_video_files_non_recursive_by_default(tmp_path: Path):
    folder = tmp_path / "drive"
    sub = folder / "nested"
    sub.mkdir(parents=True)
    (folder / "top.mp4").write_bytes(b"x")
    (sub / "deeper.mp4").write_bytes(b"x")
    flat = cli_mod._find_video_files(folder)
    assert [p.name for p in flat] == ["top.mp4"]
    rec = cli_mod._find_video_files(folder, recursive=True)
    assert {p.name for p in rec} == {"top.mp4", "deeper.mp4"}


# ---- analyze-folder behavior ----------------------------------------------

def _make_folder(tmp_path: Path, names: list[str]) -> Path:
    folder = tmp_path / "drive"
    folder.mkdir()
    for n in names:
        (folder / n).write_bytes(b"fake")
    return folder


def test_analyze_folder_picks_up_only_supported_files(
    mock_local_infrastructure, monkeypatch, tmp_path: Path,
):
    folder = _make_folder(tmp_path, [
        "a.mp4", "b.txt", "c.mov", "d.avi", "e.webm",
    ])
    _patch_providers(
        monkeypatch,
        pass1_text=json.dumps(PASS1_OLIVER_GOOD),
        pass2_text=json.dumps(PASS2_OLIVER_GOOD),
    )
    result = _invoke(tmp_path, "analyze-folder", str(folder))
    assert result.exit_code == 0, result.output
    # 3 supported files; default limit is 5 so all three are processed.
    assert "Found 3 video file(s)" in result.output
    assert "analyzing 3" in result.output
    # The .avi / .txt names are NOT picked up.
    rows = list_reports(tmp_path / "emotion_radar.db", limit=20)
    names = {r["raw_analysis"]["source_metadata"]["source_filename"] for r in rows}
    assert names == {"a.mp4", "c.mov", "e.webm"}


def test_analyze_folder_respects_default_limit(
    mock_local_infrastructure, monkeypatch, tmp_path: Path,
):
    folder = _make_folder(tmp_path, [f"clip_{i:02d}.mp4" for i in range(10)])
    _patch_providers(
        monkeypatch,
        pass1_text=json.dumps(PASS1_OLIVER_GOOD),
        pass2_text=json.dumps(PASS2_OLIVER_GOOD),
    )
    result = _invoke(tmp_path, "analyze-folder", str(folder))
    assert result.exit_code == 0, result.output
    assert "Found 10 video file(s)" in result.output
    assert "analyzing 5" in result.output  # default limit
    rows = list_reports(tmp_path / "emotion_radar.db", limit=20)
    assert len(rows) == 5


def test_analyze_folder_explicit_limit_prints_cost_warning(
    mock_local_infrastructure, monkeypatch, tmp_path: Path,
):
    folder = _make_folder(tmp_path, [f"clip_{i:02d}.mp4" for i in range(10)])
    _patch_providers(
        monkeypatch,
        pass1_text=json.dumps(PASS1_OLIVER_GOOD),
        pass2_text=json.dumps(PASS2_OLIVER_GOOD),
    )
    result = _invoke(tmp_path, "analyze-folder", str(folder), "--limit", "8")
    assert result.exit_code == 0, result.output
    assert "WARNING" in result.output
    assert "exceeds the safe default" in result.output
    rows = list_reports(tmp_path / "emotion_radar.db", limit=20)
    assert len(rows) == 8


def test_analyze_folder_no_files_raises(
    mock_local_infrastructure, monkeypatch, tmp_path: Path,
):
    folder = _make_folder(tmp_path, ["readme.txt"])
    result = _invoke(tmp_path, "analyze-folder", str(folder))
    assert result.exit_code != 0
    assert "No supported video files" in result.output


def test_analyze_folder_continues_after_per_file_failure(
    monkeypatch, tmp_path: Path,
):
    """One bad file must not kill the whole batch."""
    folder = _make_folder(tmp_path, ["good_a.mp4", "bad.mp4", "good_b.mp4"])

    class _ApifyMustNotBeCalled:
        def __init__(self, *a, **kw):
            raise AssertionError("Apify must not be called in analyze-folder")

    monkeypatch.setattr(cli_mod, "ApifyClient", _ApifyMustNotBeCalled)

    from emotion_radar.video import VideoError

    def _fake_extract(video_path, out_dir, timestamps):
        if Path(video_path).name == "bad.mp4":
            raise VideoError("simulated frame extraction failure")
        out_dir = Path(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        ps = []
        for ts in timestamps:
            p = out_dir / f"t{ts:0.2f}.jpg"
            p.write_bytes(b"\xff\xd8\xff\xe0")
            ps.append(p)
        return ps

    def _fake_sheet(frame_paths, timestamps, out_path, **kw):
        out_path = Path(out_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_bytes(b"\xff\xd8\xff\xe0")
        return out_path

    monkeypatch.setattr(cli_mod, "extract_frames", _fake_extract)
    monkeypatch.setattr(cli_mod, "build_contact_sheet", _fake_sheet)

    _patch_providers(
        monkeypatch,
        pass1_text=json.dumps(PASS1_OLIVER_GOOD),
        pass2_text=json.dumps(PASS2_OLIVER_GOOD),
    )

    result = _invoke(tmp_path, "analyze-folder", str(folder))
    assert result.exit_code == 0, result.output  # batch survives
    assert "BATCH SUMMARY" in result.output
    # The bad file is recorded as a row with error set; good ones run all passes.
    rows = list_reports(tmp_path / "emotion_radar.db", limit=20)
    assert len(rows) == 3
    by_name = {r["raw_analysis"]["source_metadata"]["source_filename"]: r for r in rows}
    assert by_name["bad.mp4"]["error"] is not None
    assert by_name["good_a.mp4"]["error"] is None
    assert by_name["good_b.mp4"]["error"] is None
    # The good files completed the full three-pass.
    assert by_name["good_a.mp4"]["raw_analysis"]["analysis_mode"] == "three_pass"


def test_analyze_folder_batch_summary_lists_report_ids_and_flows(
    mock_local_infrastructure, monkeypatch, tmp_path: Path,
):
    folder = _make_folder(tmp_path, ["alpha.mp4", "beta.mp4"])
    _patch_providers(
        monkeypatch,
        pass1_text=json.dumps(PASS1_OLIVER_GOOD),
        pass2_text=json.dumps(PASS2_OLIVER_GOOD),
    )
    result = _invoke(tmp_path, "analyze-folder", str(folder))
    assert result.exit_code == 0, result.output
    assert "BATCH SUMMARY" in result.output
    assert "Analyzed: 2" in result.output
    assert "Failed:   0" in result.output
    assert "Report IDs:" in result.output
    assert "Dominant story flows:" in result.output
    # Each Pass-2 mock declares the same dominant flow → count = 2.
    assert "public_disrespect_viewer_defense: 2" in result.output
    assert "Viral mechanics:" in result.output


def test_analyze_folder_does_not_auto_evaluate_per_file(
    mock_local_infrastructure, monkeypatch, tmp_path: Path,
):
    """Auto-fixture lookup must be off in batch mode — local seed
    clips don't have analytics and certainly don't match the Oliver
    video_id. Make sure no 'Evaluation' section spams the output."""
    folder = _make_folder(tmp_path, ["alpha.mp4", "beta.mp4"])
    _patch_providers(
        monkeypatch,
        pass1_text=json.dumps(PASS1_OLIVER_GOOD),
        pass2_text=json.dumps(PASS2_OLIVER_GOOD),
    )
    result = _invoke(tmp_path, "analyze-folder", str(folder))
    assert result.exit_code == 0, result.output
    # The per-file "--skip-evaluation" notice from analyze-link should
    # NOT appear (we don't pass --skip-evaluation; analyze-folder
    # internally suppresses auto-fixture lookup instead).
    assert "calibration check skipped" not in result.output
    assert "Evaluation" not in result.output


def test_analyze_folder_processes_files_in_sorted_order(
    mock_local_infrastructure, monkeypatch, tmp_path: Path,
):
    folder = _make_folder(tmp_path, ["zeta.mp4", "alpha.mp4", "middle.mp4"])
    _patch_providers(
        monkeypatch,
        pass1_text=json.dumps(PASS1_OLIVER_GOOD),
        pass2_text=json.dumps(PASS2_OLIVER_GOOD),
    )
    result = _invoke(tmp_path, "analyze-folder", str(folder))
    assert result.exit_code == 0, result.output
    # alpha must appear before middle, and middle before zeta in the output.
    alpha_pos = result.output.find("alpha.mp4")
    middle_pos = result.output.find("middle.mp4")
    zeta_pos = result.output.find("zeta.mp4")
    assert 0 <= alpha_pos < middle_pos < zeta_pos
