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
    assert "processing up to 5 new files" in result.output  # Phase 7.2 wording
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
    assert "processing up to 5 new files" in result.output  # Phase 7.2 wording
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
    # Phase 7.2: bucketed summary.
    assert "Analyzed (full):          2" in result.output
    assert "Partial (Pass 3 failed):  0" in result.output
    assert "Failed:                   0" in result.output
    assert "Full-success report IDs:" in result.output
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


# ---- Phase 7.1: relative-path bug ------------------------------------------

def test_analyze_folder_relative_path_no_value_error(
    mock_local_infrastructure, monkeypatch, tmp_path: Path,
):
    """Pre-fix: analyze-folder with a relative path raised
        ValueError: relative path can't be expressed as a file URI
    when _local_seed_report_stub tried .as_uri() on each file."""
    folder = _make_folder(tmp_path, ["a.mp4", "b.mp4"])

    _patch_providers(
        monkeypatch,
        pass1_text=json.dumps(PASS1_OLIVER_GOOD),
        pass2_text=json.dumps(PASS2_OLIVER_GOOD),
    )

    monkeypatch.chdir(tmp_path)
    result = _invoke(tmp_path, "analyze-folder", "drive")
    assert result.exit_code == 0, result.output
    rows = list_reports(tmp_path / "emotion_radar.db", limit=10)
    assert len(rows) == 2
    for r in rows:
        # Every submitted_url resolved to an absolute file:// URI.
        assert r["submitted_url"].startswith("file://")
        assert Path(r["raw_analysis"]["source_metadata"]["original_local_path"]).is_absolute()


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


# ============================================================================
# Phase 7.2: --skip-existing + --bank-fast + partial save + bucketed summary
# ============================================================================

def test_skip_existing_skips_files_already_processed_by_path(
    mock_local_infrastructure, monkeypatch, tmp_path: Path,
):
    """First run processes both files. Second run with --skip-existing
    should skip them both via submitted_url path match."""
    folder = _make_folder(tmp_path, ["alpha.mp4", "beta.mp4"])
    _patch_providers(
        monkeypatch,
        pass1_text=json.dumps(PASS1_OLIVER_GOOD),
        pass2_text=json.dumps(PASS2_OLIVER_GOOD),
    )
    # First pass: analyze both.
    first = _invoke(tmp_path, "analyze-folder", str(folder))
    assert first.exit_code == 0, first.output
    assert len(list_reports(tmp_path / "emotion_radar.db", limit=20)) == 2

    # Second pass with --skip-existing.
    second = _invoke(tmp_path, "analyze-folder", str(folder), "--skip-existing")
    assert second.exit_code == 0, second.output
    assert "skip alpha.mp4" in second.output
    assert "skip beta.mp4" in second.output
    assert "Skipped (already done):   2" in second.output
    # No new rows.
    assert len(list_reports(tmp_path / "emotion_radar.db", limit=20)) == 2


def test_skip_existing_filename_fallback(
    mock_local_infrastructure, monkeypatch, tmp_path: Path,
):
    """If the same filename is processed from a moved location, the
    filename fallback should still skip it."""
    # First location, first run.
    src1 = tmp_path / "drive_v1"
    src1.mkdir()
    (src1 / "shared_clip.mp4").write_bytes(b"fake")

    _patch_providers(
        monkeypatch,
        pass1_text=json.dumps(PASS1_OLIVER_GOOD),
        pass2_text=json.dumps(PASS2_OLIVER_GOOD),
    )
    res1 = _invoke(tmp_path, "analyze-folder", str(src1))
    assert res1.exit_code == 0
    assert len(list_reports(tmp_path / "emotion_radar.db", limit=20)) == 1

    # Now the "same" clip lives in a different folder; same filename.
    src2 = tmp_path / "drive_v2"
    src2.mkdir()
    (src2 / "shared_clip.mp4").write_bytes(b"fake")

    res2 = _invoke(tmp_path, "analyze-folder", str(src2), "--skip-existing")
    assert res2.exit_code == 0, res2.output
    assert "skip shared_clip.mp4" in res2.output
    # Still only one row in the DB.
    assert len(list_reports(tmp_path / "emotion_radar.db", limit=20)) == 1


def test_skip_existing_limit_counts_processed_not_skipped(
    mock_local_infrastructure, monkeypatch, tmp_path: Path,
):
    """If the first 3 sorted files already exist, --limit 2 --skip-existing
    should keep scanning until 2 NEW files are processed."""
    folder = _make_folder(tmp_path, [f"clip_{i:02d}.mp4" for i in range(8)])
    _patch_providers(
        monkeypatch,
        pass1_text=json.dumps(PASS1_OLIVER_GOOD),
        pass2_text=json.dumps(PASS2_OLIVER_GOOD),
    )
    # Pre-process the first 3 (sorted: clip_00, clip_01, clip_02).
    pre = _invoke(tmp_path, "analyze-folder", str(folder), "--limit", "3")
    assert pre.exit_code == 0
    assert len(list_reports(tmp_path / "emotion_radar.db", limit=20)) == 3

    # Now ask for 2 NEW files. With --skip-existing this should skip
    # clip_00..02 and process clip_03 and clip_04.
    res = _invoke(
        tmp_path, "analyze-folder", str(folder),
        "--limit", "2", "--skip-existing",
    )
    assert res.exit_code == 0, res.output
    assert "Skipped (already done):   3" in res.output
    assert "Analyzed (full):          2" in res.output
    rows = list_reports(tmp_path / "emotion_radar.db", limit=20)
    names = {r["raw_analysis"]["source_metadata"]["source_filename"] for r in rows}
    assert "clip_03.mp4" in names and "clip_04.mp4" in names
    # clip_05+ should NOT have been processed yet.
    assert "clip_05.mp4" not in names
    assert len(rows) == 5


def test_partial_save_pass3_failure_does_not_kill_batch(
    monkeypatch, tmp_path: Path,
):
    """Pass 3 failure: row is saved as two_pass with specificity_status
    'failed' and goes into the partial-success bucket — NOT failed."""
    folder = _make_folder(tmp_path, ["alpha.mp4", "beta.mp4"])

    class _ApifyMustNotBeCalled:
        def __init__(self, *a, **kw):
            raise AssertionError("Apify must not be called")

    monkeypatch.setattr(cli_mod, "ApifyClient", _ApifyMustNotBeCalled)

    def _fake_extract(video_path, out_dir, timestamps):
        out_dir = Path(out_dir); out_dir.mkdir(parents=True, exist_ok=True)
        ps = []
        for ts in timestamps:
            p = out_dir / f"t{ts:0.2f}.jpg"
            p.write_bytes(b"\xff\xd8\xff\xe0")
            ps.append(p)
        return ps

    def _fake_sheet(frame_paths, timestamps, out_path, **kw):
        Path(out_path).parent.mkdir(parents=True, exist_ok=True)
        Path(out_path).write_bytes(b"\xff\xd8\xff\xe0")
        return out_path

    monkeypatch.setattr(cli_mod, "extract_frames", _fake_extract)
    monkeypatch.setattr(cli_mod, "build_contact_sheet", _fake_sheet)

    # Pass 1 OK; Pass 2 OK; Pass 3 garbage; repair also garbage.
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
        # Per-file: Pass 2 ok, Pass 3 garbage, repair garbage. ×2 files.
        return _MP("s", "", [
            json.dumps(PASS2_OLIVER_GOOD), "pass3 bad", "still bad",
            json.dumps(PASS2_OLIVER_GOOD), "pass3 bad", "still bad",
        ])

    monkeypatch.setattr(cli_mod, "build_provider_for_role", _fake_build)

    result = _invoke(tmp_path, "analyze-folder", str(folder))
    assert result.exit_code == 0, result.output
    # Bucketed counts: both files are partial success.
    assert "Analyzed (full):          0" in result.output
    assert "Partial (Pass 3 failed):  2" in result.output
    assert "Failed:                   0" in result.output
    assert "Partial-success report IDs" in result.output  # full header includes "(Pass 1+2 banked, Pass 3 failed)"

    # Each row preserves Pass 1 + Pass 2 and is annotated as partial.
    rows = list_reports(tmp_path / "emotion_radar.db", limit=20)
    assert len(rows) == 2
    for r in rows:
        raw = r["raw_analysis"]
        assert raw["analysis_mode"] == "two_pass"
        assert raw["specificity_status"] == "failed"
        assert "Pass 3 specificity JSON parse failed" in raw["specificity_error"]
        assert raw["hook_strategy_pass"]["dominant_story_flow"] == "public_disrespect_viewer_defense"


def test_bank_fast_is_equivalent_to_no_specificity(
    mock_local_infrastructure, monkeypatch, tmp_path: Path,
):
    """--bank-fast should produce a two_pass row, identical in shape to
    what --no-specificity produces."""
    folder = _make_folder(tmp_path, ["alpha.mp4"])
    _patch_providers(
        monkeypatch,
        pass1_text=json.dumps(PASS1_OLIVER_GOOD),
        pass2_text=json.dumps(PASS2_OLIVER_GOOD),
    )
    result = _invoke(tmp_path, "analyze-folder", str(folder), "--bank-fast")
    assert result.exit_code == 0, result.output
    assert "Pass 3 (specificity)" not in result.output  # never ran
    rows = list_reports(tmp_path / "emotion_radar.db", limit=20)
    assert len(rows) == 1
    raw = rows[0]["raw_analysis"]
    assert raw["analysis_mode"] == "two_pass"
    # No specificity_status because Pass 3 was intentionally skipped.
    assert "specificity_status" not in raw
    # Counts as a full success (the user chose to bank, not a failure).
    assert "Analyzed (full):          1" in result.output
    assert "Partial (Pass 3 failed):  0" in result.output


def test_bank_fast_and_no_specificity_both_accepted_and_compose(
    mock_local_infrastructure, monkeypatch, tmp_path: Path,
):
    """Passing both flags must not error — they OR together."""
    folder = _make_folder(tmp_path, ["alpha.mp4"])
    _patch_providers(
        monkeypatch,
        pass1_text=json.dumps(PASS1_OLIVER_GOOD),
        pass2_text=json.dumps(PASS2_OLIVER_GOOD),
    )
    result = _invoke(
        tmp_path, "analyze-folder", str(folder),
        "--no-specificity", "--bank-fast",
    )
    assert result.exit_code == 0, result.output
    rows = list_reports(tmp_path / "emotion_radar.db", limit=20)
    assert rows[0]["raw_analysis"]["analysis_mode"] == "two_pass"


def test_layby_workflow_skip_existing_plus_bank_fast(
    mock_local_infrastructure, monkeypatch, tmp_path: Path,
):
    """End-to-end Phase 7.2 layby pattern: bank a few new clips fast,
    rerun safely with --skip-existing and bank a few more."""
    folder = _make_folder(tmp_path, [f"clip_{i:02d}.mp4" for i in range(6)])
    _patch_providers(
        monkeypatch,
        pass1_text=json.dumps(PASS1_OLIVER_GOOD),
        pass2_text=json.dumps(PASS2_OLIVER_GOOD),
    )
    # First layby batch: 2 new, bank-fast.
    r1 = _invoke(
        tmp_path, "analyze-folder", str(folder),
        "--limit", "2", "--skip-existing", "--bank-fast",
    )
    assert r1.exit_code == 0, r1.output
    assert "Analyzed (full):          2" in r1.output
    db = tmp_path / "emotion_radar.db"
    assert len(list_reports(db, limit=20)) == 2

    # Second layby batch: another 2 new, bank-fast. The first two are skipped.
    r2 = _invoke(
        tmp_path, "analyze-folder", str(folder),
        "--limit", "2", "--skip-existing", "--bank-fast",
    )
    assert r2.exit_code == 0, r2.output
    assert "Skipped (already done):   2" in r2.output
    assert "Analyzed (full):          2" in r2.output
    rows = list_reports(db, limit=20)
    assert len(rows) == 4
    # All four are two_pass; no specificity_status because bank-fast.
    for r in rows:
        raw = r["raw_analysis"]
        assert raw["analysis_mode"] == "two_pass"
        assert "specificity_status" not in raw
