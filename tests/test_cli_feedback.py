"""CLI integration tests for Phase 6 commands:
list-scenes, rate-scene, list-feedback, taste-summary.

Plus: analyze-link three-pass default + --no-specificity escape +
SPECIFIC HOOK SCENES section in the final summary.

All network is mocked."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from click.testing import CliRunner

from emotion_radar import cli as cli_mod
from emotion_radar import db as db_mod
from emotion_radar.cli import cli

# Reuse the rich mocking helpers from test_cli_analyze_link so we don't
# duplicate Apify/video/sheet plumbing.
from tests.test_cli_analyze_link import (  # type: ignore[no-redef]
    DEFAULT_PASS3,
    OLIVER_URL,
    OLIVER_VIDEO_ID,
    PASS1_OLIVER_GOOD,
    PASS2_OLIVER_GOOD,
    _patch_providers,
    mock_infrastructure,  # fixture re-export
)


def _invoke(tmp_path: Path, *args: str):
    runner = CliRunner()
    return runner.invoke(
        cli, ["--data-dir", str(tmp_path), *args], catch_exceptions=False,
    )


def _only_report_id(tmp_path: Path) -> str:
    from emotion_radar.db import list_reports
    rows = list_reports(tmp_path / "emotion_radar.db", limit=10)
    assert len(rows) == 1, f"expected exactly one report, got {len(rows)}"
    return rows[0]["id"]


# ---- analyze-link three-pass default --------------------------------------

def test_analyze_link_default_runs_three_pass(mock_infrastructure, monkeypatch, tmp_path: Path):
    _patch_providers(
        monkeypatch,
        pass1_text=json.dumps(PASS1_OLIVER_GOOD),
        pass2_text=json.dumps(PASS2_OLIVER_GOOD),
        # Pass 3 default DEFAULT_PASS3 is fine.
    )
    result = _invoke(tmp_path, "analyze-link", OLIVER_URL, "--skip-evaluation")
    assert result.exit_code == 0, result.output
    # Three-pass labels appear.
    assert "Pass 1 (visual event)" in result.output
    assert "Pass 2 (hook strategy)" in result.output
    assert "Pass 3 (specificity)" in result.output

    rid = _only_report_id(tmp_path)
    row = db_mod.get_report(tmp_path / "emotion_radar.db", rid)
    assert row["raw_analysis"]["analysis_mode"] == "three_pass"
    assert "specificity_pass" in row["raw_analysis"]


def test_analyze_link_no_specificity_falls_back_to_two_pass(mock_infrastructure, monkeypatch, tmp_path: Path):
    _patch_providers(
        monkeypatch,
        pass1_text=json.dumps(PASS1_OLIVER_GOOD),
        pass2_text=json.dumps(PASS2_OLIVER_GOOD),
    )
    result = _invoke(tmp_path, "analyze-link", OLIVER_URL, "--no-specificity", "--skip-evaluation")
    assert result.exit_code == 0, result.output
    assert "Pass 3 (specificity)" not in result.output
    rid = _only_report_id(tmp_path)
    row = db_mod.get_report(tmp_path / "emotion_radar.db", rid)
    assert row["raw_analysis"]["analysis_mode"] == "two_pass"


def test_final_summary_prints_specific_hook_scenes(mock_infrastructure, monkeypatch, tmp_path: Path):
    _patch_providers(
        monkeypatch,
        pass1_text=json.dumps(PASS1_OLIVER_GOOD),
        pass2_text=json.dumps(PASS2_OLIVER_GOOD),
    )
    result = _invoke(tmp_path, "analyze-link", OLIVER_URL, "--skip-evaluation")
    assert result.exit_code == 0, result.output
    # The new section header.
    assert "SPECIFIC HOOK SCENES" in result.output
    assert "main actionable output" in result.output
    # Scene concept names from DEFAULT_PASS3 land in output.
    assert "Receipt Wall, Pinned" in result.output
    assert "Receipt at the Stall" in result.output
    # Per-scene fields rendered.
    assert "first 2 seconds:" in result.output
    assert "onscreen text:" in result.output
    assert "social tension:" in result.output
    assert "comment impulse:" in result.output


# ---- list-scenes -----------------------------------------------------------

def _seed_three_pass_report(mock_infrastructure, monkeypatch, tmp_path: Path) -> str:
    _patch_providers(
        monkeypatch,
        pass1_text=json.dumps(PASS1_OLIVER_GOOD),
        pass2_text=json.dumps(PASS2_OLIVER_GOOD),
    )
    res = _invoke(tmp_path, "analyze-link", OLIVER_URL, "--skip-evaluation")
    assert res.exit_code == 0, res.output
    return _only_report_id(tmp_path)


def test_list_scenes_shows_indexes_and_names(mock_infrastructure, monkeypatch, tmp_path: Path):
    rid = _seed_three_pass_report(mock_infrastructure, monkeypatch, tmp_path)
    result = _invoke(tmp_path, "list-scenes", rid)
    assert result.exit_code == 0, result.output
    assert "[1] Receipt Wall, Pinned" in result.output
    assert "[2] Receipt at the Stall" in result.output
    assert "first 2 seconds:" in result.output
    assert "onscreen text:" in result.output
    assert "virality score:" in result.output


def test_list_scenes_handles_missing_specificity(mock_infrastructure, monkeypatch, tmp_path: Path):
    _patch_providers(
        monkeypatch,
        pass1_text=json.dumps(PASS1_OLIVER_GOOD),
        pass2_text=json.dumps(PASS2_OLIVER_GOOD),
    )
    res = _invoke(
        tmp_path, "analyze-link", OLIVER_URL,
        "--no-specificity", "--skip-evaluation",
    )
    assert res.exit_code == 0, res.output
    rid = _only_report_id(tmp_path)
    result = _invoke(tmp_path, "list-scenes", rid)
    assert result.exit_code == 0, result.output
    assert "no scene_concepts" in result.output


# ---- rate-scene ------------------------------------------------------------

def test_rate_scene_inserts_feedback(mock_infrastructure, monkeypatch, tmp_path: Path):
    rid = _seed_three_pass_report(mock_infrastructure, monkeypatch, tmp_path)
    result = _invoke(
        tmp_path, "rate-scene", rid, "1",
        "--rating", "fire", "--note", "tactile evidence; great",
    )
    assert result.exit_code == 0, result.output
    assert "Recorded feedback" in result.output
    # Verify it landed in the DB.
    rows = db_mod.list_feedback(tmp_path / "emotion_radar.db")
    assert len(rows) == 1
    assert rows[0]["rating"] == "fire"
    assert rows[0]["concept_index"] == 1
    assert rows[0]["concept_name"] == "Receipt Wall, Pinned"
    assert rows[0]["report_id"] == rid
    assert rows[0]["note"] == "tactile evidence; great"


def test_rate_scene_rejects_invalid_rating(mock_infrastructure, monkeypatch, tmp_path: Path):
    rid = _seed_three_pass_report(mock_infrastructure, monkeypatch, tmp_path)
    runner = CliRunner()
    result = runner.invoke(
        cli, ["--data-dir", str(tmp_path),
              "rate-scene", rid, "1", "--rating", "amazing"],
        catch_exceptions=False,
    )
    # Click rejects bad choice; exit code is non-zero.
    assert result.exit_code != 0
    assert "amazing" in result.output or "invalid" in result.output.lower()


def test_rate_scene_rejects_out_of_range_index(mock_infrastructure, monkeypatch, tmp_path: Path):
    rid = _seed_three_pass_report(mock_infrastructure, monkeypatch, tmp_path)
    result = _invoke(tmp_path, "rate-scene", rid, "99", "--rating", "fire")
    assert result.exit_code != 0
    assert "out of range" in result.output


# ---- list-feedback ---------------------------------------------------------

def test_list_feedback_empty(mock_infrastructure, monkeypatch, tmp_path: Path):
    # Just init the data dir without running anything.
    result = _invoke(tmp_path, "list-feedback")
    assert result.exit_code == 0, result.output
    assert "no feedback yet" in result.output


def test_list_feedback_shows_rows(mock_infrastructure, monkeypatch, tmp_path: Path):
    rid = _seed_three_pass_report(mock_infrastructure, monkeypatch, tmp_path)
    _invoke(tmp_path, "rate-scene", rid, "1", "--rating", "fire", "--note", "great")
    _invoke(tmp_path, "rate-scene", rid, "2", "--rating", "cringe")
    result = _invoke(tmp_path, "list-feedback")
    assert result.exit_code == 0, result.output
    assert "[fire" in result.output
    assert "[cringe" in result.output
    assert "Receipt Wall, Pinned" in result.output
    assert "Receipt at the Stall" in result.output


# ---- taste-summary ---------------------------------------------------------

def test_taste_summary_empty_when_no_feedback(mock_infrastructure, monkeypatch, tmp_path: Path):
    result = _invoke(tmp_path, "taste-summary")
    assert result.exit_code == 0, result.output
    assert "no feedback yet" in result.output


def test_taste_summary_groups_feedback(mock_infrastructure, monkeypatch, tmp_path: Path):
    rid = _seed_three_pass_report(mock_infrastructure, monkeypatch, tmp_path)
    _invoke(tmp_path, "rate-scene", rid, "1", "--rating", "fire", "--note", "tactile evidence is strong")
    _invoke(tmp_path, "rate-scene", rid, "2", "--rating", "cringe", "--note", "too polished")
    result = _invoke(tmp_path, "taste-summary")
    assert result.exit_code == 0, result.output
    assert "User tends to like:" in result.output
    assert "Receipt Wall, Pinned" in result.output
    assert "User dislikes:" in result.output
    assert "Receipt at the Stall" in result.output
    assert "Recent notes:" in result.output
    assert "tactile evidence is strong" in result.output
    assert "too polished" in result.output


# ---- Pass 3 conditioning via stored taste ---------------------------------

def test_pass3_user_prompt_includes_taste_summary_when_feedback_exists(
    mock_infrastructure, monkeypatch, tmp_path: Path,
):
    """Seed one report + feedback. Then run analyze-report (three-pass)
    and verify the strategy provider's Pass-3 call sees the taste
    summary in its user prompt."""
    # Step 1: seed a report with scene_concepts.
    rid = _seed_three_pass_report(mock_infrastructure, monkeypatch, tmp_path)
    _invoke(tmp_path, "rate-scene", rid, "1", "--rating", "fire",
            "--note", "believable public tension is what works")

    # Step 2: re-run analyze-report with a fresh provider mock that
    # records all text-call user prompts.
    captured_text_calls: list[str] = []

    class _RecordingMP:
        name = "mock"
        def __init__(self, label, image, texts):
            self.model = label
            self._image = image
            self._texts = list(texts)
            self._i = 0
        def analyze_image(self, *_a, **_kw):
            return self._image
        def analyze_text(self, system, user):
            captured_text_calls.append(user)
            if self._i < len(self._texts):
                r = self._texts[self._i]; self._i += 1; return r
            return self._texts[-1]

    def _fake_build(env, role):
        if role == "vision_event":
            return _RecordingMP("v", json.dumps(PASS1_OLIVER_GOOD), [""])
        return _RecordingMP("s", "",
            [json.dumps(PASS2_OLIVER_GOOD), json.dumps(DEFAULT_PASS3)])

    monkeypatch.setattr(cli_mod, "build_provider_for_role", _fake_build)
    res = _invoke(tmp_path, "analyze-report", rid)
    assert res.exit_code == 0, res.output

    # The last text call is Pass 3; the one before is Pass 2.
    assert len(captured_text_calls) >= 2
    pass3_user = captured_text_calls[-1]
    assert "USER TASTE PROFILE" in pass3_user
    assert "believable public tension is what works" in pass3_user
    assert "Receipt Wall, Pinned" in pass3_user
