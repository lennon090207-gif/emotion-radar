"""analyze-link CLI integration tests.

All network is mocked: ApifyClient, download_video, extract_frames,
build_contact_sheet, and the vision providers. No real Apify run,
no real vision API call, no real ffmpeg/PIL invocation."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest
from click.testing import CliRunner

from emotion_radar import cli as cli_mod
from emotion_radar.cli import cli
from emotion_radar.db import get_report
from emotion_radar.models import ApifyRunInfo


# ---- fixtures --------------------------------------------------------------

OLIVER_VIDEO_ID = "7623559389307211030"
OLIVER_URL = "https://www.tiktok.com/@olivermakesartt/video/7623559389307211030"


def _fake_apify_item(video_id: str = OLIVER_VIDEO_ID, url: str = OLIVER_URL) -> dict:
    return {
        "id": video_id,
        "webVideoUrl": url,
        "text": "please be honest, how are they?",
        "mediaUrls": [f"https://api.apify.com/v2/key-value-stores/abc/records/{video_id}.mp4"],
        "videoMeta": {
            "duration": 14.0,
            "downloadAddr": "https://fallback/dl.mp4",
            "coverUrl": "https://cdn/cover.jpg",
        },
        "authorMeta": {"name": "olivermakesartt", "nickName": "Oliver"},
        "playCount": 12345,
        "diggCount": 67,
        "commentCount": 5,
        "shareCount": 2,
        "collectCount": 1,
    }


def _fake_run_info() -> ApifyRunInfo:
    return ApifyRunInfo(
        run_id="RUN_FAKE",
        dataset_id="DS_FAKE",
        usage_total_usd=0.0083,
        charged_events={"count": 1},
    )


@pytest.fixture
def mock_infrastructure(monkeypatch, tmp_path: Path):
    """Mock everything below the CLI layer: token, Apify, download,
    frames, contact sheet. Returns control handles for tests to tweak."""
    state: dict[str, Any] = {
        "item": _fake_apify_item(),
        "run_info": _fake_run_info(),
    }

    # 1. APIFY_TOKEN — pretend it's set.
    monkeypatch.setattr(cli_mod, "get_apify_token", lambda env=None: "apify_api_fake")

    # 2. ApifyClient — replace with a thin stub that returns our item.
    class _FakeApifyClient:
        def __init__(self, *a, **kw):
            pass

        def run_actor(self, urls, **kw):
            return [state["item"]], state["run_info"]

    monkeypatch.setattr(cli_mod, "ApifyClient", _FakeApifyClient)

    # 3. download_video — write a tiny placeholder file.
    def _fake_download(url, out_dir, video_id, **kw):
        out_dir = Path(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        p = out_dir / f"{video_id}.mp4"
        p.write_bytes(b"fake mp4 bytes")
        return p

    monkeypatch.setattr(cli_mod, "download_video", _fake_download)

    # 4. extract_frames — write tiny JPEG-ish placeholders.
    def _fake_extract(video_path, out_dir, timestamps):
        out_dir = Path(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        paths = []
        for ts in timestamps:
            p = out_dir / f"t{ts:0.2f}.jpg"
            p.write_bytes(b"\xff\xd8\xff\xe0frame")
            paths.append(p)
        return paths

    monkeypatch.setattr(cli_mod, "extract_frames", _fake_extract)

    # 5. build_contact_sheet — write a tiny placeholder JPEG and return path.
    def _fake_sheet(frame_paths, timestamps, out_path, **kw):
        out_path = Path(out_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_bytes(b"\xff\xd8\xff\xe0contact-sheet-bytes\xff\xd9")
        return out_path

    monkeypatch.setattr(cli_mod, "build_contact_sheet", _fake_sheet)

    return state


# ---- mock vision providers -------------------------------------------------

PASS1_OLIVER_GOOD = {
    "frame_observations": [
        {"timestamp": "0.0s", "observation": "outdoor market stall with handmade dragon lamps on display",
         "people_visible": "maker behind stall", "object_state": "lamps intact on table",
         "action_change_from_previous": ""},
        {"timestamp": "1.0s", "observation": "stranger picks up a dragon lamp",
         "people_visible": "maker, stranger", "object_state": "lamp in stranger's hand",
         "action_change_from_previous": "stranger entered frame and grabbed lamp"},
        {"timestamp": "2.0s", "observation": "dragon lamp lies smashed on the floor",
         "people_visible": "maker reacting, stranger",
         "object_state": "lamp on floor, broken",
         "action_change_from_previous": "lamp was thrown / dropped and smashed"},
    ],
    "environment": "outdoor weekend market stall",
    "people": "underdog maker (mid-30s); stranger acting as antagonist",
    "product_or_object": "handmade dragon lamp (HTTYD-style)",
    "onscreen_text": "Please be honest, how are they?",
    "physical_action": "thrown and smashed on the floor",
    "object_state_change": "lamp starts on the display table, ends on the floor with visible damage",
    "visual_conflict_detected": True,
    "conflict_type": "smash",
    "confidence": 0.92,
    "uncertainty_notes": "",
}

PASS2_OLIVER_GOOD = {
    "visual_hook_summary": (
        "At an outdoor market stall, a stranger picks up the maker's handmade "
        "dragon lamp and throws it on the floor, where it is smashed; the "
        "mechanic is public disrespect + underdog maker triggering viewer defense."
    ),
    "emotional_mechanic": "public disrespect + underdog maker (viewer-defense instinct)",
    "viewer_role": "defender",
    "emotions_triggered": ["anger", "protectiveness", "sympathy"],
    "why_it_works": "viewer wants to step in and protect a small handmade seller",
    "cooked_parts_to_avoid": ["overly staged 'random stranger' framing"],
    "product_attachability_score": 0.78,
    "transferability_score": 0.66,
    "freshness_score": 0.71,
    "cooked_score": 0.34,
    "overall_opportunity_score": 0.74,
    "hook_mutations": [
        {"type": "safe", "idea": "Customer haggles aggressively over a handmade memorial portrait",
         "opening_scene": "wide stall shot, memorial portraits on easels",
         "onscreen_text": "she just asked me to do this for free",
         "product_niche_fit": "handmade memorial portraits / market stall",
         "why_it_might_work": "indignation engine; defender role",
         "cringe_or_cooked_risk": "tips into staged territory if acting is bad",
         "production_difficulty": "easy"},
    ],
}


def _patch_providers(monkeypatch, pass1_text: str, pass2_text: str):
    class _MP:
        name = "mock"

        def __init__(self, model_label, image_resp, text_resp):
            self.model = model_label
            self._image = image_resp
            self._text = text_resp

        def analyze_image(self, image_path, system, user):
            return self._image

        def analyze_text(self, system, user):
            return self._text

    def _fake_build(env, role):
        if role == "vision_event":
            return _MP("mock-vision-1", pass1_text, "")
        return _MP("mock-strategy-1", "", pass2_text)

    monkeypatch.setattr(cli_mod, "build_provider_for_role", _fake_build)


# ---- helpers ---------------------------------------------------------------

def _invoke(tmp_path: Path, *args: str):
    runner = CliRunner()
    return runner.invoke(
        cli,
        ["--data-dir", str(tmp_path), *args],
        catch_exceptions=False,
    )


def _only_report_id(tmp_path: Path) -> str:
    """Pull the report id of the single report inserted by the run."""
    from emotion_radar.db import list_reports
    rows = list_reports(tmp_path / "emotion_radar.db", limit=10)
    assert len(rows) == 1, f"expected exactly one report, got {len(rows)}"
    return rows[0]["id"]


# ---- analyze-link: --no-vision ---------------------------------------------

def test_no_vision_skips_provider_and_evaluation(mock_infrastructure, monkeypatch, tmp_path: Path):
    # Provider must NOT be built when --no-vision is set.
    def _explode(env, role):
        raise AssertionError("provider must not be built in --no-vision mode")
    monkeypatch.setattr(cli_mod, "build_provider_for_role", _explode)

    result = _invoke(tmp_path, "analyze-link", OLIVER_URL, "--no-vision")

    assert result.exit_code == 0, result.output
    assert "--no-vision: skipping" in result.output
    assert "Emotion Radar Report" in result.output
    # No calibration ran.
    assert "Evaluation" not in result.output

    # Report row exists, with no vision analysis.
    rid = _only_report_id(tmp_path)
    row = get_report(tmp_path / "emotion_radar.db", rid)
    assert row["video_id"] == OLIVER_VIDEO_ID
    assert row["visual_hook_summary"] is None
    assert row["emotional_mechanic"] is None


# ---- analyze-link: --dry-run-vision ----------------------------------------

def test_dry_run_vision_prints_both_prompts_and_skips_api(mock_infrastructure, monkeypatch, tmp_path: Path):
    def _explode(env, role):
        raise AssertionError("provider must not be built in --dry-run-vision mode")
    monkeypatch.setattr(cli_mod, "build_provider_for_role", _explode)

    result = _invoke(tmp_path, "analyze-link", OLIVER_URL, "--dry-run-vision")

    assert result.exit_code == 0, result.output
    assert "=== PASS 1 SYSTEM" in result.output
    assert "=== PASS 2 SYSTEM" in result.output
    assert "=== PASS 1 USER" in result.output
    assert "=== PASS 2 USER" in result.output
    assert "dry-run: no API call made" in result.output
    # Final report stub also prints.
    assert "Emotion Radar Report" in result.output


# ---- analyze-link: full two-pass + auto-evaluation (Oliver fixture) -------

def test_full_two_pass_auto_evaluates_oliver_and_passes(mock_infrastructure, monkeypatch, tmp_path: Path):
    _patch_providers(
        monkeypatch,
        pass1_text=json.dumps(PASS1_OLIVER_GOOD),
        pass2_text=json.dumps(PASS2_OLIVER_GOOD),
    )

    result = _invoke(tmp_path, "analyze-link", OLIVER_URL)

    assert result.exit_code == 0, result.output
    assert "Emotion Radar Report" in result.output
    # Pass 1 + Pass 2 both labeled in CLI output.
    assert "Pass 1 (visual event)" in result.output
    assert "Pass 2 (hook strategy)" in result.output
    # Auto-evaluation ran against the Oliver fixture.
    assert "Evaluation" in result.output
    assert "PASS" in result.output
    assert "auto, known video_id" in result.output
    # No calibration-failed warning.
    assert "Calibration failed" not in result.output

    # Report row got the merged fields written back.
    rid = _only_report_id(tmp_path)
    row = get_report(tmp_path / "emotion_radar.db", rid)
    assert row["emotional_mechanic"].startswith("public disrespect")
    assert row["onscreen_text"] == "Please be honest, how are they?"
    assert row["overall_opportunity_score"] == 0.74
    assert isinstance(row["raw_analysis"], dict)
    assert row["raw_analysis"]["analysis_mode"] == "two_pass"
    assert row["raw_analysis"]["visual_event_pass"]["conflict_type"] == "smash"


# ---- analyze-link: failed calibration ------------------------------------

def test_failed_calibration_prints_warning(mock_infrastructure, monkeypatch, tmp_path: Path):
    """Pass 2 produces a too-soft mechanic; the Oliver canary should fail
    and the CLI should print the explicit warning. Report still saved."""
    weak_pass2 = {
        **PASS2_OLIVER_GOOD,
        "visual_hook_summary": "The creator looks discouraged at his market stall.",
        "emotional_mechanic": "creator vulnerability",
    }
    weak_pass1 = {
        **PASS1_OLIVER_GOOD,
        "physical_action": "",
        "visual_conflict_detected": False,
        "conflict_type": "none",
        "object_state_change": "lamps stay on the table",
        "frame_observations": [
            {"timestamp": "0.0s", "observation": "market stall, creator looks at camera",
             "people_visible": "creator", "object_state": "lamps on table",
             "action_change_from_previous": ""},
        ],
        # remove "smashed", "thrown", "dragon lamp" etc.
        "product_or_object": "handmade lamps",
    }
    _patch_providers(
        monkeypatch,
        pass1_text=json.dumps(weak_pass1),
        pass2_text=json.dumps(weak_pass2),
    )

    result = _invoke(tmp_path, "analyze-link", OLIVER_URL)

    assert result.exit_code == 0, result.output  # warning, not crash
    assert "FAIL" in result.output
    assert "Calibration failed. Do not trust this report yet." in result.output
    # Report still saved.
    rid = _only_report_id(tmp_path)
    row = get_report(tmp_path / "emotion_radar.db", rid)
    assert row["emotional_mechanic"] == "creator vulnerability"


# ---- analyze-link: --skip-evaluation honors the flag ----------------------

def test_skip_evaluation_overrides_known_fixture(mock_infrastructure, monkeypatch, tmp_path: Path):
    _patch_providers(
        monkeypatch,
        pass1_text=json.dumps(PASS1_OLIVER_GOOD),
        pass2_text=json.dumps(PASS2_OLIVER_GOOD),
    )

    result = _invoke(tmp_path, "analyze-link", OLIVER_URL, "--skip-evaluation")

    assert result.exit_code == 0, result.output
    assert "Evaluation" not in result.output
    assert "skip-evaluation: calibration check skipped" in result.output


# ---- analyze-link: --expected SPEC ----------------------------------------

def test_explicit_expected_path_wins_over_auto_fixture(mock_infrastructure, monkeypatch, tmp_path: Path):
    _patch_providers(
        monkeypatch,
        pass1_text=json.dumps(PASS1_OLIVER_GOOD),
        pass2_text=json.dumps(PASS2_OLIVER_GOOD),
    )

    # A bespoke spec that demands a term NOT in the Oliver-good outputs.
    custom_spec = tmp_path / "custom_spec.json"
    custom_spec.write_text(json.dumps({
        "required_terms": ["mango",  # deliberately absent
                           "market stall"],  # present
    }), encoding="utf-8")

    result = _invoke(tmp_path, "analyze-link", OLIVER_URL,
                     "--expected", str(custom_spec))

    assert result.exit_code == 0, result.output
    assert "Evaluation" in result.output
    assert "FAIL" in result.output
    # Spec path printed in evaluation section.
    assert str(custom_spec) in result.output
    # NOT labeled as auto, because user passed --expected explicitly.
    assert "auto, known video_id" not in result.output


# ---- analyze-link: non-Oliver video, no auto-fixture ----------------------

def test_unknown_video_id_no_auto_evaluation(mock_infrastructure, monkeypatch, tmp_path: Path):
    mock_infrastructure["item"] = _fake_apify_item(
        video_id="99999999999999",
        url="https://www.tiktok.com/@x/video/99999999999999",
    )
    _patch_providers(
        monkeypatch,
        pass1_text=json.dumps(PASS1_OLIVER_GOOD),
        pass2_text=json.dumps(PASS2_OLIVER_GOOD),
    )

    result = _invoke(tmp_path, "analyze-link",
                     "https://www.tiktok.com/@x/video/99999999999999")

    assert result.exit_code == 0, result.output
    # No fixture for this id, so no evaluation runs.
    assert "Evaluation" not in result.output


# ---- analyze-link: final summary structure --------------------------------

def test_final_summary_groups_mutations_by_type(mock_infrastructure, monkeypatch, tmp_path: Path):
    pass2 = {
        **PASS2_OLIVER_GOOD,
        "hook_mutations": [
            {"type": "safe", "idea": "Safe idea A"},
            {"type": "safe", "idea": "Safe idea B"},
            {"type": "fresh", "idea": "Fresh idea A"},
            {"type": "fresh", "idea": "Fresh idea B"},
            {"type": "fresh", "idea": "Fresh idea C"},
            {"type": "big_swing", "idea": "Big swing idea"},
        ],
    }
    _patch_providers(
        monkeypatch,
        pass1_text=json.dumps(PASS1_OLIVER_GOOD),
        pass2_text=json.dumps(pass2),
    )
    result = _invoke(tmp_path, "analyze-link", OLIVER_URL, "--skip-evaluation")
    assert result.exit_code == 0, result.output
    # Each group label shows up with its count.
    assert "-- Safe (2) --" in result.output
    assert "-- Fresh (3) --" in result.output
    assert "-- Big Swing (1) --" in result.output


# ---- known fixture path sanity --------------------------------------------

def test_known_fixtures_dict_points_at_existing_file():
    """If someone moves or renames docs/examples/oliver_expected.json,
    analyze-link's auto-evaluation silently breaks. Catch that here."""
    assert OLIVER_VIDEO_ID in cli_mod.KNOWN_FIXTURES
    fixture_path = cli_mod.KNOWN_FIXTURES[OLIVER_VIDEO_ID]
    assert fixture_path.is_file(), f"shipped fixture missing on disk: {fixture_path}"
