"""Bank-only mode (Phase 7.3) — analysis-layer tests.

Covers:
  - BANK_EXTRACT_SYSTEM_PROMPT content (rules, library embedding, forbidden outputs),
  - build_bank_extract_user_prompt (Pass 1 embedded, optional taste),
  - extract_bank_concept makes exactly ONE text-only call,
  - analyze_bank_only orchestrates Pass 1 + bank extract (no full Pass 2/3),
  - build_bank_only_analysis_result maps top-level fields and stores
    raw_analysis.analysis_mode == "bank_only".

No real Apify, no real vision API."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from emotion_radar import analysis as A
from emotion_radar.models import AnalysisResult


# ---- Fixtures --------------------------------------------------------------

PASS1_GOOD = {
    "frame_observations": [
        {"timestamp": "0.0s", "observation": "market stall with dragon lamps"},
        {"timestamp": "1.5s", "observation": "stranger drops the lamp"},
        {"timestamp": "2.0s", "observation": "lamp broken on the ground"},
    ],
    "environment": "outdoor market stall",
    "people": "underdog maker; passerby antagonist",
    "product_or_object": "handmade dragon lamp",
    "onscreen_text": "Please be honest, how are they?",
    "physical_action": "dropped and broken on the ground",
    "object_state_change": "lamp on table -> lamp broken on ground",
    "visual_conflict_detected": True,
    "conflict_type": "drop",
    "confidence": 0.91,
    "uncertainty_notes": "",
}

BANK_GOOD = {
    "bank_concept": {
        "concept_name": "Stall Smash + Be Honest",
        "visual_hook_summary": (
            "At an outdoor market stall, a stranger drops a handmade dragon "
            "lamp and the lamp breaks on the ground while a 'please be "
            "honest' caption is on screen."
        ),
        "viral_mechanic": "public disrespect + underdog maker (viewer-defense)",
        "dominant_story_flow": "public_disrespect_viewer_defense",
        "matched_story_flows": [
            {"id": "public_disrespect_viewer_defense",
             "name": "Public Disrespect -> Viewer Defense",
             "confidence": 0.92,
             "why_matched": "stranger destroys handmade item with insult-text overlay"},
            {"id": "direct_viewer_plea_social_contract",
             "name": "Direct Viewer Plea -> Tiny Social Contract",
             "confidence": 0.4,
             "why_matched": "'please be honest' adds a plea framing"},
        ],
        "story_flow_steps_observed": [
            "stall on display",
            "passerby handles and drops lamp",
            "maker visibly reacts",
            "viewer positioned as defender",
        ],
        "viewer_role": "defender",
        "emotions_triggered": ["anger", "protectiveness"],
        "comment_trigger": "urge to verbally retaliate on behalf of the maker",
        "share_trigger": "send-to-friend for shared injustice",
        "why_it_works": "felt moral violation against an underdog self-casts the viewer as defender.",
        "scroll_stop_reason": "physical destruction of a handmade item in 2 seconds",
        "key_visual_pattern": "object goes from intact display to broken-on-ground",
        "key_text_pattern": "vulnerable-framing caption synchronized with destruction beat",
        "freshness_angle": "physical destruction tier of public-doubt mechanic",
        "cooked_elements": ["please be honest framing", "staged-stranger trope"],
        "ethical_risk_notes": "low; no protected-class vulnerability",
        "what_to_learn_from_it": "destruction + vulnerable-framing caption is the engine, not the product",
        "what_not_to_copy": "the exact 'please be honest' wording",
        "mutation_paths": [
            "swap maker for a child being defended",
            "swap destruction for verbal escalation",
            "swap public stall for online-marketplace screenshot",
        ],
        "scores": {
            "scroll_stop_strength_score": 0.86,
            "comment_likelihood_score": 0.82,
            "share_likelihood_score": 0.7,
            "viewer_role_strength_score": 0.83,
            "freshness_score": 0.7,
            "cooked_score": 0.35,
            "ethical_risk_score": 0.28,
            "virality_capability_score": 0.78,
        },
    },
}


class _MockProvider:
    name = "mock"
    model = "mock-1"

    def __init__(self, image_response: str = "", text_responses=()):
        self._image_response = image_response
        self._text_responses = list(text_responses)
        self._text_index = 0
        self.image_calls: list[dict[str, Any]] = []
        self.text_calls: list[dict[str, Any]] = []

    def analyze_image(self, image_path, system, user):
        self.image_calls.append({"image_path": image_path, "system": system, "user": user})
        return self._image_response

    def analyze_text(self, system, user):
        self.text_calls.append({"system": system, "user": user})
        if self._text_index < len(self._text_responses):
            r = self._text_responses[self._text_index]
            self._text_index += 1
            return r
        return self._text_responses[-1] if self._text_responses else "{}"


# ---- BANK_EXTRACT_SYSTEM_PROMPT content -----------------------------------

def test_bank_prompt_states_product_secondary_mechanic_primary():
    sp = A.BANK_EXTRACT_SYSTEM_PROMPT.lower()
    assert "product is secondary" in sp
    assert "mechanic is primary" in sp or "viral mechanic is" in sp


def test_bank_prompt_forbids_variations_pioneer_scenes():
    sp = A.BANK_EXTRACT_SYSTEM_PROMPT.lower()
    # The prompt names the things it must NOT produce.
    assert "variations" in sp
    assert "pioneer concepts" in sp or "pioneer_concepts" in sp
    assert "scene_concepts" in sp or "specific hook scenes" in sp
    # And the explicit "DO NOT produce" framing.
    assert "do not produce" in sp


def test_bank_prompt_describes_compact_record():
    sp = A.BANK_EXTRACT_SYSTEM_PROMPT
    # All required bank_concept schema keys appear in the prompt.
    for key in (
        "concept_name",
        "visual_hook_summary",
        "viral_mechanic",
        "dominant_story_flow",
        "matched_story_flows",
        "story_flow_steps_observed",
        "viewer_role",
        "emotions_triggered",
        "comment_trigger",
        "share_trigger",
        "why_it_works",
        "scroll_stop_reason",
        "key_visual_pattern",
        "key_text_pattern",
        "freshness_angle",
        "cooked_elements",
        "ethical_risk_notes",
        "what_to_learn_from_it",
        "what_not_to_copy",
        "mutation_paths",
    ):
        assert key in sp, f"bank prompt missing schema key: {key}"


def test_bank_prompt_lists_all_score_fields():
    sp = A.BANK_EXTRACT_SYSTEM_PROMPT
    for score in (
        "scroll_stop_strength_score",
        "comment_likelihood_score",
        "share_likelihood_score",
        "viewer_role_strength_score",
        "freshness_score",
        "cooked_score",
        "ethical_risk_score",
        "virality_capability_score",
    ):
        assert score in sp, f"bank prompt missing score field: {score}"


def test_bank_prompt_embeds_story_flow_library():
    """Bank-extract must see the same library names/ids as the full
    Pass 2; matching is done against the library here too."""
    sp = A.BANK_EXTRACT_SYSTEM_PROMPT
    for fid in (
        "public_disrespect_viewer_defense",
        "family_protection_validation",
        "moral_pressure_tiny_rescue",
        "comment_humiliation_public_witness",
        "stall_vulnerability_social_judgment",
        "wrong_audience_right_tribe",
        "shock_problem_immediate_fix",
        "ethical_edge_vulnerability_sympathy_surge",
        "direct_viewer_plea_social_contract",
        "weirdness_curiosity_reveal_loop",
    ):
        assert fid in sp, f"bank prompt missing story flow id: {fid}"


def test_bank_prompt_pass1_evidence_binding_carried_over():
    """The destruction+insult-text rule from Phase 3.1 must apply here
    too — otherwise the bank record can soften the mechanic."""
    sp = A.BANK_EXTRACT_SYSTEM_PROMPT.lower()
    assert "accidental" in sp
    assert "destruction" in sp or "destroyed" in sp
    assert "rejection" in sp or "insult" in sp
    # The conclusion phrase the rule produces.
    assert "public disrespect" in sp and "underdog maker" in sp


def test_bank_prompt_says_strict_json_only():
    sp = A.BANK_EXTRACT_SYSTEM_PROMPT
    assert "STRICT JSON only" in sp
    assert "No markdown fences" in sp


def test_bank_prompt_describes_mutation_paths_as_one_line_each():
    """Mutation paths must be SHORT — they're directions, not full ideas."""
    sp = A.BANK_EXTRACT_SYSTEM_PROMPT.lower()
    assert "mutation_paths" in sp
    assert "one line each" in sp or "broad directions" in sp


# ---- User prompt construction ---------------------------------------------

def test_user_prompt_embeds_pass1_evidence():
    up = A.build_bank_extract_user_prompt({}, PASS1_GOOD)
    assert "PASS 1 EVIDENCE" in up
    assert "dropped and broken on the ground" in up


def test_user_prompt_includes_source_metadata_when_present():
    metadata = {
        "platform": "seed_clip",
        "creator_username": "olivermakesartt",
        "raw_analysis": {
            "source_metadata": {
                "source_filename": "lobster_bag_drop.mp4",
                "source_type": "drive_seed_clip",
                "known_viral": True,
                "analytics_available": False,
            },
        },
    }
    up = A.build_bank_extract_user_prompt(metadata, PASS1_GOOD)
    assert "lobster_bag_drop.mp4" in up
    assert "known_viral" in up
    assert "analytics" in up


def test_user_prompt_omits_taste_section_when_none():
    up = A.build_bank_extract_user_prompt({}, PASS1_GOOD, taste_profile=None)
    assert "USER TASTE PROFILE" not in up


def test_user_prompt_includes_taste_section_when_provided():
    up = A.build_bank_extract_user_prompt(
        {}, PASS1_GOOD,
        taste_profile="User tends to like:\n  - believable public tension",
    )
    assert "USER TASTE PROFILE" in up
    assert "believable public tension" in up


# ---- extract_bank_concept (single text call) ------------------------------

def test_extract_bank_concept_uses_analyze_text_not_image():
    provider = _MockProvider(text_responses=[json.dumps(BANK_GOOD)])
    parsed = A.extract_bank_concept({}, PASS1_GOOD, provider)
    assert "bank_concept" in parsed
    assert parsed["bank_concept"]["dominant_story_flow"] == "public_disrespect_viewer_defense"
    assert provider.image_calls == []
    assert len(provider.text_calls) == 1
    assert provider.text_calls[0]["system"] == A.BANK_EXTRACT_SYSTEM_PROMPT


def test_extract_bank_concept_raises_with_bank_label_on_bad_json():
    provider = _MockProvider(text_responses=["definitely not json", "still bad"])
    with pytest.raises(ValueError) as ei:
        A.extract_bank_concept({}, PASS1_GOOD, provider)
    assert "Bank extract JSON parse failed" in str(ei.value)


# ---- analyze_bank_only orchestration --------------------------------------

def test_analyze_bank_only_runs_pass1_then_one_text_call(tmp_path: Path):
    sheet = tmp_path / "sheet.jpg"
    sheet.write_bytes(b"\xff\xd8\xff\xe0fake-sheet")
    vision = _MockProvider(image_response=json.dumps(PASS1_GOOD), text_responses=[""])
    strategy = _MockProvider(image_response="X", text_responses=[json.dumps(BANK_GOOD)])
    pass1, bank = A.analyze_bank_only(sheet, {}, vision, strategy)
    assert pass1["conflict_type"] == "drop"
    assert bank["bank_concept"]["dominant_story_flow"] == "public_disrespect_viewer_defense"
    # Pass 1 hits vision; bank extract is text-only.
    assert len(vision.image_calls) == 1
    assert vision.text_calls == []  # no full Pass 2; no Pass 3
    assert len(strategy.text_calls) == 1


def test_analyze_bank_only_never_invokes_full_pass2_or_pass3(tmp_path: Path):
    """Critical property: bank mode must NOT trigger generate_hook_strategy
    or run_specificity_pass. Each of those would be a second strategy
    call. Bank mode uses exactly one text call."""
    sheet = tmp_path / "sheet.jpg"
    sheet.write_bytes(b"\xff\xd8\xff\xe0")
    vision = _MockProvider(image_response=json.dumps(PASS1_GOOD), text_responses=[""])
    strategy = _MockProvider(text_responses=[json.dumps(BANK_GOOD)])
    A.analyze_bank_only(sheet, {}, vision, strategy)
    # Exactly one strategy text call (the bank extract).
    assert len(strategy.text_calls) == 1
    call = strategy.text_calls[0]
    # And it's the bank system prompt, not the full Pass 2 or Pass 3 prompt.
    assert call["system"] == A.BANK_EXTRACT_SYSTEM_PROMPT
    assert call["system"] != A.HOOK_STRATEGY_SYSTEM_PROMPT
    assert call["system"] != A.SPECIFICITY_SYSTEM_PROMPT


# ---- build_bank_only_analysis_result merge --------------------------------

def test_merge_maps_bank_concept_to_top_level_fields():
    result = A.build_bank_only_analysis_result(PASS1_GOOD, BANK_GOOD)
    assert isinstance(result, AnalysisResult)
    assert result.visual_hook_summary.startswith("At an outdoor market stall")
    assert result.onscreen_text == "Please be honest, how are they?"
    # viral_mechanic maps onto the legacy emotional_mechanic column.
    assert result.emotional_mechanic.startswith("public disrespect")
    assert result.viewer_role == "defender"
    assert result.emotions_triggered == ["anger", "protectiveness"]
    assert result.freshness_score == 0.7
    assert result.cooked_score == 0.35
    assert result.overall_opportunity_score == 0.78  # virality_capability_score
    # Bank mode intentionally produces no hook_mutations.
    assert result.hook_mutations == []


def test_merge_sets_raw_analysis_mode_bank_only():
    result = A.build_bank_only_analysis_result(PASS1_GOOD, BANK_GOOD)
    raw = result.raw_analysis
    assert raw["analysis_mode"] == "bank_only"
    assert raw["visual_event_pass"] == PASS1_GOOD
    assert raw["bank_concept"] == BANK_GOOD["bank_concept"]


def test_merge_handles_empty_bank_result():
    result = A.build_bank_only_analysis_result(PASS1_GOOD, {})
    assert result.raw_analysis["analysis_mode"] == "bank_only"
    assert result.raw_analysis["bank_concept"] == {}
    assert result.visual_hook_summary is None
    assert result.hook_mutations == []


def test_merge_handles_missing_scores_block():
    bank = {"bank_concept": {"concept_name": "x", "viral_mechanic": "y"}}
    result = A.build_bank_only_analysis_result(PASS1_GOOD, bank)
    # All score fields are None when the scores block is absent.
    assert result.freshness_score is None
    assert result.cooked_score is None
    assert result.overall_opportunity_score is None


# ============================================================================
# Phase 7.3: analyze-report upgrades a bank_only row by reusing Pass 1
# ============================================================================

def test_analyze_report_upgrades_bank_only_by_reusing_pass1(
    monkeypatch, tmp_path: Path,
):
    """End-to-end: seed a bank_only report, then run analyze-report
    REPORT_ID and verify it skips Pass 1 (reuses saved visual_event_pass)
    while still running Pass 2 + Pass 3. The vision provider should be
    instantiated but its analyze_image must NOT be called during the
    upgrade."""
    from click.testing import CliRunner
    from emotion_radar import cli as cli_mod
    from emotion_radar.cli import cli
    from emotion_radar.db import insert_report, get_report

    # Step 1: directly seed a bank_only report in the DB. Mirrors what
    # _ingest_local_video + _run_bank_only_and_update would produce,
    # without needing the contact-sheet plumbing.
    db_path = tmp_path / "emotion_radar.db"
    sheet_path = tmp_path / "contact_sheets" / "alpha.jpg"
    sheet_path.parent.mkdir(parents=True, exist_ok=True)
    sheet_path.write_bytes(b"\xff\xd8\xff\xe0fake-sheet")

    bank_only_row = {
        "platform": "seed_clip",
        "submitted_url": (tmp_path / "drive" / "alpha.mp4").resolve().as_uri(),
        "video_id": "alpha",
        "creator_username": None,
        "creator_nickname": None,
        "caption": None,
        "metrics": {"views": None, "likes": None, "comments": None,
                    "shares": None, "saves": None},
        "duration": None,
        "cover_url": None,
        "video_download_url_saved": False,
        "apify_run_id": None,
        "apify_dataset_id": None,
        "apify_usage_usd": None,
        "apify_charged_events": None,
        "contact_sheet_path": str(sheet_path),
        "visual_hook_summary": "stranger destroys lamp",
        "onscreen_text": "Please be honest, how are they?",
        "emotional_mechanic": "public disrespect + underdog maker",
        "viewer_role": "defender",
        "emotions_triggered": ["anger"],
        "freshness_score": 0.7,
        "cooked_score": 0.35,
        "overall_opportunity_score": 0.78,
        "hook_mutations": [],
        "raw_analysis": {
            "analysis_mode": "bank_only",
            "visual_event_pass": PASS1_GOOD,
            "bank_concept": BANK_GOOD["bank_concept"],
            "source_metadata": {
                "source_type": "drive_seed_clip",
                "source_filename": "alpha.mp4",
                "known_viral": True,
                "analytics_available": False,
                "original_local_path": "/tmp/drive/alpha.mp4",
            },
        },
        "error": None,
    }
    rid = insert_report(db_path, bank_only_row)

    # Step 2: mock providers. Vision provider's analyze_image MUST NOT
    # be called (the saved pass1 should be reused). Strategy provider
    # serves Pass 2 then Pass 3.
    from tests.test_cli_analyze_link import (
        DEFAULT_PASS3,
        PASS2_OLIVER_GOOD,
    )

    class _RecordingMP:
        name = "mock"
        def __init__(self, label, image, texts):
            self.model = label; self._image = image
            self._texts = list(texts); self._i = 0
            self.image_calls = 0
            self.text_calls = 0
        def analyze_image(self, *a, **kw):
            self.image_calls += 1
            return self._image
        def analyze_text(self, *a, **kw):
            self.text_calls += 1
            if self._i < len(self._texts):
                r = self._texts[self._i]; self._i += 1; return r
            return self._texts[-1]

    captured = {}
    def _fake_build(env, role):
        if role == "vision_event":
            captured["vision"] = _RecordingMP("v", json.dumps(PASS1_GOOD), [""])
            return captured["vision"]
        captured["strategy"] = _RecordingMP("s", "", [
            json.dumps(PASS2_OLIVER_GOOD),
            json.dumps(DEFAULT_PASS3),
        ])
        return captured["strategy"]

    monkeypatch.setattr(cli_mod, "build_provider_for_role", _fake_build)

    # Step 3: run analyze-report.
    runner = CliRunner()
    result = runner.invoke(
        cli, ["--data-dir", str(tmp_path), "analyze-report", rid],
        catch_exceptions=False,
    )
    assert result.exit_code == 0, result.output
    assert "Upgrading bank_only report" in result.output
    assert "reusing saved Pass 1" in result.output
    assert "Upgraded report" in result.output

    # The vision provider's analyze_image was NOT called — Pass 1 was
    # reused from the saved bank_only row.
    assert captured["vision"].image_calls == 0
    # Pass 2 + Pass 3 ran (two text calls on the strategy provider).
    assert captured["strategy"].text_calls == 2

    # The row was upgraded to three_pass.
    final = get_report(db_path, rid)
    assert final["raw_analysis"]["analysis_mode"] == "three_pass"
    # The original source_metadata was preserved through the upgrade.
    assert final["raw_analysis"]["source_metadata"]["source_filename"] == "alpha.mp4"


def test_analyze_report_normal_row_still_runs_pass1(
    monkeypatch, tmp_path: Path,
):
    """Sanity check: a non-bank_only row still triggers a Pass 1 vision
    call (no accidental reuse)."""
    from click.testing import CliRunner
    from emotion_radar import cli as cli_mod
    from emotion_radar.cli import cli
    from emotion_radar.db import insert_report

    db_path = tmp_path / "emotion_radar.db"
    sheet_path = tmp_path / "contact_sheets" / "alpha.jpg"
    sheet_path.parent.mkdir(parents=True, exist_ok=True)
    sheet_path.write_bytes(b"\xff\xd8\xff\xe0fake-sheet")

    # A typical stub row from analyze-url that has not been analyzed yet.
    stub_row = {
        "platform": "TikTok",
        "submitted_url": "https://www.tiktok.com/@u/v/1",
        "video_id": "1",
        "creator_username": "u",
        "creator_nickname": "U",
        "caption": "c",
        "metrics": {"views": 1, "likes": 0, "comments": 0, "shares": 0, "saves": 0},
        "contact_sheet_path": str(sheet_path),
        "raw_analysis": {"analysis_mode": "stub"},
    }
    rid = insert_report(db_path, stub_row)

    from tests.test_cli_analyze_link import (
        DEFAULT_PASS3,
        PASS2_OLIVER_GOOD,
    )

    class _RecordingMP:
        name = "mock"
        def __init__(self, label, image, texts):
            self.model = label; self._image = image
            self._texts = list(texts); self._i = 0
            self.image_calls = 0
            self.text_calls = 0
        def analyze_image(self, *a, **kw):
            self.image_calls += 1
            return self._image
        def analyze_text(self, *a, **kw):
            self.text_calls += 1
            if self._i < len(self._texts):
                r = self._texts[self._i]; self._i += 1; return r
            return self._texts[-1]

    captured = {}
    def _fake_build(env, role):
        if role == "vision_event":
            captured["vision"] = _RecordingMP("v", json.dumps(PASS1_GOOD), [""])
            return captured["vision"]
        captured["strategy"] = _RecordingMP("s", "", [
            json.dumps(PASS2_OLIVER_GOOD),
            json.dumps(DEFAULT_PASS3),
        ])
        return captured["strategy"]

    monkeypatch.setattr(cli_mod, "build_provider_for_role", _fake_build)

    runner = CliRunner()
    result = runner.invoke(
        cli, ["--data-dir", str(tmp_path), "analyze-report", rid],
        catch_exceptions=False,
    )
    assert result.exit_code == 0, result.output
    # Stub row: Pass 1 must run normally.
    assert captured["vision"].image_calls == 1
    # And the "reusing saved Pass 1" message must NOT appear.
    assert "reusing saved Pass 1" not in result.output
