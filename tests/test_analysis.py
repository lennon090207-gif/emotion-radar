"""Pass 1 (Visual Event Extractor) + Pass 2 (Hook Strategist) tests.
No real API calls."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from emotion_radar import analysis as A
from emotion_radar.models import AnalysisResult


# ============================================================================
# Pass 1 — Visual Event Extractor prompt
# ============================================================================

def test_pass1_prompt_emphasizes_literal_visual_observation():
    sp = A.VISUAL_EVENT_SYSTEM_PROMPT
    lowered = sp.lower()
    assert "literally visible" in lowered or "literal" in lowered
    assert "chronological" in lowered
    assert "evidence" in lowered
    assert "JSON" in sp


def test_pass1_prompt_explicitly_lists_actions_to_check():
    sp = A.VISUAL_EVENT_SYSTEM_PROMPT.lower()
    for required in (
        "pick up",
        "throw",
        "drop",
        "smash",
        "break",
        "damage",
        "market stall",
        "approach",
        "disrespect",
    ):
        assert required in sp, f"Pass 1 prompt missing action check: {required}"


def test_pass1_prompt_forbids_strategy_and_scoring():
    """Pass 1 must NOT instruct the model to score or generate ideas."""
    sp = A.VISUAL_EVENT_SYSTEM_PROMPT.lower()
    # Pass 1 mentions the schema for evidence only; it must NOT introduce
    # scoring rubrics, mutation quotas, or hook ideas.
    assert "hook_mutations" not in A.VISUAL_EVENT_SYSTEM_PROMPT
    assert "product_attachability_score" not in A.VISUAL_EVENT_SYSTEM_PROMPT
    assert "freshness_score" not in A.VISUAL_EVENT_SYSTEM_PROMPT
    # And we want a clear "do NOT generate hook ideas" guard.
    assert "generate hook ideas" in sp or "hook ideas" in sp


def test_pass1_prompt_lists_pass1_schema_keys():
    sp = A.VISUAL_EVENT_SYSTEM_PROMPT
    for key in (
        "frame_observations",
        "environment",
        "people",
        "product_or_object",
        "onscreen_text",
        "physical_action",
        "object_state_change",
        "visual_conflict_detected",
        "conflict_type",
        "confidence",
        "uncertainty_notes",
        # Per-frame fields
        "people_visible",
        "object_state",
        "action_change_from_previous",
    ):
        assert key in sp, f"Pass 1 prompt missing key: {key}"


def test_pass1_prompt_forbids_mood_softening():
    sp = A.VISUAL_EVENT_SYSTEM_PROMPT.lower()
    # The exact phrases the model previously retreated to.
    assert "discouraged" in sp
    assert "action wins" in sp


def test_build_visual_event_user_prompt_is_minimal():
    """Pass 1 user message should NOT include the caption (which biases
    evidence interpretation). It should reference frames and chronology."""
    md = {
        "platform": "TikTok",
        "creator_username": "olivermakesartt",
        "caption": "please be honest, how are they?",
    }
    up = A.build_visual_event_user_prompt(md)
    assert "TikTok" in up
    assert "chronologic" in up.lower()
    # No caption leak into Pass 1.
    assert "please be honest" not in up.lower()


# ============================================================================
# Pass 2 — Hook Strategist prompt
# ============================================================================

def test_pass2_prompt_relies_on_pass1_as_evidence_layer():
    sp = A.HOOK_STRATEGY_SYSTEM_PROMPT.lower()
    assert "pass 1" in sp or "pass-1" in sp
    assert "evidence" in sp
    # Pass 2 must NOT re-analyze the image.
    assert "do not re-analyze" in sp or "you do not re-analyze" in sp


def test_pass2_prompt_lists_pass2_schema_keys():
    sp = A.HOOK_STRATEGY_SYSTEM_PROMPT
    for key in (
        "visual_hook_summary",
        "emotional_mechanic",
        "viewer_role",
        "emotions_triggered",
        "why_it_works",
        "cooked_parts_to_avoid",
        "product_attachability_score",
        "transferability_score",
        "freshness_score",
        "cooked_score",
        "overall_opportunity_score",
        "hook_mutations",
    ):
        assert key in sp, f"Pass 2 prompt missing key: {key}"


def test_pass2_prompt_taste_target_world():
    sp = A.HOOK_STRATEGY_SYSTEM_PROMPT.lower()
    for target in (
        "handmade",
        "emotional",
        "custom",
        "fandom",
        "pet",
        "memorial",
        "market stall",
    ):
        assert target in sp, f"Pass 2 prompt missing target-world cue: {target}"


def test_pass2_prompt_explicitly_rejects_unrelated_niches():
    sp = A.HOOK_STRATEGY_SYSTEM_PROMPT.lower()
    for forbidden in (
        "street musician",
        "busker",
        "saas",
        "fitness",
        "crypto",
        "real estate",
        "dropshipping",
    ):
        assert forbidden in sp, f"Pass 2 prompt should explicitly reject: {forbidden}"


def test_pass2_prompt_mutation_quota():
    sp = A.HOOK_STRATEGY_SYSTEM_PROMPT
    assert "6" in sp
    assert '2 "safe"' in sp
    assert '3 "fresh"' in sp
    assert '1 "big_swing"' in sp


def test_pass2_prompt_lists_per_mutation_fields():
    sp = A.HOOK_STRATEGY_SYSTEM_PROMPT
    for fld in (
        "opening_scene",
        "onscreen_text",
        "product_niche_fit",
        "why_it_might_work",
        "cringe_or_cooked_risk",
        "production_difficulty",
    ):
        assert fld in sp, f"per-mutation field missing in Pass 2 prompt: {fld}"


def test_pass2_prompt_lists_cooked_phrases_to_avoid():
    sp = A.HOOK_STRATEGY_SYSTEM_PROMPT
    for cooked in (
        "Nobody will ever buy your",
        "Please be honest",
        "Would you buy one",
        "POV:",
    ):
        assert cooked in sp, f"Pass 2 prompt missing cooked-phrase warning: {cooked}"


def test_build_hook_strategy_user_prompt_embeds_pass1_json():
    md = {
        "platform": "TikTok",
        "creator_username": "olivermakesartt",
        "caption": "please be honest",
        "metrics": {"views": 100, "likes": 1, "comments": 0},
    }
    pass1 = {"physical_action": "thrown / smashed on floor", "visual_conflict_detected": True}
    up = A.build_hook_strategy_user_prompt(md, pass1)
    assert "thrown / smashed on floor" in up
    assert "visual_conflict_detected" in up
    # Caption included as weak prior, but Pass 1 must precede the prior.
    assert up.index("PASS 1 EVIDENCE LAYER") < up.index("OPTIONAL CONTEXT")
    assert "@olivermakesartt" in up
    assert "please be honest" in up


# ============================================================================
# JSON parsing (shared)
# ============================================================================

def test_parse_full_json_object():
    parsed = A.parse_analysis_json('{"a": 1, "b": [1,2]}')
    assert parsed == {"a": 1, "b": [1, 2]}


def test_parse_json_with_markdown_fence():
    raw = "Here you go:\n```json\n{\"x\": 1}\n```\nDone."
    assert A.parse_analysis_json(raw) == {"x": 1}


def test_parse_json_with_leading_trailing_prose():
    raw = "Sure: {\"y\": 2} thanks!"
    assert A.parse_analysis_json(raw) == {"y": 2}


def test_parse_json_raises_on_invalid():
    with pytest.raises(ValueError):
        A.parse_analysis_json("not json")


def test_parse_json_raises_on_array_top_level():
    with pytest.raises(ValueError):
        A.parse_analysis_json("[1,2,3]")


def test_parse_json_raises_on_empty():
    with pytest.raises(ValueError):
        A.parse_analysis_json("   ")


# ============================================================================
# Stub
# ============================================================================

def test_stub_returns_nulls_and_two_pass_mode_hint():
    result = A.analyze_contact_sheet(Path("nonexistent.jpg"), {"caption": "x"})
    assert isinstance(result, AnalysisResult)
    assert result.visual_hook_summary is None
    assert result.hook_mutations == []
    assert result.raw_analysis["analysis_mode"] == "stub"
