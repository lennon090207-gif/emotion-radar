"""Prompt construction + JSON parsing tests. No real API calls."""

from __future__ import annotations

from pathlib import Path

import pytest

from emotion_radar import analysis as A
from emotion_radar.models import AnalysisResult


# ---- prompts ---------------------------------------------------------------

def test_system_prompt_emphasizes_windowing_and_visual_focus():
    sp = A.SYSTEM_PROMPT
    assert "0-5" in sp or "0 to 5" in sp
    assert "VISUAL HOOK" in sp.upper() or "visual hook" in sp.lower()
    assert "caption" in sp.lower()  # mentions caption-is-weak
    assert "JSON" in sp
    # The taste rules must call out the non-cringe / non-AI-slop guardrails.
    assert "AI" in sp or "ai-slop" in sp.lower() or "AI marketing" in sp
    assert "safe" in sp and "fresh" in sp and "big_swing" in sp


def test_system_prompt_lists_schema_keys():
    sp = A.SYSTEM_PROMPT
    for key in (
        "visual_hook_summary",
        "environment",
        "people",
        "product_or_object",
        "action_or_conflict",
        "physical_action",
        "visual_conflict_detected",
        "onscreen_text",
        "emotional_mechanic",
        "viewer_role",
        "emotions_triggered",
        "why_it_works",
        "cooked_parts_to_avoid",
        "confidence",
        "uncertainty_notes",
        "product_attachability_score",
        "transferability_score",
        "freshness_score",
        "cooked_score",
        "overall_opportunity_score",
        "frame_observations",
        "hook_mutations",
    ):
        assert key in sp, f"system prompt missing key: {key}"


def test_system_prompt_requires_chronological_frame_reasoning():
    sp = A.SYSTEM_PROMPT
    lowered = sp.lower()
    assert "chronological" in lowered or "chronologically" in lowered
    assert "frame_observations" in sp
    # Must instruct one entry per visible timestamp tile, in order.
    assert "timestamp" in lowered
    # Should explicitly tell the model to look at CHANGES between frames,
    # not single static frames — that's the whole regression we're fixing.
    assert "change" in lowered  # "changes" / "CHANGED" / "changes between"


def test_system_prompt_lists_explicit_conflict_checks():
    sp = A.SYSTEM_PROMPT.lower()
    # The specific physical-action triggers the prompt MUST mention.
    for required_phrase in (
        "approaching",
        "picking up",
        "handling",
        "thrown",
        "smashed",
        "market stall",
        "disrespect",
        "before-and-after",
    ):
        assert required_phrase in sp, f"prompt missing explicit check: {required_phrase}"


def test_system_prompt_forbids_generic_sentiment_fallback():
    sp = A.SYSTEM_PROMPT.lower()
    # We want the prompt to actively tell the model NOT to retreat to
    # "creator looks discouraged" style readings.
    assert "discouraged" in sp
    assert "action wins" in sp or "the action wins" in sp


def test_system_prompt_taste_target_world():
    sp = A.SYSTEM_PROMPT.lower()
    for target in (
        "handmade",
        "emotional",
        "custom",
        "fandom",
        "pet",
        "memorial",
        "market stall",
    ):
        assert target in sp, f"prompt missing target-world cue: {target}"


def test_system_prompt_explicitly_rejects_unrelated_niches():
    sp = A.SYSTEM_PROMPT.lower()
    for forbidden in (
        "street musician",
        "busker",
        "saas",
        "fitness",
        "crypto",
        "real estate",
        "dropshipping",
    ):
        assert forbidden in sp, f"prompt should explicitly reject: {forbidden}"


def test_system_prompt_specifies_mutation_quota():
    sp = A.SYSTEM_PROMPT
    # Exactly 6 mutations split 2 safe / 3 fresh / 1 big_swing.
    assert "6" in sp
    assert '2 "safe"' in sp
    assert '3 "fresh"' in sp
    assert '1 "big_swing"' in sp


def test_system_prompt_lists_per_mutation_fields():
    sp = A.SYSTEM_PROMPT
    for field_name in (
        "opening_scene",
        "onscreen_text",
        "product_niche_fit",
        "why_it_might_work",
        "cringe_or_cooked_risk",
        "production_difficulty",
    ):
        assert field_name in sp, f"mutation field missing in prompt: {field_name}"


def test_build_user_prompt_includes_metadata():
    md = {
        "platform": "TikTok",
        "creator_username": "olivermakesartt",
        "creator_nickname": "Oliver",
        "caption": "please be honest,\nhow are they?",
        "metrics": {"views": 12345, "likes": 67, "comments": 5},
    }
    up = A.build_user_prompt(md)
    assert "TikTok" in up
    assert "@olivermakesartt" in up
    assert "Oliver" in up
    assert "please be honest" in up
    # caption newline should be flattened so the prompt stays one block
    assert "\nhow are they?" not in up
    assert "12345" in up


def test_build_user_prompt_handles_missing_fields():
    up = A.build_user_prompt({})
    assert "(unknown)" in up
    assert "(none)" in up


# ---- JSON parsing ----------------------------------------------------------

_FULL_PARSED = {
    "visual_hook_summary": "Man at market stall watches stranger smash his handmade lamp.",
    "environment": "outdoor weekend market, daytime, busy",
    "people": "underdog maker (mid-30s), passerby (40s) playing antagonist",
    "product_or_object": "handmade HTTYD lamp",
    "action_or_conflict": "passerby picks up lamp, drops/throws it; maker reacts",
    "onscreen_text": "Please be honest, how are they?",
    "emotional_mechanic": "public disrespect of an underdog maker triggers viewer-defense instinct",
    "viewer_role": "defender",
    "emotions_triggered": ["anger", "protectiveness", "sympathy"],
    "why_it_works": "viewer wants to step in and defend; high comment-bait",
    "cooked_parts_to_avoid": ["overly staged 'random stranger' setup"],
    "product_attachability_score": 0.78,
    "transferability_score": 0.66,
    "freshness_score": 0.71,
    "cooked_score": 0.34,
    "overall_opportunity_score": 0.74,
    "hook_mutations": [
        {
            "type": "safe",
            "idea": "Maker shows lamp; rude customer demands a discount",
            "opening_scene": "wide stall shot, hand-made lamps visible",
            "onscreen_text": "she just asked me to do this for free",
            "why_it_might_work": "indignation engine",
            "taste_risk": "tips into staged territory if acting is bad",
            "production_difficulty": "easy",
        }
    ],
}


def test_parse_full_json():
    import json
    raw = json.dumps(_FULL_PARSED)
    parsed = A.parse_analysis_json(raw)
    assert parsed["visual_hook_summary"].startswith("Man at market stall")
    assert parsed["overall_opportunity_score"] == 0.74


def test_parse_json_with_markdown_fence():
    import json
    raw = "Sure, here you go:\n\n```json\n" + json.dumps(_FULL_PARSED) + "\n```\nLet me know."
    parsed = A.parse_analysis_json(raw)
    assert parsed["emotional_mechanic"].startswith("public disrespect")


def test_parse_json_with_leading_trailing_prose():
    import json
    raw = "Here is the analysis: " + json.dumps(_FULL_PARSED) + " End of analysis."
    parsed = A.parse_analysis_json(raw)
    assert parsed["viewer_role"] == "defender"


def test_parse_json_raises_on_invalid():
    with pytest.raises(ValueError):
        A.parse_analysis_json("this is not json at all")


def test_parse_json_raises_on_array():
    with pytest.raises(ValueError):
        A.parse_analysis_json("[1, 2, 3]")


# ---- mapping to AnalysisResult ---------------------------------------------

def test_map_parsed_to_result_full():
    result = A.map_parsed_to_result(_FULL_PARSED)
    assert isinstance(result, AnalysisResult)
    assert result.visual_hook_summary.startswith("Man at market stall")
    assert result.onscreen_text == "Please be honest, how are they?"
    assert result.emotional_mechanic.startswith("public disrespect")
    assert result.viewer_role == "defender"
    assert result.emotions_triggered == ["anger", "protectiveness", "sympathy"]
    assert result.product_attachability_score == 0.78
    assert result.transferability_score == 0.66
    assert result.freshness_score == 0.71
    assert result.cooked_score == 0.34
    assert result.overall_opportunity_score == 0.74
    assert len(result.hook_mutations) == 1
    assert result.hook_mutations[0]["type"] == "safe"
    # Extra fields survive in raw_analysis.
    assert result.raw_analysis["why_it_works"]
    assert result.raw_analysis["environment"]


def test_map_parsed_clamps_scores_to_unit_interval():
    parsed = {
        "freshness_score": 1.5,
        "cooked_score": -0.3,
        "overall_opportunity_score": "0.55",  # string -> coerced
    }
    result = A.map_parsed_to_result(parsed)
    assert result.freshness_score == 1.0
    assert result.cooked_score == 0.0
    assert result.overall_opportunity_score == 0.55


def test_map_parsed_handles_missing_fields():
    result = A.map_parsed_to_result({})
    assert result.visual_hook_summary is None
    assert result.emotions_triggered == []
    assert result.hook_mutations == []
    assert result.product_attachability_score is None


def test_map_parsed_filters_non_string_emotions():
    parsed = {"emotions_triggered": ["anger", 42, "", None, "joy"]}
    result = A.map_parsed_to_result(parsed)
    assert result.emotions_triggered == ["anger", "joy"]


# ---- end-to-end with mock provider ----------------------------------------

class _MockProvider:
    name = "mock"
    model = "mock-vision-1"

    def __init__(self, response_text: str):
        self._response_text = response_text
        self.last_image_path: Path | None = None
        self.last_system_prompt: str | None = None
        self.last_user_prompt: str | None = None

    def analyze_image(self, image_path, system_prompt, user_prompt):
        self.last_image_path = image_path
        self.last_system_prompt = system_prompt
        self.last_user_prompt = user_prompt
        return self._response_text


def test_analyze_contact_sheet_with_vision_full_flow(tmp_path: Path):
    import json
    sheet = tmp_path / "sheet.jpg"
    sheet.write_bytes(b"fake jpeg bytes")
    provider = _MockProvider(json.dumps(_FULL_PARSED))
    metadata = {
        "platform": "TikTok",
        "creator_username": "olivermakesartt",
        "caption": "please be honest",
        "metrics": {"views": 100, "likes": 1, "comments": 0},
    }
    result = A.analyze_contact_sheet_with_vision(sheet, metadata, provider)
    assert provider.last_image_path == sheet
    assert "VISUAL HOOK" in provider.last_system_prompt.upper() or "visual hook" in provider.last_system_prompt.lower()
    assert "@olivermakesartt" in provider.last_user_prompt
    assert result.viewer_role == "defender"
    assert result.overall_opportunity_score == 0.74


def test_analyze_with_invalid_model_output_raises(tmp_path: Path):
    sheet = tmp_path / "sheet.jpg"
    sheet.write_bytes(b"fake jpeg bytes")
    provider = _MockProvider("the model decided to write a poem instead")
    with pytest.raises(ValueError):
        A.analyze_contact_sheet_with_vision(sheet, {}, provider)


def test_stub_returns_nulls():
    result = A.analyze_contact_sheet(Path("nonexistent.jpg"), {"caption": "x"})
    assert result.visual_hook_summary is None
    assert result.hook_mutations == []
    assert result.raw_analysis["status"] == "stub"
