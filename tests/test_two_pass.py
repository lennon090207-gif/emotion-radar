"""Two-pass orchestration tests with mocked providers. No network."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from emotion_radar import analysis as A
from emotion_radar.models import AnalysisResult


# ---- mock providers --------------------------------------------------------

class _MockVisionProvider:
    name = "mock_vision"
    model = "mock-vision-1"

    # text_response defaults to garbage so that the repair fallback (which
    # calls analyze_text after a parse failure) cannot accidentally rescue a
    # test that's specifically meant to surface a parse error. Tests that
    # want repair to succeed should set text_response explicitly.
    def __init__(self, image_response: str, text_response: str = "<<INVALID JSON FROM MOCK>>"):
        self._image_response = image_response
        self._text_response = text_response
        self.image_calls: list[dict[str, Any]] = []
        self.text_calls: list[dict[str, Any]] = []

    def analyze_image(self, image_path, system_prompt, user_prompt):
        self.image_calls.append({
            "image_path": image_path,
            "system_prompt": system_prompt,
            "user_prompt": user_prompt,
        })
        return self._image_response

    def analyze_text(self, system_prompt, user_prompt):
        self.text_calls.append({
            "system_prompt": system_prompt,
            "user_prompt": user_prompt,
        })
        return self._text_response


PASS1_GOOD = {
    "frame_observations": [
        {"timestamp": "0.0s", "observation": "market stall with dragon lamps on display",
         "people_visible": "maker behind table", "object_state": "lamps intact on table",
         "action_change_from_previous": ""},
        {"timestamp": "0.5s", "observation": "stranger reaches for a dragon lamp",
         "people_visible": "maker, stranger", "object_state": "lamp being lifted",
         "action_change_from_previous": "stranger entered frame and grabbed lamp"},
        {"timestamp": "1.0s", "observation": "stranger holds the lamp up",
         "people_visible": "maker, stranger", "object_state": "lamp in stranger's hand",
         "action_change_from_previous": "lamp lifted higher"},
        {"timestamp": "1.5s", "observation": "lamp mid-air falling toward the floor",
         "people_visible": "maker, stranger", "object_state": "lamp in flight",
         "action_change_from_previous": "lamp left stranger's hand"},
        {"timestamp": "2.0s", "observation": "lamp lies broken on the floor",
         "people_visible": "maker (reacting), stranger",
         "object_state": "lamp on floor, visibly damaged",
         "action_change_from_previous": "lamp hit the floor and broke"},
    ],
    "environment": "outdoor weekend market stall, handmade-goods display, daytime",
    "people": "underdog maker (mid-30s) behind stall; passerby acting as antagonist",
    "product_or_object": "handmade dragon lamp (HTTYD style)",
    "onscreen_text": "Please be honest, how are they?",
    "physical_action": "thrown / smashed on floor",
    "object_state_change": "lamp starts on display table, ends on the floor with visible damage",
    "visual_conflict_detected": True,
    "conflict_type": "smash",
    "confidence": 0.92,
    "uncertainty_notes": "",
}

PASS2_GOOD = {
    "visual_hook_summary": (
        "At an outdoor market stall, a stranger picks up a handmade "
        "dragon lamp and smashes it on the floor while the maker watches."
    ),
    "emotional_mechanic": "public disrespect of an underdog maker + viewer-defense instinct",
    "viewer_role": "defender",
    "emotions_triggered": ["anger", "protectiveness", "sympathy"],
    "why_it_works": "viewer wants to step in and protect; high comment-bait",
    "cooked_parts_to_avoid": ["overly staged 'random stranger' framing"],
    "product_attachability_score": 0.78,
    "transferability_score": 0.66,
    "freshness_score": 0.71,
    "cooked_score": 0.34,
    "overall_opportunity_score": 0.74,
    "hook_mutations": [
        {
            "type": "safe", "idea": "Maker shows lamp; rude customer demands a discount",
            "opening_scene": "wide stall shot, hand-made lamps visible",
            "onscreen_text": "she just asked me to do this for free",
            "product_niche_fit": "handmade fandom lamps / craft fair sellers",
            "why_it_might_work": "indignation engine; defender role",
            "cringe_or_cooked_risk": "tips into staged territory if acting is bad",
            "production_difficulty": "easy",
        }
    ],
}


# ---- extract_visual_event --------------------------------------------------

def test_extract_visual_event_calls_image_with_pass1_system_prompt(tmp_path: Path):
    sheet = tmp_path / "sheet.jpg"
    sheet.write_bytes(b"fake")
    provider = _MockVisionProvider(image_response=json.dumps(PASS1_GOOD))
    parsed = A.extract_visual_event(sheet, {"platform": "TikTok"}, provider)
    assert parsed["visual_conflict_detected"] is True
    assert parsed["conflict_type"] == "smash"
    assert len(provider.image_calls) == 1
    call = provider.image_calls[0]
    assert call["system_prompt"] == A.VISUAL_EVENT_SYSTEM_PROMPT
    assert call["image_path"] == sheet
    assert provider.text_calls == []  # Pass 1 must NOT call text-only.


def test_extract_visual_event_raises_on_bad_json(tmp_path: Path):
    sheet = tmp_path / "sheet.jpg"
    sheet.write_bytes(b"fake")
    provider = _MockVisionProvider(image_response="not json")
    with pytest.raises(ValueError):
        A.extract_visual_event(sheet, {}, provider)


# ---- generate_hook_strategy ------------------------------------------------

def test_generate_hook_strategy_is_text_only_and_embeds_pass1():
    provider = _MockVisionProvider(image_response="UNUSED", text_response=json.dumps(PASS2_GOOD))
    metadata = {"platform": "TikTok", "creator_username": "olivermakesartt",
                "caption": "please be honest", "metrics": {"views": 100}}
    parsed = A.generate_hook_strategy(metadata, PASS1_GOOD, provider)
    assert parsed["emotional_mechanic"].startswith("public disrespect")
    assert provider.image_calls == []  # Pass 2 must NOT call vision.
    assert len(provider.text_calls) == 1
    call = provider.text_calls[0]
    assert call["system_prompt"] == A.HOOK_STRATEGY_SYSTEM_PROMPT
    # Pass 1 evidence must be embedded in the Pass 2 user prompt.
    assert "thrown / smashed on floor" in call["user_prompt"]
    assert "visual_conflict_detected" in call["user_prompt"]


# ---- analyze_two_pass orchestration ----------------------------------------

def test_analyze_two_pass_runs_pass1_then_pass2(tmp_path: Path):
    sheet = tmp_path / "sheet.jpg"
    sheet.write_bytes(b"fake")
    provider = _MockVisionProvider(
        image_response=json.dumps(PASS1_GOOD),
        text_response=json.dumps(PASS2_GOOD),
    )
    pass1, pass2 = A.analyze_two_pass(sheet, {"platform": "TikTok"}, provider)
    assert pass1["conflict_type"] == "smash"
    assert pass2["viewer_role"] == "defender"
    # Order: vision first, then text.
    assert len(provider.image_calls) == 1
    assert len(provider.text_calls) == 1


def test_analyze_two_pass_uses_separate_providers_when_supplied(tmp_path: Path):
    sheet = tmp_path / "sheet.jpg"
    sheet.write_bytes(b"fake")
    vision = _MockVisionProvider(image_response=json.dumps(PASS1_GOOD))
    strategy = _MockVisionProvider(image_response="X", text_response=json.dumps(PASS2_GOOD))
    pass1, pass2 = A.analyze_two_pass(sheet, {}, vision, strategy)
    assert pass1["conflict_type"] == "smash"
    assert pass2["overall_opportunity_score"] == 0.74
    # Each provider sees exactly one call of the right kind.
    assert len(vision.image_calls) == 1 and vision.text_calls == []
    assert strategy.image_calls == [] and len(strategy.text_calls) == 1


def test_two_pass_propagates_pass1_failure(tmp_path: Path):
    sheet = tmp_path / "sheet.jpg"
    sheet.write_bytes(b"fake")
    provider = _MockVisionProvider(image_response="garbage")
    with pytest.raises(ValueError):
        A.analyze_two_pass(sheet, {}, provider)


def test_two_pass_propagates_pass2_failure(tmp_path: Path):
    sheet = tmp_path / "sheet.jpg"
    sheet.write_bytes(b"fake")
    provider = _MockVisionProvider(
        image_response=json.dumps(PASS1_GOOD),
        text_response="totally not json",
    )
    with pytest.raises(ValueError):
        A.analyze_two_pass(sheet, {}, provider)


# ---- build_two_pass_analysis_result merge ---------------------------------

def test_merge_sources_onscreen_text_from_pass1_other_fields_from_pass2():
    result = A.build_two_pass_analysis_result(PASS1_GOOD, PASS2_GOOD)
    # Pass 1-sourced
    assert result.onscreen_text == "Please be honest, how are they?"
    # Pass 2-sourced
    assert result.visual_hook_summary.startswith("At an outdoor market stall")
    assert result.emotional_mechanic.startswith("public disrespect")
    assert result.viewer_role == "defender"
    assert result.emotions_triggered == ["anger", "protectiveness", "sympathy"]
    assert result.product_attachability_score == 0.78
    assert result.transferability_score == 0.66
    assert result.freshness_score == 0.71
    assert result.cooked_score == 0.34
    assert result.overall_opportunity_score == 0.74
    assert len(result.hook_mutations) == 1


def test_merge_raw_analysis_carries_both_passes():
    result = A.build_two_pass_analysis_result(PASS1_GOOD, PASS2_GOOD)
    assert result.raw_analysis["analysis_mode"] == "two_pass"
    assert result.raw_analysis["visual_event_pass"] == PASS1_GOOD
    assert result.raw_analysis["hook_strategy_pass"] == PASS2_GOOD


def test_merge_clamps_scores_to_unit_interval():
    pass2 = {**PASS2_GOOD, "freshness_score": 1.7, "cooked_score": -0.4,
             "overall_opportunity_score": "0.55"}
    result = A.build_two_pass_analysis_result(PASS1_GOOD, pass2)
    assert result.freshness_score == 1.0
    assert result.cooked_score == 0.0
    assert result.overall_opportunity_score == 0.55


def test_merge_handles_empty_passes():
    result = A.build_two_pass_analysis_result({}, {})
    assert result.onscreen_text is None
    assert result.visual_hook_summary is None
    assert result.hook_mutations == []
    # raw_analysis still records the mode.
    assert result.raw_analysis["analysis_mode"] == "two_pass"


def test_merge_drops_non_list_hook_mutations():
    pass2 = {**PASS2_GOOD, "hook_mutations": "definitely not a list"}
    result = A.build_two_pass_analysis_result(PASS1_GOOD, pass2)
    assert result.hook_mutations == []


def test_merge_returns_analysis_result_dataclass():
    result = A.build_two_pass_analysis_result(PASS1_GOOD, PASS2_GOOD)
    assert isinstance(result, AnalysisResult)


# ============================================================================
# JSON repair fallback
# ============================================================================

def test_repair_recovers_from_dirty_pass1_output(tmp_path: Path):
    """Image returns prose-wrapped near-JSON; repair text call cleans it up."""
    sheet = tmp_path / "sheet.jpg"
    sheet.write_bytes(b"fake")
    dirty = "Sure! here is your response:\n\nthe model forgot the braces"
    provider = _MockVisionProvider(
        image_response=dirty,
        text_response=json.dumps(PASS1_GOOD),
    )
    parsed = A.extract_visual_event(sheet, {}, provider)
    assert parsed["conflict_type"] == "smash"
    assert len(provider.image_calls) == 1  # original Pass 1 call
    assert len(provider.text_calls) == 1   # one repair call


def test_repair_preserves_original_error_when_repair_also_fails(tmp_path: Path):
    """If both the initial parse AND the repair attempt fail, the user
    sees the ORIGINAL ValueError — that's the one that explains why we
    couldn't make sense of the model output."""
    sheet = tmp_path / "sheet.jpg"
    sheet.write_bytes(b"fake")
    provider = _MockVisionProvider(
        image_response="garbage from the model",
        # text_response defaults to invalid; repair attempt will also fail.
    )
    with pytest.raises(ValueError):
        A.extract_visual_event(sheet, {}, provider)


def test_repair_uses_supplied_repair_provider_not_main_provider(tmp_path: Path):
    """Pass 1 producing garbage should route the repair call to the
    explicit repair_provider, not the vision provider. Keeps cost
    sensible when Pass 1 is an expensive vision model."""
    sheet = tmp_path / "sheet.jpg"
    sheet.write_bytes(b"fake")
    vision = _MockVisionProvider(image_response="not parseable")
    repair = _MockVisionProvider(
        image_response="<<unused>>",
        text_response=json.dumps(PASS1_GOOD),
    )
    parsed = A.extract_visual_event(
        sheet, {}, vision, repair_provider=repair,
    )
    assert parsed["conflict_type"] == "smash"
    assert len(vision.image_calls) == 1
    assert vision.text_calls == []        # main provider untouched on repair
    assert len(repair.text_calls) == 1    # repair handled by the repair provider


def test_repair_off_when_repair_provider_is_None(tmp_path: Path):
    """If repair_provider is explicitly None and the initial parse
    fails, ValueError surfaces immediately with no extra calls."""
    raw = "still not json"
    with pytest.raises(ValueError):
        A._parse_or_repair(raw, repair_provider=None)


def test_analyze_two_pass_uses_strategy_provider_as_repair_target(tmp_path: Path):
    """When Pass 1 (vision) produces unparseable output, the orchestrator
    should route the repair call to the strategy (text) provider — not
    re-invoke the vision provider."""
    sheet = tmp_path / "sheet.jpg"
    sheet.write_bytes(b"fake")
    vision = _MockVisionProvider(image_response="not json at all")
    strategy = _MockVisionProvider(
        image_response="<<unused>>",
        text_response=json.dumps(PASS1_GOOD),
    )
    # The first text call on `strategy` will repair Pass 1 -> PASS1_GOOD.
    # Then generate_hook_strategy runs and needs Pass 2 JSON. Its initial
    # text response is also PASS1_GOOD, which parses fine as a dict (Pass
    # 2 just needs a JSON object). We're not asserting Pass 2 quality —
    # only that the strategy provider, not the vision provider, was used
    # for the Pass 1 repair.
    A.analyze_two_pass(sheet, {}, vision, strategy)
    assert len(vision.image_calls) == 1
    assert vision.text_calls == []        # vision never asked to repair
    assert len(strategy.image_calls) == 0
    assert len(strategy.text_calls) >= 1  # at least the Pass 1 repair
