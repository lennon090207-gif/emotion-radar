"""Pass 3 (specificity rewriter) tests. No real API calls."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from emotion_radar import analysis as A


# Re-use the same fixture shape as the two-pass tests; keep this file
# independent so tests can run in any order.
PASS1_GOOD = {
    "frame_observations": [{"timestamp": "0.0s", "observation": "market stall"}],
    "environment": "outdoor weekend market stall",
    "people": "underdog maker; passerby",
    "product_or_object": "handmade dragon lamp",
    "onscreen_text": "Please be honest, how are they?",
    "physical_action": "thrown / smashed on floor",
    "object_state_change": "lamp on table -> lamp on floor",
    "visual_conflict_detected": True,
    "conflict_type": "smash",
    "confidence": 0.92,
    "uncertainty_notes": "",
}

PASS2_GOOD = {
    "viral_mechanic": "public disrespect + underdog maker",
    "viewer_role": "defender",
    "matched_story_flows": [
        {"id": "public_disrespect_viewer_defense", "name": "Public Disrespect -> Viewer Defense",
         "confidence": 0.92, "why_matched": "stranger destroys product at the stall"},
    ],
    "dominant_story_flow": "public_disrespect_viewer_defense",
    "variations": [
        {"concept_name": "Receipt of Cruelty",
         "story_flow_id": "public_disrespect_viewer_defense",
         "first_2_seconds": "abstract",
         "emotional_trigger": "vindication", "viewer_role": "jury",
         "why_it_could_go_viral": "tactile receipt format is rare",
         "what_is_new": "printed-screenshot prop",
         "what_is_cooked_to_avoid": "please be honest",
         "believability_risk": "fabricated screenshots"},
    ],
    "pioneer_concepts": [
        {"concept_name": "Receipt Wall",
         "inspired_by_story_flow_id": "comment_humiliation_public_witness",
         "first_2_seconds": "creator pins printed DMs to a corkboard",
         "emotional_physics": "tactile evidence of cruelty",
         "why_it_is_not_a_direct_copy": "no reaction shot, no read-aloud",
         "why_it_could_be_breakout": "the prop is the hook",
         "viewer_comment_impulse": "urge to defend",
         "ethical_or_cringe_risk": "redact names"},
    ],
}

PASS3_GOOD = {
    "specificity_notes": "rewrote abstract variations into filmable scenes",
    "weak_patterns_fixed": ["dismisses a handmade item", "reveals effort"],
    "scene_concepts": [
        {"source_type": "variation",
         "source_concept_name": "Receipt of Cruelty",
         "story_flow_id": "public_disrespect_viewer_defense",
         "specific_concept_name": "Receipt at the Stall",
         "first_2_seconds": (
             "the seller silently sets a printed screenshot of a rude DM "
             "next to the lamp at his market stall"
         ),
         "onscreen_text": "she said it was 'overpriced trash' last night",
         "visual_beat": "tape pulled off the back of the screenshot, set down",
         "social_tension": "the stall is busy; people pause to read",
         "viewer_comment_impulse": "urge to defend the maker in comments",
         "why_they_keep_watching": "viewers want to see if anyone reacts in frame",
         "freshness_angle": "physical-receipt format instead of voice-over",
         "believability_risk": "fails if the DM reads written-by-the-creator",
         "cringe_risk": "screenshot must look authentic; fabricated kills it",
         "virality_potential_score": 0.81},
        {"source_type": "pioneer_concept",
         "source_concept_name": "Receipt Wall",
         "story_flow_id": "comment_humiliation_public_witness",
         "specific_concept_name": "Receipt Wall, Pinned",
         "first_2_seconds": (
             "the maker silently pins three printed rude DMs to a corkboard "
             "behind the work; the camera tilts up to show them"
         ),
         "onscreen_text": "I keep them all now.",
         "visual_beat": "the third pin going in",
         "social_tension": "no spoken word; the wall does the talking",
         "viewer_comment_impulse": "urge to add 'I'd pay extra now'",
         "why_they_keep_watching": "the wall is a slow-reveal payoff",
         "freshness_angle": "tactile evidence collage is underused",
         "believability_risk": "real names must be redacted",
         "cringe_risk": "feels staged if pinned too neatly",
         "virality_potential_score": 0.87},
    ],
}


# ---- Pass 3 system prompt content ------------------------------------------

def test_pass3_prompt_states_role_is_scene_writer_not_strategist():
    sp = A.SPECIFICITY_SYSTEM_PROMPT.lower()
    assert "scene writer" in sp
    assert "not a strategist" in sp or "not a strategist" in sp


def test_pass3_prompt_forbids_changing_story_flow():
    sp = A.SPECIFICITY_SYSTEM_PROMPT
    assert "story_flow_id" in sp
    # The rule that the story flow must survive Pass 3.
    assert "do not change the underlying story_flow_id" in sp.lower() \
        or "story_flow_id of a concept" in sp.lower()


def test_pass3_prompt_forbids_image_or_video_generation():
    sp = A.SPECIFICITY_SYSTEM_PROMPT.lower()
    assert "do not generate images" in sp or "do not generate images or videos" in sp


def test_pass3_prompt_forbids_product_or_niche_constraints():
    sp = A.SPECIFICITY_SYSTEM_PROMPT.lower()
    assert "product is secondary" in sp
    assert "viral mechanic is primary" in sp


def test_pass3_prompt_bans_vague_placeholders():
    sp = A.SPECIFICITY_SYSTEM_PROMPT.lower()
    # The exact ban list from the spec.
    for phrase in (
        '"a person',
        '"an item',
        '"a product',
        '"a creator',
        '"a unique item',
        '"a handmade item',
        '"reveals effort"',
        '"publicly doubts"',
        '"shows hidden value"',
    ):
        assert phrase.lower() in sp, f"Pass 3 prompt missing ban: {phrase}"


def test_pass3_prompt_includes_concrete_calibration_examples():
    sp = A.SPECIFICITY_SYSTEM_PROMPT
    # At least the lamp + $80 example and the daughter/comments example are
    # spelled out so the model has anchors.
    assert "$80" in sp
    assert "Did anyone like it yet?" in sp
    assert "People pay for this?" in sp


def test_pass3_prompt_lists_required_per_scene_fields():
    sp = A.SPECIFICITY_SYSTEM_PROMPT
    for fld in (
        "source_type",
        "source_concept_name",
        "story_flow_id",
        "specific_concept_name",
        "first_2_seconds",
        "onscreen_text",
        "visual_beat",
        "social_tension",
        "viewer_comment_impulse",
        "why_they_keep_watching",
        "freshness_angle",
        "believability_risk",
        "cringe_risk",
        "virality_potential_score",
    ):
        assert fld in sp, f"Pass 3 prompt missing per-scene field: {fld}"


def test_pass3_prompt_documents_uncastable_escape_hatch():
    sp = A.SPECIFICITY_SYSTEM_PROMPT
    assert "UNCASTABLE" in sp
    assert "0.3" in sp or "0.30" in sp


def test_pass3_prompt_prioritises_pioneer_concepts():
    sp = A.SPECIFICITY_SYSTEM_PROMPT.lower()
    assert "all pioneer_concepts" in sp or "all pioneer concepts" in sp
    assert "primary goal" in sp


# ---- Pass 3 user prompt -----------------------------------------------------

def test_user_prompt_embeds_pass1_and_pass2_slices():
    up = A.build_specificity_user_prompt(PASS1_GOOD, PASS2_GOOD)
    assert "PASS 1 EVIDENCE" in up
    assert "PASS 2 STRATEGY" in up
    # The relevant Pass 2 slice (variations + pioneer_concepts) is embedded.
    assert "Receipt of Cruelty" in up
    assert "Receipt Wall" in up
    # And the Pass 1 evidence is there.
    assert "thrown / smashed on floor" in up


def test_user_prompt_omits_taste_section_when_none():
    up = A.build_specificity_user_prompt(PASS1_GOOD, PASS2_GOOD, taste_profile=None)
    assert "USER TASTE PROFILE" not in up


def test_user_prompt_includes_taste_section_when_provided():
    taste = "User tends to like:\n  - believable public tension\nUser dislikes:\n  - theatrical villain scenes"
    up = A.build_specificity_user_prompt(PASS1_GOOD, PASS2_GOOD, taste_profile=taste)
    assert "USER TASTE PROFILE" in up
    assert "believable public tension" in up
    assert "theatrical villain scenes" in up


# ---- Pass 3 orchestration ---------------------------------------------------

class _MockProvider:
    name = "mock"
    model = "mock-1"

    def __init__(self, text_responses: list[str], image_response: str = ""):
        self._text_responses = list(text_responses)
        self._image_response = image_response
        self._text_index = 0
        self.image_calls: list[dict[str, Any]] = []
        self.text_calls: list[dict[str, Any]] = []

    def analyze_image(self, image_path, system, user):
        self.image_calls.append(
            {"image_path": image_path, "system": system, "user": user}
        )
        return self._image_response

    def analyze_text(self, system, user):
        self.text_calls.append({"system": system, "user": user})
        if self._text_index < len(self._text_responses):
            r = self._text_responses[self._text_index]
            self._text_index += 1
            return r
        return self._text_responses[-1] if self._text_responses else "{}"


def test_run_specificity_pass_uses_analyze_text_not_image():
    provider = _MockProvider(text_responses=[json.dumps(PASS3_GOOD)])
    parsed = A.run_specificity_pass(PASS1_GOOD, PASS2_GOOD, provider)
    assert len(parsed["scene_concepts"]) == 2
    assert provider.image_calls == []
    assert len(provider.text_calls) == 1
    call = provider.text_calls[0]
    assert call["system"] == A.SPECIFICITY_SYSTEM_PROMPT
    assert "PASS 2 STRATEGY" in call["user"]


def test_run_specificity_pass_passes_taste_profile():
    provider = _MockProvider(text_responses=[json.dumps(PASS3_GOOD)])
    A.run_specificity_pass(
        PASS1_GOOD, PASS2_GOOD, provider,
        taste_profile="User dislikes: vague product swaps",
    )
    assert "vague product swaps" in provider.text_calls[0]["user"]


def test_analyze_three_pass_runs_three_calls_in_order(tmp_path: Path):
    sheet = tmp_path / "sheet.jpg"
    sheet.write_bytes(b"fake")
    vision = _MockProvider(text_responses=[], image_response=json.dumps(PASS1_GOOD))
    strategy = _MockProvider(text_responses=[json.dumps(PASS2_GOOD), json.dumps(PASS3_GOOD)])
    pass1, pass2, pass3 = A.analyze_three_pass(sheet, {}, vision, strategy)
    assert pass1["conflict_type"] == "smash"
    assert pass2["dominant_story_flow"] == "public_disrespect_viewer_defense"
    assert len(pass3["scene_concepts"]) == 2
    # Pass 1 hits vision once; Pass 2 + Pass 3 hit strategy twice.
    assert len(vision.image_calls) == 1
    assert vision.text_calls == []
    assert len(strategy.text_calls) == 2


def test_build_three_pass_result_carries_all_three_in_raw():
    result = A.build_three_pass_analysis_result(PASS1_GOOD, PASS2_GOOD, PASS3_GOOD)
    assert result.raw_analysis["analysis_mode"] == "three_pass"
    assert result.raw_analysis["visual_event_pass"] == PASS1_GOOD
    assert result.raw_analysis["hook_strategy_pass"] == PASS2_GOOD
    assert result.raw_analysis["specificity_pass"] == PASS3_GOOD


def test_build_three_pass_result_preserves_top_level_fields_from_pass2():
    result = A.build_three_pass_analysis_result(PASS1_GOOD, PASS2_GOOD, PASS3_GOOD)
    # Top-level fields are still sourced from Pass 2 (and onscreen_text from Pass 1).
    assert result.onscreen_text == "Please be honest, how are they?"
    assert result.viewer_role == "defender"


def test_build_three_pass_result_handles_empty_pass3():
    result = A.build_three_pass_analysis_result(PASS1_GOOD, PASS2_GOOD, {})
    assert result.raw_analysis["specificity_pass"] == {}
    assert result.raw_analysis["analysis_mode"] == "three_pass"
