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
    "viral_mechanic": "public disrespect + underdog maker → viewer-defense instinct",
    "scroll_stop_reason": (
        "physical destruction of a small maker's work creates instant moral outrage"
    ),
    "viewer_role": "defender",
    "comment_trigger": "viewer wants to verbally retaliate on behalf of the maker",
    "share_trigger": "send-to-friend impulse to validate shared sense of injustice",
    "emotional_pressure": (
        "felt injustice that scrolling away would be 'letting it stand'"
    ),
    "emotional_mechanic": "public disrespect + underdog maker (viewer-defense instinct)",
    "emotions_triggered": ["anger", "protectiveness", "indignation"],
    "why_it_works": "the destruction is a public moral violation; defenders self-cast.",
    "cooked_elements": ["please be honest framing", "staged stranger trope"],
    "cooked_parts_to_avoid": ["please be honest"],
    "freshness_angle": "physical-destruction tier of public-doubt mechanic, not merely insult",
    "scroll_stop_strength_score": 0.86,
    "comment_likelihood_score": 0.81,
    "share_likelihood_score": 0.72,
    "viewer_role_strength_score": 0.83,
    "creative_transfer_potential_score": 0.74,
    "virality_capability_score": 0.79,
    "product_attachability_score": 0.62,
    "transferability_score": 0.66,
    "freshness_score": 0.71,
    "cooked_score": 0.34,
    "overall_opportunity_score": 0.78,
    "creative_hook_concepts": [
        {
            "creative_distance": "same_mechanic",
            "concept_name": "Silent Proof After Insult",
            "first_2_seconds": (
                "cut from a dismissive comment overlay to the maker silently "
                "lifting one finished piece and rotating it to show obscene detail"
            ),
            "emotional_trigger": "vindication",
            "viewer_role": "jury",
            "why_it_could_go_viral": (
                "viewer feels they have just rendered a verdict against the heckler"
            ),
            "what_to_avoid": "don't narrate; let the silence do the work",
            "believability_risk": "feels staged if the comment overlay reads written by the creator",
            "cooked_risk": "silent-reveal format is widely used; the proof must be sharp",
        }
    ],
    # Phase 5 additions
    "matched_story_flows": [
        {
            "id": "public_disrespect_viewer_defense",
            "name": "Public Disrespect -> Viewer Defense",
            "confidence": 0.92,
            "why_matched": "stranger physically destroys handmade item at a market stall",
        },
        {
            "id": "stall_vulnerability_social_judgment",
            "name": "Stall Vulnerability -> Social Judgment",
            "confidence": 0.71,
            "why_matched": "market stall display in plain view sets up public-judgment context",
        },
    ],
    "dominant_story_flow": "public_disrespect_viewer_defense",
    "story_flow_steps_observed": [
        "stranger approaches the market stall",
        "stranger handles and destroys the handmade lamp",
        "maker visibly absorbs the disrespect",
        "viewer is positioned as defender",
    ],
    "story_flow_strength_score": 0.9,
    "novelty_beyond_baseline_score": 0.45,
    "ethical_risk_score": 0.3,
    "cringe_risk_score": 0.35,
    "breakout_potential_score": 0.72,
    "variations": [
        {"concept_name": "Receipt of Cruelty", "story_flow_id": "public_disrespect_viewer_defense",
         "first_2_seconds": "maker silently holds up a printed screenshot of a rude DM next to the finished piece at the stall",
         "emotional_trigger": "vindication", "viewer_role": "jury",
         "why_it_could_go_viral": "physical receipt format is rare and tactile",
         "what_is_new": "the printed-screenshot prop instead of a verbal exchange",
         "what_is_cooked_to_avoid": "do not use 'please be honest' wording",
         "believability_risk": "fabricated-screenshot reads kill it if the DM is too on-the-nose"},
        {"concept_name": "Walked-Past Verdict", "story_flow_id": "public_disrespect_viewer_defense",
         "first_2_seconds": "wide shot of stall; a couple stops, picks the piece up, makes a face, sets it down hard, walks off",
         "emotional_trigger": "second-hand indignation", "viewer_role": "defender",
         "why_it_could_go_viral": "the put-down beat is silent and damning",
         "what_is_new": "no spoken comment, no overlay — body language carries the disrespect",
         "what_is_cooked_to_avoid": "do not zoom in on a sad reaction shot",
         "believability_risk": "needs real candid timing; staged passerby kills it"},
        {"concept_name": "Tribe Snap-Defense", "story_flow_id": "wrong_audience_right_tribe",
         "first_2_seconds": "stall passerby calls the work weird; text immediately names the tribe that would defend it",
         "emotional_trigger": "identity claim", "viewer_role": "tribe member",
         "why_it_could_go_viral": "comments fill with the tribe self-identifying",
         "what_is_new": "the tribe is named within the first 2 seconds, not the punchline",
         "what_is_cooked_to_avoid": "POV-when-X-says-Y framing is cooked",
         "believability_risk": "tribe label must be claimable, not too niche"},
        {"concept_name": "Almost-Closed Save", "story_flow_id": "stall_vulnerability_social_judgment",
         "first_2_seconds": "maker starts packing up after a quiet day; one stranger stops",
         "emotional_trigger": "second-hand-pride", "viewer_role": "rescuer",
         "why_it_could_go_viral": "viewer is positioned as one of the people who could save the moment",
         "what_is_new": "no caption tells the story; the pack-up beat is silent",
         "what_is_cooked_to_avoid": "narrated 'sad creator' captions are cooked",
         "believability_risk": "fails if the rescue timing is too clean"},
        {"concept_name": "Comment Card on the Table", "story_flow_id": "stall_vulnerability_social_judgment",
         "first_2_seconds": "a handwritten note on the stall reads a real rude comment from a previous day",
         "emotional_trigger": "moral outrage", "viewer_role": "appreciator",
         "why_it_could_go_viral": "the physical note format collapses time and creates witness role",
         "what_is_new": "asynchronous evidence rather than live confrontation",
         "what_is_cooked_to_avoid": "do not over-handwrite the note; let it look authentic",
         "believability_risk": "feels manipulative if the comment is too perfect"},
    ],
    "pioneer_concepts": [
        {"concept_name": "Receipt Wall",
         "inspired_by_story_flow_id": "comment_humiliation_public_witness",
         "first_2_seconds": "creator silently pins printed rude DMs to a corkboard behind the work; one tap, no narration",
         "emotional_physics": "physical, tactile evidence of cruelty against the maker; viewers become jurors",
         "why_it_is_not_a_direct_copy": "no creator-reaction shot, no read-aloud; the wall itself is the indictment",
         "why_it_could_be_breakout": "tactile-evidence formats are underused; the prop is the hook",
         "viewer_comment_impulse": "urge to add their own receipt or to defend",
         "ethical_or_cringe_risk": "real names must be redacted; otherwise high cringe + brand risk"},
        {"concept_name": "Bystander Camera",
         "inspired_by_story_flow_id": "public_disrespect_viewer_defense",
         "first_2_seconds": "filmed from a stranger's POV in line behind the rude customer; the maker is barely visible",
         "emotional_physics": "third-person witnessing accelerates the defender response",
         "why_it_is_not_a_direct_copy": "the maker is NOT the protagonist of the frame; the bystander is",
         "why_it_could_be_breakout": "POV inversion makes the viewer feel they were there",
         "viewer_comment_impulse": "urge to roleplay 'what I would have said'",
         "ethical_or_cringe_risk": "low if the framing reads candid, high if it reads staged"},
        {"concept_name": "Quiet Apology",
         "inspired_by_story_flow_id": "wrong_audience_right_tribe",
         "first_2_seconds": "a passerby comes back to the stall the next day and quietly apologises for their reaction yesterday",
         "emotional_physics": "delayed-justice loop; tribe sees the antagonist redeemed",
         "why_it_is_not_a_direct_copy": "the antagonist becomes the hero; no defender role for the viewer at all",
         "why_it_could_be_breakout": "redemption arcs in 1-2s are rare; high share for 'restored faith'",
         "viewer_comment_impulse": "urge to share to someone who needs faith restored",
         "ethical_or_cringe_risk": "feels manipulative if the apology is read off a script"},
        {"concept_name": "Stranger's Note",
         "inspired_by_story_flow_id": "stall_vulnerability_social_judgment",
         "first_2_seconds": "maker finds a handwritten note tucked under one of the pieces at the stall",
         "emotional_physics": "anonymous appreciation breaks the public-judgment frame",
         "why_it_is_not_a_direct_copy": "no insult, no destruction — the hook is a positive intervention witnessed in silence",
         "why_it_could_be_breakout": "appreciation-witnessed-in-silence is underused; pairs with the comment_humiliation flow as a polarity flip",
         "viewer_comment_impulse": "urge to claim 'I would have written this'",
         "ethical_or_cringe_risk": "fails if the note is too sentimental"},
        {"concept_name": "Last Item Standing",
         "inspired_by_story_flow_id": "moral_pressure_tiny_rescue",
         "first_2_seconds": "wide shot of the stall with one piece left at end-of-day; maker reaches to pack it",
         "emotional_physics": "scarcity + moral pressure to be the one who saved the last piece",
         "why_it_is_not_a_direct_copy": "no plea, no caption, no 'please don't scroll'; only the visual stake",
         "why_it_could_be_breakout": "tactile, time-pressured rescue framing without the cooked phrasing",
         "viewer_comment_impulse": "urge to ask 'is it still available?'",
         "ethical_or_cringe_risk": "feels staged if the stall is suspiciously well-lit; low otherwise"},
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
    assert pass2["overall_opportunity_score"] == 0.78
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
    assert result.emotions_triggered == ["anger", "protectiveness", "indignation"]
    assert result.product_attachability_score == 0.62
    assert result.transferability_score == 0.66
    assert result.freshness_score == 0.71
    assert result.cooked_score == 0.34
    assert result.overall_opportunity_score == 0.78
    # Phase 4: hook_mutations is now sourced from creative_hook_concepts.
    assert len(result.hook_mutations) == 1
    assert result.hook_mutations[0]["creative_distance"] == "same_mechanic"


def test_merge_prefers_creative_hook_concepts_over_legacy_hook_mutations():
    """Phase 4: when Pass 2 returns both, creative_hook_concepts wins."""
    pass2 = {
        **PASS2_GOOD,
        "hook_mutations": [{"type": "safe", "idea": "legacy entry that should lose"}],
    }
    result = A.build_two_pass_analysis_result(PASS1_GOOD, pass2)
    assert result.hook_mutations == PASS2_GOOD["creative_hook_concepts"]


def test_merge_falls_back_to_legacy_hook_mutations_if_no_concepts():
    """Older Pass-2 prompts that still emit hook_mutations work."""
    pass2 = {
        k: v for k, v in PASS2_GOOD.items() if k != "creative_hook_concepts"
    }
    pass2["hook_mutations"] = [
        {"type": "safe", "idea": "legacy idea", "opening_scene": "..."}
    ]
    result = A.build_two_pass_analysis_result(PASS1_GOOD, pass2)
    assert len(result.hook_mutations) == 1
    assert result.hook_mutations[0]["type"] == "safe"


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


def test_merge_drops_non_list_concepts_and_mutations():
    """If neither creative_hook_concepts nor legacy hook_mutations is a
    list, the merged hook_mutations is empty (no crash)."""
    pass2 = {
        k: v for k, v in PASS2_GOOD.items() if k != "creative_hook_concepts"
    }
    pass2["creative_hook_concepts"] = "not a list"
    pass2["hook_mutations"] = "also not a list"
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


# ============================================================================
# Phase 5: Story flow fields survive through merge
# ============================================================================

def test_merge_preserves_matched_story_flows_in_raw_analysis():
    result = A.build_two_pass_analysis_result(PASS1_GOOD, PASS2_GOOD)
    hsp = result.raw_analysis["hook_strategy_pass"]
    assert hsp["dominant_story_flow"] == "public_disrespect_viewer_defense"
    matched = hsp["matched_story_flows"]
    assert isinstance(matched, list) and len(matched) == 2
    ids = {m["id"] for m in matched if isinstance(m, dict)}
    assert "public_disrespect_viewer_defense" in ids


def test_merge_preserves_variations_quota_5():
    result = A.build_two_pass_analysis_result(PASS1_GOOD, PASS2_GOOD)
    variations = result.raw_analysis["hook_strategy_pass"]["variations"]
    assert isinstance(variations, list)
    assert len(variations) == 5
    # Every variation references a story_flow_id.
    for v in variations:
        assert v.get("story_flow_id"), f"variation missing story_flow_id: {v}"


def test_merge_preserves_pioneer_concepts_quota_5():
    result = A.build_two_pass_analysis_result(PASS1_GOOD, PASS2_GOOD)
    pioneers = result.raw_analysis["hook_strategy_pass"]["pioneer_concepts"]
    assert isinstance(pioneers, list)
    assert len(pioneers) == 5
    for p in pioneers:
        assert p.get("inspired_by_story_flow_id"), f"pioneer missing inspired_by_story_flow_id: {p}"
        assert p.get("concept_name"), f"pioneer missing concept_name: {p}"


def test_merge_preserves_phase5_scores():
    result = A.build_two_pass_analysis_result(PASS1_GOOD, PASS2_GOOD)
    hsp = result.raw_analysis["hook_strategy_pass"]
    for key in (
        "story_flow_strength_score",
        "novelty_beyond_baseline_score",
        "ethical_risk_score",
        "cringe_risk_score",
        "breakout_potential_score",
    ):
        assert key in hsp, f"Phase 5 score missing after merge: {key}"
        assert 0.0 <= hsp[key] <= 1.0


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
