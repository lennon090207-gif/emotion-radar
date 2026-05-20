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
    # Phase 4: viral-focused schema + legacy compatibility fields.
    for key in (
        "visual_hook_summary",
        "viral_mechanic",
        "scroll_stop_reason",
        "viewer_role",
        "comment_trigger",
        "share_trigger",
        "emotional_pressure",
        "emotional_mechanic",
        "emotions_triggered",
        "why_it_works",
        "cooked_elements",
        "cooked_parts_to_avoid",
        "freshness_angle",
        "scroll_stop_strength_score",
        "comment_likelihood_score",
        "share_likelihood_score",
        "viewer_role_strength_score",
        "creative_transfer_potential_score",
        "virality_capability_score",
        "product_attachability_score",
        "transferability_score",
        "freshness_score",
        "cooked_score",
        "overall_opportunity_score",
        "creative_hook_concepts",
    ):
        assert key in sp, f"Pass 2 prompt missing key: {key}"


def test_pass2_prompt_does_not_pin_target_world_to_handmade_only():
    """Phase 4 reorientation: the prompt must NOT restrict mutations to
    the handmade-only target world that Phase 3.1 enforced. Concepts
    should be free to live in any believable organic emotional setup."""
    sp = A.HOOK_STRATEGY_SYSTEM_PROMPT.lower()
    # The old hard-constraint section header should be gone.
    assert "target world (hard constraint)" not in sp
    # And the old "EVERY mutation MUST live" pin to handmade should be gone.
    assert "every mutation must live in this world" not in sp


def test_pass2_prompt_states_product_secondary_mechanic_primary():
    """Phase 4 core reorientation: the prompt must explicitly say the
    product is secondary and the viral mechanic is the asset."""
    sp = A.HOOK_STRATEGY_SYSTEM_PROMPT
    sp_lower = sp.lower()
    assert "product is secondary" in sp_lower
    assert "viral hook mechanic is primary" in sp_lower or "mechanic is primary" in sp_lower


def test_pass2_prompt_forbids_product_swap_lists():
    """The exact failure mode that this phase exists to fix:
    'same hook but with mugs / candles / jewelry'."""
    sp = A.HOOK_STRATEGY_SYSTEM_PROMPT.lower()
    assert "product-swap" in sp or "product swap" in sp
    # The specific bad-example vocabulary that the prompt warns against.
    for swap_token in ("mugs", "candles", "jewelry"):
        assert swap_token in sp, f"product-swap warning should mention: {swap_token}"
    # The structural rule.
    assert "mutate the emotional situation" in sp or "emotional situation" in sp


def test_pass2_prompt_requires_mutating_situation_not_object():
    sp = A.HOOK_STRATEGY_SYSTEM_PROMPT
    # The strongest phrasing of the rule.
    assert "Mutate the EMOTIONAL SITUATION" in sp or "mutate the emotional situation" in sp.lower()
    # The "only differ in object" antipattern is called out.
    assert "only differ in" in sp.lower() or "vary the *situation*" in sp.lower() or "vary the situation" in sp.lower()


def test_pass2_prompt_explicitly_rejects_unrelated_niches():
    """Carry-over: SaaS / crypto / fitness / etc. are still off-limits
    unless the source supports the leap."""
    sp = A.HOOK_STRATEGY_SYSTEM_PROMPT.lower()
    for forbidden in (
        "saas",
        "fitness",
        "crypto",
        "real estate",
        "dropshipping",
    ):
        assert forbidden in sp, f"Pass 2 prompt should explicitly reject: {forbidden}"
    # And the new escape clause.
    assert "unless the source" in sp


def test_pass2_prompt_concept_distribution_2_3_2_1():
    """Phase 4 quota: 2 same_mechanic / 3 adjacent_leap / 2 big_swing /
    1 wildcard = 8 total."""
    sp = A.HOOK_STRATEGY_SYSTEM_PROMPT
    assert "EXACTLY 8" in sp
    assert '2 "same_mechanic"' in sp
    assert '3 "adjacent_leap"' in sp
    assert '2 "big_swing"' in sp
    assert '1 "wildcard"' in sp


def test_pass2_prompt_lists_per_concept_required_fields():
    """The eight per-concept fields must be enumerated."""
    sp = A.HOOK_STRATEGY_SYSTEM_PROMPT
    for fld in (
        "creative_distance",
        "concept_name",
        "first_2_seconds",
        "emotional_trigger",
        "viewer_role",
        "why_it_could_go_viral",
        "what_to_avoid",
        "believability_risk",
        "cooked_risk",
    ):
        assert fld in sp, f"per-concept field missing in Pass 2 prompt: {fld}"


def test_pass2_prompt_lists_virality_focused_scores():
    sp = A.HOOK_STRATEGY_SYSTEM_PROMPT
    for score in (
        "scroll_stop_strength_score",
        "comment_likelihood_score",
        "share_likelihood_score",
        "viewer_role_strength_score",
        "creative_transfer_potential_score",
        "virality_capability_score",
    ):
        assert score in sp, f"virality score missing in Pass 2 prompt: {score}"


# --- Phase 5: Story Flow Library, Variations, Pioneer Concepts -------------

def test_pass2_prompt_includes_all_eight_story_flow_names():
    sp = A.HOOK_STRATEGY_SYSTEM_PROMPT
    expected_names = (
        "Public Disrespect -> Viewer Defense",
        "Family Protection -> Validation",
        "Moral Pressure -> Tiny Rescue Action",
        "Comment Humiliation -> Public Witness",
        "Stall Vulnerability -> Social Judgment",
        "Wrong Audience -> Right Tribe",
        "Shock Problem -> Immediate Fix",
        "Ethical Edge Vulnerability -> Sympathy Surge",
    )
    for name in expected_names:
        assert name in sp, f"story flow name missing in prompt: {name}"


def test_pass2_prompt_includes_all_eight_story_flow_ids():
    sp = A.HOOK_STRATEGY_SYSTEM_PROMPT
    expected_ids = (
        "public_disrespect_viewer_defense",
        "family_protection_validation",
        "moral_pressure_tiny_rescue",
        "comment_humiliation_public_witness",
        "stall_vulnerability_social_judgment",
        "wrong_audience_right_tribe",
        "shock_problem_immediate_fix",
        "ethical_edge_vulnerability_sympathy_surge",
    )
    for fid in expected_ids:
        assert fid in sp, f"story flow id missing in prompt: {fid}"


def test_pass2_prompt_variation_quota_exactly_5():
    sp = A.HOOK_STRATEGY_SYSTEM_PROMPT
    assert "Variations (EXACTLY 5)" in sp


def test_pass2_prompt_pioneer_quota_exactly_5():
    sp = A.HOOK_STRATEGY_SYSTEM_PROMPT
    # Either phrasing of "5 pioneer concepts" should be present.
    assert "Pioneer concepts (EXACTLY 5)" in sp


def test_pass2_prompt_lists_variation_required_fields():
    sp = A.HOOK_STRATEGY_SYSTEM_PROMPT
    for fld in (
        "story_flow_id",
        "what_is_new",
        "what_is_cooked_to_avoid",
        "believability_risk",
    ):
        assert fld in sp, f"variation field missing in Pass 2 prompt: {fld}"


def test_pass2_prompt_lists_pioneer_required_fields():
    sp = A.HOOK_STRATEGY_SYSTEM_PROMPT
    for fld in (
        "inspired_by_story_flow_id",
        "emotional_physics",
        "why_it_is_not_a_direct_copy",
        "why_it_could_be_breakout",
        "viewer_comment_impulse",
        "ethical_or_cringe_risk",
    ):
        assert fld in sp, f"pioneer field missing in Pass 2 prompt: {fld}"


def test_pass2_prompt_lists_phase5_scoring_fields():
    sp = A.HOOK_STRATEGY_SYSTEM_PROMPT
    for score in (
        "story_flow_strength_score",
        "novelty_beyond_baseline_score",
        "ethical_risk_score",
        "cringe_risk_score",
        "breakout_potential_score",
    ):
        assert score in sp, f"Phase 5 score missing in Pass 2 prompt: {score}"


def test_pass2_prompt_mentions_ethical_risk_for_ethical_edge_flow():
    """The ethical-edge-vulnerability flow must be specifically tied to
    a high ethical_risk_score in the prompt — that's how the model
    knows to flag the high-virality / high-risk hooks."""
    sp = A.HOOK_STRATEGY_SYSTEM_PROMPT.lower()
    assert "ethical_risk_score" in sp
    assert "ethical edge vulnerability" in sp
    # Floor guidance present.
    assert ">= 0.7" in sp or "ethical_risk_default" in sp


def test_pass2_prompt_includes_pioneer_primary_goal_phrasing():
    sp = A.HOOK_STRATEGY_SYSTEM_PROMPT
    assert "PRIMARY GOAL" in sp or "primary goal" in sp.lower()


def test_pass2_prompt_schema_includes_phase5_keys():
    sp = A.HOOK_STRATEGY_SYSTEM_PROMPT
    for key in (
        "matched_story_flows",
        "dominant_story_flow",
        "story_flow_steps_observed",
        "variations",
        "pioneer_concepts",
    ):
        assert key in sp, f"Phase 5 schema key missing in prompt: {key}"


def test_pass2_prompt_assembly_failure_modes_explicit():
    """The assembly helper raises if either anchor is missing. This
    test is a smoke check that the helper exists and is callable."""
    s = A._assemble_hook_strategy_prompt()
    assert isinstance(s, str)
    assert len(s) > 5000  # the assembled prompt should be substantially larger than the base
    # Story flow library content must have landed in the output.
    assert "public_disrespect_viewer_defense" in s
    assert "story_flow_strength_score" in s


def test_pass2_prompt_lists_cooked_phrases_to_avoid():
    sp = A.HOOK_STRATEGY_SYSTEM_PROMPT
    for cooked in (
        "Nobody will ever buy your",
        "Please be honest",
        "Would you buy one",
        "POV:",
    ):
        assert cooked in sp, f"Pass 2 prompt missing cooked-phrase warning: {cooked}"


# --- Pass 2 binding rules (Phase 3.1) --------------------------------------

def test_pass2_prompt_forbids_accidental_label_without_pass1_signal():
    """The strategist must not declare an action 'accidental' unless Pass
    1 explicitly said so. This is the rule that stopped gemini's correct
    'dropped' + 'broke on ground' evidence from being softened into
    'accidentally broken'."""
    sp = A.HOOK_STRATEGY_SYSTEM_PROMPT.lower()
    assert "accident" in sp  # the word appears in the rule
    # The phrasing of the rule — match phrases that mean "don't call it
    # accidental unless Pass 1 says it is".
    assert (
        "do not call" in sp and "accidental" in sp
    ) or "unless pass 1 explicitly" in sp


def test_pass2_prompt_destruction_plus_insult_resolves_to_public_disrespect():
    """Destruction in Pass 1 + insulting on-screen text => the mechanic
    MUST be public disrespect + underdog maker. This is the rule that
    prevents 'tension and disappointment' / 'creator validation' style
    softening."""
    sp = A.HOOK_STRATEGY_SYSTEM_PROMPT
    lowered = sp.lower()
    # Destruction vocabulary is enumerated.
    for word in ("dropped", "broken", "smashed", "shattered", "thrown",
                 "on the ground", "on the floor"):
        assert word in lowered, f"destruction term missing in Pass 2 prompt: {word}"
    # Insult vocabulary is enumerated.
    for word in ("nobody will buy", "stop making", "worthless",
                 "please be honest", "would you buy"):
        assert word in lowered, f"insult-text trigger missing in Pass 2 prompt: {word}"
    # The conclusion phrase is spelled out.
    assert "public disrespect + underdog maker" in sp
    # The soft framings are explicitly rejected.
    for soft in (
        "tension and disappointment",
        "accidentally broken",
        "creator validation",
        "creator vulnerability",
        "generic appreciation",
    ):
        assert soft in lowered, f"Pass 2 prompt should explicitly reject '{soft}'"


def test_pass2_prompt_conflict_must_be_central_when_detected():
    """If Pass 1 visual_conflict_detected is true, the conflict has to
    appear in BOTH visual_hook_summary AND emotional_mechanic."""
    sp = A.HOOK_STRATEGY_SYSTEM_PROMPT.lower()
    assert "visual_conflict_detected" in sp
    assert "central" in sp
    # The instruction names both target fields.
    assert "visual_hook_summary" in sp
    assert "emotional_mechanic" in sp


def test_pass2_prompt_mutations_keep_edge_when_source_is_conflict():
    """When Pass 1 detects conflict, every mutation must preserve the
    conflict / disrespect / underdog edge. Positive-validation
    mutations are explicitly forbidden in this regime."""
    sp = A.HOOK_STRATEGY_SYSTEM_PROMPT.lower()
    # The "preserve the edge" rule is stated.
    assert "preserve" in sp and "edge" in sp
    # Positive-validation examples are listed as forbidden.
    for soft_mutation in (
        "thumbs up",
        "takes a selfie",
        "smiles at the maker",
        "i love it",
    ):
        assert soft_mutation in sp, (
            f"Pass 2 prompt should forbid soft mutation example: {soft_mutation}"
        )
    # "Vary the niche, not the polarity" is the explicit guidance.
    assert "do not vary the polarity" in sp or "vary the niche" in sp


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
