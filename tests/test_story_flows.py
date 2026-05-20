"""Story Flow Library (Phase 5) — structural tests."""

from __future__ import annotations

from emotion_radar import story_flows as SF


REQUIRED_FLOW_IDS = (
    "public_disrespect_viewer_defense",
    "family_protection_validation",
    "moral_pressure_tiny_rescue",
    "comment_humiliation_public_witness",
    "stall_vulnerability_social_judgment",
    "wrong_audience_right_tribe",
    "shock_problem_immediate_fix",
    "ethical_edge_vulnerability_sympathy_surge",
    # Phase 7.1: added after live VPS test surfaced two flows the
    # original library couldn't cover.
    "direct_viewer_plea_social_contract",
    "weirdness_curiosity_reveal_loop",
)


def test_library_has_the_expected_flow_count():
    """Lets future phases add flows without breaking this test —
    REQUIRED_FLOW_IDS is the authoritative list."""
    assert len(SF.STORY_FLOWS) == len(REQUIRED_FLOW_IDS)


def test_library_contains_all_required_ids():
    ids = {flow.id for flow in SF.STORY_FLOWS}
    assert ids == set(REQUIRED_FLOW_IDS)


def test_library_includes_direct_viewer_plea_flow():
    """Phase 7.1: this flow now covers 'please don't scroll' /
    'stay 12 seconds' / 'could you be honest' hooks that previously
    got misrouted to public_disrespect."""
    flow = SF.STORY_FLOWS_BY_ID["direct_viewer_plea_social_contract"]
    assert "Direct Viewer Plea" in flow.name
    joined = " ".join(flow.example_labels).lower()
    assert "please don't scroll" in joined
    assert "stay 12 seconds" in joined
    assert "could you be honest" in joined


def test_library_includes_weirdness_curiosity_flow():
    """Phase 7.1: covers 'No I'm not a NORMAL adult' + craft-process
    curiosity-reveal hooks (the Download (2).mp4 misclassification on
    the VPS)."""
    flow = SF.STORY_FLOWS_BY_ID["weirdness_curiosity_reveal_loop"]
    assert "Weirdness" in flow.name and "Curiosity" in flow.name
    joined = " ".join(flow.example_labels).lower()
    assert "normal" in joined


def test_flow_ids_are_unique():
    ids = [flow.id for flow in SF.STORY_FLOWS]
    assert len(ids) == len(set(ids))


def test_every_flow_has_required_fields_filled():
    for flow in SF.STORY_FLOWS:
        assert flow.name and flow.name.strip()
        assert flow.steps and all(isinstance(s, str) and s.strip() for s in flow.steps)
        assert flow.emotional_physics and flow.emotional_physics.strip()
        assert flow.viewer_role and flow.viewer_role.strip()
        assert flow.comment_trigger and flow.comment_trigger.strip()
        assert flow.share_trigger and flow.share_trigger.strip()
        assert flow.cooked_risk and flow.cooked_risk.strip()
        assert 0.0 <= flow.ethical_risk_default <= 1.0
        assert flow.example_labels and all(
            isinstance(lbl, str) and lbl.strip() for lbl in flow.example_labels
        )


def test_ethical_edge_flow_has_high_ethical_risk_default():
    """The ethical-edge-vulnerability flow must default to a clearly
    high ethical_risk floor — this is the rule that protects against
    silently producing a high-virality but high-risk recommendation."""
    ethical = SF.STORY_FLOWS_BY_ID["ethical_edge_vulnerability_sympathy_surge"]
    assert ethical.ethical_risk_default >= 0.7


def test_other_flows_have_lower_ethical_default_than_ethical_edge():
    ethical = SF.STORY_FLOWS_BY_ID["ethical_edge_vulnerability_sympathy_surge"]
    for flow in SF.STORY_FLOWS:
        if flow.id == ethical.id:
            continue
        assert flow.ethical_risk_default < ethical.ethical_risk_default, (
            f"flow {flow.id} has ethical_risk_default >= the ethical-edge flow's"
        )


def test_render_for_prompt_includes_every_flow_name_and_id():
    rendered = SF.render_story_flows_for_prompt()
    for flow in SF.STORY_FLOWS:
        assert flow.name in rendered, f"render missing flow name: {flow.name}"
        assert flow.id in rendered, f"render missing flow id: {flow.id}"


def test_render_includes_steps_and_triggers():
    rendered = SF.render_story_flows_for_prompt()
    # Pick one flow and verify its full block is rendered.
    flow = SF.STORY_FLOWS_BY_ID["public_disrespect_viewer_defense"]
    for step in flow.steps:
        assert step in rendered, f"render missing step: {step}"
    assert flow.comment_trigger in rendered
    assert flow.share_trigger in rendered
    assert flow.cooked_risk in rendered


def test_storyflow_to_dict_round_trips_fields():
    flow = SF.STORY_FLOWS_BY_ID["family_protection_validation"]
    d = flow.to_dict()
    assert d["id"] == flow.id
    assert d["name"] == flow.name
    assert d["steps"] == list(flow.steps)
    assert d["example_labels"] == list(flow.example_labels)
    assert d["ethical_risk_default"] == flow.ethical_risk_default
