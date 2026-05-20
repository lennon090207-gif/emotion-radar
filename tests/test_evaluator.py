"""Evaluator (calibration canary) tests. No network."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from emotion_radar import evaluator as E


def _oliver_good_report() -> dict:
    """A report that correctly identifies the Oliver HTTYD-lamp hook.
    Mirrors what we expect the model to produce after the prompt fix."""
    return {
        "visual_hook_summary": (
            "Man at outdoor market stall watches a stranger pick up his "
            "handmade dragon lamp and throw it on the floor; the lamp is "
            "smashed."
        ),
        "onscreen_text": "Please be honest, how are they?",
        "emotional_mechanic": "public disrespect + underdog maker",
        "viewer_role": "defender",
        "caption": "Please be honest, how are they?",
        "raw_analysis": {
            "environment": "outdoor weekend market stall, daytime, busy",
            "people": "underdog maker (mid-30s), passerby acting as antagonist",
            "product_or_object": "handmade dragon lamp",
            "action_or_conflict": "passerby smashed the lamp on the ground",
            "physical_action": "thrown / smashed on floor",
            "visual_conflict_detected": True,
            "frame_observations": [
                {"timestamp": "0.0s", "observation": "wide of market stall with dragon lamps on display"},
                {"timestamp": "0.5s", "observation": "stranger reaches for a dragon lamp"},
                {"timestamp": "1.0s", "observation": "stranger holds the lamp above table"},
                {"timestamp": "1.5s", "observation": "lamp leaves stranger's hand, mid-air"},
                {"timestamp": "2.0s", "observation": "lamp is on the floor, broken"},
            ],
        },
    }


def _oliver_bad_report() -> dict:
    """The Phase-2 failure case: model retreated to generic sentiment
    and missed the action entirely."""
    return {
        "visual_hook_summary": "The creator looks discouraged at his stall and then is shown crafting more lamps.",
        "onscreen_text": "Please be honest, how are they?",
        "emotional_mechanic": "creator vulnerability",
        "viewer_role": "supporter",
        "raw_analysis": {
            "environment": "outdoor area",
            "people": "creator",
            "product_or_object": "handmade lamps",
            "action_or_conflict": "creator looks unsure of his work",
            "physical_action": "",
            "visual_conflict_detected": False,
        },
    }


def _oliver_expected() -> dict:
    return {
        "required_terms": [
            "market stall",
            "smashed",
            "thrown",
            "dragon lamp",
            "public disrespect",
            "underdog maker",
        ],
        "forbidden_terms": ["street musician", "SaaS"],
        "expected_mechanic": "public disrespect + underdog maker",
    }


# ---- haystack --------------------------------------------------------------

def test_build_haystack_includes_top_level_and_raw_analysis():
    report = _oliver_good_report()
    hay = E.build_haystack(report)
    assert "dragon lamp" in hay.lower()
    assert "market stall" in hay.lower()
    # nested frame_observations text must come through via raw_analysis dump
    assert "smashed" in hay.lower() or "broken" in hay.lower()


def test_build_haystack_handles_missing_fields():
    hay = E.build_haystack({})
    assert isinstance(hay, str)


# ---- evaluation -----------------------------------------------------------

def test_good_report_passes():
    result = E.evaluate_report(_oliver_good_report(), _oliver_expected())
    assert result.passed is True
    assert result.required_terms_missing == []
    assert set(result.required_terms_matched) == set(_oliver_expected()["required_terms"])
    assert result.mechanic_match is True


def test_bad_report_fails_with_specific_missing_terms():
    result = E.evaluate_report(_oliver_bad_report(), _oliver_expected())
    assert result.passed is False
    missing = set(result.required_terms_missing)
    # All the key visual-action terms should be missing in the bad reading.
    assert "smashed" in missing
    assert "thrown" in missing
    assert "dragon lamp" in missing
    assert "public disrespect" in missing
    # mechanic mismatch
    assert result.mechanic_match is False


def test_forbidden_term_flags_taste_regression():
    report = _oliver_good_report()
    # Pretend a model proposed a "street musician" hook mutation.
    report["raw_analysis"]["hook_mutations"] = [
        {"type": "fresh", "idea": "A street musician sets up next to the stall..."},
    ]
    result = E.evaluate_report(report, _oliver_expected())
    assert "street musician" in result.forbidden_terms_present
    assert result.passed is False


def test_case_insensitive_matching():
    report = _oliver_good_report()
    report["visual_hook_summary"] = report["visual_hook_summary"].upper()
    result = E.evaluate_report(report, _oliver_expected())
    assert result.passed is True


def test_mechanic_check_skipped_when_not_specified():
    expected = {"required_terms": ["market stall"]}
    result = E.evaluate_report(_oliver_good_report(), expected)
    assert result.mechanic_match is None
    assert result.passed is True


def test_empty_required_terms_is_pass():
    result = E.evaluate_report(_oliver_good_report(), {})
    assert result.passed is True
    assert result.required_terms_total == 0


def test_invalid_required_terms_raises():
    with pytest.raises(ValueError):
        E.evaluate_report({}, {"required_terms": "not-a-list"})


def test_load_expected_reads_real_file(tmp_path: Path):
    spec = {"required_terms": ["a", "b"], "expected_mechanic": "x"}
    p = tmp_path / "expected.json"
    p.write_text(json.dumps(spec), encoding="utf-8")
    loaded = E.load_expected(p)
    assert loaded == spec


def test_load_expected_missing_file(tmp_path: Path):
    with pytest.raises(FileNotFoundError):
        E.load_expected(tmp_path / "no.json")


def test_load_expected_non_object(tmp_path: Path):
    p = tmp_path / "expected.json"
    p.write_text("[1,2,3]", encoding="utf-8")
    with pytest.raises(ValueError):
        E.load_expected(p)


# ---- shipped fixture sanity ------------------------------------------------

def test_shipped_oliver_fixture_matches_a_good_report():
    """The repo's docs/examples/oliver_expected.json should pass against
    a hand-built 'good' report. This protects the fixture itself from
    drift — if someone edits oliver_expected.json incompatibly, this
    test fails."""
    fixture_path = Path(__file__).resolve().parents[1] / "docs" / "examples" / "oliver_expected.json"
    expected = E.load_expected(fixture_path)
    result = E.evaluate_report(_oliver_good_report(), expected)
    assert result.passed is True, (
        f"shipped fixture rejects a good report. "
        f"missing_terms={result.required_terms_missing} "
        f"missing_groups={result.required_any_missing}"
    )


# ============================================================================
# required_any equivalence groups (Phase 3.1)
# ============================================================================

def _gemini_style_report() -> dict:
    """A report that mirrors the gemini-2.5-flash Phase-3 VPS output:
    correct Pass 1 evidence using 'dropped' and 'broken' instead of
    'smashed' / 'thrown'. Phase 3.1 evaluator must accept this."""
    return {
        "visual_hook_summary": (
            "At an outdoor market stall, a man picks up the maker's "
            "dragon-themed lamp, drops it, and the lamp breaks on the ground."
        ),
        "onscreen_text": "Please be honest, how are they?",
        "emotional_mechanic": "public disrespect + underdog maker (viewer-defense instinct)",
        "viewer_role": "defender",
        "caption": "Please be honest, how are they?",
        "raw_analysis": {
            "analysis_mode": "two_pass",
            "visual_event_pass": {
                "environment": "outdoor weekend market stall",
                "people": "underdog handmade seller; passerby",
                "product_or_object": "handmade dragon-themed lamp (HTTYD style)",
                "physical_action": "dropped and broken on the ground",
                "object_state_change": "lamp starts on the display table, ends broken on the ground",
                "visual_conflict_detected": True,
                "conflict_type": "drop",
                "frame_observations": [
                    {"timestamp": "0.0s", "observation": "market stall with dragon lamps"},
                    {"timestamp": "1.0s", "observation": "man picks up a dragon lamp"},
                    {"timestamp": "1.5s", "observation": "man drops the lamp"},
                    {"timestamp": "2.0s", "observation": "lamp lies broken on the ground"},
                ],
            },
            "hook_strategy_pass": {
                "why_it_works": "viewer-defense engine",
            },
        },
    }


def test_required_any_passes_when_any_synonym_present():
    expected = {
        "required_any": [
            ["smashed", "thrown", "dropped", "broken"],
        ],
    }
    report = {"visual_hook_summary": "the lamp was DROPPED on the floor"}
    result = E.evaluate_report(report, expected)
    assert result.passed is True
    assert result.required_any_total == 1
    assert len(result.required_any_matched) == 1
    assert result.required_any_matched[0]["matched"] == "dropped"
    assert result.required_any_missing == []


def test_required_any_fails_when_whole_group_missing():
    expected = {
        "required_any": [
            ["smashed", "thrown", "dropped", "broken"],
            ["dragon lamp", "HTTYD lamp"],
        ],
    }
    report = {"visual_hook_summary": "creator stares sadly at his table"}
    result = E.evaluate_report(report, expected)
    assert result.passed is False
    assert result.required_any_total == 2
    assert result.required_any_matched == []
    # Both groups missing.
    assert len(result.required_any_missing) == 2
    assert ["smashed", "thrown", "dropped", "broken"] in result.required_any_missing


def test_required_any_mixed_with_required_terms():
    expected = {
        "required_terms": ["market stall"],
        "required_any": [
            ["smashed", "dropped", "broken"],
        ],
    }
    report = {"visual_hook_summary": "at the market stall, the lamp was broken"}
    result = E.evaluate_report(report, expected)
    assert result.passed is True
    assert result.required_terms_matched == ["market stall"]
    assert result.required_any_matched[0]["matched"] == "broken"


def test_required_any_case_insensitive():
    expected = {"required_any": [["smashed", "thrown"]]}
    report = {"visual_hook_summary": "lamp was THROWN onto the floor"}
    result = E.evaluate_report(report, expected)
    assert result.passed is True


def test_required_any_invalid_spec_raises():
    with pytest.raises(ValueError):
        E.evaluate_report({}, {"required_any": "not-a-list"})
    with pytest.raises(ValueError):
        E.evaluate_report({}, {"required_any": ["not-a-nested-list"]})


def test_required_any_ignores_empty_groups_and_empty_strings():
    """Defensive: empty groups and empty strings inside groups shouldn't
    explode or skew the total."""
    expected = {
        "required_any": [
            ["", "  "],          # entire group is empty -> skipped
            ["smashed", "broken"],
        ],
    }
    report = {"visual_hook_summary": "lamp broken"}
    result = E.evaluate_report(report, expected)
    assert result.passed is True
    assert result.required_any_total == 1  # the empty group was skipped


def test_mechanic_any_matches_any_synonym():
    expected = {
        "mechanic_any": [
            "public disrespect + underdog maker",
            "public rejection of underdog maker",
            "viewer-defense",
        ],
    }
    report = {"emotional_mechanic": "public rejection of underdog maker triggers defense"}
    result = E.evaluate_report(report, expected)
    assert result.mechanic_match is True
    assert result.passed is True


def test_mechanic_any_no_match_when_actual_unrelated():
    expected = {
        "mechanic_any": [
            "public disrespect + underdog maker",
            "viewer-defense",
        ],
    }
    report = {"emotional_mechanic": "creator vulnerability"}
    result = E.evaluate_report(report, expected)
    assert result.mechanic_match is False
    assert result.passed is False


def test_mechanic_match_combines_expected_and_any():
    """If expected_mechanic is set AND mechanic_any is set, matching
    EITHER counts."""
    expected = {
        "expected_mechanic": "public disrespect + underdog maker",
        "mechanic_any": ["viewer-defense"],
    }
    # Actual matches mechanic_any but not expected_mechanic exactly.
    report = {"emotional_mechanic": "drives viewer-defense instinct"}
    result = E.evaluate_report(report, expected)
    assert result.mechanic_match is True


def test_shipped_oliver_fixture_passes_with_dropped_and_broken():
    """Phase-3.1 regression: the shipped Oliver fixture must accept a
    gemini-2.5-flash-style report that uses 'dropped' / 'broken' instead
    of 'smashed' / 'thrown'."""
    fixture_path = Path(__file__).resolve().parents[1] / "docs" / "examples" / "oliver_expected.json"
    expected = E.load_expected(fixture_path)
    result = E.evaluate_report(_gemini_style_report(), expected)
    assert result.passed is True, (
        f"Oliver fixture rejects a gemini-style report. "
        f"missing_terms={result.required_terms_missing} "
        f"missing_groups={result.required_any_missing} "
        f"mechanic_match={result.mechanic_match}"
    )


def test_shipped_oliver_fixture_still_passes_with_smashed_and_thrown():
    """Belt-and-braces: the original 'smashed/thrown' wording must
    continue to pass after the fixture rewrite. The fixture should be
    *more* permissive, not less."""
    fixture_path = Path(__file__).resolve().parents[1] / "docs" / "examples" / "oliver_expected.json"
    expected = E.load_expected(fixture_path)
    result = E.evaluate_report(_oliver_good_report(), expected)
    assert result.passed is True


def test_shipped_oliver_fixture_rejects_soft_mechanic():
    """If Pass 2 softens 'public disrespect' into 'accidentally broken'
    or 'tension and disappointment' (the failure mode we are guarding
    against), the canary must still fail."""
    soft = _gemini_style_report()
    soft["emotional_mechanic"] = "tension and disappointment after accident"
    fixture_path = Path(__file__).resolve().parents[1] / "docs" / "examples" / "oliver_expected.json"
    expected = E.load_expected(fixture_path)
    result = E.evaluate_report(soft, expected)
    # Visual terms still appear in the haystack; the mechanic check is what fails.
    assert result.mechanic_match is False
    assert result.passed is False
