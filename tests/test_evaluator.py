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
    assert result.passed is True, f"shipped fixture rejects a good report. missing={result.required_terms_missing}"
