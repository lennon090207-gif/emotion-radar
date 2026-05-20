"""Calibration / evaluation for known examples.

This is intentionally NOT a semantic evaluator. It's a case-insensitive
substring check against a flattened text view of a report. Use it as a
fast canary for known hooks (e.g. the Oliver HTTYD-lamp video): if the
model output doesn't even mention the words "smashed" or "market
stall", you know prompt or pipeline regressed.

`expected.json` schema (all keys optional):
  {
    "required_terms":    ["market stall", "smashed", "thrown", ...],
    "forbidden_terms":   ["street musician", "SaaS", ...],
    "expected_mechanic": "public disrespect + underdog maker"
  }
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any


@dataclass
class EvaluationResult:
    passed: bool
    required_terms_total: int
    required_terms_matched: list[str] = field(default_factory=list)
    required_terms_missing: list[str] = field(default_factory=list)
    forbidden_terms_present: list[str] = field(default_factory=list)
    expected_mechanic: str | None = None
    mechanic_match: bool | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "passed": self.passed,
            "required_terms_total": self.required_terms_total,
            "required_terms_matched": list(self.required_terms_matched),
            "required_terms_missing": list(self.required_terms_missing),
            "forbidden_terms_present": list(self.forbidden_terms_present),
            "expected_mechanic": self.expected_mechanic,
            "mechanic_match": self.mechanic_match,
        }


_HAYSTACK_KEYS = (
    "visual_hook_summary",
    "onscreen_text",
    "emotional_mechanic",
    "viewer_role",
    "caption",
)


def build_haystack(report: dict[str, Any]) -> str:
    """Flatten all text-bearing parts of the report into one searchable
    string. raw_analysis is JSON-dumped so nested fields
    (frame_observations, physical_action, environment, etc.) all
    participate in the substring search."""
    parts: list[str] = []
    for key in _HAYSTACK_KEYS:
        v = report.get(key)
        if isinstance(v, str) and v:
            parts.append(v)
    raw = report.get("raw_analysis")
    if isinstance(raw, (dict, list)):
        parts.append(json.dumps(raw, ensure_ascii=False))
    elif isinstance(raw, str):
        parts.append(raw)
    return " \n ".join(parts)


def evaluate_report(
    report: dict[str, Any],
    expected: dict[str, Any],
) -> EvaluationResult:
    """Case-insensitive substring evaluation of a report against an
    `expected` spec. Returns an EvaluationResult; never raises on a
    failed term (only on a malformed spec)."""
    required = expected.get("required_terms") or []
    forbidden = expected.get("forbidden_terms") or []
    expected_mechanic = expected.get("expected_mechanic")

    if not isinstance(required, list):
        raise ValueError("expected.required_terms must be a list of strings.")
    if not isinstance(forbidden, list):
        raise ValueError("expected.forbidden_terms must be a list of strings.")

    haystack = build_haystack(report).lower()

    matched: list[str] = []
    missing: list[str] = []
    for term in required:
        if not isinstance(term, str) or not term.strip():
            continue
        if term.lower() in haystack:
            matched.append(term)
        else:
            missing.append(term)

    forbidden_hits: list[str] = []
    for term in forbidden:
        if not isinstance(term, str) or not term.strip():
            continue
        if term.lower() in haystack:
            forbidden_hits.append(term)

    mechanic_match: bool | None = None
    if isinstance(expected_mechanic, str) and expected_mechanic.strip():
        actual = (report.get("emotional_mechanic") or "").lower()
        em = expected_mechanic.lower()
        # Either direction counts: the actual mechanic mentions the expected
        # phrase, or the expected phrase contains the actual (handles longer
        # actual descriptions that happen to nest the canonical phrase).
        mechanic_match = (em in actual) or (actual != "" and actual in em)

    passed = (
        len(missing) == 0
        and not forbidden_hits
        and (mechanic_match is None or mechanic_match)
    )

    return EvaluationResult(
        passed=passed,
        required_terms_total=len([t for t in required if isinstance(t, str) and t.strip()]),
        required_terms_matched=matched,
        required_terms_missing=missing,
        forbidden_terms_present=forbidden_hits,
        expected_mechanic=expected_mechanic if isinstance(expected_mechanic, str) else None,
        mechanic_match=mechanic_match,
    )


def load_expected(path: str | Any) -> dict[str, Any]:
    """Read and minimally validate an expected.json file."""
    from pathlib import Path
    p = Path(path)
    if not p.is_file():
        raise FileNotFoundError(f"expected file not found: {p}")
    data = json.loads(p.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError(f"expected file must be a JSON object: {p}")
    return data
