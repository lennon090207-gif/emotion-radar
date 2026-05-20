"""Calibration / evaluation for known examples.

This is intentionally NOT a semantic evaluator. It's a case-insensitive
substring check against a flattened text view of a report. Use it as a
fast canary for known hooks (e.g. the Oliver HTTYD-lamp video): if the
model output doesn't even mention destruction or "market stall", you
know prompt or pipeline regressed.

`expected.json` schema (all keys optional):

  {
    "required_terms":    ["market stall"],              # each MUST be present
    "required_any": [                                   # each group: at least
      ["smashed", "thrown", "dropped", "broken"],       #   one synonym must
      ["dragon lamp", "HTTYD lamp"],                    #   be present
      ["public disrespect", "rejection", "mockery"],
      ["underdog maker", "handmade seller"]
    ],
    "forbidden_terms":   ["street musician", "SaaS"],
    "expected_mechanic": "public disrespect + underdog maker",
    "mechanic_any": [                                   # optional alt phrasings
      "public disrespect + underdog maker",
      "public rejection of underdog maker",
      "viewer-defense"
    ]
  }

Pass criteria (all must hold):
  - every required_term is in the haystack,
  - every required_any group has >=1 synonym in the haystack,
  - no forbidden_term is in the haystack,
  - if expected_mechanic or mechanic_any is set, the report's
    emotional_mechanic matches at least one of them (case-insensitive
    substring, either direction).
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
    # required_any: each group needs at least one matched synonym.
    required_any_total: int = 0
    # For each satisfied group: {"group": [synonyms...], "matched": "the term that hit"}
    required_any_matched: list[dict[str, Any]] = field(default_factory=list)
    # For each missed group: the full list of synonyms that were all absent.
    required_any_missing: list[list[str]] = field(default_factory=list)
    forbidden_terms_present: list[str] = field(default_factory=list)
    expected_mechanic: str | None = None
    mechanic_any: list[str] = field(default_factory=list)
    mechanic_match: bool | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "passed": self.passed,
            "required_terms_total": self.required_terms_total,
            "required_terms_matched": list(self.required_terms_matched),
            "required_terms_missing": list(self.required_terms_missing),
            "required_any_total": self.required_any_total,
            "required_any_matched": list(self.required_any_matched),
            "required_any_missing": list(self.required_any_missing),
            "forbidden_terms_present": list(self.forbidden_terms_present),
            "expected_mechanic": self.expected_mechanic,
            "mechanic_any": list(self.mechanic_any),
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


def _clean_str(v: Any) -> str | None:
    if isinstance(v, str) and v.strip():
        return v.strip()
    return None


def evaluate_report(
    report: dict[str, Any],
    expected: dict[str, Any],
) -> EvaluationResult:
    """Case-insensitive substring evaluation of a report against an
    `expected` spec. Returns an EvaluationResult; never raises on a
    failed term (only on a malformed spec)."""
    required = expected.get("required_terms") or []
    required_any = expected.get("required_any") or []
    forbidden = expected.get("forbidden_terms") or []
    expected_mechanic = expected.get("expected_mechanic")
    mechanic_any = expected.get("mechanic_any") or []

    if not isinstance(required, list):
        raise ValueError("expected.required_terms must be a list of strings.")
    if not isinstance(required_any, list):
        raise ValueError("expected.required_any must be a list of lists of strings.")
    for i, group in enumerate(required_any):
        if not isinstance(group, list):
            raise ValueError(
                f"expected.required_any[{i}] must be a list of strings."
            )
    if not isinstance(forbidden, list):
        raise ValueError("expected.forbidden_terms must be a list of strings.")
    if not isinstance(mechanic_any, list):
        raise ValueError("expected.mechanic_any must be a list of strings.")

    haystack = build_haystack(report).lower()

    # --- required_terms ---
    matched: list[str] = []
    missing: list[str] = []
    cleaned_required: list[str] = []
    for term in required:
        cleaned = _clean_str(term)
        if cleaned is None:
            continue
        cleaned_required.append(cleaned)
        (matched if cleaned.lower() in haystack else missing).append(cleaned)

    # --- required_any ---
    any_matched: list[dict[str, Any]] = []
    any_missing: list[list[str]] = []
    cleaned_groups: list[list[str]] = []
    for group in required_any:
        synonyms = [_clean_str(s) for s in group]
        synonyms = [s for s in synonyms if s]
        if not synonyms:
            continue
        cleaned_groups.append(synonyms)
        hit: str | None = None
        for syn in synonyms:
            if syn.lower() in haystack:
                hit = syn
                break
        if hit is not None:
            any_matched.append({"group": synonyms, "matched": hit})
        else:
            any_missing.append(synonyms)

    # --- forbidden_terms ---
    forbidden_hits: list[str] = []
    for term in forbidden:
        cleaned = _clean_str(term)
        if cleaned is None:
            continue
        if cleaned.lower() in haystack:
            forbidden_hits.append(cleaned)

    # --- mechanic ---
    expected_mechanic_clean = _clean_str(expected_mechanic)
    mechanic_any_clean = [s for s in (_clean_str(x) for x in mechanic_any) if s]
    mechanic_match: bool | None = None
    if expected_mechanic_clean or mechanic_any_clean:
        actual = (report.get("emotional_mechanic") or "").lower()
        candidates: list[str] = []
        if expected_mechanic_clean:
            candidates.append(expected_mechanic_clean)
        candidates.extend(mechanic_any_clean)
        matched_any = False
        if actual:
            for cand in candidates:
                cand_l = cand.lower()
                if cand_l in actual or actual in cand_l:
                    matched_any = True
                    break
        mechanic_match = matched_any

    passed = (
        len(missing) == 0
        and not any_missing
        and not forbidden_hits
        and (mechanic_match is None or mechanic_match)
    )

    return EvaluationResult(
        passed=passed,
        required_terms_total=len(cleaned_required),
        required_terms_matched=matched,
        required_terms_missing=missing,
        required_any_total=len(cleaned_groups),
        required_any_matched=any_matched,
        required_any_missing=any_missing,
        forbidden_terms_present=forbidden_hits,
        expected_mechanic=expected_mechanic_clean,
        mechanic_any=mechanic_any_clean,
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
