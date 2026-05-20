"""Visual hook analysis.

THIS IS A PLACEHOLDER. The schema is ready; the implementation is not.

Next phase will replace `analyze_contact_sheet` with a call to a vision/LLM
model. That call will:

  - read ONLY the contact sheet image (the burned-in frames from t=0..5s)
  - identify the visual event happening in those frames:
      who is on screen, where, what object/product is involved, what
      action or conflict occurs
  - read any on-screen text rendered into the frames
  - classify the emotional mechanic at play
      (e.g. "public disrespect of underdog maker → viewer-defense instinct")
  - identify the viewer role the hook conjures
      (defender, judge, voyeur, learner, accomplice, ...)
  - score (0-1):
      * freshness     — is this mechanic novel right now
      * cooked_score  — saturation of this mechanic across feeds
      * transferability — can the mechanic be reused in other niches
      * product_attachability — fit for product/offer attachment
      * overall_opportunity_score — combination above
  - generate hook_mutations: 3-5 new hook ideas that reuse the mechanic
      in *different* niches/products (NOT direct copies)

The function takes the contact sheet path plus the normalized metadata so
the model can use caption text as weak prior, but the classification
must be primarily driven by the visual frames.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from .models import AnalysisResult


def analyze_contact_sheet(
    contact_sheet_path: Path,
    metadata: dict[str, Any],
) -> AnalysisResult:
    """Placeholder. Returns an AnalysisResult with all concept fields null.

    Do NOT add heuristics here (e.g. keyword matching on the caption). The
    point of this MVP is to prove the plumbing without faking signal.
    Real signal arrives when the vision model is wired in."""
    # Touch the args so linters don't complain; explicit no-op.
    _ = contact_sheet_path
    _ = metadata
    return AnalysisResult(
        raw_analysis={"status": "stub", "note": "vision model not wired yet"},
    )
