"""Scoring stubs.

These functions are placeholders that document the *intended* scoring
dimensions so the rest of the project can evolve against a stable
vocabulary. They do NOT pretend to score accurately yet — every function
returns None.

Future dimensions (target weights TBD; will live in a config file):

  emotional_intensity      — how strongly the hook provokes a felt reaction
  viewer_role_strength     — how clearly the hook conjures a role
                              (defender, judge, voyeur, learner, ...)
  product_attachability    — how cleanly a product/offer can ride the hook
  transferability          — how reusable the mechanic is across niches
  freshness                — is this mechanic novel right now
  cooked_score             — saturation/exhaustion of this mechanic
  virality_proof           — metric signal (views/likes/comments/shares/saves)
  platform_fit             — how well-suited to TikTok/IG/FB feed grammar
  overall_opportunity      — weighted combination of the above

The metrics-side score (virality_proof) is the only one that could be
computed today from Apify counts, but doing so in isolation is
misleading: a viral cooked hook is worth less than a quiet fresh one.
So even that is held back until the vision model fills the concept
fields.
"""

from __future__ import annotations

from typing import Any


def emotional_intensity_score(analysis: dict[str, Any]) -> float | None:  # noqa: ARG001
    return None


def viewer_role_strength_score(analysis: dict[str, Any]) -> float | None:  # noqa: ARG001
    return None


def product_attachability_score(analysis: dict[str, Any]) -> float | None:  # noqa: ARG001
    return None


def transferability_score(analysis: dict[str, Any]) -> float | None:  # noqa: ARG001
    return None


def freshness_score(analysis: dict[str, Any]) -> float | None:  # noqa: ARG001
    return None


def cooked_score(analysis: dict[str, Any]) -> float | None:  # noqa: ARG001
    return None


def virality_proof_score(metrics: dict[str, Any]) -> float | None:  # noqa: ARG001
    return None


def platform_fit_score(analysis: dict[str, Any]) -> float | None:  # noqa: ARG001
    return None


def overall_opportunity_score(
    analysis: dict[str, Any],  # noqa: ARG001
    metrics: dict[str, Any],   # noqa: ARG001
) -> float | None:
    return None
