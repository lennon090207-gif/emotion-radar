"""Normalized internal data structures.

NormalizedItem = a single Apify result, mapped into our consistent shape.
Report          = what gets persisted in SQLite, including analysis stubs.
"""

from __future__ import annotations

from dataclasses import dataclass, field, asdict
from typing import Any


@dataclass
class NormalizedItem:
    """One Apify dataset item, mapped to our internal shape.
    `error` is set when the item is unusable (e.g. missing video_download_url).
    """
    platform: str
    source_url: str | None
    submitted_url: str
    video_download_url: str | None
    cover_url: str | None
    creator_username: str | None
    creator_nickname: str | None
    caption: str | None
    video_id: str | None
    duration: float | None
    metrics: dict[str, int | None]
    raw: dict[str, Any] = field(default_factory=dict)
    error: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class ApifyRunInfo:
    """Bookkeeping returned by the Apify client for cost/traceability."""
    run_id: str | None
    dataset_id: str | None
    usage_total_usd: float | None
    charged_events: dict[str, Any] | None


@dataclass
class AnalysisResult:
    """Output of analysis.py — placeholder fields now, vision model later.

    See analysis.py for the full description of what each field should
    contain once the vision/LLM step is wired in.
    """
    visual_hook_summary: str | None = None
    onscreen_text: str | None = None
    emotional_mechanic: str | None = None
    viewer_role: str | None = None
    emotions_triggered: list[str] = field(default_factory=list)
    product_attachability_score: float | None = None
    transferability_score: float | None = None
    freshness_score: float | None = None
    cooked_score: float | None = None
    overall_opportunity_score: float | None = None
    hook_mutations: list[Any] = field(default_factory=list)
    raw_analysis: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)
