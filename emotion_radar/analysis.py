"""Visual hook analysis.

Two entrypoints:

- `analyze_contact_sheet(contact_sheet_path, metadata)`
    Stub used by analyze-url so the report row is always shaped the same.
    Returns nulls. Cheap. Safe to call without any API key.

- `analyze_contact_sheet_with_vision(contact_sheet_path, metadata, provider)`
    Real Phase-2 vision analysis. Sends ONLY the contact sheet image and
    a tightly-scoped JSON-output prompt to the provider, parses the
    response, and returns an AnalysisResult.

The prompt below is deliberately strict about:
  * windowing to the visible 0-5s frames,
  * preferring visuals over the caption when they conflict,
  * producing JSON only,
  * generating hook *ideas* (not media) with explicit taste guards.
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

from .models import AnalysisResult
from .providers import VisionProvider


# ---- prompts ---------------------------------------------------------------

SYSTEM_PROMPT = """You are a senior organic-marketing researcher analyzing the visual hook of a short-form video.

You will be given a contact sheet image. Each tile is a frame from the FIRST 0-5 SECONDS of the video, with the timestamp burned in the top-left corner.

Hard rules:
- Analyze ONLY what is visible in the contact sheet (frames from t=0s to t=5s). The hook either lands in those 5 seconds or it does not exist.
- Focus on the VISUAL HOOK, not the caption. The caption may be misleading; the frames are ground truth. If the caption and frames disagree, the frames win.
- Be concrete and specific. Say what is physically happening, where, who is on screen, what object/product is involved, what action or conflict occurs.
- Extract on-screen text exactly as it appears in the frames. If no on-screen text is visible, set "onscreen_text" to an empty string.
- Classify the underlying emotional mechanic (e.g. "public disrespect of an underdog maker triggers viewer-defense instinct"), not the surface topic.
- Identify the viewer role the hook conjures (defender, judge, voyeur, learner, accomplice, witness, etc.).
- Return STRICT JSON only. No prose outside the JSON object. No markdown fences. No commentary.

Scoring rubric (each score is a float in [0, 1]):
- product_attachability_score: how cleanly a real product/offer can ride this mechanic.
- transferability_score: how well the mechanic transfers to other niches (1.0 = works almost anywhere).
- freshness_score: how novel this mechanic feels right now in organic feeds (1.0 = rare and surprising).
- cooked_score: how saturated/exhausted this mechanic is right now (1.0 = everyone is using it).
- overall_opportunity_score: weighted gut score combining the above. High = fresh, attachable, transferable, not cooked.

Hook-mutation taste rules (apply to EVERY item in `hook_mutations`):
GOOD ideas feel:
  - native to TikTok / Facebook / Instagram organic feed (not ads, not commercials),
  - believable and emotionally immediate,
  - shot in a specific real setting (a real market stall, a real kitchen, a real garage — NOT "a creator", NOT "someone"),
  - filmable in one continuous shot with minimal production,
  - the hook lands within 1-2 seconds,
  - naturally attached to a tangible product or offer,
  - written like a human, not like AI marketing copy.
BAD ideas are:
  - too polished, too dramatic, too fake, too generic,
  - too wordy, full of "transform your", "discover the secret", or other AI-slop phrases,
  - emotional but with no commercial attachment,
  - direct copies of a cooked TikTok format (lip-sync trends, "POV: you" variants that are already cooked, etc.).

You must produce 3-5 mutations spanning these three `type` values:
  - "safe":      low risk, uses a proven adjacent mechanic, easy to execute.
  - "fresh":     a novel combination of the mechanic and a different niche/setting.
  - "big_swing": higher risk, higher potential ceiling, more attention-grabbing.

Output schema (return EXACTLY these keys; no extras at the top level):

{
  "visual_hook_summary": string,
  "environment": string,
  "people": string,
  "product_or_object": string,
  "action_or_conflict": string,
  "onscreen_text": string,
  "emotional_mechanic": string,
  "viewer_role": string,
  "emotions_triggered": [string, ...],
  "why_it_works": string,
  "cooked_parts_to_avoid": [string, ...],
  "product_attachability_score": number,
  "transferability_score": number,
  "freshness_score": number,
  "cooked_score": number,
  "overall_opportunity_score": number,
  "hook_mutations": [
    {
      "type": "safe" | "fresh" | "big_swing",
      "idea": string,
      "opening_scene": string,
      "onscreen_text": string,
      "why_it_might_work": string,
      "taste_risk": string,
      "production_difficulty": "easy" | "medium" | "hard"
    }
  ]
}
"""


def build_user_prompt(metadata: dict[str, Any]) -> str:
    """User message. Caption + creator are weak priors. Frames are truth."""
    creator = metadata.get("creator_username") or "(unknown)"
    nickname = metadata.get("creator_nickname")
    platform = metadata.get("platform") or "TikTok"
    caption = metadata.get("caption") or ""
    metrics = metadata.get("metrics") or {}
    views = metrics.get("views")
    likes = metrics.get("likes")
    comments = metrics.get("comments")

    creator_line = f"@{creator}" + (f" ({nickname})" if nickname else "")
    caption_line = caption.strip().replace("\n", " ") if caption else "(none)"

    return (
        "Context (weak prior — the frames in the attached contact sheet are ground truth; "
        "if anything below conflicts with the frames, ignore it):\n"
        f"- platform: {platform}\n"
        f"- creator:  {creator_line}\n"
        f"- caption:  \"{caption_line}\"\n"
        f"- metrics:  views={views} likes={likes} comments={comments}\n"
        "\n"
        "Analyze the contact sheet (first 0-5 seconds of the video) and return the "
        "structured JSON described in the system instructions. Output JSON only."
    )


# ---- parsing ---------------------------------------------------------------

_JSON_FENCE_RE = re.compile(r"```(?:json)?\s*(.*?)\s*```", re.DOTALL | re.IGNORECASE)


def _extract_json_blob(text: str) -> str:
    """Models sometimes wrap JSON in ```json fences or add prose. Recover
    the largest plausible JSON object substring."""
    text = (text or "").strip()
    if not text:
        raise ValueError("Empty model response.")
    fence = _JSON_FENCE_RE.search(text)
    if fence:
        return fence.group(1).strip()
    # Greedy span from first { to last }.
    first = text.find("{")
    last = text.rfind("}")
    if first != -1 and last != -1 and last > first:
        return text[first : last + 1]
    return text


def _coerce_score(v: Any) -> float | None:
    if v is None:
        return None
    try:
        x = float(v)
    except (TypeError, ValueError):
        return None
    if x != x:  # NaN
        return None
    if x < 0.0:
        return 0.0
    if x > 1.0:
        return 1.0
    return x


def _coerce_str_list(v: Any) -> list[str]:
    if not isinstance(v, list):
        return []
    out: list[str] = []
    for item in v:
        if isinstance(item, str) and item.strip():
            out.append(item.strip())
    return out


def parse_analysis_json(text: str) -> dict[str, Any]:
    """Parse + minimally validate the model output. Returns the raw parsed
    dict (including any extra fields like environment/people/etc.).
    Raises ValueError on hard failures."""
    blob = _extract_json_blob(text)
    try:
        data = json.loads(blob)
    except json.JSONDecodeError as e:
        raise ValueError(f"Model output was not valid JSON: {e}") from e
    if not isinstance(data, dict):
        raise ValueError(f"Model output was not a JSON object (got {type(data).__name__}).")
    return data


def map_parsed_to_result(parsed: dict[str, Any]) -> AnalysisResult:
    """Project the parsed JSON onto AnalysisResult. Extra fields
    (environment, people, why_it_works, ...) survive in raw_analysis."""
    result = AnalysisResult(
        visual_hook_summary=(parsed.get("visual_hook_summary") or None) if isinstance(parsed.get("visual_hook_summary"), str) else None,
        onscreen_text=(parsed.get("onscreen_text") or None) if isinstance(parsed.get("onscreen_text"), str) else None,
        emotional_mechanic=(parsed.get("emotional_mechanic") or None) if isinstance(parsed.get("emotional_mechanic"), str) else None,
        viewer_role=(parsed.get("viewer_role") or None) if isinstance(parsed.get("viewer_role"), str) else None,
        emotions_triggered=_coerce_str_list(parsed.get("emotions_triggered")),
        product_attachability_score=_coerce_score(parsed.get("product_attachability_score")),
        transferability_score=_coerce_score(parsed.get("transferability_score")),
        freshness_score=_coerce_score(parsed.get("freshness_score")),
        cooked_score=_coerce_score(parsed.get("cooked_score")),
        overall_opportunity_score=_coerce_score(parsed.get("overall_opportunity_score")),
        hook_mutations=parsed.get("hook_mutations") if isinstance(parsed.get("hook_mutations"), list) else [],
        raw_analysis=parsed,
    )
    return result


# ---- public entrypoints ----------------------------------------------------

def analyze_contact_sheet(
    contact_sheet_path: Path,
    metadata: dict[str, Any],
) -> AnalysisResult:
    """Placeholder analysis used by analyze-url so reports always have a
    consistent shape. Returns nulls. No API calls. No heuristics on the
    caption — Phase 2 (analyze_contact_sheet_with_vision) fills the
    concept fields."""
    _ = contact_sheet_path
    _ = metadata
    return AnalysisResult(
        raw_analysis={"status": "stub", "note": "run analyze-report for vision analysis"},
    )


def analyze_contact_sheet_with_vision(
    contact_sheet_path: Path,
    metadata: dict[str, Any],
    provider: VisionProvider,
) -> AnalysisResult:
    """Send the contact sheet to the vision provider and return the
    parsed AnalysisResult. The provider call is the only network I/O
    here; everything else is prompt building and JSON wrangling."""
    user_prompt = build_user_prompt(metadata)
    raw_text = provider.analyze_image(
        contact_sheet_path,
        SYSTEM_PROMPT,
        user_prompt,
    )
    parsed = parse_analysis_json(raw_text)
    return map_parsed_to_result(parsed)
