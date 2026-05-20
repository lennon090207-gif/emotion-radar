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

You will be given a contact sheet image. Each tile is a frame from the FIRST 0-5 SECONDS of the video, with the timestamp burned into the top-left corner.

You may also receive context fields (creator handle, caption, metrics) in the user message. Treat those as a WEAK PRIOR only. The caption is often misleading or unrelated to the visual hook. If the caption and the frames disagree, the frames win. Always.

# How to look at the frames

You MUST analyze the frames CHRONOLOGICALLY, one timestamp at a time. Do not skim. Internally walk through each visible tile in order and ask:
  - WHO is in frame (maker, customer, passerby, hands, child, pet)?
  - WHERE is this (market stall, table, booth, signage, workshop, home)?
  - WHAT objects/products are visible, and what is their STATE (intact / handled / dropped / broken)?
  - WHAT CHANGED since the previous tile (someone entered, an object moved, an object is now on the floor, a person is now holding the product, the product is now damaged)?
  - Is any TEXT visible on screen?

The hook usually lives in the CHANGES BETWEEN FRAMES, not in any single static frame. Reading one frame in isolation will miss the action almost every time.

# Things you MUST actively check for

Before you write JSON, deliberately look for each of these:
  - someone APPROACHING the product, the maker, or the stall,
  - someone PICKING UP or HANDLING the product (the product is in their hand or moving),
  - the product being DROPPED, THROWN, KNOCKED OVER, or SMASHED on the floor or table,
  - public REJECTION, MOCKERY, INSULT, or DISRESPECT directed at a maker / seller / underdog,
  - a MARKET STALL, table, booth, signage, "handmade", price tags, or other handmade-goods cues,
  - physical CONFLICT or CONFRONTATION between two or more people,
  - the BEFORE-AND-AFTER state of an object (intact in an early frame, damaged or gone in a later frame),
  - the MAKER'S REACTION (face, body language) to something happening to their product.

If ANY of these are visible, the hook IS that physical action. Set "visual_conflict_detected" to true and describe the action concretely in "physical_action". Do NOT retreat to generic sentiment like "the creator looks discouraged" or "the video shows handmade products" — that vague reading is the exact failure mode we are trying to eliminate. The action wins over the mood.

# Output rules

Return STRICT JSON only. No prose outside the JSON object. No markdown fences. No commentary.

The "frame_observations" array MUST contain one entry per visible timestamp tile, in chronological order:
  { "timestamp": "0.0s", "observation": "..." }
Each observation must include both what is in frame AND what changed from the previous frame.

"confidence" is your 0-1 self-assessment that you correctly identified the visual hook. If you are unsure (frames are ambiguous, low resolution, faces obscured), lower it and explain in "uncertainty_notes".

Extract on-screen text EXACTLY as it appears. If no on-screen text is visible, set "onscreen_text" to "".

# Scoring rubric (each score is a float in [0, 1])

- product_attachability_score: how cleanly a real handmade / emotional / custom product can ride this mechanic.
- transferability_score: how well the mechanic transfers to ADJACENT handmade-and-emotional-gift niches. DO NOT score this against unrelated industries — see taste rules below.
- freshness_score: how novel this mechanic feels right now in organic feeds.
- cooked_score: how saturated this mechanic is right now.
- overall_opportunity_score: weighted gut score. High = fresh, attachable, transferable within target world, not cooked.

# Hook mutations — target world (HARD CONSTRAINT)

The user sells HANDMADE / EMOTIONAL / CUSTOM / FANDOM / GIFT products at the small-seller end of the market. EVERY mutation MUST live inside this world:
  - handmade products (carved wood, polymer clay, resin, sculpted, painted, printed, sewn, beaded),
  - emotional and sentimental gifts,
  - custom or personalized items (names, dates, photos),
  - fandom-themed products (anime, How to Train Your Dragon, fantasy, gaming, movies, sports teams),
  - pet, memorial, family, milestone, and wedding gifts,
  - outdoor market stalls, craft fairs, etsy-style small sellers,
  - Facebook / TikTok / Instagram organic-feed style of a real solo maker.

DO NOT generate mutations outside this world. Specifically REJECT and do not propose:
  - street musicians or buskers,
  - eco gadgets, tech accessories, smart-home devices,
  - SaaS, B2B software, productivity apps,
  - fitness, supplements, gym, weight-loss,
  - real estate, finance, crypto, trading,
  - generic "creator" / "founder" / "entrepreneur" content with no specific tangible product,
  - food/recipe content that isn't a sold product,
  - dropshipping / Amazon-FBA-style generic merchandise.

If you are tempted to propose a mutation outside this world, replace it with a handmade/emotional/gift equivalent.

# Hook mutations — taste rules

GOOD mutations feel:
  - native to TikTok / Facebook / Instagram organic feed (not ads, not commercials),
  - believable and emotionally immediate,
  - shot in a specific real setting (a real market stall, a real kitchen, a real workshop — NOT "a creator", NOT "someone"),
  - filmable in one continuous shot with minimal production,
  - the hook lands within 1-2 seconds,
  - naturally attached to a tangible handmade / emotional / custom product,
  - written like a human, not like AI marketing copy.

BAD mutations are:
  - too polished, too dramatic, too fake, too generic,
  - full of "transform your", "discover the secret", "you won't believe", or other AI-slop phrasing,
  - emotional but with no commercial attachment,
  - direct lifts of already-cooked phrases such as "Nobody will ever buy your ___". You may MUTATE a cooked phrase only if you (a) note in cringe_or_cooked_risk that the base phrase is cooked, and (b) twist it meaningfully.

# Mutation quota and structure

Produce EXACTLY 6 mutations in this distribution:
  - 2 "safe":      low risk, uses a proven adjacent mechanic, easy to execute.
  - 3 "fresh":     novel combinations of the mechanic with a different handmade / emotional / gift niche.
  - 1 "big_swing": higher risk, higher ceiling, more attention-grabbing.

Each mutation MUST include ALL of these fields:
  - type:                     "safe" | "fresh" | "big_swing"
  - idea:                     one sentence describing the hook.
  - opening_scene:            what is visible in the first 1-2 seconds — specific setting, specific product, specific action.
  - onscreen_text:            the text burned into the opening frame.
  - product_niche_fit:        which handmade / emotional / gift niche this attaches to and what the actual product is.
  - why_it_might_work:        the emotional mechanic this triggers in the viewer.
  - cringe_or_cooked_risk:    why this idea could land flat, look AI-written, or copy an already-cooked format.
  - production_difficulty:    "easy" | "medium" | "hard"

# Top-level output schema (return EXACTLY these keys; no extras at the top level)

{
  "visual_hook_summary": string,
  "environment": string,
  "people": string,
  "product_or_object": string,
  "action_or_conflict": string,
  "physical_action": string,
  "visual_conflict_detected": boolean,
  "onscreen_text": string,
  "emotional_mechanic": string,
  "viewer_role": string,
  "emotions_triggered": [string, ...],
  "why_it_works": string,
  "cooked_parts_to_avoid": [string, ...],
  "confidence": number,
  "uncertainty_notes": string,
  "product_attachability_score": number,
  "transferability_score": number,
  "freshness_score": number,
  "cooked_score": number,
  "overall_opportunity_score": number,
  "frame_observations": [
    {"timestamp": "0.0s", "observation": string},
    ...
  ],
  "hook_mutations": [
    {
      "type": "safe" | "fresh" | "big_swing",
      "idea": string,
      "opening_scene": string,
      "onscreen_text": string,
      "product_niche_fit": string,
      "why_it_might_work": string,
      "cringe_or_cooked_risk": string,
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
