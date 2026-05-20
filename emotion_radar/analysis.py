"""Two-pass visual hook analysis.

Pass 1 (Visual Event Extractor):
  Heavy vision pass. Reads the contact sheet, walks the timestamps
  chronologically, and reports what is LITERALLY VISIBLE.
  No strategy, no scoring, no hook ideas. Just evidence.

Pass 2 (Hook Strategist):
  Text-only reasoning pass. Consumes Pass 1's structured JSON plus the
  video's metadata, and produces the emotional mechanic + 6 mutations
  inside the user's target world (handmade / emotional / custom /
  fandom / gift / market-stall).

Why split this:
  In one-pass mode the model tended to short-circuit straight to mood
  ("creator looks discouraged at his stall") and miss the actual
  physical action ("stranger smashes the maker's dragon lamp on the
  floor"). Forcing Pass 1 to produce a chronological evidence record
  before any strategy reasoning makes that failure mode much harder.

Entrypoints:
  - analyze_contact_sheet(...)        : legacy stub, used by analyze-url.
  - extract_visual_event(...)         : Pass 1 entry. Returns parsed dict.
  - generate_hook_strategy(...)       : Pass 2 entry. Returns parsed dict.
  - analyze_two_pass(...)             : orchestrates Pass 1 then Pass 2.
  - build_two_pass_analysis_result(...): merges into AnalysisResult.
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

from .models import AnalysisResult
from .providers import VisionProvider


# ============================================================================
# Pass 1 — Visual Event Extractor
# ============================================================================

VISUAL_EVENT_SYSTEM_PROMPT = """You are a careful frame analyst. Your only job is to describe what is LITERALLY VISIBLE in a contact sheet showing the first 0-5 seconds of a short-form video. Each tile has a timestamp burned into the top-left corner.

This is the EVIDENCE pass (Pass 1 of 2). You produce a literal, chronological record of what happens in the frames.

You do NOT:
  - generate hook ideas,
  - score the hook or the opportunity,
  - interpret marketing strategy beyond what is literally visible,
  - soften a visible action into a mood ("creator looks discouraged", "dismissively handles a lamp").

You DO:
  - inspect every visible timestamp tile in chronological order,
  - describe what is in each frame AND what changed from the previous frame,
  - track the STATE of every visible object across frames (on the table? in someone's hand? mid-air? on the floor? intact or damaged?),
  - identify the environment, the people on screen, the product or object,
  - identify the physical action, if any,
  - extract on-screen text exactly as visible,
  - state your uncertainty if an action is genuinely ambiguous (motion blur, off-camera, low resolution).

# Active checks (answer these in your observations before writing JSON)

Before you write JSON, deliberately answer each of these. Embed the answers in `frame_observations`, `physical_action`, `object_state_change`, and `conflict_type`:

  - Where is the PRODUCT in each frame? (on the table, in someone's hand, mid-air, on the floor)
  - Does any object move from a hand or table to the floor across the frames?
  - Does anyone PICK UP, THROW, DROP, SMASH, BREAK, or DAMAGE the product?
  - Does anyone APPROACH the stall or the maker?
  - Is this PUBLIC DISRESPECT, MOCKERY, REJECTION, or PHYSICAL DAMAGE — or simple browsing / inspection?
  - Is the maker REACTING to something happening to their product? Where are they looking?
  - Is there visible signage (market stall, "handmade", price tags, fandom banners)?
  - What ON-SCREEN TEXT is visible? Transcribe exactly.

If a physical action is visible, the hook IS that action. The action wins over the mood. Always. Do not soften.

If the action is genuinely ambiguous, lower `confidence` and explain in `uncertainty_notes`. Do not invent.

# Output rules

Return STRICT JSON only. No prose outside the JSON object. No markdown fences. No commentary.

`frame_observations` MUST contain one entry per visible timestamp tile, in chronological order.
`object_state_change` is a single-sentence summary across all frames (e.g. "lamp starts on the display table, ends on the floor with visible damage").
`conflict_type` is one of:
  "smash" | "throw" | "drop" | "knock_over" | "mockery" | "rejection" |
  "verbal_disrespect" | "physical_disrespect" | "browsing_only" | "none" | "ambiguous".
If `visual_conflict_detected` is false, set `physical_action` to "" and `conflict_type` to "none" or "browsing_only".
`confidence` is in [0, 1].

# Schema (return EXACTLY these top-level keys)

{
  "frame_observations": [
    {
      "timestamp": "0.0s",
      "observation": string,
      "people_visible": string,
      "object_state": string,
      "action_change_from_previous": string
    }
  ],
  "environment": string,
  "people": string,
  "product_or_object": string,
  "onscreen_text": string,
  "physical_action": string,
  "object_state_change": string,
  "visual_conflict_detected": boolean,
  "conflict_type": string,
  "confidence": number,
  "uncertainty_notes": string
}
"""


def build_visual_event_user_prompt(metadata: dict[str, Any]) -> str:
    """Pass 1's user message is deliberately minimal — we don't want the
    caption to bias the evidence layer. We include just enough context
    that the model knows what kind of source it's looking at."""
    platform = metadata.get("platform") or "TikTok"
    return (
        f"This is a contact sheet from a {platform} short-form video, "
        "covering the first 0-5 seconds. Each tile has its timestamp burned "
        "in the top-left corner.\n\n"
        "Inspect the tiles chronologically and produce the structured JSON "
        "described in the system instructions. Report what is LITERALLY "
        "visible, including any physical action you can see across frames. "
        "Do not infer mood if a physical action is visible. Output JSON only."
    )


# ============================================================================
# Pass 2 — Hook Strategist
# ============================================================================

HOOK_STRATEGY_SYSTEM_PROMPT = """You are a senior organic-marketing researcher. You will receive (a) the structured frame-by-frame evidence from Pass 1 of a short-form video analysis, and (b) the video's metadata. Your job is to extract the emotional hook mechanic and generate fresh hook ideas.

This is the STRATEGY pass (Pass 2 of 2). The evidence has already been gathered in Pass 1 and is provided to you as JSON. Treat Pass 1's JSON as ground truth about what physically happens in the video. You do not re-analyze any image. You build interpretation and mutation on top of the evidence.

# Hard rules

1. The Pass-1 JSON is the EVIDENCE layer. If Pass 1 says the product was thrown on the floor, the mechanic involves physical disrespect — NOT "creator looks discouraged".
2. If `visual_conflict_detected` is true in Pass 1, the `emotional_mechanic` you produce MUST reflect that conflict explicitly.
3. Classify the underlying emotional mechanic, not the surface topic.
4. Identify the viewer role the hook conjures: defender, judge, voyeur, learner, accomplice, witness, etc.
5. Return STRICT JSON only. No prose outside the JSON object. No markdown fences. No commentary.

# Scoring rubric (each score is a float in [0, 1])

- product_attachability_score: how cleanly a real handmade / emotional / custom product can ride this mechanic.
- transferability_score: how well the mechanic transfers to ADJACENT handmade-and-emotional-gift niches. Do NOT score this against unrelated industries.
- freshness_score: how novel this mechanic feels in organic feeds right now.
- cooked_score: how saturated this mechanic is right now.
- overall_opportunity_score: weighted gut score. High = fresh, attachable, transferable within target world, not cooked.

# Hook mutations — target world (HARD CONSTRAINT)

The user sells HANDMADE / EMOTIONAL / CUSTOM / FANDOM / GIFT products at the small-seller end of the market. EVERY mutation MUST live in this world:
  - handmade products (carved wood, polymer clay, resin, sculpted, painted, printed, sewn, beaded),
  - emotional and sentimental gifts,
  - custom or personalized items (names, dates, photos),
  - fandom-themed products (anime, How to Train Your Dragon, fantasy, gaming, movies, sports teams),
  - pet, memorial, family, milestone, wedding gifts,
  - outdoor market stalls, craft fairs, etsy-style small sellers,
  - Facebook / TikTok / Instagram organic-feed style of a real solo maker.

DO NOT generate mutations outside this world. Specifically REJECT and do not propose:
  - street musicians, buskers,
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
  - direct copies of cooked TikTok formats.

COOKED PHRASES — do NOT lift these verbatim unless you both (a) flag them as cooked in `cringe_or_cooked_risk` and (b) twist them meaningfully:
  - "Nobody will ever buy your ___"
  - "Please be honest"
  - "Would you buy one?"
  - "POV: ..."

# Mutation quota

Produce EXACTLY 6 mutations in this distribution:
  - 2 "safe":      low risk, uses a proven adjacent mechanic, easy to execute.
  - 3 "fresh":     novel combinations of the mechanic with a different handmade/emotional/gift niche.
  - 1 "big_swing": higher risk, higher ceiling, more attention-grabbing.

Each mutation MUST include ALL of these fields:
  - type:                     "safe" | "fresh" | "big_swing"
  - idea:                     one sentence describing the hook.
  - opening_scene:            what is visible in the first 1-2 seconds — specific setting, specific product, specific action.
  - onscreen_text:            the text burned into the opening frame.
  - product_niche_fit:        which handmade/emotional/gift niche this attaches to and what the actual product is.
  - why_it_might_work:        the emotional mechanic this triggers in the viewer.
  - cringe_or_cooked_risk:    why this idea could land flat, look AI-written, or copy an already-cooked format.
  - production_difficulty:    "easy" | "medium" | "hard"

# Schema (return EXACTLY these top-level keys)

{
  "visual_hook_summary": string,
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
      "product_niche_fit": string,
      "why_it_might_work": string,
      "cringe_or_cooked_risk": string,
      "production_difficulty": "easy" | "medium" | "hard"
    }
  ]
}
"""


def build_hook_strategy_user_prompt(
    metadata: dict[str, Any],
    pass1_result: dict[str, Any],
) -> str:
    """Pass 2's user message embeds Pass 1's evidence JSON, then adds the
    metadata as a weak prior."""
    creator = metadata.get("creator_username") or "(unknown)"
    nickname = metadata.get("creator_nickname")
    platform = metadata.get("platform") or "TikTok"
    caption = (metadata.get("caption") or "").strip().replace("\n", " ")
    metrics = metadata.get("metrics") or {}
    views = metrics.get("views")
    likes = metrics.get("likes")
    comments = metrics.get("comments")

    creator_line = f"@{creator}" + (f" ({nickname})" if nickname else "")
    caption_line = caption or "(none)"

    pass1_json = json.dumps(pass1_result, indent=2, ensure_ascii=False)

    return (
        "PASS 1 EVIDENCE LAYER (treat this as ground truth about what physically happens in the video):\n"
        "```json\n"
        f"{pass1_json}\n"
        "```\n\n"
        "OPTIONAL CONTEXT (weak prior — may be misleading or unrelated to the visual hook; "
        "if it conflicts with Pass 1 evidence, Pass 1 wins):\n"
        f"- platform: {platform}\n"
        f"- creator:  {creator_line}\n"
        f"- caption:  \"{caption_line}\"\n"
        f"- metrics:  views={views} likes={likes} comments={comments}\n\n"
        "Using Pass 1 as the evidence layer, return the structured JSON "
        "described in the system instructions. Output JSON only."
    )


# ============================================================================
# JSON parsing (shared)
# ============================================================================

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
    first = text.find("{")
    last = text.rfind("}")
    if first != -1 and last != -1 and last > first:
        return text[first : last + 1]
    return text


def parse_analysis_json(text: str) -> dict[str, Any]:
    """Parse + minimally validate the model output. Returns the raw parsed
    dict (including any extra fields). Raises ValueError on hard failures."""
    blob = _extract_json_blob(text)
    try:
        data = json.loads(blob)
    except json.JSONDecodeError as e:
        raise ValueError(f"Model output was not valid JSON: {e}") from e
    if not isinstance(data, dict):
        raise ValueError(f"Model output was not a JSON object (got {type(data).__name__}).")
    return data


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


def _coerce_str_or_none(v: Any) -> str | None:
    if isinstance(v, str) and v != "":
        return v
    return None


# ============================================================================
# Two-pass orchestration
# ============================================================================

def extract_visual_event(
    contact_sheet_path: Path,
    metadata: dict[str, Any],
    provider: VisionProvider,
) -> dict[str, Any]:
    """Pass 1. Returns the parsed JSON dict (not an AnalysisResult)."""
    user_prompt = build_visual_event_user_prompt(metadata)
    raw = provider.analyze_image(
        contact_sheet_path,
        VISUAL_EVENT_SYSTEM_PROMPT,
        user_prompt,
    )
    return parse_analysis_json(raw)


def generate_hook_strategy(
    metadata: dict[str, Any],
    pass1_result: dict[str, Any],
    provider: VisionProvider,
) -> dict[str, Any]:
    """Pass 2. Text-only — consumes Pass 1's JSON evidence layer."""
    user_prompt = build_hook_strategy_user_prompt(metadata, pass1_result)
    raw = provider.analyze_text(
        HOOK_STRATEGY_SYSTEM_PROMPT,
        user_prompt,
    )
    return parse_analysis_json(raw)


def analyze_two_pass(
    contact_sheet_path: Path,
    metadata: dict[str, Any],
    vision_provider: VisionProvider,
    strategy_provider: VisionProvider | None = None,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Run Pass 1 -> Pass 2 and return both parsed JSON dicts.
    `strategy_provider` defaults to `vision_provider` (same model for both
    passes) if not supplied."""
    sp = strategy_provider or vision_provider
    pass1 = extract_visual_event(contact_sheet_path, metadata, vision_provider)
    pass2 = generate_hook_strategy(metadata, pass1, sp)
    return pass1, pass2


def build_two_pass_analysis_result(
    pass1: dict[str, Any],
    pass2: dict[str, Any],
) -> AnalysisResult:
    """Merge Pass 1 evidence + Pass 2 strategy into a single AnalysisResult
    ready to hand to db.update_report_analysis.

    Field origin per spec:
      visual_hook_summary, emotional_mechanic, viewer_role, emotions_triggered,
      product_attachability_score, transferability_score, freshness_score,
      cooked_score, overall_opportunity_score, hook_mutations  ← Pass 2
      onscreen_text                                            ← Pass 1
    raw_analysis carries both passes verbatim for auditability."""
    pass1 = pass1 or {}
    pass2 = pass2 or {}
    mutations = pass2.get("hook_mutations")
    if not isinstance(mutations, list):
        mutations = []
    return AnalysisResult(
        visual_hook_summary=_coerce_str_or_none(pass2.get("visual_hook_summary")),
        onscreen_text=_coerce_str_or_none(pass1.get("onscreen_text")),
        emotional_mechanic=_coerce_str_or_none(pass2.get("emotional_mechanic")),
        viewer_role=_coerce_str_or_none(pass2.get("viewer_role")),
        emotions_triggered=_coerce_str_list(pass2.get("emotions_triggered")),
        product_attachability_score=_coerce_score(pass2.get("product_attachability_score")),
        transferability_score=_coerce_score(pass2.get("transferability_score")),
        freshness_score=_coerce_score(pass2.get("freshness_score")),
        cooked_score=_coerce_score(pass2.get("cooked_score")),
        overall_opportunity_score=_coerce_score(pass2.get("overall_opportunity_score")),
        hook_mutations=mutations,
        raw_analysis={
            "analysis_mode": "two_pass",
            "visual_event_pass": pass1,
            "hook_strategy_pass": pass2,
        },
    )


# ============================================================================
# Stub used by analyze-url (no vision call)
# ============================================================================

def analyze_contact_sheet(
    contact_sheet_path: Path,
    metadata: dict[str, Any],
) -> AnalysisResult:
    """Placeholder analysis used by analyze-url so reports always have a
    consistent shape. Returns nulls. No API calls. Run analyze-link or
    analyze-report to fill the concept fields via the two-pass flow."""
    _ = contact_sheet_path
    _ = metadata
    return AnalysisResult(
        raw_analysis={
            "analysis_mode": "stub",
            "note": "run analyze-link or analyze-report for two-pass vision analysis",
        },
    )
