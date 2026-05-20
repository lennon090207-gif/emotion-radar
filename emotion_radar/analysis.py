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

HOOK_STRATEGY_SYSTEM_PROMPT = """You are a senior organic-marketing researcher analyzing the VIRAL MECHANIC of a short-form video. You receive (a) Pass 1's frame-by-frame visual evidence, and (b) the video's metadata. Your job is to identify what makes the hook stop the scroll, and to generate broad creative hook concepts that re-use that mechanic in new emotional situations.

This is the STRATEGY pass (Pass 2 of 2). Pass 1's JSON is the evidence layer — ground truth about what physically happens. You do not re-analyze any image.

# Mental model: product secondary, mechanic primary

THE PRODUCT IS SECONDARY. THE VIRAL HOOK MECHANIC IS PRIMARY.

The user has many products and can attach almost any mechanic to many products later. The product/niche of the source video is NOT the asset. The MECHANIC is the asset. You are casting EMOTIONAL SITUATIONS, not product categories.

Mutate the EMOTIONAL SITUATION, not the object. Two concepts that are structurally identical and only differ in what object is on screen are ONE concept, not two — reject them.

# Hard rules — what NOT to generate

You MUST NOT generate any of these as creative_hook_concepts:

  - PRODUCT-SWAP LISTS: "same hook but with mugs / candles / jewelry / earrings / soap / etc." This is the exact failure mode we are eliminating. A list of product swaps is one concept, not many. Never propose mutations whose entire variation is the object.
  - NICHE-SWAP LISTS tied to specific product categories (handmade-only, craft-fair-only, etsy-only). You are casting emotional situations, not product categories.
  - generic "support small business" / "rooting for the underdog" reframes that don't specify a sharp emotional setup.
  - unrelated industries (SaaS, crypto, fitness, supplements, real estate, dropshipping, B2B software) UNLESS the source video itself explicitly supports that leap. If you have to invent a connection, the leap is too far.
  - feel-good or pure-validation hooks when Pass 1 says `visual_conflict_detected` is true.

# Pass 1 evidence is BINDING (do not soften, do not contradict)

These rules override your priors. Pass 1 is ground truth.

- Do NOT call any action "accidental" unless Pass 1 explicitly uses the words "accident" / "accidentally" / "unintentional" OR Pass 1's `uncertainty_notes` strongly indicate the action was unintentional. If Pass 1 reports a product was picked up, dropped, broken, smashed, shattered, thrown, knocked over, or ended on the ground / on the floor, treat the destruction as INTENTIONAL by default. The natural default for a stranger handling a stranger's product and ending it on the floor is disrespect, not clumsiness.

- DESTRUCTION + INSULTING-TEXT RULE: if Pass 1 reports destruction (any of: dropped / broken / smashed / shattered / thrown / knocked over / on the ground / on the floor) AND the on-screen text contains rejection / insult / "honest-feedback" framing such as "nobody will buy", "stop making", "worthless", "this is bad", "ugly", "trash", "please be honest", "would you buy", or similar, the emotional_mechanic IS:

    "public disrespect + underdog maker + viewer-defense instinct"

  Do NOT retreat to "tension and disappointment", "accidentally broken", "creator validation", "creator vulnerability", "generic appreciation", or any other softer framing. The injustice IS the engine.

- If `visual_conflict_detected` is true, the conflict MUST be CENTRAL in BOTH `visual_hook_summary` AND `emotional_mechanic`. Do not bury it in a subordinate clause. Do not relocate it to a side note. The conflict is the lede.

- When `visual_conflict_detected` is true, EVERY creative_hook_concept must preserve a conflict / disrespect / underdog / injustice edge. NO positive-validation hooks. Specifically forbidden in this regime:
  - "customer smiles at the maker",
  - "customer takes a selfie with the product",
  - "customer gives a thumbs up",
  - "customer says 'I love it'",
  - any "appreciation"-themed opening.
  Vary the situation, NOT the polarity. Do not vary the polarity.

# What the user actually wants to know

For every video, answer these specifically:
  1. What is the VIRAL MECHANIC? (the underlying pattern, not the surface)
  2. WHY does it stop the scroll in the first 1-2 seconds?
  3. What VIEWER ROLE does it create? (defender, judge, tribe member, accomplice, witness, fixer, rescuer, insider, jury, etc.)
  4. What is the COMMENT TRIGGER? (the specific thing that compels typing a reply)
  5. What is the SHARE TRIGGER? (the specific thing that compels sending to someone else)
  6. What is the EMOTIONAL PRESSURE? (the felt tension that makes scrolling away uncomfortable)
  7. Which PARTS ARE COOKED (already overused right now in organic feeds)?
  8. 8 BROAD HOOK CONCEPTS reusing the same mechanic in NEW emotional situations.

# Concept distribution (EXACT)

Produce EXACTLY 8 creative_hook_concepts with this `creative_distance` distribution:

  - 2 "same_mechanic":   close to the source mechanic, varied situation/setting/staging. NOT a product swap. Vary the setup or the reveal, not just the object.
  - 3 "adjacent_leap":   move the mechanic into a DIFFERENT emotional situation. Same viewer-role engine, different emotional setup. Example shapes (do not copy verbatim): public-doubt → private-effort reveal; mistaken-rejection → wrong-audience reveal; almost-quit → one-person-notices.
  - 2 "big_swing":       higher risk, higher upside. Stronger emotional stakes, sharper conflict, bigger reveal. Could backfire if cast wrong — explicitly say how in cooked_risk and believability_risk.
  - 1 "wildcard":        surprising but still believable. Unexpected setting or framing. Still lands in 1-2 seconds. Still native to organic feed. NOT random — must reuse the underlying mechanic in a way no one is doing yet.

# Per-concept required fields

Each creative_hook_concept MUST include all of these fields:

  - creative_distance:    "same_mechanic" | "adjacent_leap" | "big_swing" | "wildcard"
  - concept_name:         2-5 words, memorable. NOT a sentence. NOT a product description.
  - first_2_seconds:      what is visible in the first 1-2 seconds — concrete scene, concrete people, concrete action. NOT "a creator does X". NOT "someone says Y".
  - emotional_trigger:    a specific feeling — indignation, vindication, recognition-shock, social-comeuppance, defensive instinct, second-hand pride, anticipatory shame, etc. NOT "emotional appeal" / "engagement".
  - viewer_role:          defender, judge, tribe member, accomplice, witness, fixer, rescuer, insider, jury, etc. NOT "viewer".
  - why_it_could_go_viral: specifically why this stops the scroll AND drives comments/shares.
  - what_to_avoid:        concrete instruction on how to NOT end up cringe / staged / AI-slop.
  - believability_risk:   what would make this feel fake or performed.
  - cooked_risk:          what about this is close to an already-cooked TikTok format.

# Reference concept shapes (illustrative — DO NOT copy verbatim)

Each of the following has a sharp setup + reveal/twist + clear viewer role. Match the SHAPE; do not match the literal concept.

  - "Wrong Audience / Right Tribe":   mocks something as weird → text calls out the exact tribe that would defend it. Role = tribe member.
  - "Silent Proof After Insult":      dismissive comment → creator silently shows the obscene detail/effort. Role = jury.
  - "Almost Gave Up":                 creator starts packing up after being ignored → one person notices. Role = rescuer.
  - "Hidden Emotional Value":         stranger calls something worthless → text reveals it was made for a deeply emotional reason. Role = defender.
  - "Public Doubt / Private Effort":  public rejection → private proof of effort. Role = jury.
  - "Wrong Person Rejects It":        someone dismisses it → viewer immediately understands they were never the target. Role = insider.
  - "Community Rescue":                looks like it's failing → viewer is positioned as one of the people who could save it. Role = rescuer.

# Scoring (each in [0, 1])

Virality-focused scores (NEW canonical set; weight heavily in overall_opportunity_score):

  - scroll_stop_strength_score:         how hard this stops the scroll in the first 1-2s.
  - comment_likelihood_score:           how strongly the hook provokes comments.
  - share_likelihood_score:             how strongly the hook provokes shares.
  - viewer_role_strength_score:         how clearly the hook conjures a specific viewer role.
  - creative_transfer_potential_score:  how reusable the mechanic is across DIFFERENT EMOTIONAL SITUATIONS (NOT product categories).
  - virality_capability_score:          weighted gut summary of the five above.

Legacy scores (still produce them):

  - product_attachability_score:        how cleanly a real product can ride this mechanic. Keep, but do NOT let it dominate.
  - transferability_score:              how well the mechanic transfers to adjacent situations.
  - freshness_score:                    how novel this mechanic feels in organic feeds.
  - cooked_score:                       how saturated this mechanic is now.
  - overall_opportunity_score:          weighted combination, with virality_capability_score weighted highest.

# Cooked phrases — do NOT lift verbatim

  - "Nobody will ever buy your ___"
  - "Please be honest"
  - "Would you buy one?"
  - "POV: ..."

You MAY mutate these only if you both (a) flag them in `cooked_elements` AND in the relevant concept's `cooked_risk`, and (b) twist them meaningfully.

# Schema (return EXACTLY these top-level keys)

{
  "visual_hook_summary": string,
  "viral_mechanic": string,
  "scroll_stop_reason": string,
  "viewer_role": string,
  "comment_trigger": string,
  "share_trigger": string,
  "emotional_pressure": string,
  "emotional_mechanic": string,
  "emotions_triggered": [string, ...],
  "why_it_works": string,
  "cooked_elements": [string, ...],
  "cooked_parts_to_avoid": [string, ...],
  "freshness_angle": string,
  "scroll_stop_strength_score": number,
  "comment_likelihood_score": number,
  "share_likelihood_score": number,
  "viewer_role_strength_score": number,
  "creative_transfer_potential_score": number,
  "virality_capability_score": number,
  "product_attachability_score": number,
  "transferability_score": number,
  "freshness_score": number,
  "cooked_score": number,
  "overall_opportunity_score": number,
  "creative_hook_concepts": [
    {
      "creative_distance": "same_mechanic" | "adjacent_leap" | "big_swing" | "wildcard",
      "concept_name": string,
      "first_2_seconds": string,
      "emotional_trigger": string,
      "viewer_role": string,
      "why_it_could_go_viral": string,
      "what_to_avoid": string,
      "believability_risk": string,
      "cooked_risk": string
    }
  ]
}

Return STRICT JSON only. No prose outside the JSON object. No markdown fences. No commentary.
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

_REPAIR_SYSTEM_PROMPT = (
    "You convert near-JSON to strict JSON. The user message contains text "
    "that was supposed to be a single JSON object but isn't (extra prose, "
    "fences, trailing commas, unescaped quotes, etc.). Return ONLY a single "
    "valid JSON object reflecting the same data. No prose. No markdown "
    "fences. No commentary."
)


def _parse_or_repair(raw: str, repair_provider: VisionProvider | None) -> dict[str, Any]:
    """Try to parse `raw` as JSON. On failure, optionally ask
    `repair_provider` (text-only call) to convert it into strict JSON
    and try once more. If the repair attempt also fails, the ORIGINAL
    parse error is the one that surfaces — that's the error a human
    needs to see to diagnose the underlying problem."""
    try:
        return parse_analysis_json(raw)
    except ValueError as first_err:
        if repair_provider is None:
            raise
        try:
            repaired = repair_provider.analyze_text(
                _REPAIR_SYSTEM_PROMPT,
                "Convert the following to a valid JSON object:\n\n" + raw,
            )
            return parse_analysis_json(repaired)
        except Exception:
            raise first_err


def extract_visual_event(
    contact_sheet_path: Path,
    metadata: dict[str, Any],
    provider: VisionProvider,
    repair_provider: VisionProvider | None = None,
) -> dict[str, Any]:
    """Pass 1. Returns the parsed JSON dict (not an AnalysisResult).

    `repair_provider` (text-only) is given one shot to fix near-JSON if
    the initial parse fails. Defaults to the same provider that
    produced the vision output; callers can pass a separate cheaper
    text model if they want."""
    user_prompt = build_visual_event_user_prompt(metadata)
    raw = provider.analyze_image(
        contact_sheet_path,
        VISUAL_EVENT_SYSTEM_PROMPT,
        user_prompt,
    )
    return _parse_or_repair(raw, repair_provider or provider)


def generate_hook_strategy(
    metadata: dict[str, Any],
    pass1_result: dict[str, Any],
    provider: VisionProvider,
    repair_provider: VisionProvider | None = None,
) -> dict[str, Any]:
    """Pass 2. Text-only — consumes Pass 1's JSON evidence layer."""
    user_prompt = build_hook_strategy_user_prompt(metadata, pass1_result)
    raw = provider.analyze_text(
        HOOK_STRATEGY_SYSTEM_PROMPT,
        user_prompt,
    )
    return _parse_or_repair(raw, repair_provider or provider)


def analyze_two_pass(
    contact_sheet_path: Path,
    metadata: dict[str, Any],
    vision_provider: VisionProvider,
    strategy_provider: VisionProvider | None = None,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Run Pass 1 -> Pass 2 and return both parsed JSON dicts.
    `strategy_provider` defaults to `vision_provider` (same model for
    both passes) if not supplied.

    The strategy (text-only) provider also acts as the repair model for
    Pass 1 JSON parse failures. That keeps repair cheap when Pass 1 is
    a strong vision model and Pass 2 is a cheaper text model."""
    sp = strategy_provider or vision_provider
    pass1 = extract_visual_event(
        contact_sheet_path, metadata, vision_provider, repair_provider=sp,
    )
    pass2 = generate_hook_strategy(metadata, pass1, sp, repair_provider=sp)
    return pass1, pass2


def build_two_pass_analysis_result(
    pass1: dict[str, Any],
    pass2: dict[str, Any],
) -> AnalysisResult:
    """Merge Pass 1 evidence + Pass 2 strategy into a single AnalysisResult
    ready to hand to db.update_report_analysis.

    Field origin (Phase 4):
      visual_hook_summary, emotional_mechanic, viewer_role, emotions_triggered,
      product_attachability_score, transferability_score, freshness_score,
      cooked_score, overall_opportunity_score  <- Pass 2
      onscreen_text                            <- Pass 1
      hook_mutations                           <- Pass 2 creative_hook_concepts
                                                  (fall back to legacy
                                                  hook_mutations for older
                                                  prompts / mocked tests)

    raw_analysis carries both passes verbatim, so the new Phase-4 fields
    (viral_mechanic, scroll_stop_reason, comment_trigger, share_trigger,
    emotional_pressure, cooked_elements, freshness_angle, and all the
    virality_* scores) survive in raw_analysis.hook_strategy_pass without
    needing new DB columns."""
    pass1 = pass1 or {}
    pass2 = pass2 or {}
    # Phase 4: creative_hook_concepts is the canonical list. Fall back to
    # the Phase-3 hook_mutations shape if the model still uses that.
    mutations: list[Any] = []
    cands = pass2.get("creative_hook_concepts")
    if isinstance(cands, list) and cands:
        mutations = cands
    else:
        legacy = pass2.get("hook_mutations")
        if isinstance(legacy, list):
            mutations = legacy
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
