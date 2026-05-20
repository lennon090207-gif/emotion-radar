"""The Story Flow Library — Phase 5.

A built-in catalog of dominant viral emotional story flows the user has
observed in baseline data. Pass 2 of the analysis pipeline maps the
source video against this library and produces:

  - variations:        fresh mutations of MATCHED flows.
  - pioneer_concepts:  bigger, newer concepts that preserve the same
                       emotional physics but feel like new categories.

Why a library and not free-form: when Pass 2 is unanchored, it tends to
drift into product-swap territory (mugs / candles / jewelry). Anchoring
mutations and pioneer concepts to a small, named set of viral flows
keeps the variation on the right axis (emotional setup), not the wrong
one (product category).

Authoring rules:
  - Every flow has a stable `id` (used by the prompt and the CLI).
  - `steps` should describe the FLOW from setup to viewer response,
    not a single scene.
  - `ethical_risk_default` is a floor, not a verdict — Pass 2 may bump
    it based on what's actually visible.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class StoryFlow:
    id: str
    name: str
    steps: tuple[str, ...]
    emotional_physics: str
    viewer_role: str
    comment_trigger: str
    share_trigger: str
    cooked_risk: str
    ethical_risk_default: float
    example_labels: tuple[str, ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "name": self.name,
            "steps": list(self.steps),
            "emotional_physics": self.emotional_physics,
            "viewer_role": self.viewer_role,
            "comment_trigger": self.comment_trigger,
            "share_trigger": self.share_trigger,
            "cooked_risk": self.cooked_risk,
            "ethical_risk_default": self.ethical_risk_default,
            "example_labels": list(self.example_labels),
        }


STORY_FLOWS: tuple[StoryFlow, ...] = (
    StoryFlow(
        id="public_disrespect_viewer_defense",
        name="Public Disrespect -> Viewer Defense",
        steps=(
            "visible / social disrespect happens",
            "vulnerable maker or person absorbs it",
            "viewer feels injustice",
            "viewer wants to defend, comment, or share",
        ),
        emotional_physics=(
            "public moral violation against an underdog converts to a "
            "defender response from the viewer"
        ),
        viewer_role="defender",
        comment_trigger="urge to verbally retaliate on behalf of the wronged party",
        share_trigger="shared sense of injustice; send-to-friend to validate the outrage",
        cooked_risk=(
            "staged-stranger trope and 'please be honest' framing are heavily cooked"
        ),
        ethical_risk_default=0.25,
        example_labels=(
            "lobster bag drop",
            "Oliver HTTYD lamp drop",
            "fake comment insults creator / product",
        ),
    ),
    StoryFlow(
        id="family_protection_validation",
        name="Family Protection -> Validation",
        steps=(
            "child / parent / family member is emotionally exposed",
            "someone doubts or mocks their effort",
            "viewer feels protective",
            "viewer validates or defends them",
        ),
        emotional_physics=(
            "protective instinct around a vulnerable family member; "
            "pride + validation impulse"
        ),
        viewer_role="protector / validator",
        comment_trigger="urge to validate the family member's effort publicly",
        share_trigger="bond send to family, parents, friends",
        cooked_risk="exploitative grief-bait framing is cooked",
        ethical_risk_default=0.3,
        example_labels=(
            "daughter handmade Halloween lamps",
            "child avatar book bags",
            "blue-collar dad + goth daughter",
        ),
    ),
    StoryFlow(
        id="moral_pressure_tiny_rescue",
        name="Moral Pressure -> Tiny Rescue Action",
        steps=(
            "creator or business appears vulnerable",
            "viewer is told a tiny action helps",
            "scrolling away feels morally uncomfortable",
            "viewer watches / comments / shares",
        ),
        emotional_physics=(
            "implicit guilt of scrolling away converted into a watchable / "
            "shareable rescue act"
        ),
        viewer_role="rescuer",
        comment_trigger="urge to publicly support the rescue act",
        share_trigger="visible-good-deed signaling",
        cooked_risk="'please don't scroll' phrasing is heavily cooked",
        ethical_risk_default=0.35,
        example_labels=(
            "please don't scroll",
            "stay 12 seconds to save my business",
        ),
    ),
    StoryFlow(
        id="comment_humiliation_public_witness",
        name="Comment Humiliation -> Public Witness",
        steps=(
            "a hurtful comment / message is shown on screen",
            "creator visibly reacts or is exposed by it",
            "viewer becomes witness / juror",
            "comments become collective defense or validation",
        ),
        emotional_physics=(
            "public exposure of personal pain pulls viewers into a "
            "witness / juror role"
        ),
        viewer_role="witness / juror",
        comment_trigger="urge to publicly counter the cruel comment",
        share_trigger="solidarity send",
        cooked_risk="fabricated-comment suspicion tips this flow into cringe quickly",
        ethical_risk_default=0.45,
        example_labels=(
            "blue-collar man crying at comment",
            "wife reading comments",
        ),
    ),
    StoryFlow(
        id="stall_vulnerability_social_judgment",
        name="Stall Vulnerability -> Social Judgment",
        steps=(
            "maker / product is displayed publicly",
            "people inspect, ignore, mock, or reject it",
            "the market setting creates public judgment",
            "viewer wants someone to appreciate it",
        ),
        emotional_physics=(
            "public-space exposure with low approval triggers the urge to "
            "supply the missing approval"
        ),
        viewer_role="appreciator / defender",
        comment_trigger="urge to comment the praise the source did not get in real life",
        share_trigger="signal-boost send to find the maker's tribe",
        cooked_risk="overly-staged 'no one bought today' framings are cooked",
        ethical_risk_default=0.25,
        example_labels=(
            "Emma's lamps stall videos",
            "Instagram 70M stall variation",
        ),
    ),
    StoryFlow(
        id="wrong_audience_right_tribe",
        name="Wrong Audience -> Right Tribe",
        steps=(
            "one person or group dismisses the thing",
            "viewer realizes it was never for them",
            "the true tribe feels called to defend / claim it",
            "identity-based comments and shares increase",
        ),
        emotional_physics=(
            "identity claim pulled into the open by a mismatched dismissal"
        ),
        viewer_role="tribe member",
        comment_trigger="urge to self-identify as the target audience",
        share_trigger="tribal recruitment send",
        cooked_risk="naming a tribe too niche to be claimable kills the share trigger",
        ethical_risk_default=0.2,
        example_labels=(
            "blue-collar dad + goth daughter",
            "fandom / strange-object hooks",
        ),
    ),
    StoryFlow(
        id="shock_problem_immediate_fix",
        name="Shock Problem -> Immediate Fix",
        steps=(
            "a shocking, relatable problem happens",
            "viewer feels fear / protectiveness",
            "a product or solution provides immediate relief",
            "viewer watches for the fix",
        ),
        emotional_physics=(
            "adrenaline spike from a near-miss; relief loop drives "
            "watch-time and bookmark intent"
        ),
        viewer_role="anxious witness",
        comment_trigger="urge to recommend / ask about the fix",
        share_trigger="prevention send to parents, caregivers, friends",
        cooked_risk=(
            "manufactured near-miss reads as exploitative if children "
            "are involved"
        ),
        ethical_risk_default=0.55,
        example_labels=(
            "baby head hitting floor -> product fix",
        ),
    ),
    StoryFlow(
        id="ethical_edge_vulnerability_sympathy_surge",
        name="Ethical Edge Vulnerability -> Sympathy Surge",
        steps=(
            "a vulnerable person or group is exposed",
            "social protection instinct spikes",
            "viewer feels guilt and protectiveness",
            "high virality, but high ethical and brand risk",
        ),
        emotional_physics=(
            "protected-class vulnerability triggers an outsized sympathy "
            "surge; reach is huge but consequences cut both ways"
        ),
        viewer_role="protector",
        comment_trigger="urge to defend, send love",
        share_trigger="solidarity broadcast",
        cooked_risk=(
            "performative vulnerability and exploitation flags rapidly "
            "cook this flow"
        ),
        ethical_risk_default=0.85,
        example_labels=(
            "mental illness / down syndrome fake-comment hook",
        ),
    ),
    # Phase 7.1: added after live VPS testing surfaced two recurring
    # flows that the original 8-flow library did not cover, causing
    # mismatched assignments like a "please be honest" + curiosity
    # reveal getting bucketed under public_disrespect_viewer_defense.
    StoryFlow(
        id="direct_viewer_plea_social_contract",
        name="Direct Viewer Plea -> Tiny Social Contract",
        steps=(
            "creator tells viewer they can scroll OR asks them not to",
            "viewer is given a tiny responsibility ('stay 12 seconds', 'be honest')",
            "staying / commenting feels like helping",
            "viewer watches to honor the implied social contract",
        ),
        emotional_physics=(
            "moral pressure + tiny ask + curiosity gap; "
            "the viewer is recruited as a participant, not a spectator"
        ),
        viewer_role="helper / supporter",
        comment_trigger="urge to comment and reassure / validate / honor the ask",
        share_trigger="low-effort visible support; share as a tiny good deed",
        cooked_risk=(
            "high if it uses 'please don't scroll', 'could you be honest', "
            "'every comment helps' without a fresh twist; these phrasings "
            "are widely cooked"
        ),
        ethical_risk_default=0.25,
        example_labels=(
            "please don't scroll",
            "stay 12 seconds",
            "could you be honest",
            "every comment helps motivate him",
        ),
    ),
    StoryFlow(
        id="weirdness_curiosity_reveal_loop",
        name="Weirdness / Not Normal -> Curiosity Reveal Loop",
        steps=(
            "creator frames themselves or the object as weird / not normal",
            "viewer feels curiosity or low-key social judgment",
            "the clip withholds the reveal (process, purpose, payoff)",
            "viewer stays to understand what is being made or why it matters",
        ),
        emotional_physics=(
            "self-othering + curiosity gap + reveal retention; "
            "viewer must stay to resolve the 'what is this' question"
        ),
        viewer_role="curious observer / judge",
        comment_trigger="urge to comment on weirdness, process, or final reveal",
        share_trigger="send to someone who likes weird / satisfying-process clips",
        cooked_risk=(
            "medium-high if it collapses into generic 'I'm not normal' "
            "bait without a real reveal"
        ),
        ethical_risk_default=0.20,
        example_labels=(
            "No I'm not a NORMAL adult",
            "unusual craft process",
            "strange materials",
            "weird hand gestures with process reveal",
        ),
    ),
)


STORY_FLOWS_BY_ID: dict[str, StoryFlow] = {flow.id: flow for flow in STORY_FLOWS}


def render_story_flows_for_prompt() -> str:
    """Format the library as a numbered text block suitable for embedding
    in a system prompt. Stable order, stable wording — change anything
    here and the prompt-content tests will tell you what shifted."""
    lines: list[str] = []
    for i, flow in enumerate(STORY_FLOWS, start=1):
        lines.append(f"{i}. {flow.name}  (id: {flow.id})")
        lines.append("   Steps:")
        for step in flow.steps:
            lines.append(f"     - {step}")
        lines.append(f"   Emotional physics: {flow.emotional_physics}")
        lines.append(f"   Viewer role:       {flow.viewer_role}")
        lines.append(f"   Comment trigger:   {flow.comment_trigger}")
        lines.append(f"   Share trigger:     {flow.share_trigger}")
        lines.append(f"   Cooked risk:       {flow.cooked_risk}")
        lines.append(f"   Ethical risk default: {flow.ethical_risk_default:.2f}")
        lines.append(f"   Example labels:    {', '.join(flow.example_labels)}")
        lines.append("")
    return "\n".join(lines).rstrip()
