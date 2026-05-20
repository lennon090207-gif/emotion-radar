"""analyze-link CLI integration tests.

All network is mocked: ApifyClient, download_video, extract_frames,
build_contact_sheet, and the vision providers. No real Apify run,
no real vision API call, no real ffmpeg/PIL invocation."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest
from click.testing import CliRunner

from emotion_radar import cli as cli_mod
from emotion_radar.cli import cli
from emotion_radar.db import get_report
from emotion_radar.models import ApifyRunInfo


# ---- fixtures --------------------------------------------------------------

OLIVER_VIDEO_ID = "7623559389307211030"
OLIVER_URL = "https://www.tiktok.com/@olivermakesartt/video/7623559389307211030"


def _fake_apify_item(video_id: str = OLIVER_VIDEO_ID, url: str = OLIVER_URL) -> dict:
    return {
        "id": video_id,
        "webVideoUrl": url,
        "text": "please be honest, how are they?",
        "mediaUrls": [f"https://api.apify.com/v2/key-value-stores/abc/records/{video_id}.mp4"],
        "videoMeta": {
            "duration": 14.0,
            "downloadAddr": "https://fallback/dl.mp4",
            "coverUrl": "https://cdn/cover.jpg",
        },
        "authorMeta": {"name": "olivermakesartt", "nickName": "Oliver"},
        "playCount": 12345,
        "diggCount": 67,
        "commentCount": 5,
        "shareCount": 2,
        "collectCount": 1,
    }


def _fake_run_info() -> ApifyRunInfo:
    return ApifyRunInfo(
        run_id="RUN_FAKE",
        dataset_id="DS_FAKE",
        usage_total_usd=0.0083,
        charged_events={"count": 1},
    )


@pytest.fixture
def mock_infrastructure(monkeypatch, tmp_path: Path):
    """Mock everything below the CLI layer: token, Apify, download,
    frames, contact sheet. Returns control handles for tests to tweak."""
    state: dict[str, Any] = {
        "item": _fake_apify_item(),
        "run_info": _fake_run_info(),
    }

    # 1. APIFY_TOKEN — pretend it's set.
    monkeypatch.setattr(cli_mod, "get_apify_token", lambda env=None: "apify_api_fake")

    # 2. ApifyClient — replace with a thin stub that returns our item.
    class _FakeApifyClient:
        def __init__(self, *a, **kw):
            pass

        def run_actor(self, urls, **kw):
            return [state["item"]], state["run_info"]

    monkeypatch.setattr(cli_mod, "ApifyClient", _FakeApifyClient)

    # 3. download_video — write a tiny placeholder file.
    def _fake_download(url, out_dir, video_id, **kw):
        out_dir = Path(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        p = out_dir / f"{video_id}.mp4"
        p.write_bytes(b"fake mp4 bytes")
        return p

    monkeypatch.setattr(cli_mod, "download_video", _fake_download)

    # 4. extract_frames — write tiny JPEG-ish placeholders.
    def _fake_extract(video_path, out_dir, timestamps):
        out_dir = Path(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        paths = []
        for ts in timestamps:
            p = out_dir / f"t{ts:0.2f}.jpg"
            p.write_bytes(b"\xff\xd8\xff\xe0frame")
            paths.append(p)
        return paths

    monkeypatch.setattr(cli_mod, "extract_frames", _fake_extract)

    # 5. build_contact_sheet — write a tiny placeholder JPEG and return path.
    def _fake_sheet(frame_paths, timestamps, out_path, **kw):
        out_path = Path(out_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_bytes(b"\xff\xd8\xff\xe0contact-sheet-bytes\xff\xd9")
        return out_path

    monkeypatch.setattr(cli_mod, "build_contact_sheet", _fake_sheet)

    return state


# ---- mock vision providers -------------------------------------------------

PASS1_OLIVER_GOOD = {
    "frame_observations": [
        {"timestamp": "0.0s", "observation": "outdoor market stall with handmade dragon lamps on display",
         "people_visible": "maker behind stall", "object_state": "lamps intact on table",
         "action_change_from_previous": ""},
        {"timestamp": "1.0s", "observation": "stranger picks up a dragon lamp",
         "people_visible": "maker, stranger", "object_state": "lamp in stranger's hand",
         "action_change_from_previous": "stranger entered frame and grabbed lamp"},
        {"timestamp": "2.0s", "observation": "dragon lamp lies smashed on the floor",
         "people_visible": "maker reacting, stranger",
         "object_state": "lamp on floor, broken",
         "action_change_from_previous": "lamp was thrown / dropped and smashed"},
    ],
    "environment": "outdoor weekend market stall",
    "people": "underdog maker (mid-30s); stranger acting as antagonist",
    "product_or_object": "handmade dragon lamp (HTTYD-style)",
    "onscreen_text": "Please be honest, how are they?",
    "physical_action": "thrown and smashed on the floor",
    "object_state_change": "lamp starts on the display table, ends on the floor with visible damage",
    "visual_conflict_detected": True,
    "conflict_type": "smash",
    "confidence": 0.92,
    "uncertainty_notes": "",
}

PASS2_OLIVER_GOOD = {
    "visual_hook_summary": (
        "At an outdoor market stall, a stranger picks up the maker's handmade "
        "dragon lamp and throws it on the floor, where it is smashed; the "
        "mechanic is public disrespect + underdog maker triggering viewer defense."
    ),
    "viral_mechanic": "public disrespect + underdog maker (viewer-defense instinct)",
    "scroll_stop_reason": "physical destruction of a small maker's work is an instant moral violation",
    "viewer_role": "defender",
    "comment_trigger": "viewer wants to verbally retaliate against the antagonist",
    "share_trigger": "send-to-friend for shared sense of injustice",
    "emotional_pressure": "feeling that scrolling away is 'letting this stand'",
    "emotional_mechanic": "public disrespect + underdog maker (viewer-defense instinct)",
    "emotions_triggered": ["anger", "protectiveness", "indignation"],
    "why_it_works": "moral violation against an underdog self-casts the viewer as defender",
    "cooked_elements": ["please-be-honest framing", "staged-stranger trope"],
    "cooked_parts_to_avoid": ["please be honest"],
    "freshness_angle": "physical-destruction tier of public-doubt, not mere insult",
    "scroll_stop_strength_score": 0.86,
    "comment_likelihood_score": 0.81,
    "share_likelihood_score": 0.72,
    "viewer_role_strength_score": 0.83,
    "creative_transfer_potential_score": 0.74,
    "virality_capability_score": 0.79,
    "product_attachability_score": 0.62,
    "transferability_score": 0.66,
    "freshness_score": 0.71,
    "cooked_score": 0.34,
    "overall_opportunity_score": 0.78,
    "creative_hook_concepts": [
        # 2 same_mechanic
        {"creative_distance": "same_mechanic", "concept_name": "Silent Proof After Insult",
         "first_2_seconds": "dismissive comment overlay, then maker silently rotates one finished piece to show obscene detail",
         "emotional_trigger": "vindication", "viewer_role": "jury",
         "why_it_could_go_viral": "viewer renders a verdict against the heckler",
         "what_to_avoid": "narration; let silence do the work",
         "believability_risk": "feels staged if comment overlay reads written by creator",
         "cooked_risk": "silent-reveal format is widely used; the proof must be very sharp"},
        {"creative_distance": "same_mechanic", "concept_name": "Wrong Audience",
         "first_2_seconds": "stranger calls a finished piece 'weird', text immediately names the tribe that would defend it",
         "emotional_trigger": "tribal recognition",
         "viewer_role": "tribe member",
         "why_it_could_go_viral": "viewers self-identify as the tribe and defend",
         "what_to_avoid": "naming a tribe so niche the average viewer can't claim it",
         "believability_risk": "passes if the tribe label is real and specific",
         "cooked_risk": "'POV: when X says Y' framing is cooked"},
        # 3 adjacent_leap
        {"creative_distance": "adjacent_leap", "concept_name": "Almost Gave Up",
         "first_2_seconds": "open mic / pop-up scene, performer starts packing up after being ignored, then one person notices",
         "emotional_trigger": "second-hand-pride",
         "viewer_role": "rescuer",
         "why_it_could_go_viral": "viewer is positioned as one of the people who could 'save' the moment",
         "what_to_avoid": "telegraphing the 'almost gave up' in the caption",
         "believability_risk": "fails if the timing of the rescue is too clean",
         "cooked_risk": "narrated 'sad creator' beats are cooked; let it be silent"},
        {"creative_distance": "adjacent_leap", "concept_name": "Hidden Emotional Value",
         "first_2_seconds": "stranger says 'I don't get it', then text reveals the deeply personal reason it was made",
         "emotional_trigger": "indignation flipping to recognition",
         "viewer_role": "defender",
         "why_it_could_go_viral": "the reveal punishes the dismissal",
         "what_to_avoid": "manipulative grief-bait phrasing",
         "believability_risk": "feels exploitative if the reveal is too sentimental",
         "cooked_risk": "memorial reveals are cooked; the framing must be cleaner than 'this was for my X'"},
        {"creative_distance": "adjacent_leap", "concept_name": "Wrong Person Rejects It",
         "first_2_seconds": "passerby loudly dismisses the work, viewer immediately understands they were never the target",
         "emotional_trigger": "insider recognition",
         "viewer_role": "insider",
         "why_it_could_go_viral": "comments fill with the actual target audience self-identifying",
         "what_to_avoid": "telling the viewer who the target is; let them notice",
         "believability_risk": "the dismisser must look like a real wrong audience, not a strawman",
         "cooked_risk": "low; the 'wrong-audience' framing is underused"},
        # 2 big_swing
        {"creative_distance": "big_swing", "concept_name": "Public Doubt / Private Effort",
         "first_2_seconds": "loud public rejection in one shot, hard cut to long private workshop tape with no narration",
         "emotional_trigger": "felt injustice",
         "viewer_role": "jury",
         "why_it_could_go_viral": "the asymmetry between the dismissal and the effort is the punchline",
         "what_to_avoid": "voiceover; let the cut do it",
         "believability_risk": "needs real workshop footage; staged proof kills it",
         "cooked_risk": "before/after montages are widely cooked; this must feel raw"},
        {"creative_distance": "big_swing", "concept_name": "Community Rescue",
         "first_2_seconds": "creator visibly closing up early, then one stranger stops, then the camera pans to a few people gathering",
         "emotional_trigger": "second-hand pride",
         "viewer_role": "rescuer",
         "why_it_could_go_viral": "viewer self-casts as one of the rescuers; high share for 'we saved this'",
         "what_to_avoid": "engineered crowd shots; one or two real people land harder than a fake throng",
         "believability_risk": "easily reads as staged if the timing is too neat",
         "cooked_risk": "'almost gave up' adjacent format risk; pivot the reveal to community, not creator"},
        # 1 wildcard
        {"creative_distance": "wildcard", "concept_name": "Receipt of Rudeness",
         "first_2_seconds": "creator silently holds up a printed screenshot of a rude DM next to the actual finished piece",
         "emotional_trigger": "vindication via evidence",
         "viewer_role": "jury",
         "why_it_could_go_viral": "the physical receipt format is rare and tactile",
         "what_to_avoid": "blurring the screenshot too aggressively; readability matters",
         "believability_risk": "fails if the DM is obviously fabricated",
         "cooked_risk": "screenshot-reveal posts are common; the physical-printout angle is what carries it"},
    ],
    # Phase 5 additions
    "matched_story_flows": [
        {"id": "public_disrespect_viewer_defense",
         "name": "Public Disrespect -> Viewer Defense",
         "confidence": 0.93,
         "why_matched": "stranger physically destroys handmade item at a market stall"},
        {"id": "stall_vulnerability_social_judgment",
         "name": "Stall Vulnerability -> Social Judgment",
         "confidence": 0.7,
         "why_matched": "market stall display in plain view sets up public-judgment context"},
    ],
    "dominant_story_flow": "public_disrespect_viewer_defense",
    "story_flow_steps_observed": [
        "stranger approaches the market stall",
        "stranger picks up and destroys the dragon lamp",
        "maker visibly absorbs the disrespect",
        "viewer is positioned as defender",
    ],
    "story_flow_strength_score": 0.91,
    "novelty_beyond_baseline_score": 0.42,
    "ethical_risk_score": 0.28,
    "cringe_risk_score": 0.35,
    "breakout_potential_score": 0.74,
    "variations": [
        {"concept_name": "Receipt of Cruelty", "story_flow_id": "public_disrespect_viewer_defense",
         "first_2_seconds": "maker silently holds up a printed screenshot of a rude DM next to the finished piece",
         "emotional_trigger": "vindication", "viewer_role": "jury",
         "why_it_could_go_viral": "physical receipt format is rare and tactile",
         "what_is_new": "the printed-screenshot prop instead of verbal exchange",
         "what_is_cooked_to_avoid": "do not use 'please be honest'",
         "believability_risk": "fabricated screenshots kill it if too on-the-nose"},
        {"concept_name": "Walked-Past Verdict", "story_flow_id": "public_disrespect_viewer_defense",
         "first_2_seconds": "wide shot of stall; couple stops, picks up the piece, makes a face, sets it down hard, walks off",
         "emotional_trigger": "second-hand indignation", "viewer_role": "defender",
         "why_it_could_go_viral": "the put-down beat is silent and damning",
         "what_is_new": "no spoken comment; body language carries the disrespect",
         "what_is_cooked_to_avoid": "do not zoom on a sad reaction shot",
         "believability_risk": "needs candid timing; staged passerby kills it"},
        {"concept_name": "Tribe Snap-Defense", "story_flow_id": "wrong_audience_right_tribe",
         "first_2_seconds": "passerby calls the work weird; text immediately names the tribe that would defend it",
         "emotional_trigger": "identity claim", "viewer_role": "tribe member",
         "why_it_could_go_viral": "comments fill with the tribe self-identifying",
         "what_is_new": "tribe is named within first 2 seconds, not the punchline",
         "what_is_cooked_to_avoid": "POV-when-X-says-Y framing",
         "believability_risk": "tribe label must be claimable, not too niche"},
        {"concept_name": "Almost-Closed Save", "story_flow_id": "stall_vulnerability_social_judgment",
         "first_2_seconds": "maker starts packing up after a quiet day; one stranger stops",
         "emotional_trigger": "second-hand pride", "viewer_role": "rescuer",
         "why_it_could_go_viral": "viewer positioned as one of the people who could save the moment",
         "what_is_new": "no caption tells the story; pack-up beat is silent",
         "what_is_cooked_to_avoid": "narrated 'sad creator' captions",
         "believability_risk": "fails if the rescue timing is too clean"},
        {"concept_name": "Comment Card on the Table", "story_flow_id": "stall_vulnerability_social_judgment",
         "first_2_seconds": "a handwritten note on the stall reads a real rude comment from a previous day",
         "emotional_trigger": "moral outrage", "viewer_role": "appreciator",
         "why_it_could_go_viral": "physical note format collapses time and creates witness role",
         "what_is_new": "asynchronous evidence rather than live confrontation",
         "what_is_cooked_to_avoid": "don't over-handwrite; keep authentic",
         "believability_risk": "feels manipulative if the comment is too perfect"},
    ],
    "pioneer_concepts": [
        {"concept_name": "Receipt Wall",
         "inspired_by_story_flow_id": "comment_humiliation_public_witness",
         "first_2_seconds": "creator silently pins printed rude DMs to a corkboard behind the work",
         "emotional_physics": "tactile evidence of cruelty against the maker; viewers become jurors",
         "why_it_is_not_a_direct_copy": "no creator-reaction shot, no read-aloud; the wall is the indictment",
         "why_it_could_be_breakout": "tactile-evidence formats are underused; the prop is the hook",
         "viewer_comment_impulse": "urge to add their own receipt or defend",
         "ethical_or_cringe_risk": "real names must be redacted; otherwise high brand risk"},
        {"concept_name": "Bystander Camera",
         "inspired_by_story_flow_id": "public_disrespect_viewer_defense",
         "first_2_seconds": "filmed from a stranger's POV in line behind the rude customer",
         "emotional_physics": "third-person witnessing accelerates the defender response",
         "why_it_is_not_a_direct_copy": "the maker is not the protagonist of the frame; the bystander is",
         "why_it_could_be_breakout": "POV inversion makes the viewer feel they were there",
         "viewer_comment_impulse": "urge to roleplay 'what I would have said'",
         "ethical_or_cringe_risk": "low if framing reads candid, high if staged"},
        {"concept_name": "Quiet Apology",
         "inspired_by_story_flow_id": "wrong_audience_right_tribe",
         "first_2_seconds": "passerby comes back next day and quietly apologises for their reaction yesterday",
         "emotional_physics": "delayed-justice loop; tribe sees the antagonist redeemed",
         "why_it_is_not_a_direct_copy": "antagonist becomes the hero; no defender role for the viewer",
         "why_it_could_be_breakout": "redemption arcs in 1-2s are rare; high share for restored faith",
         "viewer_comment_impulse": "urge to share to someone who needs faith restored",
         "ethical_or_cringe_risk": "feels manipulative if apology reads off a script"},
        {"concept_name": "Stranger's Note",
         "inspired_by_story_flow_id": "stall_vulnerability_social_judgment",
         "first_2_seconds": "maker finds a handwritten note tucked under one of the pieces",
         "emotional_physics": "anonymous appreciation breaks the public-judgment frame",
         "why_it_is_not_a_direct_copy": "no insult, no destruction; positive intervention witnessed in silence",
         "why_it_could_be_breakout": "appreciation-witnessed-in-silence is underused",
         "viewer_comment_impulse": "urge to claim 'I would have written this'",
         "ethical_or_cringe_risk": "fails if the note is too sentimental"},
        {"concept_name": "Last Item Standing",
         "inspired_by_story_flow_id": "moral_pressure_tiny_rescue",
         "first_2_seconds": "wide shot of stall with one piece left at end-of-day; maker reaches to pack it",
         "emotional_physics": "scarcity + moral pressure to be the one who saved the last piece",
         "why_it_is_not_a_direct_copy": "no plea, no 'please don't scroll'; only the visual stake",
         "why_it_could_be_breakout": "tactile, time-pressured rescue framing without cooked phrasing",
         "viewer_comment_impulse": "urge to ask 'is it still available?'",
         "ethical_or_cringe_risk": "feels staged if stall is suspiciously well-lit"},
    ],
}


DEFAULT_PASS3 = {
    "specificity_notes": "rewritten via mock provider",
    "weak_patterns_fixed": [],
    "scene_concepts": [
        {"source_type": "pioneer_concept",
         "source_concept_name": "Receipt Wall",
         "story_flow_id": "comment_humiliation_public_witness",
         "specific_concept_name": "Receipt Wall, Pinned",
         "first_2_seconds": "creator silently pins three printed rude DMs to a corkboard",
         "onscreen_text": "I keep them all now.",
         "visual_beat": "the third pin going in",
         "social_tension": "no spoken word; the wall does the talking",
         "viewer_comment_impulse": "urge to add 'I'd pay extra now'",
         "why_they_keep_watching": "slow-reveal payoff of the wall",
         "freshness_angle": "tactile evidence collage is underused",
         "believability_risk": "real names must be redacted",
         "cringe_risk": "feels staged if pinned too neatly",
         "virality_potential_score": 0.86},
        {"source_type": "variation",
         "source_concept_name": "Receipt of Cruelty",
         "story_flow_id": "public_disrespect_viewer_defense",
         "specific_concept_name": "Receipt at the Stall",
         "first_2_seconds": "the seller silently sets a printed rude-DM screenshot next to the lamp",
         "onscreen_text": "she said it was 'overpriced trash'",
         "visual_beat": "tape pulled off the screenshot, set down",
         "social_tension": "the stall is busy; people pause to read",
         "viewer_comment_impulse": "urge to defend the maker",
         "why_they_keep_watching": "viewers want to see if anyone reacts",
         "freshness_angle": "physical-receipt format instead of voice-over",
         "believability_risk": "fails if the DM reads written-by-the-creator",
         "cringe_risk": "fabricated screenshot kills it",
         "virality_potential_score": 0.78},
    ],
}


def _patch_providers(monkeypatch, pass1_text: str, pass2_text: str, pass3_text: str | None = None):
    """Patch build_provider_for_role with mocks that serve a queued
    sequence of text responses (Pass 2, then Pass 3, then repeat last).
    Pass 3 defaults to DEFAULT_PASS3 so existing tests that don't care
    about Pass 3 content still work under the new three-pass default."""
    if pass3_text is None:
        pass3_text = json.dumps(DEFAULT_PASS3)

    class _MP:
        name = "mock"

        def __init__(self, model_label, image_resp, text_responses):
            self.model = model_label
            self._image = image_resp
            self._text_responses = list(text_responses)
            self._text_index = 0
            self.image_calls: list = []
            self.text_calls: list = []

        def analyze_image(self, image_path, system, user):
            self.image_calls.append({"image_path": image_path, "system": system, "user": user})
            return self._image

        def analyze_text(self, system, user):
            self.text_calls.append({"system": system, "user": user})
            if self._text_index < len(self._text_responses):
                r = self._text_responses[self._text_index]
                self._text_index += 1
                return r
            return self._text_responses[-1] if self._text_responses else "{}"

    def _fake_build(env, role):
        if role == "vision_event":
            return _MP("mock-vision-1", pass1_text, [""])
        return _MP("mock-strategy-1", "", [pass2_text, pass3_text])

    monkeypatch.setattr(cli_mod, "build_provider_for_role", _fake_build)


# ---- helpers ---------------------------------------------------------------

def _invoke(tmp_path: Path, *args: str):
    runner = CliRunner()
    return runner.invoke(
        cli,
        ["--data-dir", str(tmp_path), *args],
        catch_exceptions=False,
    )


def _only_report_id(tmp_path: Path) -> str:
    """Pull the report id of the single report inserted by the run."""
    from emotion_radar.db import list_reports
    rows = list_reports(tmp_path / "emotion_radar.db", limit=10)
    assert len(rows) == 1, f"expected exactly one report, got {len(rows)}"
    return rows[0]["id"]


# ---- analyze-link: --no-vision ---------------------------------------------

def test_no_vision_skips_provider_and_evaluation(mock_infrastructure, monkeypatch, tmp_path: Path):
    # Provider must NOT be built when --no-vision is set.
    def _explode(env, role):
        raise AssertionError("provider must not be built in --no-vision mode")
    monkeypatch.setattr(cli_mod, "build_provider_for_role", _explode)

    result = _invoke(tmp_path, "analyze-link", OLIVER_URL, "--no-vision")

    assert result.exit_code == 0, result.output
    assert "--no-vision: skipping" in result.output
    assert "Emotion Radar Report" in result.output
    # No calibration ran.
    assert "Evaluation" not in result.output

    # Report row exists, with no vision analysis.
    rid = _only_report_id(tmp_path)
    row = get_report(tmp_path / "emotion_radar.db", rid)
    assert row["video_id"] == OLIVER_VIDEO_ID
    assert row["visual_hook_summary"] is None
    assert row["emotional_mechanic"] is None


# ---- analyze-link: --dry-run-vision ----------------------------------------

def test_dry_run_vision_prints_all_three_prompts_and_skips_api(mock_infrastructure, monkeypatch, tmp_path: Path):
    def _explode(env, role):
        raise AssertionError("provider must not be built in --dry-run-vision mode")
    monkeypatch.setattr(cli_mod, "build_provider_for_role", _explode)

    result = _invoke(tmp_path, "analyze-link", OLIVER_URL, "--dry-run-vision")

    assert result.exit_code == 0, result.output
    assert "=== PASS 1 SYSTEM" in result.output
    assert "=== PASS 2 SYSTEM" in result.output
    assert "=== PASS 3 SYSTEM" in result.output  # Phase 6: three-pass default
    assert "=== PASS 1 USER" in result.output
    assert "=== PASS 2 USER" in result.output
    assert "=== PASS 3 USER" in result.output
    assert "dry-run: no API call made" in result.output
    # Final report stub also prints.
    assert "Emotion Radar Report" in result.output


def test_dry_run_vision_with_no_specificity_prints_only_two_prompts(mock_infrastructure, monkeypatch, tmp_path: Path):
    def _explode(env, role):
        raise AssertionError("provider must not be built in --dry-run-vision mode")
    monkeypatch.setattr(cli_mod, "build_provider_for_role", _explode)

    result = _invoke(
        tmp_path, "analyze-link", OLIVER_URL,
        "--dry-run-vision", "--no-specificity",
    )

    assert result.exit_code == 0, result.output
    assert "=== PASS 1 SYSTEM" in result.output
    assert "=== PASS 2 SYSTEM" in result.output
    assert "=== PASS 3 SYSTEM" not in result.output


# ---- analyze-link: full two-pass + auto-evaluation (Oliver fixture) -------

def test_full_two_pass_auto_evaluates_oliver_and_passes(mock_infrastructure, monkeypatch, tmp_path: Path):
    _patch_providers(
        monkeypatch,
        pass1_text=json.dumps(PASS1_OLIVER_GOOD),
        pass2_text=json.dumps(PASS2_OLIVER_GOOD),
    )

    result = _invoke(tmp_path, "analyze-link", OLIVER_URL)

    assert result.exit_code == 0, result.output
    assert "Emotion Radar Report" in result.output
    # Pass 1 + Pass 2 both labeled in CLI output.
    assert "Pass 1 (visual event)" in result.output
    assert "Pass 2 (hook strategy)" in result.output
    # Auto-evaluation ran against the Oliver fixture.
    assert "Evaluation" in result.output
    assert "PASS" in result.output
    assert "auto, known video_id" in result.output
    # No calibration-failed warning.
    assert "Calibration failed" not in result.output

    # Report row got the merged fields written back.
    rid = _only_report_id(tmp_path)
    row = get_report(tmp_path / "emotion_radar.db", rid)
    assert row["emotional_mechanic"].startswith("public disrespect")
    assert row["onscreen_text"] == "Please be honest, how are they?"
    assert row["overall_opportunity_score"] == 0.78
    assert isinstance(row["raw_analysis"], dict)
    # Phase 6: analyze-link defaults to three-pass.
    assert row["raw_analysis"]["analysis_mode"] == "three_pass"
    assert row["raw_analysis"]["visual_event_pass"]["conflict_type"] == "smash"
    # Phase 6: specificity_pass rides in raw_analysis.
    assert "specificity_pass" in row["raw_analysis"]
    assert isinstance(row["raw_analysis"]["specificity_pass"].get("scene_concepts"), list)
    # Phase 4: new viral-focused fields ride in raw_analysis.
    hsp = row["raw_analysis"]["hook_strategy_pass"]
    assert hsp["viral_mechanic"].startswith("public disrespect")
    assert hsp["scroll_stop_reason"]
    assert hsp["comment_trigger"]
    assert hsp["share_trigger"]
    assert hsp["scroll_stop_strength_score"] == 0.86
    assert hsp["virality_capability_score"] == 0.79
    # hook_mutations is sourced from creative_hook_concepts (8 items: 2/3/2/1).
    assert len(row["hook_mutations"]) == 8
    by_dist = {c["creative_distance"] for c in row["hook_mutations"]}
    assert by_dist == {"same_mechanic", "adjacent_leap", "big_swing", "wildcard"}


# ---- analyze-link: failed calibration ------------------------------------

def test_failed_calibration_prints_warning(mock_infrastructure, monkeypatch, tmp_path: Path):
    """Pass 2 produces a too-soft mechanic; the Oliver canary should fail
    and the CLI should print the explicit warning. Report still saved."""
    weak_pass2 = {
        **PASS2_OLIVER_GOOD,
        "visual_hook_summary": "The creator looks discouraged at his market stall.",
        "emotional_mechanic": "creator vulnerability",
    }
    weak_pass1 = {
        **PASS1_OLIVER_GOOD,
        "physical_action": "",
        "visual_conflict_detected": False,
        "conflict_type": "none",
        "object_state_change": "lamps stay on the table",
        "frame_observations": [
            {"timestamp": "0.0s", "observation": "market stall, creator looks at camera",
             "people_visible": "creator", "object_state": "lamps on table",
             "action_change_from_previous": ""},
        ],
        # remove "smashed", "thrown", "dragon lamp" etc.
        "product_or_object": "handmade lamps",
    }
    _patch_providers(
        monkeypatch,
        pass1_text=json.dumps(weak_pass1),
        pass2_text=json.dumps(weak_pass2),
    )

    result = _invoke(tmp_path, "analyze-link", OLIVER_URL)

    assert result.exit_code == 0, result.output  # warning, not crash
    assert "FAIL" in result.output
    assert "Calibration failed. Do not trust this report yet." in result.output
    # Report still saved.
    rid = _only_report_id(tmp_path)
    row = get_report(tmp_path / "emotion_radar.db", rid)
    assert row["emotional_mechanic"] == "creator vulnerability"


# ---- analyze-link: --skip-evaluation honors the flag ----------------------

def test_skip_evaluation_overrides_known_fixture(mock_infrastructure, monkeypatch, tmp_path: Path):
    _patch_providers(
        monkeypatch,
        pass1_text=json.dumps(PASS1_OLIVER_GOOD),
        pass2_text=json.dumps(PASS2_OLIVER_GOOD),
    )

    result = _invoke(tmp_path, "analyze-link", OLIVER_URL, "--skip-evaluation")

    assert result.exit_code == 0, result.output
    assert "Evaluation" not in result.output
    assert "skip-evaluation: calibration check skipped" in result.output


# ---- analyze-link: --expected SPEC ----------------------------------------

def test_explicit_expected_path_wins_over_auto_fixture(mock_infrastructure, monkeypatch, tmp_path: Path):
    _patch_providers(
        monkeypatch,
        pass1_text=json.dumps(PASS1_OLIVER_GOOD),
        pass2_text=json.dumps(PASS2_OLIVER_GOOD),
    )

    # A bespoke spec that demands a term NOT in the Oliver-good outputs.
    custom_spec = tmp_path / "custom_spec.json"
    custom_spec.write_text(json.dumps({
        "required_terms": ["mango",  # deliberately absent
                           "market stall"],  # present
    }), encoding="utf-8")

    result = _invoke(tmp_path, "analyze-link", OLIVER_URL,
                     "--expected", str(custom_spec))

    assert result.exit_code == 0, result.output
    assert "Evaluation" in result.output
    assert "FAIL" in result.output
    # Spec path printed in evaluation section.
    assert str(custom_spec) in result.output
    # NOT labeled as auto, because user passed --expected explicitly.
    assert "auto, known video_id" not in result.output


# ---- analyze-link: non-Oliver video, no auto-fixture ----------------------

def test_unknown_video_id_no_auto_evaluation(mock_infrastructure, monkeypatch, tmp_path: Path):
    mock_infrastructure["item"] = _fake_apify_item(
        video_id="99999999999999",
        url="https://www.tiktok.com/@x/video/99999999999999",
    )
    _patch_providers(
        monkeypatch,
        pass1_text=json.dumps(PASS1_OLIVER_GOOD),
        pass2_text=json.dumps(PASS2_OLIVER_GOOD),
    )

    result = _invoke(tmp_path, "analyze-link",
                     "https://www.tiktok.com/@x/video/99999999999999")

    assert result.exit_code == 0, result.output
    # No fixture for this id, so no evaluation runs.
    assert "Evaluation" not in result.output


# ---- analyze-link: final summary structure --------------------------------

def test_final_summary_groups_concepts_by_creative_distance(mock_infrastructure, monkeypatch, tmp_path: Path):
    """Phase 4: concepts are grouped by creative_distance, not by legacy
    'type'. Quota is exactly 8: 2 same_mechanic, 3 adjacent_leap,
    2 big_swing, 1 wildcard."""
    _patch_providers(
        monkeypatch,
        pass1_text=json.dumps(PASS1_OLIVER_GOOD),
        pass2_text=json.dumps(PASS2_OLIVER_GOOD),
    )
    result = _invoke(tmp_path, "analyze-link", OLIVER_URL, "--skip-evaluation")
    assert result.exit_code == 0, result.output
    # Each group label shows up with its count.
    assert "-- Same Mechanic (2) --" in result.output
    assert "-- Adjacent Leap (3) --" in result.output
    assert "-- Big Swing (2) --" in result.output
    assert "-- Wildcard (1) --" in result.output


def test_final_summary_prints_viral_mechanic_sections(mock_infrastructure, monkeypatch, tmp_path: Path):
    """Phase 4: the new summary headers must appear."""
    _patch_providers(
        monkeypatch,
        pass1_text=json.dumps(PASS1_OLIVER_GOOD),
        pass2_text=json.dumps(PASS2_OLIVER_GOOD),
    )
    result = _invoke(tmp_path, "analyze-link", OLIVER_URL, "--skip-evaluation")
    assert result.exit_code == 0, result.output
    assert "Viral Mechanic Analysis" in result.output
    assert "Virality Scores" in result.output
    assert "Broad Hook Concepts" in result.output
    # Specific lines.
    assert "Viral Mechanic:" in result.output
    assert "Why Stops Scroll:" in result.output
    assert "Comment Trigger:" in result.output
    assert "Share Trigger:" in result.output
    assert "Scroll-stop strength:" in result.output
    assert "Comment likelihood:" in result.output
    assert "Share likelihood:" in result.output
    assert "Virality capability:" in result.output
    # Concept name from PASS2_OLIVER_GOOD shows up under its group.
    assert "Silent Proof After Insult" in result.output


def test_final_summary_renders_legacy_hook_mutations_without_creative_distance(
    mock_infrastructure, monkeypatch, tmp_path: Path,
):
    """Back-compat: if Pass 2 still emits the Phase-3 hook_mutations
    shape (type + idea), the printer renders them in an 'Other' bucket
    rather than crashing."""
    legacy_pass2 = {
        k: v for k, v in PASS2_OLIVER_GOOD.items() if k != "creative_hook_concepts"
    }
    legacy_pass2["hook_mutations"] = [
        {"type": "safe", "idea": "Legacy safe idea"},
        {"type": "fresh", "idea": "Legacy fresh idea"},
    ]
    _patch_providers(
        monkeypatch,
        pass1_text=json.dumps(PASS1_OLIVER_GOOD),
        pass2_text=json.dumps(legacy_pass2),
    )
    result = _invoke(tmp_path, "analyze-link", OLIVER_URL, "--skip-evaluation")
    assert result.exit_code == 0, result.output
    # All four canonical buckets render even when empty.
    assert "-- Same Mechanic (0) --" in result.output
    assert "-- Wildcard (0) --" in result.output
    # The legacy items land in the "Other" / unrecognised bucket.
    assert "Other (" in result.output
    assert "Legacy safe idea" in result.output


# ---- known fixture path sanity --------------------------------------------

# ============================================================================
# Phase 5: Story Flow Match / Variations / Pioneer Concepts in final summary
# ============================================================================

def test_final_summary_prints_story_flow_match_section(mock_infrastructure, monkeypatch, tmp_path: Path):
    _patch_providers(
        monkeypatch,
        pass1_text=json.dumps(PASS1_OLIVER_GOOD),
        pass2_text=json.dumps(PASS2_OLIVER_GOOD),
    )
    result = _invoke(tmp_path, "analyze-link", OLIVER_URL, "--skip-evaluation")
    assert result.exit_code == 0, result.output
    assert "Story Flow Match" in result.output
    assert "Dominant flow:" in result.output
    assert "public_disrespect_viewer_defense" in result.output
    assert "Steps observed in source:" in result.output
    # Phase 5 scores rendered.
    assert "Story-flow strength:" in result.output
    assert "Novelty beyond baseline:" in result.output
    assert "Ethical risk:" in result.output
    assert "Cringe risk:" in result.output
    assert "Breakout potential:" in result.output


def test_final_summary_prints_variations_section(mock_infrastructure, monkeypatch, tmp_path: Path):
    _patch_providers(
        monkeypatch,
        pass1_text=json.dumps(PASS1_OLIVER_GOOD),
        pass2_text=json.dumps(PASS2_OLIVER_GOOD),
    )
    result = _invoke(tmp_path, "analyze-link", OLIVER_URL, "--skip-evaluation")
    assert result.exit_code == 0, result.output
    # Header with the exact count.
    assert "Variations (5)" in result.output
    # Concept names from PASS2_OLIVER_GOOD's variations land in output.
    assert "Receipt of Cruelty" in result.output
    assert "Walked-Past Verdict" in result.output
    # Per-variation fields are rendered.
    assert "what is new:" in result.output
    assert "cooked to avoid:" in result.output


def test_final_summary_prints_pioneer_concepts_prominently(mock_infrastructure, monkeypatch, tmp_path: Path):
    """Pioneer Concepts is the primary goal — it should print with a
    visually heavier header and full per-concept detail."""
    _patch_providers(
        monkeypatch,
        pass1_text=json.dumps(PASS1_OLIVER_GOOD),
        pass2_text=json.dumps(PASS2_OLIVER_GOOD),
    )
    result = _invoke(tmp_path, "analyze-link", OLIVER_URL, "--skip-evaluation")
    assert result.exit_code == 0, result.output
    # The prominent header (all-caps + box) appears.
    assert "PIONEER CONCEPTS  (5)" in result.output
    # Concept names from the fixture.
    assert "Receipt Wall" in result.output
    assert "Bystander Camera" in result.output
    assert "Stranger's Note" in result.output
    # Pioneer-specific fields rendered.
    assert "inspired by:" in result.output
    assert "why it could be breakout:" in result.output
    assert "viewer comment impulse:" in result.output
    assert "ethical / cringe risk:" in result.output


def test_final_summary_handles_missing_phase5_fields_gracefully(mock_infrastructure, monkeypatch, tmp_path: Path):
    """If Pass 2 omits the Phase-5 fields entirely (older models, prompt
    regression), the printer should render empty sections and exit
    cleanly — not crash."""
    pass2 = {
        k: v for k, v in PASS2_OLIVER_GOOD.items()
        if k not in (
            "matched_story_flows", "dominant_story_flow", "story_flow_steps_observed",
            "story_flow_strength_score", "novelty_beyond_baseline_score",
            "ethical_risk_score", "cringe_risk_score", "breakout_potential_score",
            "variations", "pioneer_concepts",
        )
    }
    _patch_providers(
        monkeypatch,
        pass1_text=json.dumps(PASS1_OLIVER_GOOD),
        pass2_text=json.dumps(pass2),
    )
    result = _invoke(tmp_path, "analyze-link", OLIVER_URL, "--skip-evaluation")
    assert result.exit_code == 0, result.output
    # Sections still render their headers.
    assert "Story Flow Match" in result.output
    assert "Variations (0)" in result.output
    assert "PIONEER CONCEPTS  (0)" in result.output


def test_known_fixtures_dict_points_at_existing_file():
    """If someone moves or renames docs/examples/oliver_expected.json,
    analyze-link's auto-evaluation silently breaks. Catch that here."""
    assert OLIVER_VIDEO_ID in cli_mod.KNOWN_FIXTURES
    fixture_path = cli_mod.KNOWN_FIXTURES[OLIVER_VIDEO_ID]
    assert fixture_path.is_file(), f"shipped fixture missing on disk: {fixture_path}"
