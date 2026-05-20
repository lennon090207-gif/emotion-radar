"""concept_feedback DB layer + build_taste_summary."""

from __future__ import annotations

from pathlib import Path

import pytest

from emotion_radar import db


def _make_report_id(tmp_path: Path) -> tuple[Path, str]:
    db_path = tmp_path / "test.db"
    rid = db.insert_report(db_path, {
        "platform": "TikTok",
        "submitted_url": "https://www.tiktok.com/@u/v/1",
        "video_id": "1",
        "creator_username": "u",
        "creator_nickname": "U",
        "caption": "c",
        "metrics": {"views": 100, "likes": 1, "comments": 0, "shares": 0, "saves": 0},
    })
    return db_path, rid


def test_insert_feedback_returns_id_and_persists(tmp_path: Path):
    db_path, rid = _make_report_id(tmp_path)
    fb_id = db.insert_feedback(
        db_path,
        report_id=rid,
        concept_source_type="pioneer_concept",
        concept_name="Receipt Wall",
        concept_index=1,
        rating="fire",
        note="strong tactile evidence format",
    )
    assert isinstance(fb_id, int) and fb_id > 0
    rows = db.list_feedback(db_path)
    assert len(rows) == 1
    assert rows[0]["concept_name"] == "Receipt Wall"
    assert rows[0]["rating"] == "fire"
    assert rows[0]["note"] == "strong tactile evidence format"
    assert rows[0]["concept_index"] == 1


def test_invalid_rating_raises_feedback_error(tmp_path: Path):
    db_path, rid = _make_report_id(tmp_path)
    with pytest.raises(db.FeedbackError):
        db.insert_feedback(
            db_path, report_id=rid, concept_source_type="variation",
            concept_name="X", concept_index=1, rating="amazing",  # not allowed
        )


def test_missing_required_fields_raise(tmp_path: Path):
    db_path, _ = _make_report_id(tmp_path)
    with pytest.raises(db.FeedbackError):
        db.insert_feedback(
            db_path, report_id="", concept_source_type="variation",
            concept_name="X", concept_index=1, rating="fire",
        )


def test_allowed_ratings_constant_matches_spec():
    assert set(db.ALLOWED_RATINGS) == {"fire", "good", "meh", "cringe", "cooked"}


def test_list_feedback_orders_newest_first(tmp_path: Path):
    import time
    db_path, rid = _make_report_id(tmp_path)
    db.insert_feedback(db_path, rid, "variation", "First", 1, "good")
    time.sleep(0.01)
    db.insert_feedback(db_path, rid, "variation", "Second", 2, "meh")
    time.sleep(0.01)
    db.insert_feedback(db_path, rid, "variation", "Third", 3, "fire")
    rows = db.list_feedback(db_path)
    names = [r["concept_name"] for r in rows]
    # Newest first: Third, Second, First.
    assert names == ["Third", "Second", "First"]


def test_get_report_feedback_filters_by_report(tmp_path: Path):
    db_path, rid1 = _make_report_id(tmp_path)
    rid2 = db.insert_report(db_path, {
        "platform": "TikTok",
        "submitted_url": "https://www.tiktok.com/@u/v/2",
        "video_id": "2",
        "creator_username": "u",
        "creator_nickname": "U",
        "caption": "c",
        "metrics": {"views": 100, "likes": 1, "comments": 0, "shares": 0, "saves": 0},
    })
    db.insert_feedback(db_path, rid1, "variation", "A", 1, "fire")
    db.insert_feedback(db_path, rid2, "variation", "B", 1, "cringe")
    db.insert_feedback(db_path, rid1, "variation", "C", 2, "good")

    only_rid1 = db.get_report_feedback(db_path, rid1)
    assert {r["concept_name"] for r in only_rid1} == {"A", "C"}
    only_rid2 = db.get_report_feedback(db_path, rid2)
    assert {r["concept_name"] for r in only_rid2} == {"B"}


# ---- taste summary ---------------------------------------------------------

def test_build_taste_summary_returns_none_when_no_feedback(tmp_path: Path):
    db_path = tmp_path / "empty.db"
    db.init_db(db_path)
    assert db.build_taste_summary(db_path) is None


def test_build_taste_summary_groups_by_rating(tmp_path: Path):
    db_path, rid = _make_report_id(tmp_path)
    db.insert_feedback(db_path, rid, "pioneer_concept", "Receipt Wall", 1, "fire",
                       note="strong tactile evidence format")
    db.insert_feedback(db_path, rid, "pioneer_concept", "Bystander Camera", 2, "good")
    db.insert_feedback(db_path, rid, "variation", "Tribe Snap-Defense", 3, "meh")
    db.insert_feedback(db_path, rid, "variation", "Customer Smiles", 4, "cringe",
                       note="too polished")
    db.insert_feedback(db_path, rid, "pioneer_concept", "Generic Reveal", 5, "cooked",
                       note="please-be-honest framing is cooked")
    summary = db.build_taste_summary(db_path)
    assert summary is not None
    assert "User tends to like:" in summary
    assert "Receipt Wall" in summary
    assert "Bystander Camera" in summary
    assert "User dislikes:" in summary
    assert "Customer Smiles" in summary
    assert "Generic Reveal" in summary
    # "meh" rated items should NOT appear under like/dislike sections by name.
    # (They're internally bucketed as neither.)
    # Notes section should surface user-written notes.
    assert "Recent notes:" in summary
    assert "too polished" in summary
    assert "please-be-honest framing is cooked" in summary


def test_taste_summary_passes_through_to_pass3_user_prompt(tmp_path: Path):
    """End-to-end: a stored taste summary should be injectable into the
    Pass 3 user prompt verbatim. Smoke check that the integration
    point matches."""
    from emotion_radar import analysis as A
    db_path, rid = _make_report_id(tmp_path)
    db.insert_feedback(db_path, rid, "pioneer_concept", "Receipt Wall", 1, "fire")
    db.insert_feedback(db_path, rid, "variation", "Customer Smiles", 4, "cringe",
                       note="too polished, too staged")
    summary = db.build_taste_summary(db_path)
    assert summary is not None

    pass1 = {"environment": "market stall"}
    pass2 = {"variations": [], "pioneer_concepts": []}
    up = A.build_specificity_user_prompt(pass1, pass2, taste_profile=summary)
    assert "USER TASTE PROFILE" in up
    assert "Receipt Wall" in up
    assert "too polished, too staged" in up
