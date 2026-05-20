"""update_report_analysis behavior. No network."""

from __future__ import annotations

from pathlib import Path

from emotion_radar import db


def _seed_report(tmp_path: Path) -> tuple[Path, str]:
    db_path = tmp_path / "test.db"
    rid = db.insert_report(db_path, {
        "platform": "TikTok",
        "source_url": "https://www.tiktok.com/@u/video/1",
        "submitted_url": "https://www.tiktok.com/@u/video/1",
        "video_id": "1",
        "creator_username": "u",
        "creator_nickname": "User",
        "caption": "hi",
        "metrics": {"views": 100, "likes": 1, "comments": 0, "shares": 0, "saves": 0},
        "duration": 10.0,
        "contact_sheet_path": "data/contact_sheets/1.jpg",
        "cover_url": "https://cdn/cover.jpg",
        "apify_run_id": "RUN1",
        "apify_dataset_id": "DS1",
        "apify_usage_usd": 0.01,
    })
    return db_path, rid


def test_update_writes_all_analysis_fields(tmp_path: Path):
    db_path, rid = _seed_report(tmp_path)

    fields = {
        "visual_hook_summary": "man watches lamp smash",
        "onscreen_text": "Please be honest",
        "emotional_mechanic": "public disrespect → viewer defense",
        "viewer_role": "defender",
        "emotions_triggered": ["anger", "sympathy"],
        "product_attachability_score": 0.8,
        "transferability_score": 0.7,
        "freshness_score": 0.6,
        "cooked_score": 0.3,
        "overall_opportunity_score": 0.75,
        "hook_mutations": [{"type": "safe", "idea": "x"}],
        "raw_analysis": {"why_it_works": "indignation engine", "environment": "market"},
    }
    ok = db.update_report_analysis(db_path, rid, fields)
    assert ok is True

    row = db.get_report(db_path, rid)
    assert row["visual_hook_summary"] == "man watches lamp smash"
    assert row["onscreen_text"] == "Please be honest"
    assert row["emotional_mechanic"].startswith("public disrespect")
    assert row["viewer_role"] == "defender"
    assert row["emotions_triggered"] == ["anger", "sympathy"]
    assert row["product_attachability_score"] == 0.8
    assert row["transferability_score"] == 0.7
    assert row["freshness_score"] == 0.6
    assert row["cooked_score"] == 0.3
    assert row["overall_opportunity_score"] == 0.75
    assert row["hook_mutations"] == [{"type": "safe", "idea": "x"}]
    assert row["raw_analysis"]["why_it_works"] == "indignation engine"


def test_update_preserves_other_columns(tmp_path: Path):
    db_path, rid = _seed_report(tmp_path)
    before = db.get_report(db_path, rid)

    db.update_report_analysis(db_path, rid, {
        "visual_hook_summary": "new summary",
    })

    after = db.get_report(db_path, rid)
    assert after["visual_hook_summary"] == "new summary"
    # Untouched columns must still match.
    assert after["creator_username"] == before["creator_username"]
    assert after["caption"] == before["caption"]
    assert after["metrics"] == before["metrics"]
    assert after["apify_run_id"] == before["apify_run_id"]
    assert after["contact_sheet_path"] == before["contact_sheet_path"]


def test_update_unknown_id_returns_false(tmp_path: Path):
    db_path, _ = _seed_report(tmp_path)
    ok = db.update_report_analysis(db_path, "no-such-id", {
        "visual_hook_summary": "x",
    })
    assert ok is False


def test_update_with_empty_dict_is_noop(tmp_path: Path):
    db_path, rid = _seed_report(tmp_path)
    before = db.get_report(db_path, rid)
    ok = db.update_report_analysis(db_path, rid, {})
    assert ok is False  # nothing to write
    after = db.get_report(db_path, rid)
    assert after == before


def test_update_ignores_unknown_keys(tmp_path: Path):
    db_path, rid = _seed_report(tmp_path)
    ok = db.update_report_analysis(db_path, rid, {
        "visual_hook_summary": "summary",
        "totally_made_up_field": "ignored",
    })
    assert ok is True
    row = db.get_report(db_path, rid)
    assert row["visual_hook_summary"] == "summary"
    # And of course the unknown field doesn't appear anywhere.
    assert "totally_made_up_field" not in row
