from __future__ import annotations

from pathlib import Path

from emotion_radar import db


def _make_report(submitted_url: str = "https://www.tiktok.com/@u/video/1") -> dict:
    return {
        "platform": "TikTok",
        "source_url": "https://www.tiktok.com/@u/video/1",
        "submitted_url": submitted_url,
        "video_id": "1234567890",
        "creator_username": "u",
        "creator_nickname": "User",
        "caption": "hello\nworld",
        "metrics": {"views": 1000, "likes": 50, "comments": 3, "shares": 2, "saves": 1},
        "duration": 12.5,
        "contact_sheet_path": "data/contact_sheets/1234567890.jpg",
        "cover_url": "https://cdn/cover.jpg",
        "video_download_url_saved": False,
        "apify_run_id": "RUN1",
        "apify_dataset_id": "DS1",
        "apify_usage_usd": 0.0083,
        "apify_charged_events": {"count": 1},
        "emotions_triggered": ["anger", "defense"],
        "hook_mutations": ["niche A", "niche B"],
        "raw_analysis": {"status": "stub"},
    }


def test_insert_and_get_report(tmp_path: Path):
    db_path = tmp_path / "test.db"
    rid = db.insert_report(db_path, _make_report())
    assert rid

    fetched = db.get_report(db_path, rid)
    assert fetched is not None
    assert fetched["id"] == rid
    assert fetched["platform"] == "TikTok"
    assert fetched["metrics"]["views"] == 1000
    assert fetched["emotions_triggered"] == ["anger", "defense"]
    assert fetched["hook_mutations"] == ["niche A", "niche B"]
    assert fetched["raw_analysis"] == {"status": "stub"}
    assert fetched["video_download_url_saved"] is False
    assert fetched["apify_usage_usd"] == 0.0083


def test_list_reports_orders_desc(tmp_path: Path):
    db_path = tmp_path / "test.db"
    rid1 = db.insert_report(db_path, _make_report("https://www.tiktok.com/@u/video/1"))
    rid2 = db.insert_report(db_path, _make_report("https://www.tiktok.com/@u/video/2"))
    rid3 = db.insert_report(db_path, _make_report("https://www.tiktok.com/@u/video/3"))

    rows = db.list_reports(db_path)
    ids = [r["id"] for r in rows]
    assert set(ids) == {rid1, rid2, rid3}
    # Latest insert should be first (most recent created_at).
    assert ids[0] == rid3


def test_get_report_missing_returns_none(tmp_path: Path):
    db_path = tmp_path / "test.db"
    db.init_db(db_path)
    assert db.get_report(db_path, "nonexistent") is None


def test_insert_handles_error_only_item(tmp_path: Path):
    db_path = tmp_path / "test.db"
    minimal = {
        "platform": "TikTok",
        "submitted_url": "https://www.tiktok.com/@u/video/x",
        "error": "No downloadable video URL.",
    }
    rid = db.insert_report(db_path, minimal)
    fetched = db.get_report(db_path, rid)
    assert fetched["error"] == "No downloadable video URL."
    assert fetched["metrics"] is None
