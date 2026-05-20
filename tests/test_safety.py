"""Safety: URL-cap enforcement and Apify-item normalization (no network)."""

from __future__ import annotations

import pytest

from emotion_radar.apify_client import normalize_item, normalize_items
from emotion_radar.config import enforce_url_cap


def test_enforce_url_cap_default():
    enforce_url_cap(["a"], confirm_large=False)
    enforce_url_cap(["a", "b", "c"], confirm_large=False)
    with pytest.raises(ValueError):
        enforce_url_cap(["a", "b", "c", "d"], confirm_large=False)


def test_normalize_item_full_mapping():
    raw = {
        "id": "7623559389307211030",
        "text": "please be honest, how are they?",
        "webVideoUrl": "https://www.tiktok.com/@olivermakesartt/video/7623559389307211030",
        "mediaUrls": ["https://api.apify.com/.../video.mp4"],
        "videoMeta": {
            "duration": 14.2,
            "coverUrl": "https://cdn/cover.jpg",
            "downloadAddr": "https://fallback/dl.mp4",
        },
        "authorMeta": {"name": "olivermakesartt", "nickName": "Oliver"},
        "playCount": 123456,
        "diggCount": 789,
        "commentCount": 42,
        "shareCount": 17,
        "collectCount": 9,
    }
    norm = normalize_item(raw, submitted_url="https://www.tiktok.com/@olivermakesartt/video/7623559389307211030")
    assert norm.platform == "TikTok"
    assert norm.video_id == "7623559389307211030"
    assert norm.caption == "please be honest, how are they?"
    assert norm.video_download_url == "https://api.apify.com/.../video.mp4"
    assert norm.cover_url == "https://cdn/cover.jpg"
    assert norm.creator_username == "olivermakesartt"
    assert norm.creator_nickname == "Oliver"
    assert norm.duration == 14.2
    assert norm.metrics == {
        "views": 123456,
        "likes": 789,
        "comments": 42,
        "shares": 17,
        "saves": 9,
    }
    assert norm.error is None


def test_normalize_item_falls_back_to_downloadAddr():
    raw = {
        "id": "1",
        "mediaUrls": [],
        "videoMeta": {"downloadAddr": "https://fallback/dl.mp4"},
        "authorMeta": {"name": "u"},
    }
    norm = normalize_item(raw, submitted_url="https://x/y/1")
    assert norm.video_download_url == "https://fallback/dl.mp4"
    assert norm.error is None


def test_normalize_item_marks_error_when_no_url():
    raw = {
        "id": "1",
        "mediaUrls": [],
        "videoMeta": {},
        "authorMeta": {"name": "u"},
    }
    norm = normalize_item(raw, submitted_url="https://x/y/1")
    assert norm.video_download_url is None
    assert norm.error is not None


def test_normalize_items_matches_by_webVideoUrl():
    url_a = "https://www.tiktok.com/@a/video/1"
    url_b = "https://www.tiktok.com/@b/video/2"
    items = [
        {"id": "2", "webVideoUrl": url_b, "mediaUrls": ["x"], "videoMeta": {}, "authorMeta": {}},
        {"id": "1", "webVideoUrl": url_a, "mediaUrls": ["y"], "videoMeta": {}, "authorMeta": {}},
    ]
    out = normalize_items(items, [url_a, url_b])
    assert out[0].video_id == "1"
    assert out[1].video_id == "2"


def test_normalize_items_returns_error_for_missing_url():
    url_a = "https://www.tiktok.com/@a/video/1"
    url_b = "https://www.tiktok.com/@b/video/2"
    out = normalize_items([], [url_a, url_b])
    assert len(out) == 2
    assert all(item.error for item in out)
