"""download_video header handling. No network — requests.get is patched."""

from __future__ import annotations

from pathlib import Path

import pytest

from emotion_radar import video as video_mod


class _FakeResponse:
    """Minimal stand-in for requests.Response in streaming mode."""

    def __init__(self, body: bytes, ok: bool = True, status_code: int = 200):
        self._body = body
        self.ok = ok
        self.status_code = status_code

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        return False

    def iter_content(self, chunk_size: int = 1 << 16):
        yield self._body


def _patch_get(monkeypatch, captured: dict, body: bytes = b"FAKEMP4DATA"):
    def fake_get(url, stream=False, timeout=None, headers=None):
        captured["url"] = url
        captured["headers"] = headers
        captured["timeout"] = timeout
        captured["stream"] = stream
        return _FakeResponse(body)

    monkeypatch.setattr(video_mod.requests, "get", fake_get)


def test_default_user_agent_sent(tmp_path: Path, monkeypatch):
    captured: dict = {}
    _patch_get(monkeypatch, captured)
    out = video_mod.download_video(
        "https://example.com/video.mp4",
        tmp_path,
        "vid1",
    )
    assert out.exists()
    assert out.read_bytes() == b"FAKEMP4DATA"
    assert captured["headers"]["User-Agent"] == video_mod.DEFAULT_USER_AGENT
    assert "Authorization" not in captured["headers"]


def test_custom_headers_merge_with_user_agent(tmp_path: Path, monkeypatch):
    captured: dict = {}
    _patch_get(monkeypatch, captured)
    video_mod.download_video(
        "https://api.apify.com/v2/key-value-stores/abc/records/xyz.mp4",
        tmp_path,
        "vid2",
        headers={"Authorization": "Bearer tkn_xyz"},
    )
    headers = captured["headers"]
    assert headers["Authorization"] == "Bearer tkn_xyz"
    assert headers["User-Agent"] == video_mod.DEFAULT_USER_AGENT


def test_caller_can_override_user_agent(tmp_path: Path, monkeypatch):
    captured: dict = {}
    _patch_get(monkeypatch, captured)
    video_mod.download_video(
        "https://example.com/x.mp4",
        tmp_path,
        "vid3",
        headers={"User-Agent": "custom-agent/9"},
    )
    assert captured["headers"]["User-Agent"] == "custom-agent/9"


def test_http_error_raises_video_error(tmp_path: Path, monkeypatch):
    def fake_get(url, stream=False, timeout=None, headers=None):
        return _FakeResponse(b"", ok=False, status_code=403)

    monkeypatch.setattr(video_mod.requests, "get", fake_get)
    with pytest.raises(video_mod.VideoError):
        video_mod.download_video(
            "https://api.apify.com/v2/key-value-stores/abc/records/xyz.mp4",
            tmp_path,
            "vid4",
        )
