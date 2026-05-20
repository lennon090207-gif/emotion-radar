"""Provider factory: env-based wiring. No real API calls."""

from __future__ import annotations

import pytest

from emotion_radar import providers


def test_no_key_raises_clear_error():
    with pytest.raises(providers.VisionProviderError) as ei:
        providers.build_default_provider({})
    assert "Vision API key not configured." in str(ei.value)


def test_openai_key_uses_openai_base_url():
    p = providers.build_default_provider({"OPENAI_API_KEY": "sk-test"})
    assert isinstance(p, providers.OpenAICompatibleProvider)
    assert p.base_url == providers.DEFAULT_OPENAI_BASE_URL.rstrip("/")
    assert p.model == providers.DEFAULT_MODEL


def test_openrouter_key_uses_openrouter_base_url():
    p = providers.build_default_provider({"OPENROUTER_API_KEY": "sk-or-test"})
    assert p.base_url == providers.DEFAULT_OPENROUTER_BASE_URL.rstrip("/")


def test_explicit_base_url_wins():
    p = providers.build_default_provider({
        "OPENAI_API_KEY": "sk-test",
        "OPENAI_BASE_URL": "https://custom.example.com/v1/",
    })
    assert p.base_url == "https://custom.example.com/v1"


def test_explicit_model_wins():
    p = providers.build_default_provider({
        "OPENAI_API_KEY": "sk-test",
        "VISION_MODEL": "my-custom-vision-model",
    })
    assert p.model == "my-custom-vision-model"


def test_openai_key_preferred_over_openrouter_when_both_set():
    p = providers.build_default_provider({
        "OPENAI_API_KEY": "sk-test",
        "OPENROUTER_API_KEY": "sk-or-test",
    })
    # When OPENAI_API_KEY is set, default base url should be OpenAI's, not OpenRouter's.
    assert p.base_url == providers.DEFAULT_OPENAI_BASE_URL.rstrip("/")


class _CapturingSession:
    def __init__(self, response_text: str = '{"x": 1}', status_code: int = 200):
        self.response_text = response_text
        self.status_code = status_code
        self.calls: list[dict] = []

    def post(self, url, headers=None, json=None, timeout=None):
        self.calls.append({"url": url, "headers": headers, "json": json, "timeout": timeout})

        class _Resp:
            def __init__(self, body, status):
                self.text = body
                self.status_code = status
                self.ok = 200 <= status < 300

            def json(self):
                import json as _j
                return _j.loads(self.text)

        return _Resp(self._mock_body(), self.status_code)

    def _mock_body(self):
        import json as _j
        return _j.dumps({"choices": [{"message": {"content": self.response_text}}]})


def test_openai_provider_puts_key_in_authorization_header(tmp_path):
    img = tmp_path / "img.jpg"
    img.write_bytes(b"\xff\xd8\xff\xe0fake")
    sess = _CapturingSession(response_text='{"ok": true}')
    p = providers.OpenAICompatibleProvider(
        api_key="sk-secret",
        model="gpt-4o",
        base_url="https://api.example.com/v1",
        session=sess,
    )
    p.analyze_image(img, "system", "user")
    assert len(sess.calls) == 1
    call = sess.calls[0]
    # Key is in the Authorization header, NOT the URL.
    assert call["headers"]["Authorization"] == "Bearer sk-secret"
    assert "sk-secret" not in call["url"]
    # Payload shape sanity checks.
    body = call["json"]
    assert body["model"] == "gpt-4o"
    msgs = body["messages"]
    assert msgs[0]["role"] == "system"
    assert msgs[1]["role"] == "user"
    parts = msgs[1]["content"]
    assert any(part.get("type") == "image_url" for part in parts)
    assert any(part.get("type") == "text" for part in parts)


def test_openai_provider_raises_on_missing_image(tmp_path):
    p = providers.OpenAICompatibleProvider(api_key="sk", model="m")
    with pytest.raises(providers.VisionProviderError):
        p.analyze_image(tmp_path / "nope.jpg", "s", "u")


def test_provider_does_not_print_api_key_in_repr():
    p = providers.OpenAICompatibleProvider(
        api_key="sk-secret-do-not-print",
        model="gpt-4o",
    )
    # api_key is stored under a private attribute name; default repr
    # should not include the secret.
    assert "sk-secret-do-not-print" not in repr(p)
    # And no public attribute should expose it.
    public_attrs = {k: v for k, v in vars(p).items() if not k.startswith("_")}
    assert "sk-secret-do-not-print" not in str(public_attrs)
