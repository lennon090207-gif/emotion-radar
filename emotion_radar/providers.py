"""Vision provider abstraction.

Today: an OpenAI-compatible chat/completions backend. Set
OPENAI_BASE_URL to route through OpenRouter or another compatible host;
otherwise it defaults to OpenAI direct (or OpenRouter if only
OPENROUTER_API_KEY is set).

Tomorrow: drop in Anthropic vision / Gemini / a local VLM by writing a
new VisionProvider subclass and updating build_default_provider().

API keys are read from the merged env (process env > /root/.hermes/.env >
./.env via config.load_env). Keys are never printed.
"""

from __future__ import annotations

import base64
from abc import ABC, abstractmethod
from pathlib import Path

import requests

DEFAULT_TIMEOUT_SEC = 90
DEFAULT_MAX_TOKENS = 2048
DEFAULT_TEMPERATURE = 0.4
DEFAULT_OPENAI_BASE_URL = "https://api.openai.com/v1"
DEFAULT_OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"
DEFAULT_MODEL = "gpt-4o"


class VisionProviderError(RuntimeError):
    pass


class VisionProvider(ABC):
    name: str = "abstract"
    model: str = ""

    @abstractmethod
    def analyze_image(
        self,
        image_path: Path,
        system_prompt: str,
        user_prompt: str,
    ) -> str:
        """Return the raw text response from the model (expected to be JSON)."""


class OpenAICompatibleProvider(VisionProvider):
    name = "openai_compatible"

    def __init__(
        self,
        api_key: str,
        model: str,
        base_url: str = DEFAULT_OPENAI_BASE_URL,
        timeout: int = DEFAULT_TIMEOUT_SEC,
        max_tokens: int = DEFAULT_MAX_TOKENS,
        temperature: float = DEFAULT_TEMPERATURE,
        session: requests.Session | None = None,
    ):
        if not api_key:
            raise VisionProviderError("API key is empty.")
        self._api_key = api_key
        self.model = model
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.session = session or requests.Session()

    def analyze_image(
        self,
        image_path: Path,
        system_prompt: str,
        user_prompt: str,
    ) -> str:
        if not image_path.is_file():
            raise VisionProviderError(f"Image not found: {image_path}")
        image_b64 = base64.b64encode(image_path.read_bytes()).decode("ascii")
        data_url = f"data:image/jpeg;base64,{image_b64}"
        payload = {
            "model": self.model,
            "max_tokens": self.max_tokens,
            "temperature": self.temperature,
            "messages": [
                {"role": "system", "content": system_prompt},
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": user_prompt},
                        {"type": "image_url", "image_url": {"url": data_url}},
                    ],
                },
            ],
        }
        headers = {
            "Authorization": f"Bearer {self._api_key}",
            "Content-Type": "application/json",
        }
        url = f"{self.base_url}/chat/completions"
        r = self.session.post(url, headers=headers, json=payload, timeout=self.timeout)
        if not r.ok:
            # Body may include the API key only if the server echoes it; we
            # truncate aggressively and rely on the server not echoing.
            raise VisionProviderError(
                f"Vision API HTTP {r.status_code}: {r.text[:400]}"
            )
        try:
            data = r.json()
        except ValueError as e:
            raise VisionProviderError(f"Vision API returned non-JSON: {r.text[:400]}") from e
        try:
            return data["choices"][0]["message"]["content"]
        except (KeyError, IndexError, TypeError) as e:
            raise VisionProviderError(f"Unexpected vision API response shape: {data}") from e


def build_default_provider(env: dict[str, str]) -> VisionProvider:
    """Build a provider from the merged env. Raises VisionProviderError
    with a clear message if no key is configured."""
    openai_key = (env.get("OPENAI_API_KEY") or "").strip()
    openrouter_key = (env.get("OPENROUTER_API_KEY") or "").strip()
    api_key = openai_key or openrouter_key
    if not api_key:
        raise VisionProviderError("Vision API key not configured.")

    base_url = (env.get("OPENAI_BASE_URL") or "").strip()
    if not base_url:
        if openrouter_key and not openai_key:
            base_url = DEFAULT_OPENROUTER_BASE_URL
        else:
            base_url = DEFAULT_OPENAI_BASE_URL

    model = (env.get("VISION_MODEL") or "").strip() or DEFAULT_MODEL
    return OpenAICompatibleProvider(api_key=api_key, model=model, base_url=base_url)
