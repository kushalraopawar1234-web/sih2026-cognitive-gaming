"""Configurable LLM boundary for the cultural narrative module."""

from __future__ import annotations

import json
import os
from typing import Any, Callable, Dict, Optional
from urllib import error, request


class LLMError(RuntimeError):
    """Base error for provider and response failures."""


class MissingAPIKeyError(LLMError):
    """Raised when the configured provider needs an API key but none exists."""


class APIRequestError(LLMError):
    """Raised when the provider request fails."""


class InvalidLLMResponseError(LLMError):
    """Raised when the provider response is not the expected JSON object."""


class LLMClient:
    """Small OpenAI-compatible client that can be replaced by a backend adapter."""

    def __init__(
        self,
        api_key: Optional[str] = None,
        api_url: Optional[str] = None,
        model: Optional[str] = None,
        provider: Optional[str] = None,
        transport: Optional[Callable[..., Any]] = None,
    ) -> None:
        self.api_key = api_key if api_key is not None else os.getenv("LLM_API_KEY")
        self.api_url = api_url or os.getenv("LLM_API_URL", "https://api.openai.com/v1/chat/completions")
        self.model = model or os.getenv("LLM_MODEL", "gpt-4o-mini")
        self.provider = provider or os.getenv("LLM_PROVIDER", "openai-compatible")
        self.transport = transport or request.urlopen

    def generate_json(self, prompt: str) -> Dict[str, Any]:
        if not self.api_key:
            raise MissingAPIKeyError("LLM_API_KEY is not configured.")
        if self.provider != "openai-compatible":
            raise LLMError(f"Unsupported provider: {self.provider}")

        body = json.dumps(
            {
                "model": self.model,
                "messages": [{"role": "user", "content": prompt}],
                "temperature": 0.4,
                "response_format": {"type": "json_object"},
            }
        ).encode("utf-8")
        req = request.Request(
            self.api_url,
            data=body,
            headers={
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
            },
            method="POST",
        )
        try:
            with self.transport(req, timeout=30) as response:
                raw_response = response.read().decode("utf-8")
        except (error.URLError, error.HTTPError, TimeoutError, OSError) as exc:
            raise APIRequestError(f"LLM API request failed: {exc}") from exc

        try:
            provider_response = json.loads(raw_response)
            content = provider_response["choices"][0]["message"]["content"]
            result = json.loads(content) if isinstance(content, str) else content
        except (KeyError, IndexError, TypeError, json.JSONDecodeError) as exc:
            raise InvalidLLMResponseError("LLM response did not contain valid JSON content.") from exc

        if not isinstance(result, dict):
            raise InvalidLLMResponseError("LLM JSON response must be an object.")
        return result
