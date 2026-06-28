"""Local LLM client abstractions for RAG features.

The client layer is stdlib-only and network-optional. Docsher can be installed
and searched without any LLM server; callers opt into a configured local Ollama
or OpenAI-compatible endpoint when they need generation.
"""

from __future__ import annotations

import json
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import asdict, dataclass
from typing import Any, Protocol

from docsher.config import get_llm_settings


class LLMClientError(RuntimeError):
    """Raised when an LLM backend cannot complete a generation request."""


@dataclass(frozen=True)
class LLMMessage:
    """One chat message sent to an LLM backend."""

    role: str
    content: str

    def to_dict(self) -> dict[str, str]:
        return asdict(self)


@dataclass(frozen=True)
class LLMResponse:
    """Normalized text returned by an LLM backend."""

    text: str
    model: str
    provider: str

    def to_dict(self) -> dict[str, str]:
        return asdict(self)


class LLMClient(Protocol):
    """Pluggable local LLM client contract used by future RAG features."""

    provider: str
    model: str

    def is_available(self) -> bool:
        """Return whether the configured backend can be attempted."""

    def chat(self, messages: list[LLMMessage], *, temperature: float = 0.0) -> LLMResponse:
        """Generate a response from chat messages."""


class DummyLLMClient:
    """Deterministic client for tests and offline wiring without a model server."""

    provider = "dummy"

    def __init__(self, text: str = "dummy LLM response", *, model: str = "dummy") -> None:
        self.text = text
        self.model = model
        self.seen_messages: list[LLMMessage] = []

    def is_available(self) -> bool:
        return True

    def chat(self, messages: list[LLMMessage], *, temperature: float = 0.0) -> LLMResponse:
        del temperature
        self.seen_messages = list(messages)
        return LLMResponse(text=self.text, model=self.model, provider=self.provider)


def redact_endpoint(endpoint: str) -> str:
    """Return an endpoint string safe for user-facing errors.

    Users may accidentally place credentials in URL userinfo or query strings.
    Error messages should keep enough host/path context for troubleshooting while
    avoiding credential leakage through API/CLI responses.
    """

    try:
        parsed = urllib.parse.urlsplit(endpoint)
    except ValueError:
        return "<redacted-endpoint>"
    if not parsed.scheme or not parsed.netloc:
        return "<redacted-endpoint>"
    hostname = parsed.hostname or ""
    netloc = hostname
    if parsed.port is not None:
        netloc = f"{netloc}:{parsed.port}"
    path = parsed.path or ""
    query = "<redacted>" if parsed.query else ""
    return urllib.parse.urlunsplit((parsed.scheme, netloc, path, query, ""))


class OpenAICompatibleLLMClient:
    """Client for local OpenAI-compatible ``/v1/chat/completions`` servers."""

    provider = "openai-compatible"

    def __init__(
        self,
        *,
        endpoint: str,
        model: str,
        api_key: str | None = None,
        timeout_seconds: int = 60,
        provider: str | None = None,
    ) -> None:
        self.endpoint = endpoint
        self.model = model
        self.api_key = api_key
        self.timeout_seconds = timeout_seconds
        if provider:
            self.provider = provider

    def is_available(self) -> bool:
        return self.endpoint.startswith(("http://", "https://")) and bool(self.model)

    def chat(self, messages: list[LLMMessage], *, temperature: float = 0.0) -> LLMResponse:
        if not messages:
            raise LLMClientError("LLM chat requires at least one message")
        payload = {
            "model": self.model,
            "messages": [message.to_dict() for message in messages],
            "temperature": temperature,
            "stream": False,
        }
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        request = urllib.request.Request(
            self.endpoint,
            data=json.dumps(payload).encode("utf-8"),
            headers=headers,
            method="POST",
        )
        try:
            with urllib.request.urlopen(request, timeout=self.timeout_seconds) as response:  # noqa: S310 - user-configured local endpoint.
                response_payload = json.loads(response.read().decode("utf-8"))
        except (OSError, urllib.error.URLError, json.JSONDecodeError) as exc:
            safe_endpoint = redact_endpoint(self.endpoint)
            raise LLMClientError(f"LLM backend unavailable at {safe_endpoint}: {exc}") from exc
        text = extract_openai_message_text(response_payload)
        if not text:
            raise LLMClientError("LLM backend returned no text")
        return LLMResponse(text=text, model=self.model, provider=self.provider)


class OllamaLLMClient(OpenAICompatibleLLMClient):
    """Ollama client using Ollama's local OpenAI-compatible chat endpoint."""

    provider = "ollama"

    def __init__(self, *, endpoint: str, model: str, timeout_seconds: int = 60) -> None:
        base_endpoint = endpoint.rstrip("/")
        super().__init__(
            endpoint=f"{base_endpoint}/v1/chat/completions",
            model=model,
            timeout_seconds=timeout_seconds,
            provider=self.provider,
        )
        self.base_endpoint = base_endpoint


def extract_openai_message_text(payload: object) -> str:
    """Extract assistant text from common OpenAI-compatible response shapes."""

    if not isinstance(payload, dict):
        return ""
    choices = payload.get("choices")
    if not isinstance(choices, list) or not choices:
        return ""
    first = choices[0]
    if not isinstance(first, dict):
        return ""
    message = first.get("message")
    if isinstance(message, dict):
        content = message.get("content")
        if isinstance(content, str):
            return content.strip()
        if isinstance(content, list):
            parts: list[str] = []
            for item in content:
                if isinstance(item, dict) and isinstance(item.get("text"), str):
                    parts.append(item["text"])
            return "\n".join(part.strip() for part in parts if part.strip())
    text = first.get("text")
    if isinstance(text, str):
        return text.strip()
    return ""


def create_llm_client(config: dict[str, Any] | None = None, *, provider: str | None = None) -> LLMClient:
    """Create an LLM client from configuration or an explicit provider name."""

    settings = get_llm_settings(config or {})
    selected = (provider or settings["provider"] or "dummy").lower()
    if selected == "dummy":
        return DummyLLMClient()
    if selected == "ollama":
        return OllamaLLMClient(
            endpoint=str(settings["ollama_endpoint"]),
            model=str(settings["ollama_model"]),
            timeout_seconds=int(settings["timeout_seconds"]),
        )
    if selected in {"openai", "openai-compatible", "openai_compatible"}:
        api_key = settings["api_key"]
        return OpenAICompatibleLLMClient(
            endpoint=str(settings["openai_compatible_endpoint"]),
            model=str(settings["openai_compatible_model"]),
            api_key=str(api_key) if api_key else None,
            timeout_seconds=int(settings["timeout_seconds"]),
        )
    raise ValueError(f"Unsupported LLM provider: {selected}")
