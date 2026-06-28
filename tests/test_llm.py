from __future__ import annotations

import json
import urllib.error
from pathlib import Path
from typing import Any

import pytest

from docsher.config import ENV_CONFIG_PATH, get_llm_settings, load_config
from docsher.llm import (
    DummyLLMClient,
    LLMClientError,
    LLMMessage,
    OllamaLLMClient,
    OpenAICompatibleLLMClient,
    create_llm_client,
    extract_openai_message_text,
)
from docsher.search import search_documents


class FakeHTTPResponse:
    def __init__(self, payload: dict[str, Any]) -> None:
        self.payload = payload

    def __enter__(self) -> "FakeHTTPResponse":
        return self

    def __exit__(self, exc_type: object, exc: object, tb: object) -> None:
        return None

    def read(self) -> bytes:
        return json.dumps(self.payload).encode("utf-8")


def test_llm_settings_defaults_and_legacy_merge(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    config_path = tmp_path / "config.json"
    monkeypatch.setenv(ENV_CONFIG_PATH, str(config_path))
    config_path.write_text(json.dumps({"llm": {"provider": "ollama"}}), encoding="utf-8")

    config = load_config()
    settings = get_llm_settings(config)

    assert settings["provider"] == "ollama"
    assert settings["ollama_endpoint"] == "http://localhost:11434"
    assert settings["ollama_model"] == "qwen2.5:7b-instruct"
    assert settings["openai_compatible_endpoint"].endswith("/v1/chat/completions")
    assert settings["timeout_seconds"] == 60


def test_dummy_llm_client_is_deterministic() -> None:
    client = DummyLLMClient("offline answer")
    messages = [LLMMessage(role="user", content="질문")]

    response = client.chat(messages)

    assert response.text == "offline answer"
    assert response.provider == "dummy"
    assert client.seen_messages == messages


def test_ollama_client_uses_local_openai_compatible_endpoint() -> None:
    client = create_llm_client(
        {
            "llm": {
                "provider": "ollama",
                "ollama_endpoint": "http://127.0.0.1:11434/",
                "ollama_model": "local-small",
            }
        }
    )

    assert isinstance(client, OllamaLLMClient)
    assert client.endpoint == "http://127.0.0.1:11434/v1/chat/completions"
    assert client.model == "local-small"
    assert client.is_available()


def test_openai_compatible_client_posts_chat_payload(monkeypatch: pytest.MonkeyPatch) -> None:
    captured: dict[str, Any] = {}

    def fake_urlopen(request: Any, timeout: int) -> FakeHTTPResponse:
        captured["url"] = request.full_url
        captured["timeout"] = timeout
        captured["headers"] = dict(request.header_items())
        captured["payload"] = json.loads(request.data.decode("utf-8"))
        return FakeHTTPResponse({"choices": [{"message": {"content": " 로컬 답변 "}}]})

    monkeypatch.setattr("urllib.request.urlopen", fake_urlopen)
    client = OpenAICompatibleLLMClient(
        endpoint="http://127.0.0.1:8000/v1/chat/completions",
        model="small-local",
        api_key="test-key",
        timeout_seconds=7,
    )

    response = client.chat([LLMMessage(role="user", content="요약해줘")], temperature=0.2)

    assert response.text == "로컬 답변"
    assert response.model == "small-local"
    assert captured["url"] == "http://127.0.0.1:8000/v1/chat/completions"
    assert captured["timeout"] == 7
    assert captured["headers"]["Authorization"] == "Bearer test-key"
    assert captured["payload"] == {
        "model": "small-local",
        "messages": [{"role": "user", "content": "요약해줘"}],
        "temperature": 0.2,
        "stream": False,
    }


def test_openai_compatible_client_wraps_connection_failure(monkeypatch: pytest.MonkeyPatch) -> None:
    def fake_urlopen(request: Any, timeout: int) -> FakeHTTPResponse:
        del request, timeout
        raise urllib.error.URLError("offline")

    monkeypatch.setattr("urllib.request.urlopen", fake_urlopen)
    client = OpenAICompatibleLLMClient(
        endpoint="http://127.0.0.1:1/v1/chat/completions",
        model="small-local",
        timeout_seconds=1,
    )

    with pytest.raises(LLMClientError, match="LLM backend unavailable"):
        client.chat([LLMMessage(role="user", content="hello")])


def test_extract_openai_message_text_supports_text_and_content_parts() -> None:
    assert extract_openai_message_text({"choices": [{"text": "plain"}]}) == "plain"
    assert (
        extract_openai_message_text(
            {"choices": [{"message": {"content": [{"text": "a"}, {"text": "b"}]}}]}
        )
        == "a\nb"
    )


def test_search_still_works_when_llm_backend_fails(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    # LDS-019 only introduces the client layer. Existing search should not need
    # a reachable LLM server and should remain usable if generation fails.
    def fake_urlopen(request: Any, timeout: int) -> FakeHTTPResponse:
        del request, timeout
        raise urllib.error.URLError("offline")

    monkeypatch.setattr("urllib.request.urlopen", fake_urlopen)
    client = OpenAICompatibleLLMClient(endpoint="http://127.0.0.1:1/v1/chat/completions", model="x")
    with pytest.raises(LLMClientError):
        client.chat([LLMMessage(role="user", content="fail")])

    import sqlite3

    from docsher.db import init_database

    db_path = tmp_path / "docsher.sqlite3"
    init_database(db_path)
    with sqlite3.connect(db_path) as connection:
        cursor = connection.execute(
            """
            INSERT INTO documents(path, filename, extension, size, modified_at, status)
            VALUES (?, ?, '.txt', ?, datetime('now'), 'indexed')
            """,
            (str(tmp_path / "manual.txt"), "manual.txt", len("offline searchable manual")),
        )
        document_id = int(cursor.lastrowid)
        connection.execute(
            """
            INSERT INTO chunks(document_id, chunk_index, text, token_count)
            VALUES (?, 0, ?, ?)
            """,
            (document_id, "offline searchable manual", 3),
        )

    results = search_documents("offline", database_path=db_path)

    assert len(results) == 1
    assert results[0].path.endswith("manual.txt")
