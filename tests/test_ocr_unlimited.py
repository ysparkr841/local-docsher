from __future__ import annotations

import json
from pathlib import Path

import pytest

from docsher.cli import main
from docsher.config import default_config, get_ocr_settings
from docsher.ocr import OCRBackendError, UnlimitedOCRBackend, create_ocr_backend


class DummyResponse:
    def __init__(self, payload: dict) -> None:
        self.payload = payload

    def __enter__(self) -> "DummyResponse":
        return self

    def __exit__(self, *_args: object) -> None:
        return None

    def read(self) -> bytes:
        return json.dumps(self.payload).encode("utf-8")


def test_get_ocr_settings_merges_unlimited_defaults() -> None:
    config = default_config()
    config["ocr"] = {
        "backend": "unlimited",
        "unlimited": {
            "endpoint": "http://127.0.0.1:10000/v1/chat/completions",
            "model": "Unlimited-OCR",
        },
    }

    settings = get_ocr_settings(config)

    assert settings["backend"] == "unlimited"
    assert settings["unlimited"]["endpoint"].endswith("/v1/chat/completions")
    assert settings["unlimited"]["model"] == "Unlimited-OCR"
    assert settings["unlimited"]["prompt"] == "document parsing."


def test_unlimited_backend_posts_openai_compatible_image_payload(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    requests: list[object] = []

    def fake_urlopen(request: object, timeout: int) -> DummyResponse:
        requests.append((request, timeout))
        return DummyResponse({"choices": [{"message": {"content": "무제한 OCR 결과"}}]})

    monkeypatch.setattr("docsher.ocr.urllib.request.urlopen", fake_urlopen)
    image_path = tmp_path / "korean_image.png"
    image_path.write_bytes(b"png bytes")

    backend = UnlimitedOCRBackend(endpoint="http://127.0.0.1:10000/v1/chat/completions", timeout_seconds=7)
    result = backend.recognize(image_path)

    assert result.backend == "unlimited"
    assert result.text == "무제한 OCR 결과"
    request, timeout = requests[0]
    assert timeout == 7
    body = json.loads(request.data.decode("utf-8"))
    assert body["model"] == "Unlimited-OCR"
    content = body["messages"][0]["content"]
    assert content[0] == {"type": "text", "text": "document parsing."}
    assert content[1]["image_url"]["url"].startswith("data:image/png;base64,")


def test_unlimited_backend_reports_clear_connection_error(tmp_path: Path) -> None:
    image_path = tmp_path / "korean_image.png"
    image_path.write_bytes(b"png bytes")
    backend = UnlimitedOCRBackend(endpoint="http://127.0.0.1:1/v1/chat/completions", timeout_seconds=1)

    with pytest.raises(OCRBackendError, match="Unlimited-OCR backend unavailable"):
        backend.recognize(image_path)


def test_create_ocr_backend_exposes_unlimited_experimental_backend() -> None:
    config = default_config()
    config["ocr"]["backend"] = "unlimited"
    config["ocr"]["unlimited"] = {"endpoint": "http://127.0.0.1:10000/v1/chat/completions"}

    backend = create_ocr_backend(config)

    assert isinstance(backend, UnlimitedOCRBackend)
    assert backend.name == "unlimited"


def test_ocr_test_cli_can_select_unlimited_backend(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    capsys,
) -> None:
    def fake_urlopen(request: object, timeout: int) -> DummyResponse:
        return DummyResponse({"choices": [{"message": {"content": "CLI OCR 결과"}}]})

    monkeypatch.setattr("docsher.ocr.urllib.request.urlopen", fake_urlopen)
    image_path = tmp_path / "korean_image.png"
    image_path.write_bytes(b"png bytes")

    exit_code = main([
        "ocr-test",
        str(image_path),
        "--backend",
        "unlimited",
        "--unlimited-endpoint",
        "http://127.0.0.1:10000/v1/chat/completions",
    ])

    captured = capsys.readouterr()
    assert exit_code == 0
    assert '"backend": "unlimited"' in captured.out
    assert "CLI OCR 결과" in captured.out
