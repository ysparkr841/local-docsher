from __future__ import annotations

import sys
import types
from pathlib import Path

import pytest

from docsher.cli import main
from docsher.config import default_config, get_ocr_settings
from docsher.ocr import OCRBackendError, PaddleOCRBackend, create_ocr_backend


class DummyPaddleOCR:
    calls: list[dict[str, object]] = []

    def __init__(self, **kwargs: object) -> None:
        self.kwargs = kwargs
        DummyPaddleOCR.calls.append(kwargs)

    def ocr(self, path: str, cls: bool = True) -> list[object]:
        assert cls is True
        assert path.endswith("korean_image.png")
        return [
            [
                [[0, 0], [10, 0], [10, 10], [0, 10]],
                ("안녕하세요 로컬 문서", 0.98),
            ],
            [
                [[0, 12], [10, 12], [10, 20], [0, 20]],
                ("검색 테스트", 0.91),
            ],
        ]


def install_dummy_paddle(monkeypatch: pytest.MonkeyPatch) -> None:
    module = types.ModuleType("paddleocr")
    module.PaddleOCR = DummyPaddleOCR
    monkeypatch.setitem(sys.modules, "paddleocr", module)
    DummyPaddleOCR.calls.clear()


def test_get_ocr_settings_merges_paddle_defaults() -> None:
    config = default_config()
    config["ocr"] = {
        "backend": "paddle",
        "paddle": {
            "lang": "korean",
            "det_model_dir": "/models/det",
        },
    }

    settings = get_ocr_settings(config)

    assert settings["backend"] == "paddle"
    assert settings["paddle"]["lang"] == "korean"
    assert settings["paddle"]["det_model_dir"] == "/models/det"
    assert settings["paddle"]["rec_model_dir"] is None
    assert settings["paddle"]["use_angle_cls"] is True


def test_paddle_backend_reports_clear_error_when_dependency_missing(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setitem(sys.modules, "paddleocr", None)
    backend = PaddleOCRBackend(lang="korean")

    assert backend.is_available() is False
    with pytest.raises(OCRBackendError, match="PaddleOCR backend unavailable"):
        backend.recognize(tmp_path / "korean_image.png")


def test_paddle_backend_recognizes_korean_text_with_local_model_dirs(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    install_dummy_paddle(monkeypatch)
    image_path = tmp_path / "korean_image.png"
    image_path.write_bytes(b"\x89PNG\r\n\x1a\n")

    backend = PaddleOCRBackend(
        lang="korean",
        det_model_dir=tmp_path / "det",
        rec_model_dir=tmp_path / "rec",
        cls_model_dir=tmp_path / "cls",
        use_angle_cls=True,
    )
    result = backend.recognize(image_path)

    assert result.backend == "paddle"
    assert result.text == "안녕하세요 로컬 문서\n검색 테스트"
    assert DummyPaddleOCR.calls == [
        {
            "lang": "korean",
            "use_angle_cls": True,
            "det_model_dir": str(tmp_path / "det"),
            "rec_model_dir": str(tmp_path / "rec"),
            "cls_model_dir": str(tmp_path / "cls"),
        }
    ]


def test_create_ocr_backend_uses_selected_paddle_backend(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    install_dummy_paddle(monkeypatch)
    config = default_config()
    config["ocr"]["backend"] = "paddle"
    config["ocr"]["paddle"] = {"lang": "korean", "det_model_dir": str(tmp_path / "det")}

    backend = create_ocr_backend(config)
    result = backend.recognize(tmp_path / "korean_image.png")

    assert backend.name == "paddle"
    assert "안녕하세요" in result.text


def test_ocr_test_cli_can_select_paddle_backend(monkeypatch: pytest.MonkeyPatch, tmp_path: Path, capsys) -> None:
    install_dummy_paddle(monkeypatch)
    image_path = tmp_path / "korean_image.png"
    image_path.write_bytes(b"\x89PNG\r\n\x1a\n")

    exit_code = main(["ocr-test", str(image_path), "--backend", "paddle", "--paddle-lang", "korean"])

    captured = capsys.readouterr()
    assert exit_code == 0
    assert '"backend": "paddle"' in captured.out
    assert "안녕하세요 로컬 문서" in captured.out


def test_ocr_test_cli_returns_clear_paddle_install_error(monkeypatch: pytest.MonkeyPatch, tmp_path: Path, capsys) -> None:
    monkeypatch.setitem(sys.modules, "paddleocr", None)
    image_path = tmp_path / "korean_image.png"
    image_path.write_bytes(b"\x89PNG\r\n\x1a\n")

    exit_code = main(["ocr-test", str(image_path), "--backend", "paddle"])

    captured = capsys.readouterr()
    assert exit_code == 2
    assert "OCR test error: PaddleOCR backend unavailable" in captured.out
