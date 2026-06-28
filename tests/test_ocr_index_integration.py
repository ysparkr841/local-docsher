from __future__ import annotations

import sqlite3
from pathlib import Path

from docsher.api import create_app
from docsher.config import default_config
from docsher.indexer import index_pending_documents
from docsher.ocr import OCRResult, OCR_STATUS_COMPLETED, OCR_STATUS_FAILED, process_next_ocr_job
from docsher.scanner import scan
from docsher.search import search_documents

try:
    from fastapi.testclient import TestClient
except Exception:  # pragma: no cover - FastAPI optional in minimal installs.
    TestClient = None  # type: ignore[assignment]


class CountingOCRBackend:
    name = "fake"

    def __init__(self, text: str, *, fail: bool = False) -> None:
        self.text = text
        self.fail = fail
        self.calls = 0

    def is_available(self) -> bool:
        return True

    def recognize(self, path: str | Path) -> OCRResult:
        self.calls += 1
        if self.fail:
            raise RuntimeError("intentional OCR failure")
        return OCRResult(text=self.text, backend=self.name)


def make_config(tmp_path: Path, root: Path) -> dict:
    config = default_config()
    config["documents"]["roots"] = [str(root)]
    config["storage"]["database_path"] = str(tmp_path / "docsher.sqlite3")
    return config


def test_ocr_result_cache_reuses_same_hash_and_searches_cached_text(tmp_path: Path) -> None:
    root = tmp_path / "docs"
    root.mkdir()
    image_bytes = b"\x89PNG\r\n\x1a\nidentical scanned image bytes"
    (root / "first.png").write_bytes(image_bytes)
    (root / "second.png").write_bytes(image_bytes)
    config = make_config(tmp_path, root)
    database_path = Path(config["storage"]["database_path"])

    scan(config)
    index_pending_documents(database_path)
    backend = CountingOCRBackend("스캔 문서 안의 단어 캐시본문")

    first_job = process_next_ocr_job(database_path, backend)
    second_job = process_next_ocr_job(database_path, backend)

    assert first_job is not None
    assert second_job is not None
    assert first_job.status == OCR_STATUS_COMPLETED
    assert second_job.status == OCR_STATUS_COMPLETED
    assert backend.calls == 1
    results = search_documents("캐시본문", database_path=database_path, top_k=10)
    assert sorted(result.filename for result in results) == ["first.png", "second.png"]
    assert {result.page_number for result in results} == {None}
    with sqlite3.connect(database_path) as connection:
        cache_rows = connection.execute(
            "SELECT backend, page_number, result_text FROM ocr_result_cache"
        ).fetchall()
        document_rows = connection.execute(
            "SELECT filename, status, ocr_status FROM documents ORDER BY filename"
        ).fetchall()
    assert cache_rows == [("fake", 0, "스캔 문서 안의 단어 캐시본문")]
    assert document_rows == [
        ("first.png", "indexed", OCR_STATUS_COMPLETED),
        ("second.png", "indexed", OCR_STATUS_COMPLETED),
    ]


def test_ocr_failure_status_is_visible_in_status_api_and_ui(tmp_path: Path) -> None:
    root = tmp_path / "docs"
    root.mkdir()
    (root / "bad.png").write_bytes(b"\x89PNG\r\n\x1a\nfails")
    config = make_config(tmp_path, root)
    database_path = Path(config["storage"]["database_path"])

    scan(config)
    index_pending_documents(database_path)
    failed_job = process_next_ocr_job(database_path, CountingOCRBackend("unused", fail=True))

    assert failed_job is not None
    assert failed_job.status == OCR_STATUS_FAILED
    with sqlite3.connect(database_path) as connection:
        document = connection.execute(
            "SELECT status, ocr_status FROM documents WHERE filename = 'bad.png'"
        ).fetchone()
    assert document == ("pending", OCR_STATUS_FAILED)

    if TestClient is None:
        return
    client = TestClient(create_app(database_path=database_path))
    status_payload = client.get("/status").json()
    assert status_payload["ocr_failed_count"] == 1
    assert status_payload["ocr_queued_count"] == 0
    assert status_payload["failed_documents"][0]["filename"] == "bad.png"
    assert "intentional OCR failure" in status_payload["failed_documents"][0]["error_message"]

    ui_response = client.get("/ui/status")
    assert ui_response.status_code == 200
    assert "OCR failed" in ui_response.text
    assert "bad.png" in ui_response.text
