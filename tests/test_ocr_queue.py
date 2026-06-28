from __future__ import annotations

import sqlite3
from pathlib import Path

import pytest

from docsher.db import init_database
from docsher.ocr import (
    DEFAULT_OCR_BACKEND,
    FakeOCRBackend,
    OCR_STATUS_COMPLETED,
    OCR_STATUS_FAILED,
    OCR_STATUS_QUEUED,
    enqueue_ocr_document,
    list_ocr_jobs,
    process_next_ocr_job,
)


def seed_document(database_path: Path, document_path: Path) -> int:
    init_database(database_path)
    with sqlite3.connect(database_path) as connection:
        cursor = connection.execute(
            """
            INSERT INTO documents(path, filename, extension, size, modified_at, status, ocr_status)
            VALUES (?, ?, ?, 10, '2026-01-01T00:00:00+00:00', 'indexed', 'not_required')
            """,
            (str(document_path), document_path.name, document_path.suffix),
        )
        return int(cursor.lastrowid)


def get_document_ocr_status(database_path: Path, document_id: int) -> str:
    with sqlite3.connect(database_path) as connection:
        return str(
            connection.execute(
                "SELECT ocr_status FROM documents WHERE id = ?",
                (document_id,),
            ).fetchone()[0]
        )


def test_ocr_backend_interface_fake_availability_and_result(tmp_path: Path) -> None:
    backend = FakeOCRBackend("테스트 OCR 텍스트")

    assert backend.is_available() is True
    result = backend.recognize(tmp_path / "image.png")
    assert result.text == "테스트 OCR 텍스트"
    assert result.backend == "fake"


def test_enqueue_ocr_document_creates_queue_row_and_updates_document_status(tmp_path: Path) -> None:
    database_path = tmp_path / "docsher.sqlite3"
    document_path = tmp_path / "scan.png"
    document_path.write_text("fake image bytes", encoding="utf-8")
    document_id = seed_document(database_path, document_path)

    job = enqueue_ocr_document(database_path, document_id=document_id, backend="fake")

    assert job.document_id == document_id
    assert job.backend == "fake"
    assert job.status == OCR_STATUS_QUEUED
    assert job.attempts == 0
    assert get_document_ocr_status(database_path, document_id) == OCR_STATUS_QUEUED
    assert list_ocr_jobs(database_path, status=OCR_STATUS_QUEUED) == (job,)


def test_process_next_ocr_job_records_fake_backend_result(tmp_path: Path) -> None:
    database_path = tmp_path / "docsher.sqlite3"
    document_path = tmp_path / "scan.png"
    document_path.write_text("fake image bytes", encoding="utf-8")
    document_id = seed_document(database_path, document_path)
    enqueue_ocr_document(database_path, document_id=document_id, backend="fake")

    job = process_next_ocr_job(database_path, FakeOCRBackend("recognized korean text"))

    assert job is not None
    assert job.status == OCR_STATUS_COMPLETED
    assert job.attempts == 1
    assert job.error_message is None
    assert job.result_text == "recognized korean text"
    assert get_document_ocr_status(database_path, document_id) == OCR_STATUS_COMPLETED


def test_default_backend_enqueue_is_idempotent_and_processable(tmp_path: Path) -> None:
    database_path = tmp_path / "docsher.sqlite3"
    document_path = tmp_path / "scan.png"
    document_path.write_text("fake image bytes", encoding="utf-8")
    document_id = seed_document(database_path, document_path)

    first = enqueue_ocr_document(database_path, document_id=document_id)
    second = enqueue_ocr_document(database_path, document_id=document_id)

    assert first.id == second.id
    assert second.backend == DEFAULT_OCR_BACKEND
    assert len(list_ocr_jobs(database_path, status=OCR_STATUS_QUEUED)) == 1

    job = process_next_ocr_job(database_path, FakeOCRBackend("default backend text"))
    assert job is not None
    assert job.status == OCR_STATUS_COMPLETED
    assert job.result_text == "default backend text"


def test_process_next_ocr_job_captures_backend_failure_without_raising(tmp_path: Path) -> None:
    database_path = tmp_path / "docsher.sqlite3"
    document_path = tmp_path / "scan.png"
    document_path.write_text("fake image bytes", encoding="utf-8")
    document_id = seed_document(database_path, document_path)
    enqueue_ocr_document(database_path, document_id=document_id, backend="fake")

    job = process_next_ocr_job(database_path, FakeOCRBackend(fail=True))

    assert job is not None
    assert job.status == OCR_STATUS_FAILED
    assert job.attempts == 1
    assert "fake OCR failed" in str(job.error_message)
    assert get_document_ocr_status(database_path, document_id) == OCR_STATUS_FAILED


def test_enqueue_unknown_document_is_clear_error(tmp_path: Path) -> None:
    database_path = init_database(tmp_path / "docsher.sqlite3")

    with pytest.raises(ValueError, match="Document not found"):
        enqueue_ocr_document(database_path, document_id=999, backend="fake")
