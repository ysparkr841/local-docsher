"""OCR backend interface and queue primitives for Local Docsher."""

from __future__ import annotations

import sqlite3
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Protocol

from docsher.chunker import chunk_text
from docsher.db import connect, init_database

OCR_STATUS_QUEUED = "queued"
OCR_STATUS_PROCESSING = "processing"
OCR_STATUS_COMPLETED = "completed"
OCR_STATUS_FAILED = "failed"
OCR_STATUS_NOT_REQUIRED = "not_required"
DEFAULT_OCR_BACKEND = "default"


@dataclass(frozen=True)
class OCRResult:
    """Text returned by an OCR backend for a local document/image path."""

    text: str
    backend: str
    page_number: int | None = None

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


class OCRBackend(Protocol):
    """Pluggable OCR backend contract used by OCR pipeline implementations."""

    name: str

    def is_available(self) -> bool:
        """Return whether the backend can run in the current offline environment."""

    def recognize(self, path: str | Path) -> OCRResult:
        """Run OCR for one local document/image path."""


@dataclass(frozen=True)
class OCRJob:
    """One queued OCR job."""

    id: int
    document_id: int
    backend: str | None
    status: str
    attempts: int
    error_message: str | None
    result_text: str | None
    input_path: str | None
    page_number: int
    created_at: str
    updated_at: str

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


class FakeOCRBackend:
    """Deterministic fake backend for tests and offline pipeline wiring."""

    name = "fake"

    def __init__(
        self,
        text: str = "fake OCR text",
        *,
        available: bool = True,
        fail: bool = False,
        page_number: int | None = None,
    ) -> None:
        self.text = text
        self.available = available
        self.fail = fail
        self.page_number = page_number

    def is_available(self) -> bool:
        return self.available

    def recognize(self, path: str | Path) -> OCRResult:
        if self.fail:
            raise RuntimeError(f"fake OCR failed for {Path(path).name}")
        return OCRResult(text=self.text, backend=self.name, page_number=self.page_number)


def _utc_now_iso() -> str:
    return datetime.now(UTC).isoformat()


def _store_ocr_result_chunks(
    connection: sqlite3.Connection,
    *,
    document_id: int,
    result: OCRResult,
) -> int:
    """Replace document chunks with OCR text while preserving OCR page metadata."""

    stored_page_number = result.page_number if result.page_number and result.page_number > 0 else None
    if stored_page_number is None:
        connection.execute("DELETE FROM chunks WHERE document_id = ?", (document_id,))
    else:
        connection.execute(
            "DELETE FROM chunks WHERE document_id = ? AND page_number = ?",
            (document_id, stored_page_number),
        )
    next_index = int(
        connection.execute(
            "SELECT COALESCE(MAX(chunk_index) + 1, 0) FROM chunks WHERE document_id = ?",
            (document_id,),
        ).fetchone()[0]
    )
    text_chunks = chunk_text(result.text)
    connection.executemany(
        """
        INSERT INTO chunks(
            document_id, chunk_index, text, page_number, sheet_name,
            slide_number, section_title, token_count
        )
        VALUES (?, ?, ?, ?, NULL, NULL, NULL, ?)
        """,
        (
            (
                document_id,
                next_index + chunk.chunk_index,
                chunk.text,
                stored_page_number,
                chunk.token_count,
            )
            for chunk in text_chunks
        ),
    )
    return len(text_chunks)


def _row_to_job(row: sqlite3.Row) -> OCRJob:
    return OCRJob(
        id=int(row["id"]),
        document_id=int(row["document_id"]),
        backend=row["backend"],
        status=str(row["status"]),
        attempts=int(row["attempts"]),
        error_message=row["error_message"],
        result_text=row["result_text"],
        input_path=row["input_path"],
        page_number=int(row["page_number"]),
        created_at=str(row["created_at"]),
        updated_at=str(row["updated_at"]),
    )


def enqueue_ocr_document(
    database_path: str | Path,
    *,
    document_id: int,
    backend: str | None = None,
    input_path: str | Path | None = None,
    page_number: int | None = None,
) -> OCRJob:
    """Queue a document for OCR and mark document OCR status as queued."""

    resolved_database_path = init_database(database_path)
    backend_name = backend or DEFAULT_OCR_BACKEND
    queued_input_path = str(Path(input_path).expanduser().resolve(strict=False)) if input_path is not None else None
    queued_page_number = page_number or 0
    with connect(resolved_database_path) as connection:
        connection.row_factory = sqlite3.Row
        with connection:
            document = connection.execute("SELECT id FROM documents WHERE id = ?", (document_id,)).fetchone()
            if document is None:
                raise ValueError(f"Document not found: {document_id}")
            connection.execute(
                """
                INSERT INTO ocr_jobs(document_id, backend, status, attempts, error_message, result_text, input_path, page_number)
                VALUES (?, ?, ?, 0, NULL, NULL, ?, ?)
                ON CONFLICT(document_id, backend, page_number) DO UPDATE SET
                    status = CASE
                        WHEN ocr_jobs.status = 'completed' THEN ocr_jobs.status
                        ELSE excluded.status
                    END,
                    input_path = excluded.input_path,
                    error_message = CASE
                        WHEN ocr_jobs.status = 'completed' THEN ocr_jobs.error_message
                        ELSE NULL
                    END,
                    result_text = CASE
                        WHEN ocr_jobs.status = 'completed' THEN ocr_jobs.result_text
                        ELSE NULL
                    END,
                    updated_at = datetime('now')
                """,
                (document_id, backend_name, OCR_STATUS_QUEUED, queued_input_path, queued_page_number),
            )
            connection.execute(
                "UPDATE documents SET ocr_status = ? WHERE id = ?",
                (OCR_STATUS_QUEUED, document_id),
            )
        row = connection.execute(
            "SELECT * FROM ocr_jobs WHERE document_id = ? AND backend = ? AND page_number = ?",
            (document_id, backend_name, queued_page_number),
        ).fetchone()
    return _row_to_job(row)


def list_ocr_jobs(database_path: str | Path, *, status: str | None = None) -> tuple[OCRJob, ...]:
    """List OCR jobs, optionally filtered by status."""

    resolved_database_path = init_database(database_path)
    with connect(resolved_database_path) as connection:
        connection.row_factory = sqlite3.Row
        if status is None:
            rows = connection.execute("SELECT * FROM ocr_jobs ORDER BY id").fetchall()
        else:
            rows = connection.execute(
                "SELECT * FROM ocr_jobs WHERE status = ? ORDER BY id",
                (status,),
            ).fetchall()
    return tuple(_row_to_job(row) for row in rows)


def process_next_ocr_job(database_path: str | Path, backend: OCRBackend) -> OCRJob | None:
    """Process one queued OCR job with a backend.

    Backend failures are captured on the job and document OCR status instead of
    aborting the surrounding indexing workflow.
    """

    resolved_database_path = init_database(database_path)
    with connect(resolved_database_path) as connection:
        connection.row_factory = sqlite3.Row
        row = connection.execute(
            """
            SELECT ocr_jobs.*, documents.path AS document_path
            FROM ocr_jobs
            JOIN documents ON documents.id = ocr_jobs.document_id
            WHERE ocr_jobs.status = ? AND ocr_jobs.backend IN (?, ?)
            ORDER BY ocr_jobs.id
            LIMIT 1
            """,
            (OCR_STATUS_QUEUED, backend.name, DEFAULT_OCR_BACKEND),
        ).fetchone()
        if row is None:
            return None

        job_id = int(row["id"])
        document_id = int(row["document_id"])
        document_path = str(row["input_path"] or row["document_path"])
        queued_page_number = int(row["page_number"])
        attempts = int(row["attempts"]) + 1
        with connection:
            connection.execute(
                "UPDATE ocr_jobs SET status = ?, attempts = ?, updated_at = datetime('now') WHERE id = ?",
                (OCR_STATUS_PROCESSING, attempts, job_id),
            )
            connection.execute(
                "UPDATE documents SET ocr_status = ? WHERE id = ?",
                (OCR_STATUS_PROCESSING, document_id),
            )

        try:
            if not backend.is_available():
                raise RuntimeError(f"OCR backend unavailable: {backend.name}")
            result = backend.recognize(document_path)
            if result.page_number is None and queued_page_number > 0:
                result = OCRResult(text=result.text, backend=result.backend, page_number=queued_page_number)
        except Exception as exc:  # noqa: BLE001 - backend boundary must isolate failures.
            with connection:
                connection.execute(
                    """
                    UPDATE ocr_jobs
                    SET status = ?, attempts = ?, error_message = ?, updated_at = datetime('now')
                    WHERE id = ?
                    """,
                    (OCR_STATUS_FAILED, attempts, str(exc), job_id),
                )
                connection.execute(
                    "UPDATE documents SET ocr_status = ? WHERE id = ?",
                    (OCR_STATUS_FAILED, document_id),
                )
        else:
            with connection:
                _store_ocr_result_chunks(connection, document_id=document_id, result=result)
                connection.execute(
                    """
                    UPDATE ocr_jobs
                    SET status = ?, attempts = ?, error_message = NULL, result_text = ?, updated_at = datetime('now')
                    WHERE id = ?
                    """,
                    (OCR_STATUS_COMPLETED, attempts, result.text, job_id),
                )
                remaining_jobs = int(
                    connection.execute(
                        """
                        SELECT COUNT(*) FROM ocr_jobs
                        WHERE document_id = ? AND id != ? AND status IN (?, ?)
                        """,
                        (document_id, job_id, OCR_STATUS_QUEUED, OCR_STATUS_PROCESSING),
                    ).fetchone()[0]
                )
                document_status = 'indexed' if remaining_jobs == 0 else 'pending'
                document_ocr_status = OCR_STATUS_COMPLETED if remaining_jobs == 0 else OCR_STATUS_QUEUED
                connection.execute(
                    """
                    UPDATE documents
                    SET status = ?, indexed_at = ?, error_message = NULL,
                        parser_name = ?, ocr_status = ?
                    WHERE id = ?
                    """,
                    (document_status, _utc_now_iso(), f"ocr:{backend.name}", document_ocr_status, document_id),
                )

        final_row = connection.execute("SELECT * FROM ocr_jobs WHERE id = ?", (job_id,)).fetchone()
    return _row_to_job(final_row)
