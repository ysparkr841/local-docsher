"""OCR backend interface and queue primitives for Local Docsher."""

from __future__ import annotations

import sqlite3
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Protocol

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
    created_at: str
    updated_at: str

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


class FakeOCRBackend:
    """Deterministic fake backend for tests and offline pipeline wiring."""

    name = "fake"

    def __init__(self, text: str = "fake OCR text", *, available: bool = True, fail: bool = False) -> None:
        self.text = text
        self.available = available
        self.fail = fail

    def is_available(self) -> bool:
        return self.available

    def recognize(self, path: str | Path) -> OCRResult:
        if self.fail:
            raise RuntimeError(f"fake OCR failed for {Path(path).name}")
        return OCRResult(text=self.text, backend=self.name)


def _row_to_job(row: sqlite3.Row) -> OCRJob:
    return OCRJob(
        id=int(row["id"]),
        document_id=int(row["document_id"]),
        backend=row["backend"],
        status=str(row["status"]),
        attempts=int(row["attempts"]),
        error_message=row["error_message"],
        result_text=row["result_text"],
        created_at=str(row["created_at"]),
        updated_at=str(row["updated_at"]),
    )


def enqueue_ocr_document(
    database_path: str | Path,
    *,
    document_id: int,
    backend: str | None = None,
) -> OCRJob:
    """Queue a document for OCR and mark document OCR status as queued."""

    resolved_database_path = init_database(database_path)
    backend_name = backend or DEFAULT_OCR_BACKEND
    with connect(resolved_database_path) as connection:
        connection.row_factory = sqlite3.Row
        with connection:
            document = connection.execute("SELECT id FROM documents WHERE id = ?", (document_id,)).fetchone()
            if document is None:
                raise ValueError(f"Document not found: {document_id}")
            connection.execute(
                """
                INSERT INTO ocr_jobs(document_id, backend, status, attempts, error_message, result_text)
                VALUES (?, ?, ?, 0, NULL, NULL)
                ON CONFLICT(document_id, backend) DO UPDATE SET
                    status = excluded.status,
                    error_message = NULL,
                    result_text = NULL,
                    updated_at = datetime('now')
                """,
                (document_id, backend_name, OCR_STATUS_QUEUED),
            )
            connection.execute(
                "UPDATE documents SET ocr_status = ? WHERE id = ?",
                (OCR_STATUS_QUEUED, document_id),
            )
        row = connection.execute(
            "SELECT * FROM ocr_jobs WHERE document_id = ? AND backend = ?",
            (document_id, backend_name),
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
        document_path = str(row["document_path"])
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
                connection.execute(
                    """
                    UPDATE ocr_jobs
                    SET status = ?, attempts = ?, error_message = NULL, result_text = ?, updated_at = datetime('now')
                    WHERE id = ?
                    """,
                    (OCR_STATUS_COMPLETED, attempts, result.text, job_id),
                )
                connection.execute(
                    "UPDATE documents SET ocr_status = ? WHERE id = ?",
                    (OCR_STATUS_COMPLETED, document_id),
                )

        final_row = connection.execute("SELECT * FROM ocr_jobs WHERE id = ?", (job_id,)).fetchone()
    return _row_to_job(final_row)
