"""Indexing status reporting for Local Docsher."""

from __future__ import annotations

import json
import sqlite3
from dataclasses import asdict, dataclass
from pathlib import Path

from docsher.db import connect, init_database
from docsher.indexer import SUPPORTED_INDEX_EXTENSIONS

FAILED_STATUS = "failed"
DELETED_STATUS = "deleted"
INDEXED_STATUS = "indexed"
PENDING_STATUS = "pending"


@dataclass(frozen=True)
class FailedDocumentStatus:
    """User-facing status for one failed document."""

    id: int
    path: str
    filename: str
    extension: str | None
    status: str
    error_message: str | None
    parser_name: str | None
    retryable: bool
    retryable_reason: str

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


@dataclass(frozen=True)
class IndexStatus:
    """Aggregate indexing status for the document database."""

    database_path: str
    total_documents: int
    active_documents: int
    indexed_count: int
    failed_count: int
    deleted_count: int
    pending_count: int
    other_count: int
    last_indexed_at: str | None
    failed_documents: tuple[FailedDocumentStatus, ...]

    def to_dict(self) -> dict[str, object]:
        payload = asdict(self)
        payload["failed_documents"] = [document.to_dict() for document in self.failed_documents]
        return payload


def _is_retryable_failed_document(row: sqlite3.Row) -> tuple[bool, str]:
    status = str(row["status"])
    if status == DELETED_STATUS:
        return False, "document is marked deleted"

    path = Path(str(row["path"]))
    if not path.exists():
        return False, "file is missing"
    if not path.is_file():
        return False, "path is not a regular file"

    extension = (row["extension"] or path.suffix or "").lower()
    if extension not in SUPPORTED_INDEX_EXTENSIONS:
        return False, "extension is not supported by an indexer"

    return True, "file exists and extension is supported"


def get_index_status(database_path: str | Path | None = None) -> IndexStatus:
    """Return aggregate status and failed-document details for a Docsher database."""

    resolved_database_path = init_database(database_path)
    with connect(resolved_database_path) as connection:
        connection.row_factory = sqlite3.Row
        total_documents = int(connection.execute("SELECT COUNT(*) FROM documents").fetchone()[0])
        rows = connection.execute(
            """
            SELECT status, COUNT(*) AS count
            FROM documents
            GROUP BY status
            """
        ).fetchall()
        status_counts = {str(row["status"]): int(row["count"]) for row in rows}
        last_indexed_at = connection.execute(
            """
            SELECT MAX(indexed_at)
            FROM documents
            WHERE status = ? AND indexed_at IS NOT NULL
            """,
            (INDEXED_STATUS,),
        ).fetchone()[0]
        failed_rows = connection.execute(
            """
            SELECT id, path, filename, extension, status, error_message, parser_name
            FROM documents
            WHERE status = ?
            ORDER BY path
            """,
            (FAILED_STATUS,),
        ).fetchall()

    indexed_count = status_counts.get(INDEXED_STATUS, 0)
    failed_count = status_counts.get(FAILED_STATUS, 0)
    deleted_count = status_counts.get(DELETED_STATUS, 0)
    pending_count = status_counts.get(PENDING_STATUS, 0)
    known_count = indexed_count + failed_count + deleted_count + pending_count
    failed_documents = []
    for row in failed_rows:
        retryable, retryable_reason = _is_retryable_failed_document(row)
        failed_documents.append(
            FailedDocumentStatus(
                id=int(row["id"]),
                path=str(row["path"]),
                filename=str(row["filename"]),
                extension=row["extension"],
                status=str(row["status"]),
                error_message=row["error_message"],
                parser_name=row["parser_name"],
                retryable=retryable,
                retryable_reason=retryable_reason,
            )
        )

    return IndexStatus(
        database_path=str(resolved_database_path),
        total_documents=total_documents,
        active_documents=total_documents - deleted_count,
        indexed_count=indexed_count,
        failed_count=failed_count,
        deleted_count=deleted_count,
        pending_count=pending_count,
        other_count=total_documents - known_count,
        last_indexed_at=str(last_indexed_at) if last_indexed_at is not None else None,
        failed_documents=tuple(failed_documents),
    )


def format_index_status(status: IndexStatus, *, json_output: bool = False) -> str:
    """Format index status for CLI output."""

    if json_output:
        return json.dumps(status.to_dict(), ensure_ascii=False, indent=2, sort_keys=True)

    lines = [
        "Index status:",
        f"Database: {status.database_path}",
        f"Total documents: {status.total_documents}",
        f"Active documents: {status.active_documents}",
        f"Indexed documents: {status.indexed_count}",
        f"Failed documents: {status.failed_count}",
        f"Pending documents: {status.pending_count}",
        f"Deleted documents: {status.deleted_count}",
        f"Other status documents: {status.other_count}",
        f"Last indexed at: {status.last_indexed_at or 'never'}",
    ]
    if not status.failed_documents:
        lines.append("Failed document details: none")
        return "\n".join(lines)

    lines.append("Failed document details:")
    for document in status.failed_documents:
        retryable = "yes" if document.retryable else "no"
        lines.extend(
            [
                f"- {document.filename} ({document.path})",
                f"  Error: {document.error_message or ''}",
                f"  Retryable: {retryable} ({document.retryable_reason})",
            ]
        )
    return "\n".join(lines)
