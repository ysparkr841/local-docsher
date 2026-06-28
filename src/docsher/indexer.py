"""Apply parser/chunker work for pending documents."""

from __future__ import annotations

import sqlite3
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

from docsher.chunker import TextChunk, chunk_text
from docsher.db import connect, init_database
from docsher.parsers_office import (
    PARSER_NAME as OFFICE_PARSER_NAME,
    SUPPORTED_OFFICE_EXTENSIONS,
    OfficeParserError,
    ParsedOfficeSegment,
    parse_office_document,
)
from docsher.parsers_text import (
    PARSER_NAME as TEXT_PARSER_NAME,
    SUPPORTED_TEXT_EXTENSIONS,
    TextParserError,
    parse_text_document,
)

INDEXED_STATUS = "indexed"
PENDING_STATUS = "pending"
FAILED_STATUS = "failed"
SUPPORTED_INDEX_EXTENSIONS = SUPPORTED_TEXT_EXTENSIONS | SUPPORTED_OFFICE_EXTENSIONS


@dataclass(frozen=True)
class IndexResult:
    """Summary of pending document indexing."""

    parsed_documents: int = 0
    failed_documents: int = 0
    skipped_pending_documents: int = 0
    created_chunks: int = 0


# Backwards-compatible public alias from LDS-005.
TextIndexResult = IndexResult


@dataclass(frozen=True)
class _PendingDocument:
    id: int
    path: str
    extension: str | None


@dataclass(frozen=True)
class _IndexChunk:
    chunk_index: int
    text: str
    token_count: int
    page_number: int | None = None
    sheet_name: str | None = None
    slide_number: int | None = None
    section_title: str | None = None


def _utc_now_iso() -> str:
    return datetime.now(UTC).isoformat()


def _load_pending_documents(connection: sqlite3.Connection) -> tuple[_PendingDocument, ...]:
    rows = connection.execute(
        """
        SELECT id, path, extension
        FROM documents
        WHERE status = ?
        ORDER BY path
        """,
        (PENDING_STATUS,),
    ).fetchall()
    return tuple(
        _PendingDocument(id=int(row[0]), path=str(row[1]), extension=row[2])
        for row in rows
    )


def _text_chunks_to_index_chunks(chunks: tuple[TextChunk, ...]) -> tuple[_IndexChunk, ...]:
    return tuple(
        _IndexChunk(
            chunk_index=chunk.chunk_index,
            text=chunk.text,
            token_count=chunk.token_count,
        )
        for chunk in chunks
    )


def _office_segments_to_index_chunks(segments: tuple[ParsedOfficeSegment, ...]) -> tuple[_IndexChunk, ...]:
    chunks: list[_IndexChunk] = []
    for segment in segments:
        for text_chunk in chunk_text(segment.text):
            chunks.append(
                _IndexChunk(
                    chunk_index=len(chunks),
                    text=text_chunk.text,
                    token_count=text_chunk.token_count,
                    page_number=segment.page_number,
                    sheet_name=segment.sheet_name,
                    slide_number=segment.slide_number,
                    section_title=segment.section_title,
                )
            )
    return tuple(chunks)


def _replace_document_chunks(
    connection: sqlite3.Connection,
    *,
    document_id: int,
    chunks: tuple[_IndexChunk, ...],
) -> None:
    connection.execute("DELETE FROM chunks WHERE document_id = ?", (document_id,))
    connection.executemany(
        """
        INSERT INTO chunks(
            document_id, chunk_index, text, page_number, sheet_name,
            slide_number, section_title, token_count
        )
        VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            (
                document_id,
                chunk.chunk_index,
                chunk.text,
                chunk.page_number,
                chunk.sheet_name,
                chunk.slide_number,
                chunk.section_title,
                chunk.token_count,
            )
            for chunk in chunks
        ),
    )


def _mark_document_indexed(
    connection: sqlite3.Connection,
    *,
    document_id: int,
    indexed_at: str,
    parser_name: str,
) -> None:
    connection.execute(
        """
        UPDATE documents
        SET status = ?, indexed_at = ?, error_message = NULL,
            parser_name = ?, ocr_status = 'not_required'
        WHERE id = ?
        """,
        (INDEXED_STATUS, indexed_at, parser_name, document_id),
    )


def _mark_document_failed(
    connection: sqlite3.Connection,
    *,
    document_id: int,
    error_message: str,
    parser_name: str,
) -> None:
    connection.execute("DELETE FROM chunks WHERE document_id = ?", (document_id,))
    connection.execute(
        """
        UPDATE documents
        SET status = ?, indexed_at = NULL, error_message = ?,
            parser_name = ?, ocr_status = 'not_required'
        WHERE id = ?
        """,
        (FAILED_STATUS, error_message, parser_name, document_id),
    )


def _parse_document(document: _PendingDocument, *, max_text_bytes: int | None) -> tuple[str, tuple[_IndexChunk, ...]]:
    extension = (document.extension or "").lower()
    if extension in SUPPORTED_TEXT_EXTENSIONS:
        parse_kwargs = {}
        if max_text_bytes is not None:
            parse_kwargs["max_bytes"] = max_text_bytes
        parsed = parse_text_document(document.path, **parse_kwargs)
        return parsed.parser_name, _text_chunks_to_index_chunks(chunk_text(parsed.text))
    if extension in SUPPORTED_OFFICE_EXTENSIONS:
        parsed = parse_office_document(document.path)
        if not parsed.segments:
            raise OfficeParserError("No extractable office text found")
        return parsed.parser_name, _office_segments_to_index_chunks(parsed.segments)
    raise ValueError(f"Unsupported pending document extension: {extension}")


def index_pending_documents(
    database_path: str | Path,
    *,
    max_text_bytes: int | None = None,
) -> IndexResult:
    """Parse pending supported documents and store their chunks.

    Failures are recorded per document so one bad file does not abort the whole
    index run. Pending documents without an MVP parser remain pending and are
    counted as skipped.
    """

    resolved_database_path = init_database(database_path)
    parsed_documents = 0
    failed_documents = 0
    skipped_pending_documents = 0
    created_chunks = 0

    with connect(resolved_database_path) as connection:
        pending_documents = _load_pending_documents(connection)

    for document in pending_documents:
        extension = (document.extension or "").lower()
        if extension not in SUPPORTED_INDEX_EXTENSIONS:
            skipped_pending_documents += 1
            continue

        parser_name = OFFICE_PARSER_NAME if extension in SUPPORTED_OFFICE_EXTENSIONS else TEXT_PARSER_NAME
        try:
            parser_name, chunks = _parse_document(document, max_text_bytes=max_text_bytes)
        except (OSError, TextParserError, OfficeParserError, ValueError) as exc:
            with connect(resolved_database_path) as connection:
                with connection:
                    _mark_document_failed(
                        connection,
                        document_id=document.id,
                        error_message=str(exc),
                        parser_name=parser_name,
                    )
            failed_documents += 1
            continue

        indexed_at = _utc_now_iso()
        with connect(resolved_database_path) as connection:
            with connection:
                _replace_document_chunks(
                    connection,
                    document_id=document.id,
                    chunks=chunks,
                )
                _mark_document_indexed(
                    connection,
                    document_id=document.id,
                    indexed_at=indexed_at,
                    parser_name=parser_name,
                )
        parsed_documents += 1
        created_chunks += len(chunks)

    return IndexResult(
        parsed_documents=parsed_documents,
        failed_documents=failed_documents,
        skipped_pending_documents=skipped_pending_documents,
        created_chunks=created_chunks,
    )


def index_pending_text_documents(
    database_path: str | Path,
    *,
    max_text_bytes: int | None = None,
) -> IndexResult:
    """Backward-compatible wrapper for the LDS-005 public function name."""

    return index_pending_documents(database_path, max_text_bytes=max_text_bytes)


def format_index_result(result: IndexResult) -> str:
    """Format an indexing result for CLI output."""

    return "\n".join(
        [
            "Document indexing:",
            f"Parsed documents: {result.parsed_documents}",
            f"Failed documents: {result.failed_documents}",
            f"Skipped pending unsupported documents: {result.skipped_pending_documents}",
            f"Created chunks: {result.created_chunks}",
        ]
    )


def format_text_index_result(result: IndexResult) -> str:
    """Backward-compatible formatter name used by older callers."""

    return format_index_result(result)
