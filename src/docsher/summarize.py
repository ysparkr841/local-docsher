"""Document summarization workflows for Local Docsher."""

from __future__ import annotations

import json
import sqlite3
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from docsher.db import connect, init_database
from docsher.llm import LLMClient, LLMMessage

SYSTEM_PROMPT = """You are Local Docsher, an offline-first local document summarizer.
Summarize only the document text provided by the user.
Do not use outside knowledge.
Return JSON with these keys: summary, keywords, document_type_candidate.
- summary: concise Korean or source-language summary of the core content.
- keywords: an array of 3 to 8 important keywords from the document.
- document_type_candidate: a short candidate type such as manual, policy, meeting-note, report, spreadsheet, slide, or unknown."""

MAX_PROMPT_CHARS = 12_000


@dataclass(frozen=True)
class DocumentSummary:
    """Structured document summary persisted in ``document_summaries``."""

    document_id: int
    summary: str
    keywords: tuple[str, ...]
    document_type_candidate: str
    generated_at: str
    model_name: str
    reused: bool = False
    reuse_reason: str | None = None

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


@dataclass(frozen=True)
class DocumentForSummary:
    id: int
    path: str
    filename: str
    extension: str | None
    content_hash: str | None
    chunks: tuple[dict[str, Any], ...]


def _utc_now() -> str:
    return datetime.now(UTC).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def _decode_keywords(value: str | None) -> tuple[str, ...]:
    if not value:
        return ()
    try:
        payload = json.loads(value)
    except json.JSONDecodeError:
        return tuple(part.strip() for part in value.split(",") if part.strip())
    if isinstance(payload, list):
        return tuple(str(item).strip() for item in payload if str(item).strip())
    if isinstance(payload, dict) and isinstance(payload.get("keywords"), list):
        return tuple(str(item).strip() for item in payload["keywords"] if str(item).strip())
    return ()


def _encode_keywords(keywords: tuple[str, ...]) -> str:
    return json.dumps(list(keywords), ensure_ascii=False)


def _extract_document_type(summary: str) -> str:
    for line in summary.splitlines():
        normalized = line.strip().lstrip("-*# ").strip()
        lowered = normalized.lower()
        for prefix in ("document type candidate:", "document type:", "문서 유형 후보:", "문서 유형:"):
            if lowered.startswith(prefix):
                return normalized[len(prefix) :].strip() or "unknown"
    return "unknown"


def _compose_stored_summary(core_summary: str, document_type_candidate: str) -> str:
    core = core_summary.strip()
    doc_type = document_type_candidate.strip() or "unknown"
    if _extract_document_type(core) != "unknown":
        return core
    return f"{core}\n\nDocument type candidate: {doc_type}" if core else f"Document type candidate: {doc_type}"


def _row_to_summary(row: sqlite3.Row, *, reused: bool = False, reuse_reason: str | None = None) -> DocumentSummary:
    summary = str(row["summary"] or "")
    return DocumentSummary(
        document_id=int(row["document_id"]),
        summary=summary,
        keywords=_decode_keywords(row["keywords"]),
        document_type_candidate=_extract_document_type(summary),
        generated_at=str(row["generated_at"] or ""),
        model_name=str(row["model_name"] or ""),
        reused=reused,
        reuse_reason=reuse_reason,
    )


def get_document_summary(
    document_id: int,
    *,
    database_path: str | Path | None = None,
) -> DocumentSummary | None:
    """Return a stored summary for a document, if one exists."""

    if document_id < 1:
        raise ValueError("document_id must be a positive integer")
    resolved_database_path = init_database(database_path)
    with connect(resolved_database_path) as connection:
        connection.row_factory = sqlite3.Row
        row = connection.execute(
            """
            SELECT document_id, summary, keywords, generated_at, model_name
            FROM document_summaries
            WHERE document_id = ?
            """,
            (document_id,),
        ).fetchone()
    if row is None:
        return None
    return _row_to_summary(row, reused=True, reuse_reason="existing_document_summary")


def _fetch_document_for_summary(connection: sqlite3.Connection, document_id: int) -> DocumentForSummary:
    document = connection.execute(
        """
        SELECT id, path, filename, extension, content_hash, status
        FROM documents
        WHERE id = ?
        """,
        (document_id,),
    ).fetchone()
    if document is None:
        raise ValueError(f"Document not found: {document_id}")
    if document["status"] == "deleted":
        raise ValueError(f"Document is deleted: {document_id}")
    chunks = connection.execute(
        """
        SELECT id, chunk_index, text, page_number, sheet_name, slide_number, section_title
        FROM chunks
        WHERE document_id = ?
        ORDER BY chunk_index, id
        """,
        (document_id,),
    ).fetchall()
    if not chunks:
        raise ValueError(f"Document has no indexed chunks: {document_id}")
    return DocumentForSummary(
        id=int(document["id"]),
        path=str(document["path"]),
        filename=str(document["filename"]),
        extension=document["extension"],
        content_hash=document["content_hash"],
        chunks=tuple({key: chunk[key] for key in chunk.keys()} for chunk in chunks),
    )


def _find_same_hash_summary(
    connection: sqlite3.Connection,
    *,
    document_id: int,
    content_hash: str | None,
) -> sqlite3.Row | None:
    if not content_hash:
        return None
    return connection.execute(
        """
        SELECT source_summary.document_id, source_summary.summary, source_summary.keywords,
               source_summary.generated_at, source_summary.model_name
        FROM document_summaries AS source_summary
        JOIN documents AS source_document ON source_document.id = source_summary.document_id
        WHERE source_document.content_hash = ?
          AND source_document.id != ?
          AND source_document.status != 'deleted'
        ORDER BY source_summary.generated_at DESC, source_summary.document_id
        LIMIT 1
        """,
        (content_hash, document_id),
    ).fetchone()


def _upsert_summary(
    connection: sqlite3.Connection,
    summary: DocumentSummary,
) -> DocumentSummary:
    connection.execute(
        """
        INSERT INTO document_summaries(document_id, summary, keywords, generated_at, model_name)
        VALUES (?, ?, ?, ?, ?)
        ON CONFLICT(document_id) DO UPDATE SET
            summary = excluded.summary,
            keywords = excluded.keywords,
            generated_at = excluded.generated_at,
            model_name = excluded.model_name
        """,
        (
            summary.document_id,
            summary.summary,
            _encode_keywords(summary.keywords),
            summary.generated_at,
            summary.model_name,
        ),
    )
    return summary


def build_summary_prompt(document: DocumentForSummary) -> str:
    """Build a bounded prompt from document chunk text and metadata."""

    chunks: list[str] = []
    used_chars = 0
    for chunk in document.chunks:
        location_parts = [f"chunk {chunk['chunk_index']}"]
        if chunk.get("page_number") is not None:
            location_parts.append(f"page {chunk['page_number']}")
        if chunk.get("slide_number") is not None:
            location_parts.append(f"slide {chunk['slide_number']}")
        if chunk.get("sheet_name"):
            location_parts.append(f"sheet {chunk['sheet_name']}")
        if chunk.get("section_title"):
            location_parts.append(f"section {chunk['section_title']}")
        text = str(chunk.get("text") or "")
        remaining = MAX_PROMPT_CHARS - used_chars
        if remaining <= 0:
            break
        clipped = text[:remaining]
        used_chars += len(clipped)
        chunks.append(f"[{', '.join(location_parts)}]\n{clipped}")
        if len(text) > len(clipped):
            break

    metadata = {
        "document_id": document.id,
        "filename": document.filename,
        "path": document.path,
        "extension": document.extension,
    }
    chunk_text = "\n\n".join(chunks)
    return (
        f"Document metadata:\n{json.dumps(metadata, ensure_ascii=False, sort_keys=True)}\n\n"
        "Document text chunks:\n"
        f"{chunk_text}\n\n"
        "Return only JSON with keys summary, keywords, document_type_candidate."
    )


def parse_summary_response(text: str) -> tuple[str, tuple[str, ...], str]:
    """Parse structured LLM summary output with a safe plain-text fallback."""

    stripped = text.strip()
    if not stripped:
        raise ValueError("LLM returned an empty summary")
    if stripped.startswith("```"):
        stripped = stripped.strip("`").strip()
        if stripped.lower().startswith("json"):
            stripped = stripped[4:].strip()
    try:
        payload = json.loads(stripped)
    except json.JSONDecodeError:
        return stripped, (), "unknown"
    if not isinstance(payload, dict):
        return stripped, (), "unknown"
    summary = str(payload.get("summary") or "").strip()
    if not summary:
        summary = stripped
    raw_keywords = payload.get("keywords")
    if isinstance(raw_keywords, list):
        keywords = tuple(str(item).strip() for item in raw_keywords if str(item).strip())
    elif isinstance(raw_keywords, str):
        keywords = tuple(part.strip() for part in raw_keywords.split(",") if part.strip())
    else:
        keywords = ()
    document_type_candidate = str(payload.get("document_type_candidate") or "unknown").strip() or "unknown"
    return summary, keywords, document_type_candidate


def summarize_document(
    document_id: int,
    *,
    llm_client: LLMClient,
    database_path: str | Path | None = None,
    force: bool = False,
) -> DocumentSummary:
    """Generate or reuse a document summary and persist it to SQLite.

    Reuse policy:
    - By default, an existing summary for the same document_id is reused.
    - If no per-document summary exists, an existing summary from another indexed
      document with the same content_hash is copied and reused.
    - Pass ``force=True`` to bypass both reuse paths and call the LLM again.
    """

    if document_id < 1:
        raise ValueError("document_id must be a positive integer")
    resolved_database_path = init_database(database_path)
    with connect(resolved_database_path) as connection:
        connection.row_factory = sqlite3.Row
        document = _fetch_document_for_summary(connection, document_id)
        if not force:
            existing = connection.execute(
                """
                SELECT document_id, summary, keywords, generated_at, model_name
                FROM document_summaries
                WHERE document_id = ?
                """,
                (document_id,),
            ).fetchone()
            if existing is not None:
                return _row_to_summary(existing, reused=True, reuse_reason="existing_document_summary")

            same_hash = _find_same_hash_summary(
                connection,
                document_id=document_id,
                content_hash=document.content_hash,
            )
            if same_hash is not None:
                copied = DocumentSummary(
                    document_id=document_id,
                    summary=str(same_hash["summary"] or ""),
                    keywords=_decode_keywords(same_hash["keywords"]),
                    document_type_candidate=_extract_document_type(str(same_hash["summary"] or "")),
                    generated_at=_utc_now(),
                    model_name=str(same_hash["model_name"] or ""),
                    reused=True,
                    reuse_reason="same_content_hash",
                )
                with connection:
                    _upsert_summary(connection, copied)
                return copied

        messages = [
            LLMMessage(role="system", content=SYSTEM_PROMPT),
            LLMMessage(role="user", content=build_summary_prompt(document)),
        ]
        response = llm_client.chat(messages, temperature=0.0)
        core_summary, keywords, document_type_candidate = parse_summary_response(response.text)
        generated = DocumentSummary(
            document_id=document_id,
            summary=_compose_stored_summary(core_summary, document_type_candidate),
            keywords=keywords,
            document_type_candidate=document_type_candidate,
            generated_at=_utc_now(),
            model_name=response.model,
            reused=False,
        )
        with connection:
            _upsert_summary(connection, generated)
        return generated


def format_document_summary(summary: DocumentSummary) -> str:
    """Format a stored/generated summary for human-readable CLI output."""

    lines = [
        f"Document ID: {summary.document_id}",
        f"Model: {summary.model_name}",
        f"Generated at: {summary.generated_at}",
        f"Reused: {'yes' if summary.reused else 'no'}",
    ]
    if summary.reuse_reason:
        lines.append(f"Reuse reason: {summary.reuse_reason}")
    lines.extend(
        [
            "",
            "Summary:",
            summary.summary,
            "",
            f"Keywords: {', '.join(summary.keywords) if summary.keywords else '(none)'}",
            f"Document type candidate: {summary.document_type_candidate}",
        ]
    )
    return "\n".join(lines)
