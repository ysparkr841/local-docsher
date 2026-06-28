"""SQLite FTS5 search support for Local Docsher."""

from __future__ import annotations

import sqlite3
from dataclasses import asdict, dataclass
from pathlib import Path

from docsher.db import connect, init_database


class SearchError(RuntimeError):
    """Raised when a search query cannot be executed."""


@dataclass(frozen=True)
class SearchResult:
    """A single chunk-level search result."""

    chunk_id: int
    document_id: int
    chunk_index: int
    snippet: str
    path: str
    filename: str
    extension: str | None
    page_number: int | None = None
    slide_number: int | None = None
    sheet_name: str | None = None
    section_title: str | None = None
    rank: float | None = None

    def to_dict(self) -> dict[str, object]:
        """Return a deterministic JSON-serializable representation."""

        return asdict(self)


def _normalize_extension(extension: str | None) -> str | None:
    if extension is None:
        return None
    normalized = extension.strip().lower()
    if not normalized:
        return None
    if not normalized.startswith("."):
        normalized = f".{normalized}"
    return normalized


def _escape_like(value: str) -> str:
    return value.replace("\\", "\\\\").replace("%", "\\%").replace("_", "\\_")


def _quote_fts_query(query: str) -> str:
    return '"' + query.replace('"', '""') + '"'


def _build_search_sql(*, extension: str | None, path_filter: str | None) -> tuple[str, list[object]]:
    where_clauses = ["chunks_fts MATCH ?", "documents.status != 'deleted'"]
    params: list[object] = []

    if extension is not None:
        where_clauses.append("lower(documents.extension) = ?")
        params.append(extension)
    if path_filter:
        where_clauses.append("documents.path LIKE ? ESCAPE '\\'")
        params.append(f"%{_escape_like(path_filter)}%")

    sql = f"""
        SELECT
            chunks.id AS chunk_id,
            chunks.document_id AS document_id,
            chunks.chunk_index AS chunk_index,
            snippet(chunks_fts, 0, '[', ']', '…', 24) AS snippet,
            documents.path AS path,
            documents.filename AS filename,
            documents.extension AS extension,
            chunks.page_number AS page_number,
            chunks.slide_number AS slide_number,
            chunks.sheet_name AS sheet_name,
            chunks.section_title AS section_title,
            bm25(chunks_fts) AS rank
        FROM chunks_fts
        JOIN chunks ON chunks.id = chunks_fts.rowid
        JOIN documents ON documents.id = chunks.document_id
        WHERE {' AND '.join(where_clauses)}
        ORDER BY rank, documents.path, chunks.chunk_index, chunks.id
        LIMIT ?
    """
    return sql, params


def _rows_to_results(rows: list[sqlite3.Row]) -> tuple[SearchResult, ...]:
    return tuple(
        SearchResult(
            chunk_id=int(row["chunk_id"]),
            document_id=int(row["document_id"]),
            chunk_index=int(row["chunk_index"]),
            snippet=str(row["snippet"] or ""),
            path=str(row["path"]),
            filename=str(row["filename"]),
            extension=row["extension"],
            page_number=row["page_number"],
            slide_number=row["slide_number"],
            sheet_name=row["sheet_name"],
            section_title=row["section_title"],
            rank=float(row["rank"]) if row["rank"] is not None else None,
        )
        for row in rows
    )


def search_documents(
    query: str,
    *,
    database_path: str | Path | None = None,
    extension: str | None = None,
    path_filter: str | None = None,
    top_k: int = 10,
) -> tuple[SearchResult, ...]:
    """Search indexed chunks with SQLite FTS5.

    The query is matched against chunk text plus indexed filename/path/section-title
    fields in ``chunks_fts``. ``path_filter`` is a SQLite LIKE substring filter
    over the stored document path.
    """

    cleaned_query = query.strip()
    if not cleaned_query:
        raise SearchError("Search query must not be empty")
    if top_k < 1:
        raise SearchError("--top-k must be a positive integer")

    normalized_extension = _normalize_extension(extension)
    resolved_database_path = init_database(database_path)
    sql, filter_params = _build_search_sql(
        extension=normalized_extension,
        path_filter=path_filter,
    )

    with connect(resolved_database_path) as connection:
        connection.row_factory = sqlite3.Row
        try:
            rows = connection.execute(sql, [cleaned_query, *filter_params, top_k]).fetchall()
        except sqlite3.OperationalError as exc:
            # Common for user input containing FTS operators/punctuation, e.g. a
            # full filename like "release-notes.txt". Retry as a phrase query so
            # filename search remains ergonomic while still using FTS binding.
            try:
                rows = connection.execute(
                    sql,
                    [_quote_fts_query(cleaned_query), *filter_params, top_k],
                ).fetchall()
            except sqlite3.OperationalError as retry_exc:
                raise SearchError(f"Invalid FTS search query: {query!r}") from retry_exc
            if not rows and "fts5" not in str(exc).lower():
                raise SearchError(f"Search failed: {exc}") from exc

    return _rows_to_results(rows)


def format_search_results(results: tuple[SearchResult, ...]) -> str:
    """Format search results for human-readable CLI output."""

    if not results:
        return "No results found."

    lines: list[str] = []
    for index, result in enumerate(results, start=1):
        lines.append(f"{index}. {result.filename} — {result.path}")
        location_parts = [f"chunk {result.chunk_index}"]
        if result.page_number is not None:
            location_parts.append(f"page {result.page_number}")
        if result.slide_number is not None:
            location_parts.append(f"slide {result.slide_number}")
        if result.sheet_name:
            location_parts.append(f"sheet {result.sheet_name}")
        if result.section_title:
            location_parts.append(f"section {result.section_title}")
        lines.append(f"   Location: {', '.join(location_parts)}")
        lines.append(f"   Snippet: {result.snippet}")
    return "\n".join(lines)
