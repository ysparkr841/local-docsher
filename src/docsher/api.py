"""FastAPI application for Local Docsher search, documents, and indexing."""

from __future__ import annotations

import sqlite3
from dataclasses import asdict, is_dataclass
from html import escape
from pathlib import Path
from typing import Any

from docsher import __version__
from docsher.ask import ask_question
from docsher.config import get_indexing_schedule, load_config, load_config_with_location, save_config, validate_indexing_schedule
from docsher.db import connect, init_database
from docsher.llm import LLMClientError, create_llm_client
from docsher.scheduler import run_scheduled_index_once
from docsher.search import SearchError, search_documents
from docsher.status import get_index_status

try:  # pragma: no cover - exercised implicitly when FastAPI is installed.
    from fastapi import Body, FastAPI, HTTPException, Query, Request
    from fastapi.responses import HTMLResponse, JSONResponse
except ImportError as exc:  # pragma: no cover - defensive optional dependency guard.
    raise RuntimeError(
        "FastAPI API dependencies are not installed. Install local-docsher[api] "
        "or install fastapi and uvicorn."
    ) from exc


DEFAULT_PREVIEW_CHARS = 300
MAX_PREVIEW_CHARS = 2000
MAX_TOP_K = 100
UI_SEARCH_TOP_K = 25


def _html_page(*, title: str, body: str) -> str:
    """Return a small self-contained HTML page for the local MVP UI."""

    escaped_title = escape(title)
    return f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>{escaped_title}</title>
  <style>
    :root {{ color-scheme: light dark; }}
    body {{ font-family: system-ui, -apple-system, Segoe UI, sans-serif; margin: 2rem; line-height: 1.5; }}
    main {{ max-width: 980px; margin: 0 auto; }}
    form.search {{ display: flex; gap: .5rem; margin: 1rem 0 1.5rem; }}
    input[type="search"] {{ flex: 1; padding: .65rem .75rem; font-size: 1rem; }}
    button {{ padding: .65rem 1rem; font-size: 1rem; cursor: pointer; }}
    .muted {{ color: #667085; }}
    .state {{ border: 1px solid #d0d5dd; border-radius: .5rem; padding: .75rem 1rem; margin: 1rem 0; }}
    .stats {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(140px, 1fr)); gap: .75rem; margin: 1rem 0; }}
    .stat {{ border: 1px solid #d0d5dd; border-radius: .65rem; padding: .8rem; }}
    .stat strong {{ display: block; font-size: 1.5rem; }}
    .error {{ border-color: #f04438; background: #fff1f0; color: #b42318; }}
    .result, .chunk {{ border: 1px solid #d0d5dd; border-radius: .65rem; padding: 1rem; margin: .8rem 0; }}
    .result h2, .chunk h2 {{ font-size: 1.1rem; margin: 0 0 .25rem; }}
    .path, .location, .meta {{ font-size: .92rem; color: #667085; overflow-wrap: anywhere; }}
    .snippet, .preview {{ white-space: pre-wrap; margin-top: .6rem; }}
    dl {{ display: grid; grid-template-columns: max-content 1fr; gap: .35rem 1rem; }}
    dt {{ font-weight: 700; }}
    dd {{ margin: 0; overflow-wrap: anywhere; }}
    a {{ color: #175cd3; }}
    mark {{ padding: 0 .1em; }}
    @media (prefers-color-scheme: dark) {{
      body {{ background: #101828; color: #f2f4f7; }}
      .muted, .path, .location, .meta {{ color: #98a2b3; }}
      .state, .result, .chunk, .stat {{ border-color: #475467; }}
      .error {{ background: #55160c; color: #fecdc8; }}
      a {{ color: #84caff; }}
    }}
  </style>
</head>
<body>
  <main>{body}</main>
</body>
</html>"""


def _format_location(payload: dict[str, Any]) -> str:
    """Format chunk-level location metadata for UI display."""

    parts = [f"chunk {payload.get('chunk_index')}"]
    if payload.get("page_number") is not None:
        parts.append(f"page {payload['page_number']}")
    if payload.get("slide_number") is not None:
        parts.append(f"slide {payload['slide_number']}")
    if payload.get("sheet_name"):
        parts.append(f"sheet {payload['sheet_name']}")
    if payload.get("section_title"):
        parts.append(f"section {payload['section_title']}")
    return ", ".join(str(part) for part in parts)


def _render_search_page(
    *,
    query: str = "",
    results: list[dict[str, Any]] | None = None,
    error: str | None = None,
) -> str:
    query_value = escape(query, quote=True)
    body_parts = [
        "<h1>Local Docsher</h1>",
        '<p class="muted">Search indexed local documents and open document details.</p>',
        '<p><a href="/ui/status">Index status dashboard</a> · <a href="/ui/settings">Settings</a></p>',
        '<form class="search" method="get" action="/">',
        (
            '<input type="search" name="q" placeholder="Search documents" '
            f'value="{query_value}" aria-label="Search query" autofocus>'
        ),
        '<button type="submit">Search</button>',
        "</form>",
    ]
    if error:
        body_parts.append(f'<div class="state error" role="alert">{escape(error)}</div>')
    elif query and results == []:
        body_parts.append('<div class="state" role="status">No results found.</div>')
    elif results:
        body_parts.append(
            f'<p class="muted" role="status">Showing {len(results)} result(s) for '
            f'<strong>{escape(query)}</strong>.</p>'
        )
        body_parts.append('<section aria-label="Search results">')
        for result in results:
            filename = escape(str(result.get("filename") or "Untitled"))
            path = escape(str(result.get("path") or ""))
            snippet = escape(str(result.get("snippet") or ""))
            location = escape(_format_location(result))
            document_id = escape(str(result.get("document_id")))
            body_parts.append(
                '<article class="result">'
                f'<h2><a href="/ui/documents/{document_id}">{filename}</a></h2>'
                f'<div class="path">{path}</div>'
                f'<div class="location">Location: {location}</div>'
                f'<div class="snippet">{snippet}</div>'
                '</article>'
            )
        body_parts.append("</section>")
    else:
        body_parts.append('<div class="state" role="status">Enter a query to search indexed documents.</div>')

    body_parts.append(
        """
<script>
// Lightweight enhancement hook: the form is fully functional without JavaScript.
document.querySelector('form.search').addEventListener('submit', () => {
  document.body.dataset.searchState = 'loading';
});
</script>"""
    )
    return _html_page(title="Local Docsher", body="\n".join(body_parts))


def _render_status_page(payload: dict[str, Any]) -> str:
    body_parts = [
        '<p><a href="/">← Back to search</a> · <a href="/ui/settings">Settings</a></p>',
        "<h1>Index status</h1>",
        f'<p class="muted">Database: {escape(str(payload.get("database_path") or ""))}</p>',
        '<section class="stats" aria-label="Indexing status counts">',
    ]
    stats = (
        ("Total documents", "total_documents"),
        ("Indexed", "indexed_count"),
        ("Failed", "failed_count"),
        ("Pending", "pending_count"),
        ("Deleted", "deleted_count"),
        ("OCR queued", "ocr_queued_count"),
        ("OCR completed", "ocr_completed_count"),
        ("OCR failed", "ocr_failed_count"),
        ("OCR processing", "ocr_processing_count"),
    )
    for label, key in stats:
        body_parts.append(
            f'<div class="stat"><span>{escape(label)}</span><strong>{escape(str(payload.get(key, 0)))}</strong></div>'
        )
    body_parts.extend(
        [
            "</section>",
            f'<p><strong>Last indexed at:</strong> {escape(str(payload.get("last_indexed_at") or "never"))}</p>',
            '<section aria-label="Failed documents">',
            "<h2>Failed documents</h2>",
        ]
    )
    failed_documents = payload.get("failed_documents") or []
    if not failed_documents:
        body_parts.append('<div class="state" role="status">No failed documents.</div>')
    else:
        for document in failed_documents:
            if not isinstance(document, dict):
                continue
            filename = escape(str(document.get("filename") or "Untitled"))
            path = escape(str(document.get("path") or ""))
            error_message = escape(str(document.get("error_message") or ""))
            retryable = "yes" if document.get("retryable") else "no"
            retryable_reason = escape(str(document.get("retryable_reason") or ""))
            body_parts.append(
                '<article class="result">'
                f"<h2>{filename}</h2>"
                f'<div class="path">{path}</div>'
                f'<div class="meta"><strong>Retryable:</strong> {retryable} — {retryable_reason}</div>'
                f'<div class="snippet"><strong>Error:</strong> {error_message}</div>'
                "</article>"
            )
    body_parts.append("</section>")
    body_parts.append(
        '<form method="post" action="/index/run">'
        '<button type="submit">Run indexing now</button>'
        '</form>'
    )
    return _html_page(title="Index status — Local Docsher", body="\n".join(body_parts))


def _render_settings_page(
    *,
    schedule: dict[str, Any],
    config_path: str,
    message: str | None = None,
    error: str | None = None,
) -> str:
    checked_enabled = " checked" if schedule.get("schedule_enabled") else ""
    checked_incremental = " checked" if schedule.get("incremental") else ""
    selected = {str(schedule.get("schedule")): " selected"}
    body_parts = [
        '<p><a href="/">← Back to search</a> · <a href="/ui/status">Index status</a></p>',
        "<h1>Settings</h1>",
        f'<p class="muted">Config: {escape(config_path)}</p>',
    ]
    if message:
        body_parts.append(f'<div class="state" role="status">{escape(message)}</div>')
    if error:
        body_parts.append(f'<div class="state error" role="alert">{escape(error)}</div>')
    body_parts.extend(
        [
            '<section aria-label="Indexing schedule settings">',
            "<h2>Indexing schedule</h2>",
            '<form method="post" action="/ui/settings">',
            f'<label><input type="checkbox" name="schedule_enabled" value="true"{checked_enabled}> Enable scheduled indexing</label><br>',
            '<label>Schedule '
            '<select name="schedule">'
            f'<option value="daily"{selected.get("daily", "")}>daily</option>'
            f'<option value="hourly"{selected.get("hourly", "")}>hourly</option>'
            f'<option value="manual"{selected.get("manual", "")}>manual</option>'
            '</select></label><br>',
            f'<label>Time <input type="time" name="time" value="{escape(str(schedule.get("time") or "03:00"))}"></label><br>',
            f'<label><input type="checkbox" name="incremental" value="true"{checked_incremental}> Incremental indexing</label><br>',
            '<button type="submit">Save settings</button>',
            '</form>',
            '<form method="post" action="/index/run"><button type="submit">Run indexing now</button></form>',
            '<p class="muted">Background scheduling is intentionally an interface placeholder for MVP-A; the scheduler daemon can attach to these settings later.</p>',
            '</section>',
        ]
    )
    return _html_page(title="Settings — Local Docsher", body="\n".join(body_parts))


def _render_document_page(payload: dict[str, Any]) -> str:
    document = payload["document"]
    filename = str(document.get("filename") or f"Document {document.get('id')}")
    body_parts = [
        '<p><a href="/">← Back to search</a></p>',
        f"<h1>{escape(filename)}</h1>",
        '<section aria-label="Document metadata">',
        "<h2>Metadata</h2>",
        "<dl>",
    ]
    for key in (
        "id",
        "path",
        "filename",
        "extension",
        "size",
        "modified_at",
        "content_hash",
        "indexed_at",
        "status",
        "error_message",
        "parser_name",
        "ocr_status",
    ):
        value = document.get(key)
        body_parts.append(f"<dt>{escape(key)}</dt><dd>{escape('' if value is None else str(value))}</dd>")
    body_parts.extend(["</dl>", "</section>", '<section aria-label="Chunk previews">', "<h2>Chunk previews</h2>"])
    chunks = payload.get("chunks") or []
    if not chunks:
        body_parts.append('<div class="state" role="status">No chunks are available for this document.</div>')
    for chunk in chunks:
        location = escape(_format_location(chunk))
        preview = escape(str(chunk.get("text_preview") or ""))
        truncated = "…" if chunk.get("preview_truncated") else ""
        body_parts.append(
            '<article class="chunk">'
            f'<h2>Chunk {escape(str(chunk.get("chunk_index")))}</h2>'
            f'<div class="location">Location: {location}</div>'
            f'<div class="meta">Token count: {escape(str(chunk.get("token_count") or ""))}</div>'
            f'<div class="preview">{preview}{truncated}</div>'
            '</article>'
        )
    body_parts.append("</section>")
    return _html_page(title=f"{filename} — Local Docsher", body="\n".join(body_parts))


def _dataclass_to_dict(value: Any) -> dict[str, Any]:
    if is_dataclass(value):
        return asdict(value)
    raise TypeError(f"Expected dataclass value, got {type(value)!r}")


def _resolve_config_and_database(
    *,
    config_path: str | Path | None = None,
    database_path: str | Path | None = None,
) -> tuple[dict[str, Any], str | Path]:
    if config_path is not None:
        config = load_config(Path(config_path).expanduser())
    else:
        config, _location = load_config_with_location()

    resolved_database_path = database_path or config.get("storage", {}).get("database_path")
    if not resolved_database_path or not isinstance(resolved_database_path, (str, Path)):
        raise ValueError("Invalid config: storage.database_path must be a non-empty string")
    return config, resolved_database_path


def _resolve_config_for_write(config_path: str | Path | None = None) -> tuple[dict[str, Any], Path]:
    if config_path is not None:
        path = Path(config_path).expanduser()
        return load_config(path), path
    config, location = load_config_with_location()
    return config, location.path


def _coerce_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return value.lower() in {"1", "true", "yes", "on"}
    return bool(value)


def _normalize_schedule_payload(payload: dict[str, Any]) -> dict[str, Any]:
    allowed = {"schedule_enabled", "schedule", "time", "incremental"}
    unknown = sorted(set(payload) - allowed)
    if unknown:
        raise ValueError(f"Unknown schedule setting(s): {', '.join(unknown)}")
    normalized: dict[str, Any] = {}
    if "schedule_enabled" in payload:
        normalized["schedule_enabled"] = _coerce_bool(payload["schedule_enabled"])
    if "schedule" in payload:
        normalized["schedule"] = str(payload["schedule"])
    if "time" in payload:
        normalized["time"] = str(payload["time"])
    if "incremental" in payload:
        normalized["incremental"] = _coerce_bool(payload["incremental"])
    return normalized


def _fetch_document_payload(
    *,
    database_path: str | Path,
    document_id: int,
    preview_chars: int = DEFAULT_PREVIEW_CHARS,
) -> dict[str, Any] | None:
    resolved_database_path = init_database(database_path)
    with connect(resolved_database_path) as connection:
        connection.row_factory = sqlite3.Row
        document = connection.execute(
            """
            SELECT
                id, path, filename, extension, size, modified_at, content_hash,
                indexed_at, status, error_message, parser_name, ocr_status
            FROM documents
            WHERE id = ?
            """,
            (document_id,),
        ).fetchone()
        if document is None:
            return None

        chunks = connection.execute(
            """
            SELECT
                id, chunk_index, text, page_number, sheet_name, slide_number,
                section_title, token_count
            FROM chunks
            WHERE document_id = ?
            ORDER BY chunk_index, id
            """,
            (document_id,),
        ).fetchall()

    chunk_payloads: list[dict[str, Any]] = []
    for chunk in chunks:
        text = str(chunk["text"] or "")
        chunk_payloads.append(
            {
                "id": int(chunk["id"]),
                "chunk_index": int(chunk["chunk_index"]),
                "text_preview": text[:preview_chars],
                "preview_truncated": len(text) > preview_chars,
                "page_number": chunk["page_number"],
                "sheet_name": chunk["sheet_name"],
                "slide_number": chunk["slide_number"],
                "section_title": chunk["section_title"],
                "token_count": chunk["token_count"],
            }
        )

    return {
        "document": {key: document[key] for key in document.keys()},
        "chunks": chunk_payloads,
    }


def _scan_result_to_dict(result: Any) -> dict[str, Any]:
    payload = _dataclass_to_dict(result)
    payload["new_files_count"] = len(result.new_files)
    payload["modified_files_count"] = len(result.modified_files)
    payload["deleted_files_count"] = len(result.deleted_files)
    return payload


def create_app(
    *,
    config_path: str | Path | None = None,
    database_path: str | Path | None = None,
) -> FastAPI:
    """Create a configured FastAPI application instance."""

    app = FastAPI(
        title="Local Docsher API",
        version=__version__,
        description="Local API for document health, search, metadata, and indexing.",
    )
    app.state.config_path = str(config_path) if config_path is not None else None
    app.state.database_path = str(database_path) if database_path is not None else None

    @app.exception_handler(SearchError)
    async def search_error_handler(_request: Any, exc: SearchError) -> JSONResponse:
        return JSONResponse(status_code=400, content={"detail": str(exc)})

    @app.exception_handler(ValueError)
    async def value_error_handler(_request: Any, exc: ValueError) -> JSONResponse:
        return JSONResponse(status_code=400, content={"detail": str(exc)})

    @app.get("/", response_class=HTMLResponse)
    def search_ui(q: str = Query("", description="Search query")) -> HTMLResponse:
        query = q.strip()
        if not query:
            return HTMLResponse(_render_search_page(query=q))

        try:
            _config, resolved_database_path = _resolve_config_and_database(
                config_path=app.state.config_path,
                database_path=app.state.database_path,
            )
            results = search_documents(
                query,
                database_path=resolved_database_path,
                top_k=UI_SEARCH_TOP_K,
            )
        except (SearchError, ValueError) as exc:
            return HTMLResponse(
                _render_search_page(query=q, error=str(exc)),
                status_code=400,
            )
        return HTMLResponse(
            _render_search_page(
                query=query,
                results=[result.to_dict() for result in results],
            )
        )

    @app.get("/ui/documents/{document_id}", response_class=HTMLResponse)
    def document_ui(document_id: int) -> HTMLResponse:
        try:
            _config, resolved_database_path = _resolve_config_and_database(
                config_path=app.state.config_path,
                database_path=app.state.database_path,
            )
            payload = _fetch_document_payload(
                database_path=resolved_database_path,
                document_id=document_id,
                preview_chars=DEFAULT_PREVIEW_CHARS,
            )
        except ValueError as exc:
            return HTMLResponse(
                _html_page(
                    title="Local Docsher error",
                    body=f'<h1>Document error</h1><div class="state error" role="alert">{escape(str(exc))}</div>',
                ),
                status_code=400,
            )
        if payload is None:
            return HTMLResponse(
                _html_page(
                    title="Document not found — Local Docsher",
                    body='<p><a href="/">← Back to search</a></p><h1>Document not found</h1>'
                    '<div class="state error" role="alert">Document not found.</div>',
                ),
                status_code=404,
            )
        return HTMLResponse(_render_document_page(payload))

    @app.get("/health")
    def health() -> dict[str, Any]:
        config, resolved_database_path = _resolve_config_and_database(
            config_path=app.state.config_path,
            database_path=app.state.database_path,
        )
        roots = config.get("documents", {}).get("roots", [])
        return {
            "status": "ok",
            "version": __version__,
            "database_path": str(Path(resolved_database_path).expanduser()),
            "roots_count": len(roots) if isinstance(roots, list) else 0,
        }

    @app.get("/status")
    def status() -> dict[str, Any]:
        _config, resolved_database_path = _resolve_config_and_database(
            config_path=app.state.config_path,
            database_path=app.state.database_path,
        )
        return get_index_status(resolved_database_path).to_dict()

    @app.get("/ui/status", response_class=HTMLResponse)
    def status_ui() -> HTMLResponse:
        try:
            _config, resolved_database_path = _resolve_config_and_database(
                config_path=app.state.config_path,
                database_path=app.state.database_path,
            )
            payload = get_index_status(resolved_database_path).to_dict()
        except ValueError as exc:
            return HTMLResponse(
                _html_page(
                    title="Local Docsher error",
                    body=f'<h1>Status error</h1><div class="state error" role="alert">{escape(str(exc))}</div>',
                ),
                status_code=400,
            )
        return HTMLResponse(_render_status_page(payload))

    @app.get("/settings")
    def settings() -> dict[str, Any]:
        config, config_file = _resolve_config_for_write(app.state.config_path)
        return {
            "config_path": str(config_file),
            "indexing": get_indexing_schedule(config),
        }

    @app.post("/settings")
    def update_settings(payload: dict[str, Any] = Body(...)) -> dict[str, Any]:
        config, config_file = _resolve_config_for_write(app.state.config_path)
        updated = get_indexing_schedule(config)
        updated.update(_normalize_schedule_payload(payload))
        updated = validate_indexing_schedule(updated)
        config.setdefault("indexing", {}).update(updated)
        save_config(config, config_file)
        return {
            "status": "updated",
            "config_path": str(config_file),
            "indexing": updated,
        }

    @app.get("/ui/settings", response_class=HTMLResponse)
    def settings_ui() -> HTMLResponse:
        try:
            config, config_file = _resolve_config_for_write(app.state.config_path)
            schedule = get_indexing_schedule(config)
        except ValueError as exc:
            return HTMLResponse(
                _render_settings_page(schedule=get_indexing_schedule({}), config_path="", error=str(exc)),
                status_code=400,
            )
        return HTMLResponse(_render_settings_page(schedule=schedule, config_path=str(config_file)))

    @app.post("/ui/settings", response_class=HTMLResponse)
    async def update_settings_ui(request: Request) -> HTMLResponse:
        config, config_file = _resolve_config_for_write(app.state.config_path)
        form = await request.form()
        payload = {
            "schedule_enabled": "schedule_enabled" in form,
            "schedule": str(form.get("schedule", "daily")),
            "time": str(form.get("time", "03:00")),
            "incremental": "incremental" in form,
        }
        try:
            updated = validate_indexing_schedule(payload)
        except ValueError as exc:
            return HTMLResponse(
                _render_settings_page(
                    schedule=get_indexing_schedule({"indexing": payload}),
                    config_path=str(config_file),
                    error=str(exc),
                ),
                status_code=400,
            )
        config.setdefault("indexing", {}).update(updated)
        save_config(config, config_file)
        return HTMLResponse(
            _render_settings_page(
                schedule=updated,
                config_path=str(config_file),
                message="Settings saved.",
            )
        )

    @app.get("/search")
    def search(
        q: str = Query(..., min_length=1, description="Search query"),
        top_k: int = Query(10, ge=1, le=MAX_TOP_K),
        ext: str | None = Query(None, description="Extension filter, e.g. txt or .md"),
        path: str | None = Query(None, description="Stored path substring filter"),
    ) -> dict[str, Any]:
        _config, resolved_database_path = _resolve_config_and_database(
            config_path=app.state.config_path,
            database_path=app.state.database_path,
        )
        results = search_documents(
            q,
            database_path=resolved_database_path,
            extension=ext,
            path_filter=path,
            top_k=top_k,
        )
        return {
            "query": q,
            "count": len(results),
            "results": [result.to_dict() for result in results],
        }

    @app.post("/ask")
    def ask(payload: dict[str, Any] = Body(...)) -> dict[str, Any]:
        question = payload.get("question")
        if not isinstance(question, str) or not question.strip():
            raise HTTPException(status_code=422, detail="question must be a non-empty string")
        top_k_value = payload.get("top_k", 5)
        if not isinstance(top_k_value, int) or top_k_value < 1 or top_k_value > MAX_TOP_K:
            raise HTTPException(status_code=422, detail=f"top_k must be an integer between 1 and {MAX_TOP_K}")

        config, resolved_database_path = _resolve_config_and_database(
            config_path=app.state.config_path,
            database_path=app.state.database_path,
        )
        llm_client = getattr(app.state, "llm_client", None)
        if llm_client is None:
            llm_client = create_llm_client(config)
        try:
            response = ask_question(
                question,
                llm_client=llm_client,
                database_path=resolved_database_path,
                top_k=top_k_value,
            )
        except LLMClientError as exc:
            raise HTTPException(status_code=502, detail=str(exc)) from exc
        return response.to_dict()

    @app.get("/documents/{document_id}")
    def get_document(
        document_id: int,
        preview_chars: int = Query(DEFAULT_PREVIEW_CHARS, ge=1, le=MAX_PREVIEW_CHARS),
    ) -> dict[str, Any]:
        _config, resolved_database_path = _resolve_config_and_database(
            config_path=app.state.config_path,
            database_path=app.state.database_path,
        )
        payload = _fetch_document_payload(
            database_path=resolved_database_path,
            document_id=document_id,
            preview_chars=preview_chars,
        )
        if payload is None:
            raise HTTPException(status_code=404, detail="Document not found")
        return payload

    @app.post("/index/run")
    def run_index(payload: dict[str, Any] | None = Body(default=None)) -> dict[str, Any]:
        config, resolved_database_path = _resolve_config_and_database(
            config_path=app.state.config_path,
            database_path=app.state.database_path,
        )
        roots = None
        if payload is not None and "roots" in payload:
            if not isinstance(payload["roots"], list) or not all(
                isinstance(root, str) for root in payload["roots"]
            ):
                raise HTTPException(status_code=422, detail="roots must be a list of strings")
            roots = payload["roots"]

        result = run_scheduled_index_once(config, database_path=resolved_database_path, roots=roots)
        return {
            "status": "completed",
            "scan": _scan_result_to_dict(result.scan),
            "index": _dataclass_to_dict(result.index),
        }

    return app


app = create_app()
