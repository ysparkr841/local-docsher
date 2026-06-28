"""OCR backend interface and queue primitives for Local Docsher."""

from __future__ import annotations

import base64
import inspect
import json
import sqlite3
import urllib.error
import urllib.request
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Protocol

from docsher.chunker import chunk_text
from docsher.config import get_ocr_settings
from docsher.db import connect, init_database

OCR_STATUS_QUEUED = "queued"
OCR_STATUS_PROCESSING = "processing"
OCR_STATUS_COMPLETED = "completed"
OCR_STATUS_FAILED = "failed"
OCR_STATUS_NOT_REQUIRED = "not_required"
DEFAULT_OCR_BACKEND = "default"


class OCRBackendError(RuntimeError):
    """Raised when an optional OCR backend cannot run or parse its result."""


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


class PaddleOCRBackend:
    """PaddleOCR backend with offline model directory support.

    The implementation imports ``paddleocr`` lazily so Docsher remains usable in
    minimal/offline installs. For offline deployments, pre-download PaddleOCR
    detection/recognition/classification models and pass their local directories.
    """

    name = "paddle"

    def __init__(
        self,
        *,
        lang: str = "korean",
        det_model_dir: str | Path | None = None,
        rec_model_dir: str | Path | None = None,
        cls_model_dir: str | Path | None = None,
        use_angle_cls: bool = True,
        show_log: bool = False,
    ) -> None:
        self.lang = lang
        self.det_model_dir = str(det_model_dir) if det_model_dir is not None else None
        self.rec_model_dir = str(rec_model_dir) if rec_model_dir is not None else None
        self.cls_model_dir = str(cls_model_dir) if cls_model_dir is not None else None
        self.use_angle_cls = use_angle_cls
        self.show_log = show_log
        self._engine: Any | None = None
        self._import_error: Exception | None = None

    def _load_engine(self) -> Any:
        if self._engine is not None:
            return self._engine
        try:
            from paddleocr import PaddleOCR  # type: ignore[import-not-found]
        except Exception as exc:  # noqa: BLE001 - optional dependency boundary.
            self._import_error = exc
            raise OCRBackendError(
                "PaddleOCR backend unavailable: install the optional paddleocr dependency "
                "and provide local model directories for offline use."
            ) from exc

        options = self._build_paddle_options(PaddleOCR)
        try:
            self._engine = PaddleOCR(**options)
        except Exception as exc:  # noqa: BLE001 - optional dependency/model boundary.
            raise OCRBackendError(
                f"PaddleOCR backend unavailable: {exc}. Install paddleocr plus paddlepaddle, "
                "and provide local model directories for offline use."
            ) from exc
        return self._engine

    def _build_paddle_options(self, paddle_cls: object) -> dict[str, object]:
        signature = inspect.signature(paddle_cls)
        parameters = signature.parameters
        supports_v3_names = "text_detection_model_dir" in parameters
        options: dict[str, object] = {"lang": self.lang}
        if supports_v3_names:
            options["use_doc_orientation_classify"] = False
            options["use_doc_unwarping"] = False
            options["use_textline_orientation"] = self.use_angle_cls
            if self.det_model_dir:
                options["text_detection_model_dir"] = self.det_model_dir
            if self.rec_model_dir:
                options["text_recognition_model_dir"] = self.rec_model_dir
            if self.cls_model_dir:
                options["textline_orientation_model_dir"] = self.cls_model_dir
            return options

        options["use_angle_cls"] = self.use_angle_cls
        if "show_log" in parameters:
            options["show_log"] = self.show_log
        if self.det_model_dir:
            options["det_model_dir"] = self.det_model_dir
        if self.rec_model_dir:
            options["rec_model_dir"] = self.rec_model_dir
        if self.cls_model_dir:
            options["cls_model_dir"] = self.cls_model_dir
        return options

    def is_available(self) -> bool:
        try:
            self._load_engine()
        except OCRBackendError:
            return False
        return True

    def recognize(self, path: str | Path) -> OCRResult:
        image_path = Path(path).expanduser().resolve(strict=False)
        engine = self._load_engine()
        try:
            try:
                raw_result = engine.ocr(str(image_path), cls=True)
            except TypeError as exc:
                if "unexpected keyword argument 'cls'" not in str(exc):
                    raise
                raw_result = engine.ocr(str(image_path))
        except Exception as exc:  # noqa: BLE001 - optional backend boundary.
            raise OCRBackendError(f"PaddleOCR failed for {image_path}: {exc}") from exc

        lines = _extract_paddle_text_lines(raw_result)
        if not lines:
            raise OCRBackendError(f"PaddleOCR returned no text for {image_path}")
        return OCRResult(text="\n".join(lines), backend=self.name)


def _extract_paddle_text_lines(raw_result: object) -> list[str]:
    """Normalize PaddleOCR output across common package versions."""

    lines: list[str] = []

    def visit(value: object) -> None:
        if isinstance(value, tuple) and len(value) >= 2 and isinstance(value[0], str):
            lines.append(value[0])
            return
        if isinstance(value, list):
            if len(value) >= 2 and isinstance(value[1], tuple) and value[1] and isinstance(value[1][0], str):
                lines.append(value[1][0])
                return
            for item in value:
                visit(item)

    visit(raw_result)
    return [line.strip() for line in lines if line.strip()]


class UnlimitedOCRBackend:
    """Experimental backend for a local Unlimited-OCR OpenAI-compatible server.

    Unlimited-OCR is too large to run as an in-process optional dependency for
    the Docsher MVP. This backend targets the project's documented SGLang
    OpenAI-compatible server mode so Docsher can use it when the user has already
    launched the model locally.
    """

    name = "unlimited"

    def __init__(
        self,
        *,
        endpoint: str = "http://127.0.0.1:10000/v1/chat/completions",
        model: str = "Unlimited-OCR",
        prompt: str = "document parsing.",
        timeout_seconds: int = 1200,
    ) -> None:
        self.endpoint = endpoint
        self.model = model
        self.prompt = prompt
        self.timeout_seconds = timeout_seconds

    def is_available(self) -> bool:
        return self.endpoint.startswith(("http://", "https://"))

    def recognize(self, path: str | Path) -> OCRResult:
        image_path = Path(path).expanduser().resolve(strict=False)
        if not image_path.exists():
            raise OCRBackendError(f"Unlimited-OCR input not found: {image_path}")
        image_payload = _image_data_url(image_path)
        payload = {
            "model": self.model,
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": self.prompt},
                        {"type": "image_url", "image_url": {"url": image_payload}},
                    ],
                }
            ],
            "temperature": 0,
            "stream": False,
        }
        request = urllib.request.Request(
            self.endpoint,
            data=json.dumps(payload).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        try:
            with urllib.request.urlopen(request, timeout=self.timeout_seconds) as response:  # noqa: S310 - user-configured local endpoint.
                response_payload = json.loads(response.read().decode("utf-8"))
        except (OSError, urllib.error.URLError, json.JSONDecodeError) as exc:
            raise OCRBackendError(
                f"Unlimited-OCR backend unavailable at {self.endpoint}: {exc}"
            ) from exc
        text = _extract_openai_message_text(response_payload)
        if not text:
            raise OCRBackendError("Unlimited-OCR returned no text")
        return OCRResult(text=text, backend=self.name)


def _image_data_url(path: Path) -> str:
    extension = path.suffix.lower().lstrip(".") or "png"
    mime = "image/jpeg" if extension in {"jpg", "jpeg"} else f"image/{extension}"
    return f"data:{mime};base64,{base64.b64encode(path.read_bytes()).decode('ascii')}"


def _extract_openai_message_text(payload: object) -> str:
    if not isinstance(payload, dict):
        return ""
    choices = payload.get("choices")
    if not isinstance(choices, list) or not choices:
        return ""
    first = choices[0]
    if not isinstance(first, dict):
        return ""
    message = first.get("message")
    if not isinstance(message, dict):
        return ""
    content = message.get("content")
    if isinstance(content, str):
        return content.strip()
    if isinstance(content, list):
        parts: list[str] = []
        for item in content:
            if isinstance(item, dict) and isinstance(item.get("text"), str):
                parts.append(item["text"])
        return "\n".join(part.strip() for part in parts if part.strip())
    return ""


def create_ocr_backend(config: dict[str, Any] | None = None, *, backend_name: str | None = None) -> OCRBackend:
    """Create an OCR backend from configuration or an explicit backend name."""

    settings = get_ocr_settings(config or {})
    selected = (backend_name or settings["backend"] or "fake").lower()
    if selected in {"fake", DEFAULT_OCR_BACKEND}:
        return FakeOCRBackend()
    if selected in {"paddle", "paddleocr"}:
        paddle = settings["paddle"]
        return PaddleOCRBackend(
            lang=str(paddle["lang"]),
            det_model_dir=paddle["det_model_dir"],
            rec_model_dir=paddle["rec_model_dir"],
            cls_model_dir=paddle["cls_model_dir"],
            use_angle_cls=bool(paddle["use_angle_cls"]),
        )
    if selected in {"unlimited", "unlimited-ocr"}:
        unlimited = settings["unlimited"]
        return UnlimitedOCRBackend(
            endpoint=str(unlimited["endpoint"]),
            model=str(unlimited["model"]),
            prompt=str(unlimited["prompt"]),
            timeout_seconds=int(unlimited["timeout_seconds"]),
        )
    raise ValueError(f"Unsupported OCR backend: {selected}")


def _utc_now_iso() -> str:
    return datetime.now(UTC).isoformat()


def _load_cached_ocr_result(
    connection: sqlite3.Connection,
    *,
    content_hash: str | None,
    backend: str,
    page_number: int,
) -> OCRResult | None:
    """Return a cached OCR result for the same content hash/backend/page, if any."""

    if not content_hash:
        return None
    row = connection.execute(
        """
        SELECT result_text
        FROM ocr_result_cache
        WHERE content_hash = ? AND backend = ? AND page_number = ?
        """,
        (content_hash, backend, page_number),
    ).fetchone()
    if row is None:
        return None
    stored_page_number = page_number if page_number > 0 else None
    return OCRResult(text=str(row["result_text"]), backend=backend, page_number=stored_page_number)


def _cache_ocr_result(
    connection: sqlite3.Connection,
    *,
    content_hash: str | None,
    result: OCRResult,
    backend: str | None = None,
    page_number: int | None = None,
) -> None:
    """Store OCR output so future files with the same hash can reuse it."""

    if not content_hash:
        return
    cache_backend = backend or result.backend
    cache_page_number = page_number if page_number is not None else result.page_number
    stored_page_number = cache_page_number if cache_page_number and cache_page_number > 0 else 0
    connection.execute(
        """
        INSERT INTO ocr_result_cache(content_hash, backend, page_number, result_text)
        VALUES (?, ?, ?, ?)
        ON CONFLICT(content_hash, backend, page_number) DO UPDATE SET
            result_text = excluded.result_text,
            updated_at = datetime('now')
        """,
        (content_hash, cache_backend, stored_page_number, result.text),
    )


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
            SELECT ocr_jobs.*, documents.path AS document_path, documents.content_hash AS content_hash
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

        content_hash = row["content_hash"]
        result = _load_cached_ocr_result(
            connection,
            content_hash=content_hash,
            backend=backend.name,
            page_number=queued_page_number,
        )
        if result is None:
            try:
                if not backend.is_available():
                    raise RuntimeError(f"OCR backend unavailable: {backend.name}")
                result = backend.recognize(document_path)
                result_page_number = (
                    queued_page_number
                    if queued_page_number > 0
                    else result.page_number
                )
                result = OCRResult(
                    text=result.text,
                    backend=backend.name,
                    page_number=result_page_number,
                )
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
                result = None

        if result is not None:
            with connection:
                _store_ocr_result_chunks(connection, document_id=document_id, result=result)
                _cache_ocr_result(
                    connection,
                    content_hash=content_hash,
                    result=result,
                    backend=backend.name,
                    page_number=queued_page_number,
                )
                connection.execute(
                    """
                    UPDATE ocr_jobs
                    SET status = ?, attempts = ?, error_message = NULL, result_text = ?, updated_at = datetime('now')
                    WHERE id = ?
                    """,
                    (OCR_STATUS_COMPLETED, attempts, result.text, job_id),
                )
                active_jobs = int(
                    connection.execute(
                        """
                        SELECT COUNT(*) FROM ocr_jobs
                        WHERE document_id = ? AND id != ? AND status IN (?, ?)
                        """,
                        (document_id, job_id, OCR_STATUS_QUEUED, OCR_STATUS_PROCESSING),
                    ).fetchone()[0]
                )
                failed_jobs = int(
                    connection.execute(
                        """
                        SELECT COUNT(*) FROM ocr_jobs
                        WHERE document_id = ? AND status = ?
                        """,
                        (document_id, OCR_STATUS_FAILED),
                    ).fetchone()[0]
                )
                if failed_jobs > 0:
                    connection.execute(
                        """
                        UPDATE documents
                        SET status = 'pending', parser_name = ?, ocr_status = ?
                        WHERE id = ?
                        """,
                        (f"ocr:{backend.name}", OCR_STATUS_FAILED, document_id),
                    )
                elif active_jobs == 0:
                    connection.execute(
                        """
                        UPDATE documents
                        SET status = 'indexed', indexed_at = ?, error_message = NULL,
                            parser_name = ?, ocr_status = ?
                        WHERE id = ?
                        """,
                        (_utc_now_iso(), f"ocr:{backend.name}", OCR_STATUS_COMPLETED, document_id),
                    )
                else:
                    connection.execute(
                        """
                        UPDATE documents
                        SET status = 'pending', parser_name = ?, ocr_status = ?
                        WHERE id = ?
                        """,
                        (f"ocr:{backend.name}", OCR_STATUS_QUEUED, document_id),
                    )

        final_row = connection.execute("SELECT * FROM ocr_jobs WHERE id = ?", (job_id,)).fetchone()
    return _row_to_job(final_row)
