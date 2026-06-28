"""SQLite database schema and migrations for Local Docsher."""

from __future__ import annotations

import sqlite3
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path

from docsher.config import load_config


@dataclass(frozen=True)
class Migration:
    """A single database schema migration."""

    version: int
    name: str
    statements: tuple[str, ...]


MIGRATIONS: tuple[Migration, ...] = (
    Migration(
        version=1,
        name="initial_schema",
        statements=(
            """
            CREATE TABLE IF NOT EXISTS documents (
                id INTEGER PRIMARY KEY,
                path TEXT NOT NULL UNIQUE,
                filename TEXT NOT NULL,
                extension TEXT,
                size INTEGER,
                modified_at TEXT,
                content_hash TEXT,
                indexed_at TEXT,
                status TEXT NOT NULL DEFAULT 'pending',
                error_message TEXT,
                parser_name TEXT,
                ocr_status TEXT DEFAULT 'not_required'
            )
            """,
            """
            CREATE INDEX IF NOT EXISTS idx_documents_status
            ON documents(status)
            """,
            """
            CREATE INDEX IF NOT EXISTS idx_documents_content_hash
            ON documents(content_hash)
            """,
            """
            CREATE TABLE IF NOT EXISTS chunks (
                id INTEGER PRIMARY KEY,
                document_id INTEGER NOT NULL,
                chunk_index INTEGER NOT NULL,
                text TEXT NOT NULL,
                page_number INTEGER,
                sheet_name TEXT,
                slide_number INTEGER,
                section_title TEXT,
                token_count INTEGER,
                FOREIGN KEY (document_id) REFERENCES documents(id) ON DELETE CASCADE,
                UNIQUE (document_id, chunk_index)
            )
            """,
            """
            CREATE INDEX IF NOT EXISTS idx_chunks_document_id
            ON chunks(document_id)
            """,
            """
            CREATE VIRTUAL TABLE IF NOT EXISTS chunks_fts
            USING fts5(
                text,
                filename,
                path,
                section_title
            )
            """,
            """
            CREATE TRIGGER IF NOT EXISTS chunks_ai AFTER INSERT ON chunks BEGIN
                INSERT INTO chunks_fts(rowid, text, filename, path, section_title)
                SELECT new.id, new.text, documents.filename, documents.path, new.section_title
                FROM documents
                WHERE documents.id = new.document_id;
            END
            """,
            """
            CREATE TRIGGER IF NOT EXISTS chunks_ad AFTER DELETE ON chunks BEGIN
                DELETE FROM chunks_fts WHERE rowid = old.id;
            END
            """,
            """
            CREATE TRIGGER IF NOT EXISTS chunks_au AFTER UPDATE ON chunks BEGIN
                DELETE FROM chunks_fts WHERE rowid = old.id;
                INSERT INTO chunks_fts(rowid, text, filename, path, section_title)
                SELECT new.id, new.text, documents.filename, documents.path, new.section_title
                FROM documents
                WHERE documents.id = new.document_id;
            END
            """,
            """
            CREATE TRIGGER IF NOT EXISTS documents_au_fts
            AFTER UPDATE OF filename, path ON documents BEGIN
                DELETE FROM chunks_fts
                WHERE rowid IN (SELECT id FROM chunks WHERE document_id = new.id);
                INSERT INTO chunks_fts(rowid, text, filename, path, section_title)
                SELECT chunks.id, chunks.text, new.filename, new.path, chunks.section_title
                FROM chunks
                WHERE chunks.document_id = new.id;
            END
            """,
            """
            CREATE TABLE IF NOT EXISTS document_summaries (
                document_id INTEGER PRIMARY KEY,
                summary TEXT,
                keywords TEXT,
                generated_at TEXT,
                model_name TEXT,
                FOREIGN KEY (document_id) REFERENCES documents(id) ON DELETE CASCADE
            )
            """,
            """
            CREATE INDEX IF NOT EXISTS idx_document_summaries_document_id
            ON document_summaries(document_id)
            """,
            """
            CREATE TABLE IF NOT EXISTS insight_reports (
                id INTEGER PRIMARY KEY,
                type TEXT NOT NULL,
                title TEXT NOT NULL,
                body_markdown TEXT NOT NULL,
                generated_at TEXT NOT NULL DEFAULT (datetime('now')),
                model_name TEXT
            )
            """,
            """
            CREATE INDEX IF NOT EXISTS idx_insight_reports_type_generated_at
            ON insight_reports(type, generated_at)
            """,
        ),
    ),
    Migration(
        version=2,
        name="lds003_schema_repair_marker",
        statements=(),
    ),
    Migration(
        version=3,
        name="ocr_jobs_queue",
        statements=(
            """
            CREATE TABLE IF NOT EXISTS ocr_jobs (
                id INTEGER PRIMARY KEY,
                document_id INTEGER NOT NULL,
                backend TEXT NOT NULL DEFAULT 'default',
                status TEXT NOT NULL DEFAULT 'queued',
                attempts INTEGER NOT NULL DEFAULT 0,
                error_message TEXT,
                result_text TEXT,
                created_at TEXT NOT NULL DEFAULT (datetime('now')),
                updated_at TEXT NOT NULL DEFAULT (datetime('now')),
                FOREIGN KEY (document_id) REFERENCES documents(id) ON DELETE CASCADE,
                UNIQUE(document_id, backend)
            )
            """,
            """
            CREATE INDEX IF NOT EXISTS idx_ocr_jobs_status
            ON ocr_jobs(status)
            """,
            """
            CREATE INDEX IF NOT EXISTS idx_ocr_jobs_document_id
            ON ocr_jobs(document_id)
            """,
        ),
    ),
    Migration(
        version=4,
        name="ocr_jobs_page_inputs",
        statements=(
            """
            CREATE TABLE IF NOT EXISTS ocr_jobs_v4 (
                id INTEGER PRIMARY KEY,
                document_id INTEGER NOT NULL,
                backend TEXT NOT NULL DEFAULT 'default',
                status TEXT NOT NULL DEFAULT 'queued',
                attempts INTEGER NOT NULL DEFAULT 0,
                error_message TEXT,
                result_text TEXT,
                input_path TEXT,
                page_number INTEGER NOT NULL DEFAULT 0,
                created_at TEXT NOT NULL DEFAULT (datetime('now')),
                updated_at TEXT NOT NULL DEFAULT (datetime('now')),
                FOREIGN KEY (document_id) REFERENCES documents(id) ON DELETE CASCADE,
                UNIQUE(document_id, backend, page_number)
            )
            """,
            """
            INSERT OR IGNORE INTO ocr_jobs_v4(
                id, document_id, backend, status, attempts, error_message,
                result_text, input_path, page_number, created_at, updated_at
            )
            SELECT id, document_id, backend, status, attempts, error_message,
                result_text, NULL, 0, created_at, updated_at
            FROM ocr_jobs
            """,
            "DROP TABLE IF EXISTS ocr_jobs",
            "ALTER TABLE ocr_jobs_v4 RENAME TO ocr_jobs",
            "CREATE INDEX IF NOT EXISTS idx_ocr_jobs_status ON ocr_jobs(status)",
            "CREATE INDEX IF NOT EXISTS idx_ocr_jobs_document_id ON ocr_jobs(document_id)",
        ),
    ),
)


OCR_JOB_COLUMN_DEFINITIONS: dict[str, str] = {
    "input_path": "TEXT",
    "page_number": "INTEGER NOT NULL DEFAULT 0",
}


DOCUMENT_COLUMN_DEFINITIONS: dict[str, str] = {
    "extension": "TEXT",
    "size": "INTEGER",
    "modified_at": "TEXT",
    "content_hash": "TEXT",
    "indexed_at": "TEXT",
    "status": "TEXT NOT NULL DEFAULT 'pending'",
    "error_message": "TEXT",
    "parser_name": "TEXT",
    "ocr_status": "TEXT DEFAULT 'not_required'",
}

CHUNK_COLUMN_DEFINITIONS: dict[str, str] = {
    "page_number": "INTEGER",
    "sheet_name": "TEXT",
    "slide_number": "INTEGER",
    "section_title": "TEXT",
    "token_count": "INTEGER",
}

REQUIRED_TABLE_COLUMNS: dict[str, set[str]] = {
    "documents": {
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
    },
    "chunks": {
        "id",
        "document_id",
        "chunk_index",
        "text",
        "page_number",
        "sheet_name",
        "slide_number",
        "section_title",
        "token_count",
    },
    "chunks_fts": {"text", "filename", "path", "section_title"},
    "document_summaries": {"document_id", "summary", "keywords", "generated_at", "model_name"},
    "insight_reports": {"id", "type", "title", "body_markdown", "generated_at", "model_name"},
    "ocr_jobs": {
        "id",
        "document_id",
        "backend",
        "status",
        "attempts",
        "error_message",
        "result_text",
        "input_path",
        "page_number",
        "created_at",
        "updated_at",
    },
    "schema_migrations": {"version", "name", "applied_at"},
}


def _table_exists(connection: sqlite3.Connection, table_name: str) -> bool:
    row = connection.execute(
        "SELECT 1 FROM sqlite_master WHERE type IN ('table', 'view') AND name = ?",
        (table_name,),
    ).fetchone()
    return row is not None


def _table_columns(connection: sqlite3.Connection, table_name: str) -> set[str]:
    rows = connection.execute(f"PRAGMA table_info({table_name})").fetchall()
    return {str(row[1]) for row in rows}


def _create_missing_core_tables(connection: sqlite3.Connection) -> None:
    """Create missing row tables before repairing columns and dependent objects."""

    for statement_index in (0, 3, 10, 12):
        connection.execute(MIGRATIONS[0].statements[statement_index])
    connection.execute(MIGRATIONS[2].statements[0])


def _create_dependent_schema_objects(connection: sqlite3.Connection) -> None:
    """Create indexes and triggers after repaired tables have the required columns."""

    for statement_index in (1, 2, 4, 6, 7, 8, 9, 11, 13):
        connection.execute(MIGRATIONS[0].statements[statement_index])
    for statement in MIGRATIONS[2].statements[1:]:
        connection.execute(statement)


def _add_missing_optional_columns(
    connection: sqlite3.Connection,
    table_name: str,
    column_definitions: dict[str, str],
) -> None:
    existing_columns = _table_columns(connection, table_name)
    for column_name, column_definition in column_definitions.items():
        if column_name not in existing_columns:
            connection.execute(
                f"ALTER TABLE {table_name} ADD COLUMN {column_name} {column_definition}"
            )


def _drop_fts_triggers(connection: sqlite3.Connection) -> None:
    for trigger_name in ("chunks_ai", "chunks_ad", "chunks_au", "documents_au_fts"):
        connection.execute(f"DROP TRIGGER IF EXISTS {trigger_name}")


def _ensure_chunks_fts_shape(connection: sqlite3.Connection) -> None:
    required_columns = REQUIRED_TABLE_COLUMNS["chunks_fts"]
    existing_columns = _table_columns(connection, "chunks_fts")
    if existing_columns and required_columns.issubset(existing_columns):
        return

    connection.execute("DROP TABLE IF EXISTS chunks_fts")
    connection.execute(
        """
        CREATE VIRTUAL TABLE chunks_fts
        USING fts5(
            text,
            filename,
            path,
            section_title
        )
        """
    )


def _backfill_chunks_fts(connection: sqlite3.Connection) -> None:
    """Refresh FTS rows from the current documents/chunks tables."""

    if not (_table_exists(connection, "chunks") and _table_exists(connection, "documents")):
        return

    chunks_columns = _table_columns(connection, "chunks")
    document_columns = _table_columns(connection, "documents")
    if not {"id", "document_id", "text", "section_title"}.issubset(
        chunks_columns
    ) or not {"id", "filename", "path"}.issubset(document_columns):
        return

    connection.execute("DELETE FROM chunks_fts")
    connection.execute(
        """
        INSERT INTO chunks_fts(rowid, text, filename, path, section_title)
        SELECT chunks.id, chunks.text, documents.filename, documents.path, chunks.section_title
        FROM chunks
        JOIN documents ON documents.id = chunks.document_id
        """
    )


def _repair_current_schema(connection: sqlite3.Connection) -> None:
    """Repair known LDS-003 schema drift from early version-1 databases."""

    with connection:
        _create_missing_core_tables(connection)
        _add_missing_optional_columns(
            connection, "documents", DOCUMENT_COLUMN_DEFINITIONS
        )
        _add_missing_optional_columns(connection, "chunks", CHUNK_COLUMN_DEFINITIONS)
        if _table_exists(connection, "ocr_jobs"):
            _add_missing_optional_columns(connection, "ocr_jobs", OCR_JOB_COLUMN_DEFINITIONS)
        _drop_fts_triggers(connection)
        _ensure_chunks_fts_shape(connection)
        _backfill_chunks_fts(connection)
        _create_dependent_schema_objects(connection)


def _validate_current_schema(connection: sqlite3.Connection) -> None:
    missing: list[str] = []
    for table_name, required_columns in REQUIRED_TABLE_COLUMNS.items():
        if not _table_exists(connection, table_name):
            missing.append(f"table {table_name}")
            continue
        missing_columns = sorted(required_columns - _table_columns(connection, table_name))
        if missing_columns:
            missing.append(f"columns {table_name}.({', '.join(missing_columns)})")

    if missing:
        details = "; ".join(missing)
        raise RuntimeError(f"Database schema validation failed; missing {details}")


def resolve_database_path(database_path: str | Path | None = None) -> Path:
    """Resolve the SQLite database path from an explicit path or active config."""

    if database_path is not None:
        return Path(database_path).expanduser()

    config = load_config()
    configured_path = config.get("storage", {}).get("database_path")
    if not configured_path or not isinstance(configured_path, str):
        raise ValueError("Invalid config: storage.database_path must be a non-empty string")
    return Path(configured_path).expanduser()


def connect(database_path: str | Path) -> sqlite3.Connection:
    """Open a SQLite connection with Local Docsher defaults."""

    connection = sqlite3.connect(Path(database_path))
    connection.execute("PRAGMA foreign_keys = ON")
    return connection


def _ensure_migrations_table(connection: sqlite3.Connection) -> None:
    connection.execute(
        """
        CREATE TABLE IF NOT EXISTS schema_migrations (
            version INTEGER PRIMARY KEY,
            name TEXT NOT NULL,
            applied_at TEXT NOT NULL DEFAULT (datetime('now'))
        )
        """
    )


def _applied_versions(connection: sqlite3.Connection) -> set[int]:
    rows = connection.execute("SELECT version FROM schema_migrations").fetchall()
    return {int(row[0]) for row in rows}


def apply_migrations(
    connection: sqlite3.Connection,
    migrations: Iterable[Migration] = MIGRATIONS,
) -> list[Migration]:
    """Apply missing migrations and return the migrations applied in this call."""

    _ensure_migrations_table(connection)
    applied_versions = _applied_versions(connection)
    applied_now: list[Migration] = []

    for migration in sorted(migrations, key=lambda item: item.version):
        if migration.version in applied_versions:
            continue
        with connection:
            for statement in migration.statements:
                connection.execute(statement)
            connection.execute(
                "INSERT INTO schema_migrations(version, name) VALUES (?, ?)",
                (migration.version, migration.name),
            )
        applied_versions.add(migration.version)
        applied_now.append(migration)

    _repair_current_schema(connection)
    _validate_current_schema(connection)

    return applied_now


def init_database(database_path: str | Path | None = None) -> Path:
    """Create or migrate the SQLite database and return its resolved path."""

    resolved_path = resolve_database_path(database_path)
    resolved_path.parent.mkdir(parents=True, exist_ok=True)
    with connect(resolved_path) as connection:
        apply_migrations(connection)
    return resolved_path
