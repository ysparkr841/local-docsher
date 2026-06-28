from __future__ import annotations

import json
import os
import sqlite3
import subprocess
import sys
from pathlib import Path

import pytest

from docsher.config import ENV_CONFIG_PATH
from docsher.db import init_database


REQUIRED_TABLES = {
    "documents",
    "chunks",
    "chunks_fts",
    "document_summaries",
    "insight_reports",
    "ocr_jobs",
    "ocr_result_cache",
    "schema_migrations",
}


def table_names(database_path: Path) -> set[str]:
    with sqlite3.connect(database_path) as connection:
        rows = connection.execute(
            "SELECT name FROM sqlite_master WHERE type = 'table'"
        ).fetchall()
    return {row[0] for row in rows}


def test_init_database_creates_required_schema_in_temp_db(tmp_path: Path) -> None:
    database_path = tmp_path / "docsher.sqlite3"

    returned_path = init_database(database_path)

    assert returned_path == database_path
    assert database_path.exists()
    assert REQUIRED_TABLES.issubset(table_names(database_path))


@pytest.mark.parametrize(
    ("table", "expected_columns"),
    [
        (
            "documents",
            {
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
        ),
        (
            "chunks",
            {
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
        ),
        (
            "document_summaries",
            {"document_id", "summary", "keywords", "generated_at", "model_name"},
        ),
        (
            "insight_reports",
            {"id", "type", "title", "body_markdown", "generated_at", "model_name"},
        ),
        (
            "ocr_jobs",
            {
                "id",
                "document_id",
                "backend",
                "status",
                "attempts",
                "error_message",
                "result_text",
                "created_at",
                "updated_at",
            },
        ),
        (
            "ocr_result_cache",
            {
                "id",
                "content_hash",
                "backend",
                "page_number",
                "result_text",
                "created_at",
                "updated_at",
            },
        ),
    ],
)
def test_core_tables_have_expected_columns(
    tmp_path: Path, table: str, expected_columns: set[str]
) -> None:
    database_path = init_database(tmp_path / "docsher.sqlite3")

    with sqlite3.connect(database_path) as connection:
        rows = connection.execute(f"PRAGMA table_info({table})").fetchall()

    columns = {row[1] for row in rows}
    assert expected_columns.issubset(columns)


def test_chunks_fts_has_plan_aligned_search_columns(tmp_path: Path) -> None:
    database_path = init_database(tmp_path / "docsher.sqlite3")

    with sqlite3.connect(database_path) as connection:
        rows = connection.execute("PRAGMA table_info(chunks_fts)").fetchall()

    columns = {row[1] for row in rows}
    assert {"text", "filename", "path", "section_title"}.issubset(columns)


def test_init_database_is_idempotent(tmp_path: Path) -> None:
    database_path = tmp_path / "docsher.sqlite3"

    init_database(database_path)
    init_database(database_path)

    with sqlite3.connect(database_path) as connection:
        migration_rows = connection.execute(
            "SELECT version, name FROM schema_migrations ORDER BY version"
        ).fetchall()
        documents_count = connection.execute("SELECT COUNT(*) FROM documents").fetchone()[0]

    assert migration_rows == [
        (1, "initial_schema"),
        (2, "lds003_schema_repair_marker"),
        (3, "ocr_jobs_queue"),
        (4, "ocr_jobs_page_inputs"),
        (5, "ocr_result_cache"),
    ]
    assert documents_count == 0


def test_init_database_repairs_stale_v1_schema(tmp_path: Path) -> None:
    database_path = tmp_path / "stale.sqlite3"
    with sqlite3.connect(database_path) as connection:
        connection.executescript(
            """
            CREATE TABLE schema_migrations (
                version INTEGER PRIMARY KEY,
                name TEXT NOT NULL,
                applied_at TEXT NOT NULL DEFAULT (datetime('now'))
            );
            INSERT INTO schema_migrations(version, name) VALUES (1, 'initial_schema');

            CREATE TABLE documents (
                id INTEGER PRIMARY KEY,
                path TEXT NOT NULL UNIQUE,
                filename TEXT NOT NULL,
                status TEXT NOT NULL DEFAULT 'pending'
            );
            INSERT INTO documents(path, filename, status)
            VALUES ('/tmp/example.md', 'example.md', 'indexed');

            CREATE TABLE chunks (
                id INTEGER PRIMARY KEY,
                document_id INTEGER NOT NULL,
                chunk_index INTEGER NOT NULL,
                text TEXT NOT NULL,
                FOREIGN KEY (document_id) REFERENCES documents(id) ON DELETE CASCADE,
                UNIQUE (document_id, chunk_index)
            );
            INSERT INTO chunks(document_id, chunk_index, text)
            VALUES (1, 0, 'stale fts schema should be repaired');

            CREATE VIRTUAL TABLE chunks_fts USING fts5(text);
            """
        )

    returned_path = init_database(database_path)

    assert returned_path == database_path
    assert REQUIRED_TABLES.issubset(table_names(database_path))
    with sqlite3.connect(database_path) as connection:
        migration_rows = connection.execute(
            "SELECT version, name FROM schema_migrations ORDER BY version"
        ).fetchall()
        documents_columns = {
            row[1] for row in connection.execute("PRAGMA table_info(documents)").fetchall()
        }
        chunks_columns = {
            row[1] for row in connection.execute("PRAGMA table_info(chunks)").fetchall()
        }
        fts_columns = {
            row[1] for row in connection.execute("PRAGMA table_info(chunks_fts)").fetchall()
        }
        fts_rows = connection.execute(
            "SELECT rowid, text, filename, path, section_title FROM chunks_fts"
        ).fetchall()

    assert migration_rows == [
        (1, "initial_schema"),
        (2, "lds003_schema_repair_marker"),
        (3, "ocr_jobs_queue"),
        (4, "ocr_jobs_page_inputs"),
        (5, "ocr_result_cache"),
    ]
    assert {
        "extension",
        "size",
        "modified_at",
        "content_hash",
        "indexed_at",
        "error_message",
        "parser_name",
        "ocr_status",
    }.issubset(documents_columns)
    assert {
        "page_number",
        "sheet_name",
        "slide_number",
        "section_title",
        "token_count",
    }.issubset(chunks_columns)
    assert {"text", "filename", "path", "section_title"}.issubset(fts_columns)
    assert fts_rows == [
        (1, "stale fts schema should be repaired", "example.md", "/tmp/example.md", None)
    ]


def test_init_database_replaces_shape_correct_stale_fts_triggers(
    tmp_path: Path,
) -> None:
    database_path = tmp_path / "stale_trigger.sqlite3"
    with sqlite3.connect(database_path) as connection:
        connection.executescript(
            """
            CREATE TABLE schema_migrations (
                version INTEGER PRIMARY KEY,
                name TEXT NOT NULL,
                applied_at TEXT NOT NULL DEFAULT (datetime('now'))
            );
            INSERT INTO schema_migrations(version, name) VALUES (1, 'initial_schema');

            CREATE TABLE documents (
                id INTEGER PRIMARY KEY,
                path TEXT NOT NULL UNIQUE,
                filename TEXT NOT NULL,
                status TEXT NOT NULL DEFAULT 'pending'
            );
            CREATE TABLE chunks (
                id INTEGER PRIMARY KEY,
                document_id INTEGER NOT NULL,
                chunk_index INTEGER NOT NULL,
                text TEXT NOT NULL,
                section_title TEXT,
                FOREIGN KEY (document_id) REFERENCES documents(id) ON DELETE CASCADE,
                UNIQUE (document_id, chunk_index)
            );
            CREATE VIRTUAL TABLE chunks_fts
            USING fts5(text, filename, path, section_title);
            CREATE TRIGGER chunks_ai AFTER INSERT ON chunks BEGIN
                INSERT INTO chunks_fts(rowid, text) VALUES (new.id, new.text);
            END;
            """
        )

    init_database(database_path)

    with sqlite3.connect(database_path) as connection:
        cursor = connection.execute(
            "INSERT INTO documents(path, filename, status) VALUES (?, ?, ?)",
            (str(tmp_path / "policies" / "fresh.md"), "fresh.md", "indexed"),
        )
        document_id = cursor.lastrowid
        connection.execute(
            """
            INSERT INTO chunks(document_id, chunk_index, text, section_title)
            VALUES (?, ?, ?, ?)
            """,
            (document_id, 0, "new chunk should get full fts context", "Fresh Section"),
        )
        fts_rows = connection.execute(
            "SELECT rowid, text, filename, path, section_title FROM chunks_fts"
        ).fetchall()
        filename_matches = connection.execute(
            "SELECT rowid FROM chunks_fts WHERE chunks_fts MATCH ?",
            ("filename:fresh",),
        ).fetchall()

    assert fts_rows == [
        (
            1,
            "new chunk should get full fts context",
            "fresh.md",
            str(tmp_path / "policies" / "fresh.md"),
            "Fresh Section",
        )
    ]
    assert filename_matches == [(1,)]


def test_init_database_backfills_shape_correct_empty_fts(
    tmp_path: Path,
) -> None:
    database_path = tmp_path / "empty_fts.sqlite3"
    document_path = str(tmp_path / "runbooks" / "restore-guide.md")
    with sqlite3.connect(database_path) as connection:
        connection.executescript(
            """
            CREATE TABLE schema_migrations (
                version INTEGER PRIMARY KEY,
                name TEXT NOT NULL,
                applied_at TEXT NOT NULL DEFAULT (datetime('now'))
            );
            INSERT INTO schema_migrations(version, name) VALUES (1, 'initial_schema');

            CREATE TABLE documents (
                id INTEGER PRIMARY KEY,
                path TEXT NOT NULL UNIQUE,
                filename TEXT NOT NULL,
                status TEXT NOT NULL DEFAULT 'pending'
            );
            CREATE TABLE chunks (
                id INTEGER PRIMARY KEY,
                document_id INTEGER NOT NULL,
                chunk_index INTEGER NOT NULL,
                text TEXT NOT NULL,
                section_title TEXT,
                FOREIGN KEY (document_id) REFERENCES documents(id) ON DELETE CASCADE,
                UNIQUE (document_id, chunk_index)
            );
            CREATE VIRTUAL TABLE chunks_fts
            USING fts5(text, filename, path, section_title);
            """
        )
        connection.execute(
            "INSERT INTO documents(path, filename, status) VALUES (?, ?, ?)",
            (document_path, "restore-guide.md", "indexed"),
        )
        connection.execute(
            """
            INSERT INTO chunks(document_id, chunk_index, text, section_title)
            VALUES (?, ?, ?, ?)
            """,
            (1, 0, "backup recovery checklist", "Disaster Recovery"),
        )

    init_database(database_path)

    with sqlite3.connect(database_path) as connection:
        search_results = {
            query: connection.execute(
                """
                SELECT rowid, text, filename, path, section_title
                FROM chunks_fts
                WHERE chunks_fts MATCH ?
                """,
                (query,),
            ).fetchall()
            for query in (
                "text:checklist",
                "filename:restore",
                "path:runbooks",
                "section_title:Recovery",
            )
        }

    expected = [
        (
            1,
            "backup recovery checklist",
            "restore-guide.md",
            document_path,
            "Disaster Recovery",
        )
    ]
    assert search_results == {query: expected for query in search_results}


def test_chunks_fts_table_indexes_text_and_document_context(tmp_path: Path) -> None:
    database_path = init_database(tmp_path / "docsher.sqlite3")

    with sqlite3.connect(database_path) as connection:
        connection.execute("PRAGMA foreign_keys = ON")
        cursor = connection.execute(
            """
            INSERT INTO documents(path, filename, extension, size, modified_at, status)
            VALUES (?, ?, ?, ?, ?, ?)
            """,
            (
                str(tmp_path / "policies" / "handbook.md"),
                "handbook.md",
                ".md",
                123,
                "2026-06-24T00:00:00Z",
                "indexed",
            ),
        )
        document_id = cursor.lastrowid
        connection.execute(
            """
            INSERT INTO chunks(document_id, chunk_index, text, section_title, token_count)
            VALUES (?, ?, ?, ?, ?)
            """,
            (
                document_id,
                0,
                "local docsher sqlite full text search",
                "Search Architecture",
                6,
            ),
        )

        search_results = {
            query: connection.execute(
                """
                SELECT rowid, text, filename, path, section_title
                FROM chunks_fts
                WHERE chunks_fts MATCH ?
                """,
                (query,),
            ).fetchall()
            for query in (
                "text:sqlite",
                "filename:handbook",
                "path:policies",
                "section_title:Architecture",
            )
        }

    expected = [
        (
            1,
            "local docsher sqlite full text search",
            "handbook.md",
            str(tmp_path / "policies" / "handbook.md"),
            "Search Architecture",
        )
    ]
    assert search_results == {query: expected for query in search_results}


def test_chunks_fts_updates_document_filename_and_path_context(tmp_path: Path) -> None:
    database_path = init_database(tmp_path / "docsher.sqlite3")

    with sqlite3.connect(database_path) as connection:
        connection.execute("PRAGMA foreign_keys = ON")
        cursor = connection.execute(
            "INSERT INTO documents(path, filename, status) VALUES (?, ?, ?)",
            (str(tmp_path / "draft.md"), "draft.md", "indexed"),
        )
        document_id = cursor.lastrowid
        connection.execute(
            "INSERT INTO chunks(document_id, chunk_index, text) VALUES (?, ?, ?)",
            (document_id, 0, "policy review notes"),
        )
        connection.execute(
            "UPDATE documents SET path = ?, filename = ? WHERE id = ?",
            (str(tmp_path / "published" / "final.md"), "final.md", document_id),
        )

        old_filename_rows = connection.execute(
            "SELECT rowid FROM chunks_fts WHERE chunks_fts MATCH ?",
            ("filename:draft",),
        ).fetchall()
        new_filename_rows = connection.execute(
            "SELECT rowid, filename, path FROM chunks_fts WHERE chunks_fts MATCH ?",
            ("filename:final",),
        ).fetchall()

    assert old_filename_rows == []
    assert new_filename_rows == [
        (1, "final.md", str(tmp_path / "published" / "final.md"))
    ]


def test_init_database_repairs_stale_ocr_result_cache_missing_unique_key(tmp_path: Path) -> None:
    database_path = tmp_path / "stale_cache.sqlite3"
    with sqlite3.connect(database_path) as connection:
        connection.executescript(
            """
            CREATE TABLE schema_migrations (
                version INTEGER PRIMARY KEY,
                name TEXT NOT NULL,
                applied_at TEXT NOT NULL DEFAULT (datetime('now'))
            );
            INSERT INTO schema_migrations(version, name) VALUES
                (1, 'initial_schema'),
                (2, 'lds003_schema_repair_marker'),
                (3, 'ocr_jobs_queue'),
                (4, 'ocr_jobs_page_inputs'),
                (5, 'ocr_result_cache');
            CREATE TABLE ocr_result_cache (
                id INTEGER PRIMARY KEY,
                content_hash TEXT NOT NULL,
                backend TEXT NOT NULL,
                page_number INTEGER NOT NULL DEFAULT 0,
                result_text TEXT NOT NULL,
                created_at TEXT NOT NULL DEFAULT (datetime('now')),
                updated_at TEXT NOT NULL DEFAULT (datetime('now'))
            );
            """
        )

    init_database(database_path)

    from docsher.ocr import OCRResult, _cache_ocr_result

    with sqlite3.connect(database_path) as connection:
        _cache_ocr_result(
            connection,
            content_hash="same-content",
            result=OCRResult(text="first text", backend="fake", page_number=1),
        )
        _cache_ocr_result(
            connection,
            content_hash="same-content",
            result=OCRResult(text="updated text", backend="fake", page_number=1),
        )
        cache_rows = connection.execute(
            "SELECT content_hash, backend, page_number, result_text FROM ocr_result_cache"
        ).fetchall()
        has_unique_key = False
        for index_row in connection.execute("PRAGMA index_list(ocr_result_cache)").fetchall():
            if not bool(index_row[2]):
                continue
            index_columns = tuple(
                row[2]
                for row in connection.execute(f"PRAGMA index_info({index_row[1]})").fetchall()
            )
            has_unique_key = has_unique_key or index_columns == (
                "content_hash",
                "backend",
                "page_number",
            )

    assert cache_rows == [("same-content", "fake", 1, "updated text")]
    assert has_unique_key


def test_init_database_uses_configured_storage_path(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config_path = tmp_path / "config.json"
    database_path = tmp_path / "configured.sqlite3"
    config_path.write_text(
        json.dumps({"storage": {"database_path": str(database_path)}}),
        encoding="utf-8",
    )
    monkeypatch.setenv(ENV_CONFIG_PATH, str(config_path))

    returned_path = init_database()

    assert returned_path == database_path
    assert REQUIRED_TABLES.issubset(table_names(database_path))


def test_cli_db_init_supports_explicit_database_path(tmp_path: Path) -> None:
    config_path = tmp_path / "config.json"
    database_path = tmp_path / "cli.sqlite3"
    env = os.environ.copy()
    env[ENV_CONFIG_PATH] = str(config_path)

    completed = subprocess.run(
        [sys.executable, "-m", "docsher", "db", "init", "--database-path", str(database_path)],
        check=False,
        capture_output=True,
        text=True,
        cwd=tmp_path,
        env=env,
    )

    assert completed.returncode == 0, completed.stderr
    assert f"Initialized database: {database_path}" in completed.stdout
    assert REQUIRED_TABLES.issubset(table_names(database_path))
