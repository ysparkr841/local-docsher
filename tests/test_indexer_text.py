from __future__ import annotations

import json
import os
import sqlite3
import subprocess
import sys
from pathlib import Path

from docsher.config import ENV_CONFIG_PATH, default_config
from docsher.indexer import index_pending_text_documents
from docsher.scanner import scan


def make_config(tmp_path: Path, root: Path) -> dict:
    config = default_config()
    config["documents"]["roots"] = [str(root)]
    config["storage"]["database_path"] = str(tmp_path / "docsher.sqlite3")
    return config


def fetch_document(connection: sqlite3.Connection, filename: str) -> sqlite3.Row:
    connection.row_factory = sqlite3.Row
    row = connection.execute(
        """
        SELECT id, path, filename, status, error_message, indexed_at, parser_name, ocr_status
        FROM documents
        WHERE filename = ?
        """,
        (filename,),
    ).fetchone()
    assert row is not None
    return row


def test_indexer_parses_pending_txt_and_md_into_chunks_and_fts(tmp_path: Path) -> None:
    root = tmp_path / "docs"
    root.mkdir()
    txt = root / "notes.txt"
    txt.write_bytes("alpha beta\n\n검색 대상 텍스트".encode("utf-8"))
    md = root / "guide.md"
    md.write_bytes("# Guide\n\nmarkdown body".encode("utf-8"))
    config = make_config(tmp_path, root)
    database_path = Path(config["storage"]["database_path"])

    scan_result = scan(config)
    index_result = index_pending_text_documents(database_path)

    assert [change.action for change in scan_result.changes] == ["new", "new"]
    assert index_result.parsed_documents == 2
    assert index_result.failed_documents == 0
    assert index_result.created_chunks == 2
    with sqlite3.connect(database_path) as connection:
        txt_row = fetch_document(connection, "notes.txt")
        md_row = fetch_document(connection, "guide.md")
        chunks = [tuple(row) for row in connection.execute(
            """
            SELECT documents.filename, chunks.chunk_index, chunks.text, chunks.token_count
            FROM chunks
            JOIN documents ON documents.id = chunks.document_id
            ORDER BY documents.filename, chunks.chunk_index
            """
        ).fetchall()]
        fts_matches = [tuple(row) for row in connection.execute(
            "SELECT filename FROM chunks_fts WHERE chunks_fts MATCH ? ORDER BY filename",
            ("text:markdown OR text:alpha",),
        ).fetchall()]

    assert txt_row["status"] == "indexed"
    assert txt_row["indexed_at"] is not None
    assert txt_row["error_message"] is None
    assert txt_row["parser_name"] == "text"
    assert txt_row["ocr_status"] == "not_required"
    assert md_row["status"] == "indexed"
    assert chunks == [
        ("guide.md", 0, "# Guide\n\nmarkdown body", 5),
        ("notes.txt", 0, "alpha beta\n\n검색 대상 텍스트", 5),
    ]
    assert fts_matches == [("guide.md",), ("notes.txt",)]


def test_indexer_marks_empty_document_indexed_with_zero_chunks(tmp_path: Path) -> None:
    root = tmp_path / "docs"
    root.mkdir()
    empty = root / "empty.txt"
    empty.write_text("", encoding="utf-8")
    config = make_config(tmp_path, root)
    database_path = Path(config["storage"]["database_path"])

    scan(config)
    result = index_pending_text_documents(database_path)

    assert result.parsed_documents == 1
    assert result.created_chunks == 0
    with sqlite3.connect(database_path) as connection:
        row = fetch_document(connection, "empty.txt")
        chunk_count = connection.execute("SELECT COUNT(*) FROM chunks").fetchone()[0]
    assert row["status"] == "indexed"
    assert row["error_message"] is None
    assert chunk_count == 0


def test_indexer_records_failure_for_too_large_document_and_continues(tmp_path: Path) -> None:
    root = tmp_path / "docs"
    root.mkdir()
    (root / "large.txt").write_text("0123456789", encoding="utf-8")
    (root / "ok.md").write_text("tiny", encoding="utf-8")
    config = make_config(tmp_path, root)
    database_path = Path(config["storage"]["database_path"])

    scan(config)
    result = index_pending_text_documents(database_path, max_text_bytes=5)

    assert result.parsed_documents == 1
    assert result.failed_documents == 1
    with sqlite3.connect(database_path) as connection:
        large_row = fetch_document(connection, "large.txt")
        ok_row = fetch_document(connection, "ok.md")
        chunk_count = connection.execute("SELECT COUNT(*) FROM chunks").fetchone()[0]
    assert large_row["status"] == "failed"
    assert "too large" in large_row["error_message"]
    assert large_row["indexed_at"] is None
    assert ok_row["status"] == "indexed"
    assert chunk_count == 1


def test_cli_index_apply_scans_and_indexes_text_documents(tmp_path: Path) -> None:
    root = tmp_path / "sample_docs_text"
    root.mkdir()
    doc = root / "sample.md"
    doc.write_text("CLI searchable sample", encoding="utf-8")
    config = make_config(tmp_path, root)
    database_path = Path(config["storage"]["database_path"])
    config_path = tmp_path / "config.json"
    config_path.write_text(json.dumps(config), encoding="utf-8")
    env = os.environ.copy()
    env[ENV_CONFIG_PATH] = str(config_path)

    completed = subprocess.run(
        [sys.executable, "-m", "docsher", "index", "--root", str(root)],
        cwd=Path(__file__).resolve().parents[1],
        env=env,
        check=False,
        capture_output=True,
        text=True,
    )

    assert completed.returncode == 0, completed.stderr
    assert "Index scan (APPLIED)" in completed.stdout
    assert "Document indexing:" in completed.stdout
    assert "Parsed documents: 1" in completed.stdout
    assert "Created chunks: 1" in completed.stdout
    with sqlite3.connect(database_path) as connection:
        status = connection.execute("SELECT status FROM documents WHERE filename = 'sample.md'").fetchone()[0]
        chunk_text = connection.execute("SELECT text FROM chunks").fetchone()[0]
    assert status == "indexed"
    assert chunk_text == "CLI searchable sample"
