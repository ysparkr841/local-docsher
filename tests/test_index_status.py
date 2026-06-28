from __future__ import annotations

import json
import sqlite3
import subprocess
import sys
from pathlib import Path

from fastapi.testclient import TestClient

from docsher.api import create_app
from docsher.config import default_config
from docsher.db import init_database
from docsher.search import search_documents
from docsher.status import get_index_status


def make_config(tmp_path: Path, root: Path) -> tuple[Path, Path]:
    config = default_config()
    config["documents"]["roots"] = [str(root)]
    database_path = tmp_path / "docsher.sqlite3"
    config["storage"]["database_path"] = str(database_path)
    config_path = tmp_path / "config.json"
    config_path.write_text(json.dumps(config), encoding="utf-8")
    return config_path, database_path


def seed_status_database(tmp_path: Path) -> tuple[Path, Path]:
    root = tmp_path / "docs"
    root.mkdir()
    retryable_failed_path = root / "retryable.txt"
    retryable_failed_path.write_text("retryable file remains on disk", encoding="utf-8")
    missing_failed_path = root / "missing.txt"
    indexed_path = root / "indexed.txt"
    deleted_path = root / "deleted.txt"
    database_path = init_database(tmp_path / "docsher.sqlite3")

    with sqlite3.connect(database_path) as connection:
        cursor = connection.execute(
            """
            INSERT INTO documents(path, filename, extension, size, modified_at, indexed_at, status)
            VALUES (?, 'indexed.txt', '.txt', 10, '2026-01-01T00:00:00+00:00', '2026-01-02T03:04:05+00:00', 'indexed')
            """,
            (str(indexed_path),),
        )
        indexed_id = int(cursor.lastrowid)
        connection.execute(
            """
            INSERT INTO chunks(document_id, chunk_index, text, token_count)
            VALUES (?, 0, 'visible unique_status_term', 2)
            """,
            (indexed_id,),
        )
        cursor = connection.execute(
            """
            INSERT INTO documents(path, filename, extension, size, modified_at, indexed_at, status)
            VALUES (?, 'deleted.txt', '.txt', 10, '2026-01-01T00:00:00+00:00', '2026-01-02T04:04:05+00:00', 'deleted')
            """,
            (str(deleted_path),),
        )
        deleted_id = int(cursor.lastrowid)
        connection.execute(
            """
            INSERT INTO chunks(document_id, chunk_index, text, token_count)
            VALUES (?, 0, 'hidden deleted_unique_term', 2)
            """,
            (deleted_id,),
        )
        connection.execute(
            """
            INSERT INTO documents(path, filename, extension, size, modified_at, status, error_message, parser_name)
            VALUES (?, 'retryable.txt', '.txt', 10, '2026-01-01T00:00:00+00:00', 'failed', 'Could not decode retryable.txt', 'text')
            """,
            (str(retryable_failed_path),),
        )
        connection.execute(
            """
            INSERT INTO documents(path, filename, extension, size, modified_at, status, error_message, parser_name)
            VALUES (?, 'missing.txt', '.txt', 10, '2026-01-01T00:00:00+00:00', 'failed', 'File vanished during parse', 'text')
            """,
            (str(missing_failed_path),),
        )
        connection.execute(
            """
            INSERT INTO documents(path, filename, extension, size, modified_at, status)
            VALUES (?, 'pending.md', '.md', 10, '2026-01-01T00:00:00+00:00', 'pending')
            """,
            (str(root / "pending.md"),),
        )
    return database_path, root


def test_index_status_counts_last_indexed_and_failed_retryability(tmp_path: Path) -> None:
    database_path, _root = seed_status_database(tmp_path)

    status = get_index_status(database_path)

    assert status.total_documents == 5
    assert status.active_documents == 4
    assert status.indexed_count == 1
    assert status.failed_count == 2
    assert status.pending_count == 1
    assert status.deleted_count == 1
    assert status.last_indexed_at == "2026-01-02T03:04:05+00:00"
    failed_by_filename = {document.filename: document for document in status.failed_documents}
    assert failed_by_filename["retryable.txt"].error_message == "Could not decode retryable.txt"
    assert failed_by_filename["retryable.txt"].retryable is True
    assert "exists" in failed_by_filename["retryable.txt"].retryable_reason
    assert failed_by_filename["missing.txt"].retryable is False
    assert "missing" in failed_by_filename["missing.txt"].retryable_reason


def test_cli_status_outputs_counts_failures_and_json(tmp_path: Path) -> None:
    database_path, _root = seed_status_database(tmp_path)
    repo_root = Path(__file__).resolve().parents[1]

    human = subprocess.run(
        [sys.executable, "-m", "docsher", "status", "--database-path", str(database_path)],
        cwd=repo_root,
        check=False,
        capture_output=True,
        text=True,
    )
    machine = subprocess.run(
        [sys.executable, "-m", "docsher", "status", "--database-path", str(database_path), "--json"],
        cwd=repo_root,
        check=False,
        capture_output=True,
        text=True,
    )

    assert human.returncode == 0, human.stderr
    assert "Index status:" in human.stdout
    assert "Total documents: 5" in human.stdout
    assert "Failed documents: 2" in human.stdout
    assert "Last indexed at: 2026-01-02T03:04:05+00:00" in human.stdout
    assert "Could not decode retryable.txt" in human.stdout
    assert "Retryable: yes" in human.stdout
    assert machine.returncode == 0, machine.stderr
    payload = json.loads(machine.stdout)
    assert payload["indexed_count"] == 1
    assert payload["failed_documents"][0]["error_message"]
    assert {document["retryable"] for document in payload["failed_documents"]} == {False, True}


def test_api_and_web_status_dashboard_show_failures(tmp_path: Path) -> None:
    database_path, root = seed_status_database(tmp_path)
    config_path, _configured_database_path = make_config(tmp_path, root)
    client = TestClient(create_app(config_path=config_path, database_path=database_path))

    response = client.get("/status")
    assert response.status_code == 200, response.text
    payload = response.json()
    assert payload["total_documents"] == 5
    assert payload["indexed_count"] == 1
    assert payload["failed_count"] == 2
    assert payload["deleted_count"] == 1
    assert payload["last_indexed_at"] == "2026-01-02T03:04:05+00:00"
    assert any(document["retryable"] for document in payload["failed_documents"])

    dashboard = client.get("/ui/status")
    assert dashboard.status_code == 200, dashboard.text
    assert dashboard.headers["content-type"].startswith("text/html")
    assert "Index status" in dashboard.text
    assert "Total documents" in dashboard.text
    assert "Failed" in dashboard.text
    assert "Could not decode retryable.txt" in dashboard.text
    assert "Retryable:</strong> yes" in dashboard.text

    root_page = client.get("/")
    assert root_page.status_code == 200
    assert "/ui/status" in root_page.text


def test_deleted_documents_are_excluded_from_search_even_if_stale_fts_exists(tmp_path: Path) -> None:
    database_path, _root = seed_status_database(tmp_path)

    assert [result.filename for result in search_documents("unique_status_term", database_path=database_path)] == [
        "indexed.txt"
    ]
    assert search_documents("deleted_unique_term", database_path=database_path) == ()
