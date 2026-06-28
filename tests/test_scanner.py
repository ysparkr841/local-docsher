from __future__ import annotations

import json
import os
import shutil
import sqlite3
import subprocess
import sys
from pathlib import Path

import docsher.scanner as scanner
from docsher.config import ENV_CONFIG_PATH, default_config
from docsher.db import init_database
from docsher.scanner import scan


def make_config(tmp_path: Path, root: Path) -> dict:
    config = default_config()
    config["documents"]["roots"] = [str(root)]
    config["storage"]["database_path"] = str(tmp_path / "docsher.sqlite3")
    return config


def document_rows(database_path: Path) -> list[sqlite3.Row]:
    with sqlite3.connect(database_path) as connection:
        connection.row_factory = sqlite3.Row
        return connection.execute(
            """
            SELECT path, filename, extension, size, modified_at, content_hash, status, indexed_at
            FROM documents
            ORDER BY path
            """
        ).fetchall()


def test_dry_run_reports_new_supported_files_without_mutating_db(tmp_path: Path) -> None:
    root = tmp_path / "docs"
    root.mkdir()
    supported = root / "guide.md"
    supported.write_text("hello docsher", encoding="utf-8")
    (root / "ignore.exe").write_text("not a document", encoding="utf-8")
    database_path = tmp_path / "docsher.sqlite3"
    config = make_config(tmp_path, root)

    result = scan(config, dry_run=True)

    assert result.dry_run is True
    assert result.scanned_files == 1
    assert result.skipped_files == 1
    assert [(change.action, change.path) for change in result.changes] == [
        ("new", str(supported.resolve()))
    ]
    assert not database_path.exists()


def test_scan_persists_new_files_as_pending_with_metadata(tmp_path: Path) -> None:
    root = tmp_path / "docs"
    nested = root / "nested"
    nested.mkdir(parents=True)
    doc = nested / "manual.txt"
    doc.write_text("manual contents", encoding="utf-8")
    config = make_config(tmp_path, root)
    database_path = Path(config["storage"]["database_path"])

    result = scan(config)

    assert [change.action for change in result.changes] == ["new"]
    rows = document_rows(database_path)
    assert len(rows) == 1
    row = rows[0]
    assert row["path"] == str(doc.resolve())
    assert row["filename"] == "manual.txt"
    assert row["extension"] == ".txt"
    assert row["size"] == len("manual contents")
    assert row["modified_at"]
    assert len(row["content_hash"]) == 64
    assert row["status"] == "pending"
    assert row["indexed_at"] is None


def test_unchanged_second_scan_does_not_hash_or_report_changes(
    tmp_path: Path,
    monkeypatch,
) -> None:
    root = tmp_path / "docs"
    root.mkdir()
    doc = root / "guide.md"
    doc.write_text("same", encoding="utf-8")
    config = make_config(tmp_path, root)

    first = scan(config)
    original_content_hash = scanner.content_hash
    hash_calls = 0

    def counting_hash(path: Path, *, chunk_size: int = 1024 * 1024) -> str:
        nonlocal hash_calls
        hash_calls += 1
        return original_content_hash(path, chunk_size=chunk_size)

    monkeypatch.setattr(scanner, "content_hash", counting_hash)
    second = scan(config)

    assert [change.action for change in first.changes] == ["new"]
    assert second.changes == ()
    assert hash_calls == 0
    assert len(document_rows(Path(config["storage"]["database_path"]))) == 1


def test_metadata_only_mtime_change_hashes_once_refreshes_metadata_without_reindex(
    tmp_path: Path,
    monkeypatch,
) -> None:
    root = tmp_path / "docs"
    root.mkdir()
    doc = root / "guide.md"
    doc.write_text("same content", encoding="utf-8")
    config = make_config(tmp_path, root)
    database_path = Path(config["storage"]["database_path"])

    first = scan(config)
    original_row = document_rows(database_path)[0]
    original_stat = doc.stat()
    with sqlite3.connect(database_path) as connection:
        document_id = connection.execute("SELECT id FROM documents WHERE path = ?", (str(doc.resolve()),)).fetchone()[0]
        connection.execute(
            "INSERT INTO chunks(document_id, chunk_index, text) VALUES (?, 0, 'indexed text')",
            (document_id,),
        )
        connection.execute("UPDATE documents SET status = 'indexed', indexed_at = datetime('now') WHERE id = ?", (document_id,))
        connection.commit()

    os.utime(doc, (original_stat.st_atime + 3600, original_stat.st_mtime + 3600))
    original_content_hash = scanner.content_hash
    hash_calls = 0

    def counting_hash(path: Path, *, chunk_size: int = 1024 * 1024) -> str:
        nonlocal hash_calls
        hash_calls += 1
        return original_content_hash(path, chunk_size=chunk_size)

    monkeypatch.setattr(scanner, "content_hash", counting_hash)
    result = scan(config)

    assert [change.action for change in first.changes] == ["new"]
    assert result.changes == ()
    assert result.modified_files == ()
    assert hash_calls == 1
    refreshed_row = document_rows(database_path)[0]
    assert refreshed_row["modified_at"] != original_row["modified_at"]
    assert refreshed_row["content_hash"] == original_row["content_hash"]
    assert refreshed_row["status"] == "indexed"
    assert refreshed_row["indexed_at"] is not None
    with sqlite3.connect(database_path) as connection:
        assert connection.execute("SELECT COUNT(*) FROM chunks").fetchone()[0] == 1

    hash_calls = 0
    subsequent = scan(config)

    assert subsequent.changes == ()
    assert hash_calls == 0


def test_metadata_only_mtime_change_dry_run_does_not_refresh_metadata(tmp_path: Path) -> None:
    root = tmp_path / "docs"
    root.mkdir()
    doc = root / "guide.md"
    doc.write_text("same content", encoding="utf-8")
    config = make_config(tmp_path, root)
    database_path = Path(config["storage"]["database_path"])

    first = scan(config)
    original_row = document_rows(database_path)[0]
    original_stat = doc.stat()
    os.utime(doc, (original_stat.st_atime + 3600, original_stat.st_mtime + 3600))

    dry_run = scan(config, dry_run=True)

    assert [change.action for change in first.changes] == ["new"]
    assert dry_run.changes == ()
    assert dry_run.modified_files == ()
    assert document_rows(database_path)[0]["modified_at"] == original_row["modified_at"]


def test_modified_file_is_marked_pending_for_reindex_and_chunks_are_removed(tmp_path: Path) -> None:
    root = tmp_path / "docs"
    root.mkdir()
    doc = root / "guide.md"
    doc.write_text("first", encoding="utf-8")
    config = make_config(tmp_path, root)
    database_path = Path(config["storage"]["database_path"])
    scan(config)

    with sqlite3.connect(database_path) as connection:
        document_id = connection.execute("SELECT id FROM documents WHERE path = ?", (str(doc.resolve()),)).fetchone()[0]
        connection.execute(
            """
            INSERT INTO chunks(document_id, chunk_index, text, section_title)
            VALUES (?, 0, 'old indexed text', 'old')
            """,
            (document_id,),
        )
        connection.execute(
            """
            UPDATE documents
            SET status = 'indexed', indexed_at = datetime('now')
            WHERE id = ?
            """,
            (document_id,),
        )
        connection.commit()

    doc.write_text("second", encoding="utf-8")

    result = scan(config)

    assert [change.action for change in result.changes] == ["modified"]
    row = document_rows(database_path)[0]
    assert row["status"] == "pending"
    assert row["indexed_at"] is None
    assert row["content_hash"] == result.changes[0].metadata.content_hash
    with sqlite3.connect(database_path) as connection:
        chunk_count = connection.execute("SELECT COUNT(*) FROM chunks").fetchone()[0]
        fts_count = connection.execute("SELECT COUNT(*) FROM chunks_fts").fetchone()[0]
    assert chunk_count == 0
    assert fts_count == 0


def test_deleted_file_is_marked_deleted_and_removed_from_index(tmp_path: Path) -> None:
    root = tmp_path / "docs"
    root.mkdir()
    doc = root / "remove.md"
    doc.write_text("delete me", encoding="utf-8")
    config = make_config(tmp_path, root)
    database_path = Path(config["storage"]["database_path"])
    scan(config)

    with sqlite3.connect(database_path) as connection:
        document_id = connection.execute("SELECT id FROM documents WHERE path = ?", (str(doc.resolve()),)).fetchone()[0]
        connection.execute(
            "INSERT INTO chunks(document_id, chunk_index, text) VALUES (?, 0, 'stale')",
            (document_id,),
        )
        connection.execute("UPDATE documents SET status = 'indexed' WHERE id = ?", (document_id,))
        connection.commit()

    doc.unlink()

    result = scan(config)

    assert [change.action for change in result.changes] == ["deleted"]
    rows = document_rows(database_path)
    assert len(rows) == 1
    assert rows[0]["status"] == "deleted"
    with sqlite3.connect(database_path) as connection:
        chunk_count = connection.execute("SELECT COUNT(*) FROM chunks").fetchone()[0]
        fts_count = connection.execute("SELECT COUNT(*) FROM chunks_fts").fetchone()[0]
    assert chunk_count == 0
    assert fts_count == 0


def test_root_override_scan_does_not_delete_documents_under_other_roots(tmp_path: Path) -> None:
    root_one = tmp_path / "docs-one"
    root_two = tmp_path / "docs-two"
    root_one.mkdir()
    root_two.mkdir()
    doc_one = root_one / "one.md"
    doc_two = root_two / "two.md"
    doc_one.write_text("one", encoding="utf-8")
    doc_two.write_text("two", encoding="utf-8")
    config = make_config(tmp_path, root_one)
    config["documents"]["roots"] = [str(root_one), str(root_two)]
    database_path = Path(config["storage"]["database_path"])

    first = scan(config)
    second = scan(config, roots=[root_one])

    assert [change.action for change in first.changes] == ["new", "new"]
    assert second.changes == ()
    rows = {row["path"]: row for row in document_rows(database_path)}
    assert rows[str(doc_one.resolve())]["status"] == "pending"
    assert rows[str(doc_two.resolve())]["status"] == "pending"


def test_missing_configured_root_does_not_delete_documents_under_that_root(tmp_path: Path) -> None:
    root = tmp_path / "disappearing-docs"
    root.mkdir()
    doc = root / "keep.md"
    doc.write_text("network drive content", encoding="utf-8")
    config = make_config(tmp_path, root)
    database_path = Path(config["storage"]["database_path"])

    first = scan(config)
    shutil.rmtree(root)
    second = scan(config)

    assert [change.action for change in first.changes] == ["new"]
    assert second.changes == ()
    rows = document_rows(database_path)
    assert len(rows) == 1
    assert rows[0]["path"] == str(doc.resolve())
    assert rows[0]["status"] == "pending"


def test_deletion_still_applies_within_successfully_scanned_root(tmp_path: Path) -> None:
    root = tmp_path / "docs"
    root.mkdir()
    doc = root / "remove.md"
    doc.write_text("delete me", encoding="utf-8")
    config = make_config(tmp_path, root)
    database_path = Path(config["storage"]["database_path"])
    scan(config)

    doc.unlink()
    result = scan(config)

    assert [change.action for change in result.changes] == ["deleted"]
    assert document_rows(database_path)[0]["status"] == "deleted"


def test_cli_index_dry_run_uses_isolated_config_and_prints_planned_changes(tmp_path: Path) -> None:
    root = tmp_path / "docs"
    root.mkdir()
    doc = root / "guide.md"
    doc.write_text("cli dry run", encoding="utf-8")
    database_path = tmp_path / "isolated.sqlite3"
    config_path = tmp_path / "config.json"
    config = make_config(tmp_path, root)
    config["storage"]["database_path"] = str(database_path)
    config_path.write_text(json.dumps(config), encoding="utf-8")

    env = os.environ.copy()
    env[ENV_CONFIG_PATH] = str(config_path)
    completed = subprocess.run(
        [sys.executable, "-m", "docsher", "index", "--dry-run"],
        cwd=Path(__file__).resolve().parents[1],
        env=env,
        check=False,
        capture_output=True,
        text=True,
    )

    assert completed.returncode == 0, completed.stderr
    assert "Index scan (DRY RUN)" in completed.stdout
    assert "NEW:" in completed.stdout
    assert str(doc.resolve()) in completed.stdout
    assert not database_path.exists()


def test_cli_index_dry_run_can_read_existing_database_without_mutation(tmp_path: Path) -> None:
    root = tmp_path / "docs"
    root.mkdir()
    doc = root / "guide.md"
    doc.write_text("existing", encoding="utf-8")
    config = make_config(tmp_path, root)
    database_path = Path(config["storage"]["database_path"])
    config_path = tmp_path / "config.json"
    config_path.write_text(json.dumps(config), encoding="utf-8")
    init_database(database_path)

    with sqlite3.connect(database_path) as connection:
        connection.execute(
            """
            INSERT INTO documents(path, filename, extension, size, modified_at, content_hash, status)
            VALUES ('/missing.md', 'missing.md', '.md', 7, 'old', 'abc', 'indexed')
            """
        )
        connection.commit()

    env = os.environ.copy()
    env[ENV_CONFIG_PATH] = str(config_path)
    completed = subprocess.run(
        [sys.executable, "-m", "docsher", "index", "--dry-run"],
        cwd=Path(__file__).resolve().parents[1],
        env=env,
        check=False,
        capture_output=True,
        text=True,
    )

    assert completed.returncode == 0, completed.stderr
    assert "NEW:" in completed.stdout
    assert "DELETED: /missing.md" not in completed.stdout
    rows = document_rows(database_path)
    assert len(rows) == 1
    assert rows[0]["status"] == "indexed"
