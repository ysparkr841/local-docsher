from __future__ import annotations

import json
import os
import sqlite3
import subprocess
import sys
from pathlib import Path

from docsher.config import ENV_CONFIG_PATH, default_config
from docsher.db import init_database
from docsher.scanner import scan
from docsher.indexer import index_pending_documents
from docsher.search import SearchError, search_documents


def make_config(tmp_path: Path, root: Path) -> dict:
    config = default_config()
    config["documents"]["roots"] = [str(root)]
    config["storage"]["database_path"] = str(tmp_path / "docsher.sqlite3")
    return config


def insert_indexed_chunk(
    database_path: Path,
    *,
    path: str,
    filename: str,
    extension: str,
    text: str,
    chunk_index: int = 0,
    page_number: int | None = None,
    slide_number: int | None = None,
    sheet_name: str | None = None,
    section_title: str | None = None,
) -> int:
    init_database(database_path)
    with sqlite3.connect(database_path) as connection:
        cursor = connection.execute(
            """
            INSERT INTO documents(path, filename, extension, size, modified_at, status)
            VALUES (?, ?, ?, ?, datetime('now'), 'indexed')
            """,
            (path, filename, extension, len(text.encode("utf-8"))),
        )
        document_id = int(cursor.lastrowid)
        cursor = connection.execute(
            """
            INSERT INTO chunks(
                document_id, chunk_index, text, page_number, sheet_name,
                slide_number, section_title, token_count
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                document_id,
                chunk_index,
                text,
                page_number,
                sheet_name,
                slide_number,
                section_title,
                len(text.split()),
            ),
        )
        return int(cursor.lastrowid)


def test_chunks_and_fts_stay_synchronized_for_search(tmp_path: Path) -> None:
    database_path = tmp_path / "docsher.sqlite3"
    chunk_id = insert_indexed_chunk(
        database_path,
        path=str(tmp_path / "docs" / "korean.txt"),
        filename="korean.txt",
        extension=".txt",
        text="이 문서는 테스트 검색을 검증합니다.",
        page_number=3,
        section_title="개요",
    )

    with sqlite3.connect(database_path) as connection:
        fts_row = connection.execute(
            "SELECT rowid, text, filename, path, section_title FROM chunks_fts WHERE rowid = ?",
            (chunk_id,),
        ).fetchone()

    assert fts_row == (
        chunk_id,
        "이 문서는 테스트 검색을 검증합니다.",
        "korean.txt",
        str(tmp_path / "docs" / "korean.txt"),
        "개요",
    )
    results = search_documents("테스트", database_path=database_path)
    assert [result.chunk_id for result in results] == [chunk_id]


def test_keyword_and_filename_search_return_metadata_and_snippet(tmp_path: Path) -> None:
    database_path = tmp_path / "docsher.sqlite3"
    insert_indexed_chunk(
        database_path,
        path=str(tmp_path / "docs" / "alpha.txt"),
        filename="alpha.txt",
        extension=".txt",
        text="alpha 테스트 keyword body",
        page_number=7,
    )
    filename_chunk_id = insert_indexed_chunk(
        database_path,
        path=str(tmp_path / "docs" / "release-notes.md"),
        filename="release-notes.md",
        extension=".md",
        text="body without the filename token",
        sheet_name="Sheet1",
    )

    keyword_results = search_documents("테스트", database_path=database_path)
    assert len(keyword_results) == 1
    assert keyword_results[0].filename == "alpha.txt"
    assert keyword_results[0].extension == ".txt"
    assert keyword_results[0].page_number == 7
    assert "[테스트]" in keyword_results[0].snippet

    filename_results = search_documents("release-notes.md", database_path=database_path)
    assert [result.chunk_id for result in filename_results] == [filename_chunk_id]
    assert filename_results[0].filename == "release-notes.md"
    assert filename_results[0].sheet_name == "Sheet1"


def test_search_filters_ext_path_and_top_k(tmp_path: Path) -> None:
    database_path = tmp_path / "docsher.sqlite3"
    insert_indexed_chunk(
        database_path,
        path=str(tmp_path / "docs" / "keep" / "one.txt"),
        filename="one.txt",
        extension=".txt",
        text="commonterm first",
    )
    insert_indexed_chunk(
        database_path,
        path=str(tmp_path / "docs" / "keep" / "two.md"),
        filename="two.md",
        extension=".md",
        text="commonterm second",
    )
    insert_indexed_chunk(
        database_path,
        path=str(tmp_path / "docs" / "skip" / "three.txt"),
        filename="three.txt",
        extension=".txt",
        text="commonterm third",
    )

    ext_results = search_documents("commonterm", database_path=database_path, extension="txt")
    assert {result.filename for result in ext_results} == {"one.txt", "three.txt"}

    path_results = search_documents("commonterm", database_path=database_path, path_filter="keep")
    assert {result.filename for result in path_results} == {"one.txt", "two.md"}

    limited_results = search_documents("commonterm", database_path=database_path, top_k=2)
    assert len(limited_results) == 2


def test_search_rejects_empty_query_and_bad_top_k(tmp_path: Path) -> None:
    database_path = tmp_path / "docsher.sqlite3"
    init_database(database_path)

    for kwargs in ({"query": "   "}, {"query": "테스트", "top_k": 0}):
        try:
            search_documents(database_path=database_path, **kwargs)
        except SearchError:
            pass
        else:  # pragma: no cover - assertion branch
            raise AssertionError("expected SearchError")


def test_cli_search_outputs_human_and_json(tmp_path: Path) -> None:
    root = tmp_path / "sample_docs"
    root.mkdir()
    (root / "sample.txt").write_text("CLI 테스트 searchable sample", encoding="utf-8")
    config = make_config(tmp_path, root)
    database_path = Path(config["storage"]["database_path"])
    config_path = tmp_path / "config.json"
    config_path.write_text(json.dumps(config), encoding="utf-8")
    env = os.environ.copy()
    env[ENV_CONFIG_PATH] = str(config_path)
    scan(config)
    index_pending_documents(database_path)

    human = subprocess.run(
        [sys.executable, "-m", "docsher", "search", "테스트", "--top-k", "5", "--database-path", str(database_path)],
        cwd=Path(__file__).resolve().parents[1],
        env=env,
        check=False,
        capture_output=True,
        text=True,
    )
    json_output = subprocess.run(
        [sys.executable, "-m", "docsher", "search", "테스트", "--json", "--database-path", str(database_path)],
        cwd=Path(__file__).resolve().parents[1],
        env=env,
        check=False,
        capture_output=True,
        text=True,
    )

    assert human.returncode == 0, human.stderr
    assert "sample.txt" in human.stdout
    assert "Snippet:" in human.stdout
    assert "Location:" in human.stdout
    assert json_output.returncode == 0, json_output.stderr
    payload = json.loads(json_output.stdout)
    assert payload[0]["filename"] == "sample.txt"
    assert payload[0]["extension"] == ".txt"
    assert "테스트" in payload[0]["snippet"]
