from __future__ import annotations

import json
from pathlib import Path

from fastapi.testclient import TestClient

from docsher.api import create_app
from docsher.config import default_config


def make_config(tmp_path: Path, root: Path) -> tuple[Path, Path]:
    config = default_config()
    config["documents"]["roots"] = [str(root)]
    database_path = tmp_path / "docsher.sqlite3"
    config["storage"]["database_path"] = str(database_path)
    config_path = tmp_path / "config.json"
    config_path.write_text(json.dumps(config), encoding="utf-8")
    return config_path, database_path


def make_client(tmp_path: Path, root: Path) -> TestClient:
    config_path, database_path = make_config(tmp_path, root)
    return TestClient(create_app(config_path=config_path, database_path=database_path))


def test_root_serves_search_ui(tmp_path: Path) -> None:
    root = tmp_path / "docs"
    root.mkdir()
    client = make_client(tmp_path, root)

    response = client.get("/")

    assert response.status_code == 200
    assert response.headers["content-type"].startswith("text/html")
    assert "<h1>Local Docsher</h1>" in response.text
    assert 'form class="search"' in response.text
    assert 'name="q"' in response.text
    assert "Enter a query to search indexed documents." in response.text


def test_search_ui_displays_results_with_required_metadata_and_detail_link(tmp_path: Path) -> None:
    root = tmp_path / "docs"
    root.mkdir()
    document_path = root / "korean.txt"
    document_path.write_text("이 문서는 웹 UI 검색 결과 위치 정보를 검증합니다. " * 20, encoding="utf-8")
    client = make_client(tmp_path, root)
    index_response = client.post("/index/run")
    assert index_response.status_code == 200, index_response.text

    response = client.get("/", params={"q": "검색"})

    assert response.status_code == 200, response.text
    assert "Showing" in response.text
    assert "korean.txt" in response.text
    assert str(document_path) in response.text
    assert "Location: chunk" in response.text
    assert "검색" in response.text
    assert '<div class="snippet">' in response.text
    assert 'href="/ui/documents/' in response.text


def test_search_ui_displays_empty_and_error_states(tmp_path: Path) -> None:
    root = tmp_path / "docs"
    root.mkdir()
    client = make_client(tmp_path, root)

    empty_response = client.get("/", params={"q": "missingterm"})
    assert empty_response.status_code == 200, empty_response.text
    assert "No results found." in empty_response.text

    bad_config = default_config()
    bad_config["documents"]["roots"] = [str(root)]
    bad_config["storage"]["database_path"] = ""
    bad_config_path = tmp_path / "bad-config.json"
    bad_config_path.write_text(json.dumps(bad_config), encoding="utf-8")
    bad_client = TestClient(create_app(config_path=bad_config_path))

    error_response = bad_client.get("/", params={"q": "anything"})
    assert error_response.status_code == 400
    assert "role=\"alert\"" in error_response.text
    assert "storage.database_path" in error_response.text


def test_document_detail_ui_displays_metadata_chunk_previews_and_missing_state(tmp_path: Path) -> None:
    root = tmp_path / "docs"
    root.mkdir()
    document_path = root / "detail.txt"
    document_path.write_text("detail preview text for the document detail screen " * 25, encoding="utf-8")
    client = make_client(tmp_path, root)
    index_response = client.post("/index/run")
    assert index_response.status_code == 200, index_response.text
    search_response = client.get("/search", params={"q": "detail"})
    assert search_response.status_code == 200, search_response.text
    document_id = search_response.json()["results"][0]["document_id"]

    response = client.get(f"/ui/documents/{document_id}")

    assert response.status_code == 200, response.text
    assert response.headers["content-type"].startswith("text/html")
    assert "Metadata" in response.text
    assert "Chunk previews" in response.text
    assert "detail.txt" in response.text
    assert str(document_path) in response.text
    assert "status</dt><dd>indexed" in response.text
    assert "detail preview text" in response.text
    assert "Location: chunk" in response.text

    missing_response = client.get("/ui/documents/999999")
    assert missing_response.status_code == 404
    assert "Document not found" in missing_response.text
