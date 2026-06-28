from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

from fastapi.testclient import TestClient

from docsher.api import create_app
from docsher.config import ENV_CONFIG_PATH, default_config, load_config


def make_config(tmp_path: Path, root: Path) -> tuple[dict, Path, Path]:
    config = default_config()
    config["documents"]["roots"] = [str(root)]
    database_path = tmp_path / "docsher.sqlite3"
    config["storage"]["database_path"] = str(database_path)
    config_path = tmp_path / "config.json"
    config_path.write_text(json.dumps(config), encoding="utf-8")
    return config, config_path, database_path


def make_client(tmp_path: Path, root: Path) -> tuple[TestClient, Path]:
    _config, config_path, database_path = make_config(tmp_path, root)
    app = create_app(config_path=config_path, database_path=database_path)
    return TestClient(app), database_path


def test_health_returns_json_payload(tmp_path: Path) -> None:
    root = tmp_path / "docs"
    root.mkdir()
    client, database_path = make_client(tmp_path, root)

    response = client.get("/health")

    assert response.status_code == 200
    assert response.headers["content-type"].startswith("application/json")
    payload = response.json()
    assert payload["status"] == "ok"
    assert payload["database_path"] == str(database_path)
    assert payload["roots_count"] == 1


def test_index_run_search_and_document_preview(tmp_path: Path) -> None:
    root = tmp_path / "docs"
    root.mkdir()
    document_path = root / "korean.txt"
    document_path.write_text("이 문서는 테스트 검색 API를 검증합니다. " * 20, encoding="utf-8")
    client, _database_path = make_client(tmp_path, root)

    index_response = client.post("/index/run")

    assert index_response.status_code == 200, index_response.text
    index_payload = index_response.json()
    assert index_payload["status"] == "completed"
    assert index_payload["scan"]["scanned_files"] == 1
    assert index_payload["scan"]["new_files_count"] == 1
    assert index_payload["index"]["parsed_documents"] == 1
    assert index_payload["index"]["created_chunks"] >= 1

    search_response = client.get("/search", params={"q": "테스트", "top_k": 5, "ext": "txt"})
    assert search_response.status_code == 200, search_response.text
    search_payload = search_response.json()
    assert search_payload["query"] == "테스트"
    assert search_payload["count"] >= 1
    first_result = search_payload["results"][0]
    assert first_result["filename"] == "korean.txt"
    assert first_result["extension"] == ".txt"
    assert "테스트" in first_result["snippet"]

    document_response = client.get(
        f"/documents/{first_result['document_id']}",
        params={"preview_chars": 24},
    )
    assert document_response.status_code == 200, document_response.text
    document_payload = document_response.json()
    assert document_payload["document"]["filename"] == "korean.txt"
    assert document_payload["document"]["status"] == "indexed"
    assert document_payload["chunks"]
    assert document_payload["chunks"][0]["text_preview"].startswith("이 문서는 테스트")
    assert len(document_payload["chunks"][0]["text_preview"]) <= 24
    assert "chunk_index" in document_payload["chunks"][0]


def test_document_missing_and_search_validation_errors_are_json(tmp_path: Path) -> None:
    root = tmp_path / "docs"
    root.mkdir()
    client, _database_path = make_client(tmp_path, root)

    missing_response = client.get("/documents/999")
    assert missing_response.status_code == 404
    assert missing_response.headers["content-type"].startswith("application/json")
    assert missing_response.json() == {"detail": "Document not found"}

    search_response = client.get("/search", params={"q": "   "})
    assert search_response.status_code == 400
    assert search_response.headers["content-type"].startswith("application/json")
    assert "must not be empty" in search_response.json()["detail"]

    validation_response = client.get("/search", params={"q": "테스트", "top_k": 0})
    assert validation_response.status_code == 422
    assert validation_response.headers["content-type"].startswith("application/json")
    assert "detail" in validation_response.json()


def test_index_run_accepts_roots_override(tmp_path: Path) -> None:
    configured_root = tmp_path / "configured"
    override_root = tmp_path / "override"
    configured_root.mkdir()
    override_root.mkdir()
    (configured_root / "configured.txt").write_text("configured only", encoding="utf-8")
    (override_root / "override.txt").write_text("override 테스트 only", encoding="utf-8")
    client, _database_path = make_client(tmp_path, configured_root)

    response = client.post("/index/run", json={"roots": [str(override_root)]})
    assert response.status_code == 200, response.text
    payload = response.json()
    assert payload["scan"]["roots"] == [str(override_root)]
    assert payload["index"]["parsed_documents"] == 1

    search_response = client.get("/search", params={"q": "override"})
    assert search_response.status_code == 200
    assert [result["filename"] for result in search_response.json()["results"]] == ["override.txt"]


def test_cli_serve_command_is_registered(tmp_path: Path) -> None:
    root = tmp_path / "docs"
    root.mkdir()
    _config, config_path, _database_path = make_config(tmp_path, root)
    env = os.environ.copy()
    env[ENV_CONFIG_PATH] = str(config_path)

    completed = subprocess.run(
        [sys.executable, "-m", "docsher", "serve", "--help"],
        cwd=Path(__file__).resolve().parents[1],
        env=env,
        check=False,
        capture_output=True,
        text=True,
    )

    assert completed.returncode == 0, completed.stderr
    assert "Run the Local Docsher FastAPI server" in completed.stdout
    assert "--database-path" in completed.stdout


def test_settings_api_and_web_ui_update_schedule(tmp_path: Path) -> None:
    root = tmp_path / "docs"
    root.mkdir()
    _config, config_path, database_path = make_config(tmp_path, root)
    client = TestClient(create_app(config_path=config_path, database_path=database_path))

    settings_response = client.get("/settings")
    assert settings_response.status_code == 200
    assert settings_response.json()["indexing"]["schedule"] == "daily"

    update_response = client.post(
        "/settings",
        json={
            "schedule_enabled": False,
            "schedule": "manual",
            "time": "22:45",
            "incremental": False,
        },
    )
    assert update_response.status_code == 200, update_response.text
    assert update_response.json()["indexing"] == {
        "schedule_enabled": False,
        "schedule": "manual",
        "time": "22:45",
        "incremental": False,
    }

    ui_response = client.get("/ui/settings")
    assert ui_response.status_code == 200
    assert "Indexing schedule" in ui_response.text
    assert "Run indexing now" in ui_response.text
    assert "value=\"22:45\"" in ui_response.text

    form_response = client.post(
        "/ui/settings",
        data={"schedule_enabled": "true", "schedule": "daily", "time": "05:30", "incremental": "true"},
    )
    assert form_response.status_code == 200, form_response.text
    assert "Settings saved" in form_response.text
    assert load_config(config_path)["indexing"] == {
        "schedule_enabled": True,
        "schedule": "daily",
        "time": "05:30",
        "incremental": True,
    }
