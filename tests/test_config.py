from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

from docsher.config import (
    ENV_CONFIG_PATH,
    add_root,
    default_user_config_path,
    get_indexing_schedule,
    list_roots,
    load_config,
    normalize_root,
    remove_root,
    resolve_config_location,
    update_indexing_schedule,
)


def test_default_config_contains_lds_002_fields(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    config_path = tmp_path / "config.json"
    monkeypatch.setenv(ENV_CONFIG_PATH, str(config_path))

    config = load_config()

    assert resolve_config_location().path == config_path
    assert config["documents"]["roots"] == []
    assert config["indexing"]["schedule"] == "daily"
    assert config["indexing"]["schedule_enabled"] is True
    assert config["indexing"]["incremental"] is True
    assert config["offline_mode"] is True
    assert config["storage"]["database_path"]
    assert config["llm"]["ollama_endpoint"] == "http://localhost:11434"
    assert config["llm"]["ollama_model"]
    assert config["ocr"]["backend"]


def test_config_location_uses_project_config_before_home(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.delenv(ENV_CONFIG_PATH, raising=False)
    project_config = tmp_path / ".docsher" / "config.json"
    project_config.parent.mkdir()
    project_config.write_text("{}", encoding="utf-8")

    location = resolve_config_location(cwd=tmp_path)

    assert location.path == project_config
    assert location.source == "project"


def test_config_location_falls_back_to_user_home(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv(ENV_CONFIG_PATH, raising=False)

    location = resolve_config_location(cwd=Path.cwd())

    if (Path.cwd() / ".docsher" / "config.json").exists():
        assert location.source == "project"
    else:
        assert location.path == default_user_config_path()
        assert location.source == "user"


def test_add_list_remove_roots(tmp_path: Path) -> None:
    config_path = tmp_path / "config.json"
    docs_root = tmp_path / "sample_docs"

    normalized, added = add_root(docs_root, config_path=config_path)
    assert added is True
    assert normalized == normalize_root(docs_root)
    assert list_roots(load_config(config_path)) == [normalized]

    duplicate, added_again = add_root(docs_root, config_path=config_path)
    assert duplicate == normalized
    assert added_again is False
    assert list_roots(load_config(config_path)) == [normalized]

    removed_root, removed = remove_root(docs_root, config_path=config_path)
    assert removed_root == normalized
    assert removed is True
    assert list_roots(load_config(config_path)) == []


def run_docsher(args: list[str], *, config_path: Path, cwd: Path) -> subprocess.CompletedProcess[str]:
    env = os.environ.copy()
    env[ENV_CONFIG_PATH] = str(config_path)
    return subprocess.run(
        [sys.executable, "-m", "docsher", *args],
        check=False,
        capture_output=True,
        text=True,
        cwd=cwd,
        env=env,
    )


def test_cli_config_show_and_roots_round_trip(tmp_path: Path) -> None:
    config_path = tmp_path / "docsher-config.json"

    show = run_docsher(["config", "show"], config_path=config_path, cwd=tmp_path)
    assert show.returncode == 0
    payload = json.loads(show.stdout)
    assert payload["config_path"] == str(config_path)
    assert payload["config_source"] == "environment"
    assert payload["config"]["indexing"]["schedule"] == "daily"

    add = run_docsher(["roots", "add", "./sample_docs"], config_path=config_path, cwd=tmp_path)
    assert add.returncode == 0
    assert "Added root:" in add.stdout

    roots_list = run_docsher(["roots", "list"], config_path=config_path, cwd=tmp_path)
    assert roots_list.returncode == 0
    assert str((tmp_path / "sample_docs").resolve(strict=False)) in roots_list.stdout


def test_indexing_schedule_update_and_cli_round_trip(tmp_path: Path) -> None:
    config_path = tmp_path / "docsher-config.json"

    updated, written_path = update_indexing_schedule(
        schedule_enabled=False,
        schedule="manual",
        time="22:15",
        incremental=False,
        config_path=config_path,
    )

    assert written_path == config_path
    assert updated == {
        "schedule_enabled": False,
        "schedule": "manual",
        "time": "22:15",
        "incremental": False,
    }
    assert get_indexing_schedule(load_config(config_path)) == updated

    show = run_docsher(["config", "schedule", "show"], config_path=config_path, cwd=tmp_path)
    assert show.returncode == 0, show.stderr
    assert json.loads(show.stdout)["indexing"] == updated

    set_result = run_docsher(
        ["config", "schedule", "set", "--enabled", "--schedule", "daily", "--time", "04:30", "--incremental"],
        config_path=config_path,
        cwd=tmp_path,
    )
    assert set_result.returncode == 0, set_result.stderr
    assert "Updated indexing schedule" in set_result.stdout
    assert get_indexing_schedule(load_config(config_path)) == {
        "schedule_enabled": True,
        "schedule": "daily",
        "time": "04:30",
        "incremental": True,
    }


def test_cli_index_accepts_changed_only_verification_path(tmp_path: Path) -> None:
    config_path = tmp_path / "docsher-config.json"
    docs_root = tmp_path / "docs"
    docs_root.mkdir()
    (docs_root / "changed.txt").write_text("changed only indexing path", encoding="utf-8")
    config = load_config(config_path)
    config["documents"]["roots"] = [str(docs_root)]
    config["storage"]["database_path"] = str(tmp_path / "docsher.sqlite3")
    config_path.write_text(json.dumps(config), encoding="utf-8")

    completed = run_docsher(["index", "--changed-only"], config_path=config_path, cwd=tmp_path)

    assert completed.returncode == 0, completed.stderr
    assert "Scanned supported files: 1" in completed.stdout
    assert "Parsed documents: 1" in completed.stdout
