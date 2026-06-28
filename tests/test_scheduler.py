from __future__ import annotations

from pathlib import Path

from docsher.config import default_config
from docsher.scheduler import build_schedule_plan, load_schedule_plan, run_scheduled_index_once


def test_scheduler_plan_exposes_attachment_contract(tmp_path: Path) -> None:
    config = default_config()
    config["indexing"] = {
        "schedule_enabled": False,
        "schedule": "manual",
        "time": "21:10",
        "incremental": False,
    }
    config_path = tmp_path / "config.json"
    config_path.write_text(
        '{"indexing":{"schedule_enabled":false,"schedule":"manual","time":"21:10","incremental":false}}',
        encoding="utf-8",
    )

    assert build_schedule_plan(config).to_dict() == {
        "enabled": False,
        "schedule": "manual",
        "time": "21:10",
        "incremental": False,
    }
    assert load_schedule_plan(config_path).to_dict() == build_schedule_plan(config).to_dict()


def test_run_scheduled_index_once_uses_manual_indexing_path(tmp_path: Path) -> None:
    root = tmp_path / "docs"
    root.mkdir()
    (root / "manual.txt").write_text("manual scheduler term", encoding="utf-8")
    config = default_config()
    config["documents"]["roots"] = [str(root)]
    database_path = tmp_path / "docsher.sqlite3"
    config["storage"]["database_path"] = str(database_path)

    result = run_scheduled_index_once(config, changed_only=True)

    assert result.scan.scanned_files == 1
    assert result.index.parsed_documents == 1
    assert result.index.created_chunks >= 1
