"""Scheduler attachment interface for Local Docsher indexing.

MVP-A does not run a background daemon. This module defines the stable contract
that a future scheduler daemon, Windows Task Scheduler bridge, or Hermes cron
job can use to read schedule settings and trigger the same manual indexing path.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from docsher.config import get_indexing_schedule, load_config
from docsher.indexer import IndexResult, index_pending_documents
from docsher.scanner import ScanResult, scan


@dataclass(frozen=True)
class IndexSchedulePlan:
    """Normalized schedule settings consumed by external schedulers."""

    enabled: bool
    schedule: str
    time: str
    incremental: bool

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class ScheduledIndexRunResult:
    """Result returned by a scheduler-triggered indexing run."""

    scan: ScanResult
    index: IndexResult

    def to_dict(self) -> dict[str, Any]:
        return {
            "scan": asdict(self.scan),
            "index": asdict(self.index),
        }


def build_schedule_plan(config: dict[str, Any]) -> IndexSchedulePlan:
    """Return the scheduler-facing plan from loaded config."""

    schedule = get_indexing_schedule(config)
    return IndexSchedulePlan(
        enabled=bool(schedule["schedule_enabled"]),
        schedule=str(schedule["schedule"]),
        time=str(schedule["time"]),
        incremental=bool(schedule["incremental"]),
    )


def load_schedule_plan(config_path: str | Path | None = None) -> IndexSchedulePlan:
    """Load the scheduler-facing plan from config storage."""

    return build_schedule_plan(load_config(Path(config_path).expanduser() if config_path else None))


def run_scheduled_index_once(
    config: dict[str, Any],
    *,
    database_path: str | Path | None = None,
    roots: list[str] | None = None,
    changed_only: bool = True,
) -> ScheduledIndexRunResult:
    """Run the scheduler/manual indexing path once.

    ``changed_only`` is part of the public scheduler contract even though MVP-A
    indexing is already incremental by scanner status. Passing ``False`` is
    reserved for a future full-reindex implementation.
    """

    _ = changed_only
    resolved_database_path = database_path or config["storage"]["database_path"]
    scan_result = scan(config, database_path=resolved_database_path, roots=roots)
    index_result = index_pending_documents(resolved_database_path)
    return ScheduledIndexRunResult(scan=scan_result, index=index_result)
