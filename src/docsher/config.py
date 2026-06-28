"""Configuration management for Local Docsher.

The configuration layer is intentionally stdlib-only so CLI, future API routes,
and the web UI can share the same read/write behavior without adding runtime
dependencies.
"""

from __future__ import annotations

import json
import os
import re
from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path
from typing import Any

ENV_CONFIG_PATH = "DOCSHER_CONFIG_PATH"
DEFAULT_CONFIG_DIRNAME = ".docsher"
DEFAULT_CONFIG_FILENAME = "config.json"
PROJECT_CONFIG_PATH = Path(DEFAULT_CONFIG_DIRNAME) / DEFAULT_CONFIG_FILENAME


DEFAULT_CONFIG: dict[str, Any] = {
    "version": 1,
    "documents": {
        "roots": [],
    },
    "indexing": {
        "schedule_enabled": True,
        "schedule": "daily",
        "time": "03:00",
        "incremental": True,
    },
    "storage": {
        "database_path": str(Path.home() / DEFAULT_CONFIG_DIRNAME / "docsher.sqlite3"),
    },
    "offline_mode": True,
    "llm": {
        "provider": "ollama",
        "ollama_endpoint": "http://localhost:11434",
        "ollama_model": "qwen2.5:7b-instruct",
    },
    "ocr": {
        "enabled": True,
        "backend": "paddle",
        "fallback_backend": "tesseract",
        "paddle": {
            "lang": "korean",
            "det_model_dir": None,
            "rec_model_dir": None,
            "cls_model_dir": None,
            "use_angle_cls": True,
        },
    },
}


@dataclass(frozen=True)
class ConfigLocation:
    """Resolved configuration path and why it was selected."""

    path: Path
    source: str


def default_user_config_path() -> Path:
    """Return the default per-user configuration path."""

    return Path.home() / DEFAULT_CONFIG_DIRNAME / DEFAULT_CONFIG_FILENAME


def resolve_config_location(cwd: Path | None = None) -> ConfigLocation:
    """Resolve the active configuration path.

    Override rules are deliberately simple and consistent:

    1. ``DOCSHER_CONFIG_PATH`` always wins. This is used by tests, automation,
       and users who want an explicit profile without touching their real home
       configuration.
    2. If the current project directory already contains ``.docsher/config.json``,
       use it as the project-local configuration.
    3. Otherwise use the per-user default at ``~/.docsher/config.json``.
    """

    explicit_path = os.environ.get(ENV_CONFIG_PATH)
    if explicit_path:
        return ConfigLocation(Path(explicit_path).expanduser(), "environment")

    current_dir = Path.cwd() if cwd is None else cwd
    project_path = current_dir / PROJECT_CONFIG_PATH
    if project_path.exists():
        return ConfigLocation(project_path, "project")

    return ConfigLocation(default_user_config_path(), "user")


def default_config() -> dict[str, Any]:
    """Return an independent copy of the default configuration."""

    return deepcopy(DEFAULT_CONFIG)


def _deep_merge_defaults(config: dict[str, Any], defaults: dict[str, Any]) -> dict[str, Any]:
    """Merge missing default keys into a loaded config without dropping values."""

    merged = deepcopy(defaults)
    for key, value in config.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = _deep_merge_defaults(value, merged[key])
        else:
            merged[key] = value
    return merged


def load_config(path: Path | None = None) -> dict[str, Any]:
    """Load configuration, returning defaults when the file does not exist."""

    config_path = path or resolve_config_location().path
    if not config_path.exists():
        return default_config()

    try:
        loaded = json.loads(config_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError(f"Invalid JSON config at {config_path}: {exc}") from exc

    if not isinstance(loaded, dict):
        raise ValueError(f"Invalid config at {config_path}: top-level value must be an object")

    return _deep_merge_defaults(loaded, DEFAULT_CONFIG)


def save_config(config: dict[str, Any], path: Path | None = None) -> Path:
    """Persist configuration as pretty JSON and return the path written."""

    config_path = path or resolve_config_location().path
    config_path.parent.mkdir(parents=True, exist_ok=True)
    config_path.write_text(
        json.dumps(config, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return config_path


def load_config_with_location() -> tuple[dict[str, Any], ConfigLocation]:
    """Load the active config and return it with the resolved location."""

    location = resolve_config_location()
    return load_config(location.path), location


SCHEDULE_CHOICES = {"daily", "hourly", "manual"}
_TIME_RE = re.compile(r"^(?:[01]\d|2[0-3]):[0-5]\d$")


def get_indexing_schedule(config: dict[str, Any]) -> dict[str, Any]:
    """Return indexing schedule settings with defaults merged in."""

    indexing = config.setdefault("indexing", {})
    defaults = DEFAULT_CONFIG["indexing"]
    return {
        "schedule_enabled": bool(indexing.get("schedule_enabled", defaults["schedule_enabled"])),
        "schedule": str(indexing.get("schedule", defaults["schedule"])),
        "time": str(indexing.get("time", defaults["time"])),
        "incremental": bool(indexing.get("incremental", defaults["incremental"])),
    }


def get_ocr_settings(config: dict[str, Any]) -> dict[str, Any]:
    """Return OCR settings with defaults merged in.

    PaddleOCR supports offline operation when model directories are pre-downloaded
    and provided through ``ocr.paddle.*_model_dir``. Keeping this shape in the
    stdlib config layer lets CLI, API, and background workers select the same
    backend without importing optional OCR dependencies.
    """

    ocr = config.setdefault("ocr", {})
    defaults = DEFAULT_CONFIG["ocr"]
    paddle_defaults = defaults["paddle"]
    paddle_config = ocr.get("paddle", {})
    if not isinstance(paddle_config, dict):
        paddle_config = {}
    return {
        "enabled": bool(ocr.get("enabled", defaults["enabled"])),
        "backend": str(ocr.get("backend", defaults["backend"])),
        "fallback_backend": ocr.get("fallback_backend", defaults["fallback_backend"]),
        "paddle": {
            "lang": str(paddle_config.get("lang", paddle_defaults["lang"])),
            "det_model_dir": paddle_config.get("det_model_dir", paddle_defaults["det_model_dir"]),
            "rec_model_dir": paddle_config.get("rec_model_dir", paddle_defaults["rec_model_dir"]),
            "cls_model_dir": paddle_config.get("cls_model_dir", paddle_defaults["cls_model_dir"]),
            "use_angle_cls": bool(paddle_config.get("use_angle_cls", paddle_defaults["use_angle_cls"])),
        },
    }


def validate_indexing_schedule(schedule: dict[str, Any]) -> dict[str, Any]:
    """Validate and normalize user-facing indexing schedule settings."""

    normalized = get_indexing_schedule({"indexing": schedule})
    if normalized["schedule"] not in SCHEDULE_CHOICES:
        choices = ", ".join(sorted(SCHEDULE_CHOICES))
        raise ValueError(f"Invalid schedule: must be one of {choices}")
    if not _TIME_RE.match(normalized["time"]):
        raise ValueError("Invalid schedule time: expected HH:MM in 24-hour format")
    return normalized


def update_indexing_schedule(
    *,
    schedule_enabled: bool | None = None,
    schedule: str | None = None,
    time: str | None = None,
    incremental: bool | None = None,
    config_path: Path | None = None,
) -> tuple[dict[str, Any], Path]:
    """Update persisted indexing schedule settings and return the normalized schedule."""

    config = load_config(config_path)
    current = get_indexing_schedule(config)
    if schedule_enabled is not None:
        current["schedule_enabled"] = schedule_enabled
    if schedule is not None:
        current["schedule"] = schedule
    if time is not None:
        current["time"] = time
    if incremental is not None:
        current["incremental"] = incremental

    normalized = validate_indexing_schedule(current)
    config.setdefault("indexing", {}).update(normalized)
    written_path = save_config(config, config_path)
    return normalized, written_path


def normalize_root(root: str | Path, base_dir: Path | None = None) -> str:
    """Normalize a document root to an absolute path string.

    Existing paths are resolved with symlinks. Missing paths are still accepted
    and normalized lexically so users can register roots before creating them.
    """

    root_path = Path(root).expanduser()
    if not root_path.is_absolute():
        root_path = (base_dir or Path.cwd()) / root_path
    return str(root_path.resolve(strict=False))


def list_roots(config: dict[str, Any]) -> list[str]:
    """Return configured document roots."""

    roots = config.setdefault("documents", {}).setdefault("roots", [])
    if not isinstance(roots, list):
        raise ValueError("Invalid config: documents.roots must be a list")
    return roots


def add_root(root: str | Path, *, config_path: Path | None = None, base_dir: Path | None = None) -> tuple[str, bool]:
    """Add a document root if missing.

    Returns ``(normalized_root, added)`` where ``added`` is false when the root
    already existed.
    """

    config = load_config(config_path)
    roots = list_roots(config)
    normalized = normalize_root(root, base_dir=base_dir)
    if normalized in roots:
        return normalized, False
    roots.append(normalized)
    save_config(config, config_path)
    return normalized, True


def remove_root(root: str | Path, *, config_path: Path | None = None, base_dir: Path | None = None) -> tuple[str, bool]:
    """Remove a document root if present.

    Returns ``(normalized_root, removed)`` where ``removed`` is false when the
    root was not configured.
    """

    config = load_config(config_path)
    roots = list_roots(config)
    normalized = normalize_root(root, base_dir=base_dir)
    if normalized not in roots:
        return normalized, False
    roots.remove(normalized)
    save_config(config, config_path)
    return normalized, True
