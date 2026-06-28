"""Command line interface for Local Docsher."""

from __future__ import annotations

import argparse
import json
from collections.abc import Sequence
from pathlib import Path

from docsher import __version__
from docsher.config import (
    add_root,
    get_indexing_schedule,
    list_roots,
    load_config_with_location,
    remove_root,
    update_indexing_schedule,
)
from docsher.db import init_database
from docsher.indexer import format_index_result
from docsher.scanner import format_scan_result, scan
from docsher.scheduler import run_scheduled_index_once
from docsher.search import SearchError, format_search_results, search_documents
from docsher.status import format_index_status, get_index_status


def _cmd_config_show(_args: argparse.Namespace) -> int:
    config, location = load_config_with_location()
    payload = {
        "config_path": str(location.path),
        "config_source": location.source,
        "config": config,
    }
    print(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


def _cmd_config_schedule_show(_args: argparse.Namespace) -> int:
    config, location = load_config_with_location()
    payload = {
        "config_path": str(location.path),
        "config_source": location.source,
        "indexing": get_indexing_schedule(config),
    }
    print(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


def _cmd_config_schedule_set(args: argparse.Namespace) -> int:
    schedule_enabled = None
    if args.enabled:
        schedule_enabled = True
    if args.disabled:
        schedule_enabled = False
    incremental = None
    if args.incremental:
        incremental = True
    if args.full:
        incremental = False

    _config, location = load_config_with_location()
    try:
        schedule, written_path = update_indexing_schedule(
            schedule_enabled=schedule_enabled,
            schedule=args.schedule,
            time=args.time,
            incremental=incremental,
            config_path=location.path,
        )
    except ValueError as exc:
        print(f"Config error: {exc}")
        return 2
    print(f"Updated indexing schedule: {written_path}")
    print(json.dumps(schedule, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


def _cmd_roots_add(args: argparse.Namespace) -> int:
    normalized, added = add_root(args.path, base_dir=Path.cwd())
    if added:
        print(f"Added root: {normalized}")
    else:
        print(f"Root already exists: {normalized}")
    return 0


def _cmd_roots_list(_args: argparse.Namespace) -> int:
    config, _location = load_config_with_location()
    roots = list_roots(config)
    if not roots:
        print("No document roots configured.")
        return 0
    for root in roots:
        print(root)
    return 0


def _cmd_roots_remove(args: argparse.Namespace) -> int:
    normalized, removed = remove_root(args.path, base_dir=Path.cwd())
    if removed:
        print(f"Removed root: {normalized}")
        return 0
    print(f"Root not configured: {normalized}")
    return 1


def _cmd_db_init(args: argparse.Namespace) -> int:
    database_path = init_database(args.database_path)
    print(f"Initialized database: {database_path}")
    return 0


def _cmd_index(args: argparse.Namespace) -> int:
    config, _location = load_config_with_location()
    if args.dry_run:
        result = scan(
            config,
            database_path=args.database_path,
            dry_run=True,
            roots=args.root,
        )
        print(format_scan_result(result))
        return 0

    database_path = args.database_path or config["storage"]["database_path"]
    result = run_scheduled_index_once(
        config,
        database_path=database_path,
        roots=args.root,
        changed_only=args.changed_only,
    )
    print(format_scan_result(result.scan))
    print(format_index_result(result.index))
    return 0


def _cmd_search(args: argparse.Namespace) -> int:
    try:
        results = search_documents(
            args.query,
            database_path=args.database_path,
            extension=args.ext,
            path_filter=args.path,
            top_k=args.top_k,
        )
    except SearchError as exc:
        print(f"Search error: {exc}")
        return 2

    if args.json:
        print(
            json.dumps(
                [result.to_dict() for result in results],
                ensure_ascii=False,
                indent=2,
                sort_keys=True,
            )
        )
    else:
        print(format_search_results(results))
    return 0


def _cmd_status(args: argparse.Namespace) -> int:
    status = get_index_status(args.database_path)
    print(format_index_status(status, json_output=args.json))
    return 0


def _cmd_serve(args: argparse.Namespace) -> int:
    try:
        import uvicorn
    except ImportError:
        print(
            "Serve error: uvicorn is not installed. Install local-docsher[api] "
            "or install fastapi and uvicorn."
        )
        return 2

    from docsher.api import create_app

    app = create_app(config_path=args.config_path, database_path=args.database_path)
    uvicorn.run(app, host=args.host, port=args.port, log_level=args.log_level)
    return 0


def build_parser() -> argparse.ArgumentParser:
    """Build the top-level CLI argument parser."""
    parser = argparse.ArgumentParser(
        prog="docsher",
        description=(
            "Local Docsher indexes local document folders and provides "
            "offline-first search and knowledge workflows."
        ),
    )
    parser.add_argument(
        "--version",
        action="version",
        version=f"docsher {__version__}",
    )

    subparsers = parser.add_subparsers(dest="command")

    config_parser = subparsers.add_parser("config", help="Manage Docsher configuration.")
    config_subparsers = config_parser.add_subparsers(dest="config_command", required=True)
    config_show_parser = config_subparsers.add_parser("show", help="Show the active configuration.")
    config_show_parser.set_defaults(func=_cmd_config_show)

    schedule_parser = config_subparsers.add_parser("schedule", help="Show or update indexing schedule settings.")
    schedule_subparsers = schedule_parser.add_subparsers(dest="schedule_command", required=True)

    schedule_show_parser = schedule_subparsers.add_parser("show", help="Show indexing schedule settings.")
    schedule_show_parser.set_defaults(func=_cmd_config_schedule_show)

    schedule_set_parser = schedule_subparsers.add_parser("set", help="Update indexing schedule settings.")
    schedule_set_parser.add_argument("--enabled", action="store_true", help="Enable scheduled indexing.")
    schedule_set_parser.add_argument("--disabled", action="store_true", help="Disable scheduled indexing.")
    schedule_set_parser.add_argument("--schedule", choices=("daily", "hourly", "manual"), help="Schedule cadence placeholder.")
    schedule_set_parser.add_argument("--time", help="Daily schedule time in HH:MM 24-hour format.")
    schedule_set_parser.add_argument("--incremental", action="store_true", help="Run scheduled/manual indexing incrementally.")
    schedule_set_parser.add_argument("--full", action="store_true", help="Reserve interface for future full reindex behavior.")
    schedule_set_parser.set_defaults(func=_cmd_config_schedule_set)

    roots_parser = subparsers.add_parser("roots", help="Manage configured document roots.")
    roots_subparsers = roots_parser.add_subparsers(dest="roots_command", required=True)

    roots_add_parser = roots_subparsers.add_parser("add", help="Add a document root.")
    roots_add_parser.add_argument("path", help="Path to register as a document root.")
    roots_add_parser.set_defaults(func=_cmd_roots_add)

    roots_list_parser = roots_subparsers.add_parser("list", help="List configured document roots.")
    roots_list_parser.set_defaults(func=_cmd_roots_list)

    roots_remove_parser = roots_subparsers.add_parser("remove", help="Remove a document root.")
    roots_remove_parser.add_argument("path", help="Path to remove from configured document roots.")
    roots_remove_parser.set_defaults(func=_cmd_roots_remove)

    db_parser = subparsers.add_parser("db", help="Manage the Docsher SQLite database.")
    db_subparsers = db_parser.add_subparsers(dest="db_command", required=True)

    db_init_parser = db_subparsers.add_parser("init", help="Initialize or migrate the database schema.")
    db_init_parser.add_argument(
        "--database-path",
        help="SQLite database path to initialize instead of storage.database_path from config.",
    )
    db_init_parser.set_defaults(func=_cmd_db_init)

    index_parser = subparsers.add_parser(
        "index",
        help="Scan configured document roots and detect incremental file changes.",
    )
    index_parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show planned changes without mutating the database.",
    )
    index_parser.add_argument(
        "--changed-only",
        action="store_true",
        default=True,
        help="Run the incremental changed-file indexing path (default for MVP-A).",
    )
    index_parser.add_argument(
        "--database-path",
        help="SQLite database path to use instead of storage.database_path from config.",
    )
    index_parser.add_argument(
        "--root",
        action="append",
        help="Root to scan instead of configured document roots. May be repeated.",
    )
    index_parser.set_defaults(func=_cmd_index)

    search_parser = subparsers.add_parser(
        "search",
        help="Search indexed document chunks with SQLite FTS5.",
    )
    search_parser.add_argument(
        "query",
        help="FTS5 search query to match against text, filename, path, and section title.",
    )
    search_parser.add_argument(
        "--database-path",
        help="SQLite database path to use instead of storage.database_path from config.",
    )
    search_parser.add_argument(
        "--ext",
        help="Filter by document extension, with or without a leading dot (for example: txt or .md).",
    )
    search_parser.add_argument(
        "--path",
        help="Filter to documents whose stored path contains this substring.",
    )
    search_parser.add_argument(
        "--top-k",
        type=int,
        default=10,
        help="Maximum number of results to return (default: 10).",
    )
    search_parser.add_argument(
        "--json",
        action="store_true",
        help="Emit deterministic machine-readable JSON output.",
    )
    search_parser.set_defaults(func=_cmd_search)

    status_parser = subparsers.add_parser(
        "status",
        help="Show indexing status, failures, and retryability.",
    )
    status_parser.add_argument(
        "--database-path",
        help="SQLite database path to use instead of storage.database_path from config.",
    )
    status_parser.add_argument(
        "--json",
        action="store_true",
        help="Emit deterministic machine-readable JSON output.",
    )
    status_parser.set_defaults(func=_cmd_status)

    serve_parser = subparsers.add_parser(
        "serve",
        help="Run the Local Docsher FastAPI server.",
        description="Run the Local Docsher FastAPI server.",
    )
    serve_parser.add_argument(
        "--host",
        default="127.0.0.1",
        help="Host interface to bind (default: 127.0.0.1).",
    )
    serve_parser.add_argument(
        "--port",
        type=int,
        default=8000,
        help="TCP port to bind (default: 8000).",
    )
    serve_parser.add_argument(
        "--database-path",
        help="SQLite database path to use instead of storage.database_path from config.",
    )
    serve_parser.add_argument(
        "--config-path",
        help="Config path to use instead of normal config resolution.",
    )
    serve_parser.add_argument(
        "--log-level",
        default="info",
        choices=("critical", "error", "warning", "info", "debug", "trace"),
        help="Uvicorn log level (default: info).",
    )
    serve_parser.set_defaults(func=_cmd_serve)

    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Run the Local Docsher CLI."""
    parser = build_parser()
    args = parser.parse_args(argv)
    if hasattr(args, "func"):
        return args.func(args)
    return 0
