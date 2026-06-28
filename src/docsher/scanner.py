"""Document file scanner and incremental change detection."""

from __future__ import annotations

import hashlib
import os
import sqlite3
from dataclasses import dataclass, field, replace
from datetime import UTC, datetime
from pathlib import Path
from typing import Iterable

from docsher.config import list_roots
from docsher.db import connect, init_database

SUPPORTED_EXTENSIONS: frozenset[str] = frozenset(
    {
        ".txt",
        ".md",
        ".pdf",
        ".docx",
        ".pptx",
        ".xlsx",
    }
)

PENDING_STATUS = "pending"
DELETED_STATUS = "deleted"


@dataclass(frozen=True)
class FileMetadata:
    """Filesystem metadata stored in the documents table."""

    path: str
    filename: str
    extension: str
    size: int
    modified_at: str
    content_hash: str | None = None


@dataclass(frozen=True)
class PlannedChange:
    """A scanner-planned document state transition."""

    action: str
    path: str
    metadata: FileMetadata | None = None
    previous_status: str | None = None


@dataclass(frozen=True)
class ScanResult:
    """Summary of one scanner run."""

    roots: tuple[str, ...]
    supported_extensions: tuple[str, ...]
    changes: tuple[PlannedChange, ...] = field(default_factory=tuple)
    scanned_files: int = 0
    skipped_files: int = 0
    dry_run: bool = False

    @property
    def new_files(self) -> tuple[PlannedChange, ...]:
        return tuple(change for change in self.changes if change.action == "new")

    @property
    def modified_files(self) -> tuple[PlannedChange, ...]:
        return tuple(change for change in self.changes if change.action == "modified")

    @property
    def deleted_files(self) -> tuple[PlannedChange, ...]:
        return tuple(change for change in self.changes if change.action == "deleted")


@dataclass(frozen=True)
class _ScanPlan:
    """Scanner plan split into user-visible changes and internal metadata refreshes."""

    changes: tuple[PlannedChange, ...]
    metadata_refreshes: tuple[PlannedChange, ...] = field(default_factory=tuple)


@dataclass(frozen=True)
class _ExistingDocument:
    path: str
    filename: str
    extension: str | None
    size: int | None
    modified_at: str | None
    content_hash: str | None
    status: str


def normalize_document_path(path: str | Path) -> str:
    """Return the canonical absolute path representation used by the scanner."""

    return str(Path(path).expanduser().resolve(strict=False))


def is_supported_file(path: Path, extensions: Iterable[str] = SUPPORTED_EXTENSIONS) -> bool:
    """Return true when ``path`` has an extension supported by the MVP scanner."""

    normalized_extensions = {extension.lower() for extension in extensions}
    return path.is_file() and path.suffix.lower() in normalized_extensions


def content_hash(path: Path, *, chunk_size: int = 1024 * 1024) -> str:
    """Compute a SHA-256 content hash for a file."""

    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def file_metadata(path: Path, *, include_hash: bool = True) -> FileMetadata:
    """Build database-ready metadata for a supported document file."""

    resolved = path.expanduser().resolve(strict=False)
    stat = resolved.stat()
    return FileMetadata(
        path=str(resolved),
        filename=resolved.name,
        extension=resolved.suffix.lower(),
        size=stat.st_size,
        modified_at=datetime.fromtimestamp(stat.st_mtime, UTC).isoformat(),
        content_hash=content_hash(resolved) if include_hash else None,
    )


def _metadata_with_hash(metadata: FileMetadata) -> FileMetadata:
    """Return metadata with content_hash populated, hashing only when needed."""

    if metadata.content_hash is not None:
        return metadata
    return replace(metadata, content_hash=content_hash(Path(metadata.path)))


def discover_files(
    roots: Iterable[str | Path],
    *,
    supported_extensions: Iterable[str] = SUPPORTED_EXTENSIONS,
) -> tuple[dict[str, FileMetadata], int]:
    """Walk configured roots and return supported file metadata keyed by path.

    Missing or non-directory roots are ignored. The second return value is a best
    effort count of regular files skipped because their extensions are not
    supported.
    """

    candidates, skipped_files, _scanned_roots = _discover_files_and_scanned_roots(
        roots,
        supported_extensions=supported_extensions,
    )
    return candidates, skipped_files


def _discover_files_and_scanned_roots(
    roots: Iterable[str | Path],
    *,
    supported_extensions: Iterable[str] = SUPPORTED_EXTENSIONS,
) -> tuple[dict[str, FileMetadata], int, tuple[str, ...]]:
    """Walk valid roots and return files, skipped count, and roots actually scanned."""

    candidates: dict[str, FileMetadata] = {}
    skipped_files = 0
    scanned_roots: list[str] = []
    normalized_extensions = frozenset(extension.lower() for extension in supported_extensions)

    for root in roots:
        root_path = Path(root).expanduser().resolve(strict=False)
        if not root_path.is_dir():
            continue
        scanned_roots.append(str(root_path))
        for path in sorted(root_path.rglob("*")):
            if not path.is_file():
                continue
            if path.suffix.lower() not in normalized_extensions:
                skipped_files += 1
                continue
            metadata = file_metadata(path, include_hash=False)
            candidates[metadata.path] = metadata

    return candidates, skipped_files, tuple(scanned_roots)


def _is_path_under_any_root(path: str | Path, roots: Iterable[str | Path]) -> bool:
    """Return true when path is contained by any root using path-aware comparison."""

    normalized_path = os.path.normcase(str(Path(path).expanduser().resolve(strict=False)))
    for root in roots:
        normalized_root = os.path.normcase(str(Path(root).expanduser().resolve(strict=False)))
        try:
            if os.path.commonpath([normalized_path, normalized_root]) == normalized_root:
                return True
        except ValueError:
            # Different drives or otherwise incompatible paths are not contained.
            continue
    return False


def _load_existing_documents(connection: sqlite3.Connection) -> dict[str, _ExistingDocument]:
    rows = connection.execute(
        """
        SELECT path, filename, extension, size, modified_at, content_hash, status
        FROM documents
        """
    ).fetchall()
    return {
        str(row[0]): _ExistingDocument(
            path=str(row[0]),
            filename=str(row[1]),
            extension=row[2],
            size=row[3],
            modified_at=row[4],
            content_hash=row[5],
            status=str(row[6]),
        )
        for row in rows
    }


def _has_same_fast_metadata(existing: _ExistingDocument, current: FileMetadata) -> bool:
    return existing.size == current.size and existing.modified_at == current.modified_at


def _is_modified(existing: _ExistingDocument, current: FileMetadata) -> bool:
    if existing.status == DELETED_STATUS:
        return True

    if _has_same_fast_metadata(existing, current):
        return False

    return existing.content_hash != current.content_hash


def _plan_scan(
    current_files: dict[str, FileMetadata],
    existing_documents: dict[str, _ExistingDocument],
    *,
    deletion_roots: Iterable[str | Path] | None = None,
) -> _ScanPlan:
    """Compare filesystem state with DB rows and return public/internal changes."""

    changes: list[PlannedChange] = []
    metadata_refreshes: list[PlannedChange] = []

    for path, metadata in sorted(current_files.items()):
        existing = existing_documents.get(path)
        if existing is None:
            changes.append(PlannedChange("new", path, metadata=_metadata_with_hash(metadata)))
            continue

        if existing.status != DELETED_STATUS and _has_same_fast_metadata(existing, metadata):
            continue

        hashed_metadata = _metadata_with_hash(metadata)
        if _is_modified(existing, hashed_metadata):
            changes.append(
                PlannedChange(
                    "modified",
                    path,
                    metadata=hashed_metadata,
                    previous_status=existing.status,
                )
            )
        elif not _has_same_fast_metadata(existing, hashed_metadata):
            metadata_refreshes.append(
                PlannedChange(
                    "metadata_refresh",
                    path,
                    metadata=hashed_metadata,
                    previous_status=existing.status,
                )
            )

    for path, existing in sorted(existing_documents.items()):
        if existing.status == DELETED_STATUS:
            continue
        if deletion_roots is not None and not _is_path_under_any_root(path, deletion_roots):
            continue
        if path not in current_files:
            changes.append(
                PlannedChange("deleted", path, previous_status=existing.status)
            )

    return _ScanPlan(tuple(changes), tuple(metadata_refreshes))


def plan_scan(
    current_files: dict[str, FileMetadata],
    existing_documents: dict[str, _ExistingDocument],
    *,
    deletion_roots: Iterable[str | Path] | None = None,
) -> tuple[PlannedChange, ...]:
    """Compare filesystem state with DB rows and return user-visible changes."""

    return _plan_scan(
        current_files,
        existing_documents,
        deletion_roots=deletion_roots,
    ).changes


def _apply_changes(connection: sqlite3.Connection, changes: Iterable[PlannedChange]) -> None:
    with connection:
        for change in changes:
            if change.action in {"new", "modified"}:
                if change.metadata is None:
                    raise ValueError(f"Missing metadata for {change.action} change: {change.path}")
                metadata = change.metadata
                connection.execute(
                    """
                    INSERT INTO documents(
                        path, filename, extension, size, modified_at,
                        content_hash, indexed_at, status, error_message,
                        parser_name, ocr_status
                    )
                    VALUES (?, ?, ?, ?, ?, ?, NULL, ?, NULL, NULL, 'not_required')
                    ON CONFLICT(path) DO UPDATE SET
                        filename = excluded.filename,
                        extension = excluded.extension,
                        size = excluded.size,
                        modified_at = excluded.modified_at,
                        content_hash = excluded.content_hash,
                        indexed_at = NULL,
                        status = excluded.status,
                        error_message = NULL,
                        parser_name = NULL,
                        ocr_status = excluded.ocr_status
                    """,
                    (
                        metadata.path,
                        metadata.filename,
                        metadata.extension,
                        metadata.size,
                        metadata.modified_at,
                        metadata.content_hash,
                        PENDING_STATUS,
                    ),
                )
                connection.execute(
                    """
                    DELETE FROM chunks
                    WHERE document_id = (SELECT id FROM documents WHERE path = ?)
                    """,
                    (metadata.path,),
                )
            elif change.action == "deleted":
                connection.execute(
                    """
                    DELETE FROM chunks
                    WHERE document_id = (SELECT id FROM documents WHERE path = ?)
                    """,
                    (change.path,),
                )
                connection.execute(
                    """
                    UPDATE documents
                    SET status = ?, indexed_at = NULL, error_message = NULL
                    WHERE path = ?
                    """,
                    (DELETED_STATUS, change.path),
                )
            elif change.action == "metadata_refresh":
                if change.metadata is None:
                    raise ValueError(f"Missing metadata for {change.action} change: {change.path}")
                metadata = change.metadata
                connection.execute(
                    """
                    UPDATE documents
                    SET filename = ?, extension = ?, size = ?, modified_at = ?, content_hash = ?
                    WHERE path = ?
                    """,
                    (
                        metadata.filename,
                        metadata.extension,
                        metadata.size,
                        metadata.modified_at,
                        metadata.content_hash,
                        metadata.path,
                    ),
                )
            else:
                raise ValueError(f"Unknown scanner action: {change.action}")


def scan(
    config: dict,
    *,
    database_path: str | Path | None = None,
    dry_run: bool = False,
    roots: Iterable[str | Path] | None = None,
) -> ScanResult:
    """Scan document roots, detect incremental changes, and optionally persist them."""

    configured_roots = tuple(str(root) for root in (roots if roots is not None else list_roots(config)))
    current_files, skipped_files, scanned_roots = _discover_files_and_scanned_roots(configured_roots)
    configured_database_path = database_path or config.get("storage", {}).get("database_path")
    if not configured_database_path or not isinstance(configured_database_path, (str, Path)):
        raise ValueError("Invalid config: storage.database_path must be a non-empty string")
    resolved_database_path = Path(configured_database_path).expanduser()

    if dry_run:
        if not resolved_database_path.exists():
            existing_documents: dict[str, _ExistingDocument] = {}
        else:
            with connect(resolved_database_path) as connection:
                try:
                    existing_documents = _load_existing_documents(connection)
                except sqlite3.OperationalError as exc:
                    if "no such table: documents" not in str(exc):
                        raise
                    existing_documents = {}
    else:
        init_database(resolved_database_path)
        with connect(resolved_database_path) as connection:
            existing_documents = _load_existing_documents(connection)

    scan_plan = _plan_scan(
        current_files,
        existing_documents,
        deletion_roots=scanned_roots,
    )
    changes = scan_plan.changes

    if not dry_run:
        with connect(resolved_database_path) as connection:
            _apply_changes(connection, (*changes, *scan_plan.metadata_refreshes))

    return ScanResult(
        roots=configured_roots,
        supported_extensions=tuple(sorted(SUPPORTED_EXTENSIONS)),
        changes=changes,
        scanned_files=len(current_files),
        skipped_files=skipped_files,
        dry_run=dry_run,
    )


def format_scan_result(result: ScanResult) -> str:
    """Format a scanner result for CLI output."""

    mode = "DRY RUN" if result.dry_run else "APPLIED"
    lines = [
        f"Index scan ({mode})",
        f"Roots: {len(result.roots)}",
        f"Supported extensions: {', '.join(result.supported_extensions)}",
        f"Scanned supported files: {result.scanned_files}",
        f"Skipped unsupported files: {result.skipped_files}",
        (
            "Planned changes: "
            f"{len(result.changes)} "
            f"(new={len(result.new_files)}, "
            f"modified={len(result.modified_files)}, "
            f"deleted={len(result.deleted_files)})"
        ),
    ]
    if not result.roots:
        lines.append("No document roots configured.")
    if not result.changes:
        lines.append("No changes detected.")
        return "\n".join(lines)

    for change in result.changes:
        lines.append(f"{change.action.upper()}: {change.path}")
    return "\n".join(lines)
