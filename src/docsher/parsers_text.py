"""Text and Markdown parser support for the indexing pipeline."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

SUPPORTED_TEXT_EXTENSIONS: frozenset[str] = frozenset({".txt", ".md"})
PARSER_NAME = "text"
DEFAULT_MAX_TEXT_BYTES = 10 * 1024 * 1024
TEXT_ENCODINGS: tuple[str, ...] = ("utf-8", "cp949", "euc-kr")


class TextParserError(Exception):
    """Base class for text parser failures."""


class UnsupportedTextExtensionError(TextParserError):
    """Raised when the text parser is asked to parse an unsupported extension."""


class TextDocumentTooLargeError(TextParserError):
    """Raised when a document exceeds the configured safe parser byte limit."""


class TextEncodingError(TextParserError):
    """Raised when a document cannot be decoded with supported encodings."""


@dataclass(frozen=True)
class ParsedTextDocument:
    """Decoded text document payload."""

    text: str
    encoding: str
    parser_name: str = PARSER_NAME


def is_text_document(path: str | Path) -> bool:
    """Return true when ``path`` has a text-parser-supported extension."""

    return Path(path).suffix.lower() in SUPPORTED_TEXT_EXTENSIONS


def parse_text_document(
    path: str | Path,
    *,
    max_bytes: int = DEFAULT_MAX_TEXT_BYTES,
    encodings: tuple[str, ...] = TEXT_ENCODINGS,
) -> ParsedTextDocument:
    """Parse a UTF-8-first txt/md document with Korean legacy fallbacks.

    The file is read as bytes only after its size has been checked against
    ``max_bytes`` to avoid accidental unbounded ingestion in the MVP pipeline.
    """

    document_path = Path(path)
    extension = document_path.suffix.lower()
    if extension not in SUPPORTED_TEXT_EXTENSIONS:
        raise UnsupportedTextExtensionError(f"Unsupported text document extension: {extension}")

    size = document_path.stat().st_size
    if size > max_bytes:
        raise TextDocumentTooLargeError(
            f"Text document is too large to parse safely: {size} bytes > {max_bytes} bytes"
        )

    raw = document_path.read_bytes()
    failures: list[str] = []
    for encoding in encodings:
        try:
            return ParsedTextDocument(text=raw.decode(encoding), encoding=encoding)
        except UnicodeDecodeError as exc:
            failures.append(f"{encoding}: {exc.reason}")

    failure_details = "; ".join(failures) if failures else "no encodings configured"
    raise TextEncodingError(f"Could not decode {document_path} ({failure_details})")
