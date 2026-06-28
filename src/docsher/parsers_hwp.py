"""HWP/HWPX parser adapter contract.

This module intentionally does not implement HWP/HWPX extraction or wire the
scanner/indexer. It defines the stable interface that future built-in and
external extractor backends can satisfy.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

SUPPORTED_HWP_EXTENSIONS: frozenset[str] = frozenset({".hwp", ".hwpx"})
PARSER_NAME = "hwp"


class HwpParserError(Exception):
    """Base class for HWP/HWPX parser failures."""


class UnsupportedHwpExtensionError(HwpParserError):
    """Raised when an unsupported extension is parsed by the HWP parser."""


class HwpToolUnavailableError(HwpParserError):
    """Raised when no HWP/HWPX extraction backend is available."""


class HwpParseError(HwpParserError):
    """Raised when a supported HWP/HWPX document cannot be parsed."""


@dataclass(frozen=True)
class ParsedHwpSegment:
    """A location-preserving parsed HWP/HWPX text segment."""

    text: str
    page_number: int | None = None
    section_title: str | None = None


@dataclass(frozen=True)
class ParsedHwpDocument:
    """Parsed HWP/HWPX document payload."""

    segments: tuple[ParsedHwpSegment, ...]
    parser_name: str = PARSER_NAME
    source: str | None = None


class HwpTextExtractor(Protocol):
    """Protocol implemented by HWP/HWPX extraction backends."""

    @property
    def name(self) -> str:
        """Human-readable backend name used in diagnostics/metadata."""
        ...

    def is_available(self) -> bool:
        """Return true when the backend can run in the current environment."""
        ...

    def extract(self, path: str | Path) -> ParsedHwpDocument:
        """Extract text segments from a supported HWP/HWPX document."""
        ...


def is_hwp_document(path: str | Path) -> bool:
    """Return true when ``path`` has a HWP-parser-supported extension."""

    return Path(path).suffix.lower() in SUPPORTED_HWP_EXTENSIONS


def validate_hwp_extension(path: str | Path) -> str:
    """Validate and return the normalized HWP/HWPX extension for ``path``."""

    extension = Path(path).suffix.lower()
    if extension not in SUPPORTED_HWP_EXTENSIONS:
        raise UnsupportedHwpExtensionError(f"Unsupported HWP/HWPX document extension: {extension or '<none>'}")
    return extension


class UnavailableHwpExtractor:
    """Placeholder extractor used until an HWP/HWPX backend is configured."""

    name = "unavailable-hwp-extractor"

    def is_available(self) -> bool:
        """The placeholder backend is never available."""

        return False

    def extract(self, path: str | Path) -> ParsedHwpDocument:
        """Raise a clear error explaining that HWP/HWPX support is not configured."""

        raise HwpToolUnavailableError(
            "No HWP/HWPX extractor is configured. Future supported options include "
            "the built-in HWPX ZIP/XML baseline and optional external sidecars such as "
            "@ssabrojs/hwpxjs or kordoc."
        )


def parse_hwp_document(path: str | Path, extractor: HwpTextExtractor | None = None) -> ParsedHwpDocument:
    """Parse a HWP/HWPX document through the supplied extractor contract.

    LDS-017 prepares the adapter boundary only. Callers may inject a fake or
    future real extractor; without one this function raises a clear unavailable
    error after extension validation.
    """

    document_path = Path(path)
    validate_hwp_extension(document_path)
    selected_extractor = extractor or UnavailableHwpExtractor()
    if not selected_extractor.is_available():
        raise HwpToolUnavailableError(f"HWP/HWPX extractor '{selected_extractor.name}' is not available")
    try:
        parsed = selected_extractor.extract(document_path)
    except HwpParserError:
        raise
    except Exception as exc:
        raise HwpParseError(f"Could not parse {document_path.name} with {selected_extractor.name}: {exc}") from exc
    return parsed
