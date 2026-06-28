from __future__ import annotations

from pathlib import Path

import pytest

from docsher.parsers_hwp import (
    HwpParseError,
    HwpToolUnavailableError,
    ParsedHwpDocument,
    ParsedHwpSegment,
    UnsupportedHwpExtensionError,
    is_hwp_document,
    parse_hwp_document,
    validate_hwp_extension,
)


class FakeHwpExtractor:
    name = "fake-hwp"

    def __init__(self, *, available: bool = True, fail: bool = False) -> None:
        self.available = available
        self.fail = fail
        self.seen_path: Path | None = None

    def is_available(self) -> bool:
        return self.available

    def extract(self, path: str | Path) -> ParsedHwpDocument:
        self.seen_path = Path(path)
        if self.fail:
            raise RuntimeError("boom")
        return ParsedHwpDocument(
            segments=(
                ParsedHwpSegment(
                    text="고유검색토큰-한글-HWP",
                    page_number=2,
                    section_title="본문",
                ),
            ),
            source=self.name,
        )


def test_is_hwp_document_matches_supported_extensions_case_insensitively() -> None:
    assert is_hwp_document("report.hwp")
    assert is_hwp_document("report.HWPX")
    assert not is_hwp_document("report.docx")


def test_validate_hwp_extension_returns_normalized_extension() -> None:
    assert validate_hwp_extension("REPORT.HWP") == ".hwp"
    assert validate_hwp_extension("REPORT.HWPX") == ".hwpx"


def test_validate_hwp_extension_rejects_unsupported_extension() -> None:
    with pytest.raises(UnsupportedHwpExtensionError, match="Unsupported HWP/HWPX document extension: .txt"):
        validate_hwp_extension("notes.txt")


def test_parse_hwp_document_uses_fake_extractor_path(tmp_path: Path) -> None:
    document = tmp_path / "sample.hwpx"
    document.write_text("placeholder", encoding="utf-8")
    extractor = FakeHwpExtractor()

    parsed = parse_hwp_document(document, extractor=extractor)

    assert extractor.seen_path == document
    assert parsed.parser_name == "hwp"
    assert parsed.source == "fake-hwp"
    assert parsed.segments[0].text == "고유검색토큰-한글-HWP"


def test_parse_hwp_document_unavailable_extractor_has_clear_error(tmp_path: Path) -> None:
    document = tmp_path / "sample.hwp"
    document.write_bytes(b"placeholder")

    with pytest.raises(HwpToolUnavailableError, match="extractor 'unavailable-hwp-extractor' is not available"):
        parse_hwp_document(document)


def test_parse_hwp_document_rejects_unsupported_extension_before_extractor(tmp_path: Path) -> None:
    document = tmp_path / "sample.pdf"
    document.write_bytes(b"placeholder")
    extractor = FakeHwpExtractor()

    with pytest.raises(UnsupportedHwpExtensionError):
        parse_hwp_document(document, extractor=extractor)
    assert extractor.seen_path is None


def test_parse_hwp_document_wraps_unexpected_extractor_errors(tmp_path: Path) -> None:
    document = tmp_path / "broken.hwpx"
    document.write_bytes(b"placeholder")

    with pytest.raises(HwpParseError, match="Could not parse broken.hwpx with fake-hwp: boom"):
        parse_hwp_document(document, extractor=FakeHwpExtractor(fail=True))


def test_segment_metadata_fields_are_preserved() -> None:
    segment = ParsedHwpSegment(text="표검색토큰-한글-HWPX", page_number=3, section_title="표 섹션")
    parsed = ParsedHwpDocument(segments=(segment,), source="fixture")

    assert parsed.segments == (segment,)
    assert parsed.segments[0].page_number == 3
    assert parsed.segments[0].section_title == "표 섹션"
    assert parsed.source == "fixture"
