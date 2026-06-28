from __future__ import annotations

from pathlib import Path

import pytest

from docsher.parsers_text import (
    TextDocumentTooLargeError,
    TextEncodingError,
    is_text_document,
    parse_text_document,
)


def test_txt_parser_reads_utf8_first(tmp_path: Path) -> None:
    doc = tmp_path / "note.txt"
    doc.write_text("hello UTF-8 문서", encoding="utf-8")

    parsed = parse_text_document(doc)

    assert parsed.text == "hello UTF-8 문서"
    assert parsed.encoding == "utf-8"
    assert parsed.parser_name == "text"


def test_md_parser_reads_markdown_as_text(tmp_path: Path) -> None:
    doc = tmp_path / "guide.md"
    doc.write_bytes("# 제목\n\n- 항목 **강조**".encode("utf-8"))

    parsed = parse_text_document(doc)

    assert parsed.text == "# 제목\n\n- 항목 **강조**"
    assert parsed.encoding == "utf-8"


def test_parser_falls_back_to_cp949_for_korean_text(tmp_path: Path) -> None:
    doc = tmp_path / "legacy.txt"
    expected = "레거시 한글 문서"
    doc.write_bytes(expected.encode("cp949"))

    parsed = parse_text_document(doc)

    assert parsed.text == expected
    assert parsed.encoding == "cp949"


def test_parser_reports_clear_encoding_failure(tmp_path: Path) -> None:
    doc = tmp_path / "binary.txt"
    doc.write_bytes(b"\xff\xfe\x00\x81")

    with pytest.raises(TextEncodingError, match="Could not decode"):
        parse_text_document(doc, encodings=("utf-8",))


def test_parser_rejects_too_large_documents_before_reading(tmp_path: Path) -> None:
    doc = tmp_path / "large.md"
    doc.write_text("0123456789", encoding="utf-8")

    with pytest.raises(TextDocumentTooLargeError, match="too large"):
        parse_text_document(doc, max_bytes=5)


def test_is_text_document_matches_txt_and_md_case_insensitively(tmp_path: Path) -> None:
    assert is_text_document(tmp_path / "A.TXT")
    assert is_text_document(tmp_path / "B.Md")
    assert not is_text_document(tmp_path / "C.pdf")
