from __future__ import annotations

import json
import sqlite3
import zipfile
import zlib
from pathlib import Path

from docsher.config import default_config
from docsher.indexer import index_pending_documents
from docsher.parsers_office import parse_office_document
from docsher.scanner import scan


def _write_zip(path: Path, files: dict[str, str]) -> None:
    with zipfile.ZipFile(path, "w") as archive:
        for name, content in files.items():
            archive.writestr(name, content)


def make_docx(path: Path, text: str = "Docx paragraph text") -> None:
    _write_zip(
        path,
        {
            "word/document.xml": f"""
                <w:document xmlns:w="http://schemas.openxmlformats.org/wordprocessingml/2006/main">
                  <w:body><w:p><w:r><w:t>{text}</w:t></w:r></w:p></w:body>
                </w:document>
            """,
        },
    )


def make_pptx(path: Path) -> None:
    _write_zip(
        path,
        {
            "ppt/slides/slide1.xml": """
                <p:sld xmlns:p="http://schemas.openxmlformats.org/presentationml/2006/main"
                       xmlns:a="http://schemas.openxmlformats.org/drawingml/2006/main">
                  <p:cSld><p:spTree><p:sp><p:txBody><a:p><a:r><a:t>Slide one text</a:t></a:r></a:p></p:txBody></p:sp></p:spTree></p:cSld>
                </p:sld>
            """,
            "ppt/slides/slide2.xml": """
                <p:sld xmlns:p="http://schemas.openxmlformats.org/presentationml/2006/main"
                       xmlns:a="http://schemas.openxmlformats.org/drawingml/2006/main">
                  <p:cSld><p:spTree><p:sp><p:txBody><a:p><a:r><a:t>Slide two text</a:t></a:r></a:p></p:txBody></p:sp></p:spTree></p:cSld>
                </p:sld>
            """,
        },
    )


def make_xlsx(path: Path, *, sheet_id: str = "1", target_sheet: str = "sheet1.xml", sheet_name: str = "Budget") -> None:
    _write_zip(
        path,
        {
            "xl/workbook.xml": f"""
                <workbook xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main"
                  xmlns:r="http://schemas.openxmlformats.org/officeDocument/2006/relationships">
                  <sheets><sheet name="{sheet_name}" sheetId="{sheet_id}" r:id="rId1"/></sheets>
                </workbook>
            """,
            "xl/_rels/workbook.xml.rels": f"""
                <Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">
                  <Relationship Id="rId1" Target="worksheets/{target_sheet}"/>
                </Relationships>
            """,
            f"xl/worksheets/{target_sheet}": """
                <worksheet xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main">
                  <sheetData><row><c t="inlineStr"><is><t>Xlsx cell text</t></is></c></row></sheetData>
                </worksheet>
            """,
        },
    )


def make_pdf(path: Path) -> None:
    path.write_bytes(
        b"%PDF-1.4\n"
        b"1 0 obj << /Type /Catalog /Pages 2 0 R >> endobj\n"
        b"2 0 obj << /Type /Pages /Kids [3 0 R] /Count 1 >> endobj\n"
        b"3 0 obj << /Type /Page /Parent 2 0 R /Contents 4 0 R >> endobj\n"
        b"4 0 obj << /Length 44 >> stream\n"
        b"BT /F1 12 Tf 72 720 Td (Pdf page text) Tj ET\n"
        b"endstream endobj\n%%EOF\n"
    )


def make_flate_pdf(path: Path) -> None:
    payload = zlib.compress(b"BT /F1 12 Tf 72 720 Td (Compressed pdf text) Tj ET")
    path.write_bytes(
        b"%PDF-1.4\n"
        b"1 0 obj << /Type /Catalog /Pages 2 0 R >> endobj\n"
        b"2 0 obj << /Type /Pages /Kids [3 0 R] /Count 1 >> endobj\n"
        b"3 0 obj << /Type /Page /Parent 2 0 R /Contents 4 0 R >> endobj\n"
        + b"4 0 obj << /Filter /FlateDecode /Length " + str(len(payload)).encode() + b" >> stream\n"
        + payload
        + b"\nendstream endobj\n%%EOF\n"
    )


def make_multi_stream_one_page_pdf(path: Path) -> None:
    path.write_bytes(
        b"%PDF-1.4\n"
        b"1 0 obj << /Type /Catalog /Pages 2 0 R >> endobj\n"
        b"2 0 obj << /Type /Pages /Kids [3 0 R] /Count 1 >> endobj\n"
        b"3 0 obj << /Type /Page /Parent 2 0 R /Contents [4 0 R 5 0 R] >> endobj\n"
        b"4 0 obj << /Length 20 >> stream\nBT (First stream) Tj ET\nendstream endobj\n"
        b"5 0 obj << /Length 21 >> stream\nBT (Second stream) Tj ET\nendstream endobj\n%%EOF\n"
    )


def make_kids_order_pdf(path: Path) -> None:
    path.write_bytes(
        b"%PDF-1.4\n"
        b"1 0 obj << /Type /Catalog /Pages 2 0 R >> endobj\n"
        b"2 0 obj << /Type /Pages /Kids [4 0 R 3 0 R] /Count 2 >> endobj\n"
        b"3 0 obj << /Type /Page /Parent 2 0 R /Contents 5 0 R >> endobj\n"
        b"4 0 obj << /Type /Page /Parent 2 0 R /Contents 6 0 R >> endobj\n"
        b"5 0 obj << /Length 27 >> stream\nBT (Second page text) Tj ET\nendstream endobj\n"
        b"6 0 obj << /Length 26 >> stream\nBT (First page text) Tj ET\nendstream endobj\n%%EOF\n"
    )


def make_config(tmp_path: Path, root: Path) -> dict:
    config = default_config()
    config["documents"]["roots"] = [str(root)]
    config["storage"]["database_path"] = str(tmp_path / "docsher.sqlite3")
    return config


def test_pdf_parser_extracts_text_and_page_number(tmp_path: Path) -> None:
    pdf = tmp_path / "sample.pdf"
    make_pdf(pdf)

    parsed = parse_office_document(pdf)

    assert parsed.parser_name == "office"
    assert parsed.segments[0].text == "Pdf page text"
    assert parsed.segments[0].page_number == 1


def test_pdf_parser_decodes_flate_streams(tmp_path: Path) -> None:
    pdf = tmp_path / "compressed.pdf"
    make_flate_pdf(pdf)

    parsed = parse_office_document(pdf)

    assert parsed.segments[0].text == "Compressed pdf text"
    assert parsed.segments[0].page_number == 1


def test_pdf_parser_keeps_multiple_content_streams_on_same_page(tmp_path: Path) -> None:
    pdf = tmp_path / "multi-stream.pdf"
    make_multi_stream_one_page_pdf(pdf)

    parsed = parse_office_document(pdf)

    assert len(parsed.segments) == 1
    assert parsed.segments[0].text == "First stream\nSecond stream"
    assert parsed.segments[0].page_number == 1


def test_pdf_parser_uses_page_tree_kids_order_for_page_numbers(tmp_path: Path) -> None:
    pdf = tmp_path / "kids-order.pdf"
    make_kids_order_pdf(pdf)

    parsed = parse_office_document(pdf)

    assert [(segment.page_number, segment.text) for segment in parsed.segments] == [
        (1, "First page text"),
        (2, "Second page text"),
    ]


def test_docx_parser_extracts_text(tmp_path: Path) -> None:
    docx = tmp_path / "sample.docx"
    make_docx(docx)

    parsed = parse_office_document(docx)

    assert parsed.segments[0].text == "Docx paragraph text"


def test_pptx_parser_extracts_text_and_slide_numbers(tmp_path: Path) -> None:
    pptx = tmp_path / "deck.pptx"
    make_pptx(pptx)

    parsed = parse_office_document(pptx)

    assert [(segment.slide_number, segment.text) for segment in parsed.segments] == [
        (1, "Slide one text"),
        (2, "Slide two text"),
    ]


def test_xlsx_parser_extracts_text_and_sheet_name(tmp_path: Path) -> None:
    xlsx = tmp_path / "sheet.xlsx"
    make_xlsx(xlsx)

    parsed = parse_office_document(xlsx)

    assert parsed.segments[0].text == "Xlsx cell text"
    assert parsed.segments[0].sheet_name == "Budget"


def test_xlsx_parser_uses_relationship_sheet_name_when_sheet_id_differs(tmp_path: Path) -> None:
    xlsx = tmp_path / "sheet.xlsx"
    make_xlsx(xlsx, sheet_id="7", target_sheet="sheet2.xml", sheet_name="NamedSheet")

    parsed = parse_office_document(xlsx)

    assert parsed.segments[0].text == "Xlsx cell text"
    assert parsed.segments[0].sheet_name == "NamedSheet"


def test_indexer_parses_office_documents_with_location_metadata(tmp_path: Path) -> None:
    root = tmp_path / "sample_docs_office"
    root.mkdir()
    make_pdf(root / "sample.pdf")
    make_docx(root / "sample.docx")
    make_pptx(root / "sample.pptx")
    make_xlsx(root / "sample.xlsx")
    config = make_config(tmp_path, root)
    database_path = Path(config["storage"]["database_path"])

    scan_result = scan(config)
    index_result = index_pending_documents(database_path)

    assert len(scan_result.changes) == 4
    assert index_result.parsed_documents == 4
    assert index_result.failed_documents == 0
    assert index_result.created_chunks == 5
    with sqlite3.connect(database_path) as connection:
        docs = connection.execute(
            "SELECT filename, status, parser_name, error_message FROM documents ORDER BY filename"
        ).fetchall()
        chunks = connection.execute(
            """
            SELECT d.filename, c.text, c.page_number, c.slide_number, c.sheet_name
            FROM chunks c JOIN documents d ON d.id = c.document_id
            ORDER BY d.filename, c.chunk_index
            """
        ).fetchall()
        fts_count = connection.execute("SELECT COUNT(*) FROM chunks_fts").fetchone()[0]

    assert docs == [
        ("sample.docx", "indexed", "office", None),
        ("sample.pdf", "indexed", "office", None),
        ("sample.pptx", "indexed", "office", None),
        ("sample.xlsx", "indexed", "office", None),
    ]
    assert ("sample.pdf", "Pdf page text", 1, None, None) in chunks
    assert ("sample.pptx", "Slide one text", None, 1, None) in chunks
    assert ("sample.pptx", "Slide two text", None, 2, None) in chunks
    assert ("sample.xlsx", "Xlsx cell text", None, None, "Budget") in chunks
    assert fts_count == 5


def test_indexer_marks_office_parse_failure_and_continues(tmp_path: Path) -> None:
    root = tmp_path / "docs"
    root.mkdir()
    (root / "bad.pdf").write_text("not actually a pdf", encoding="utf-8")
    make_docx(root / "ok.docx", text="still indexed")
    config = make_config(tmp_path, root)
    database_path = Path(config["storage"]["database_path"])

    scan(config)
    result = index_pending_documents(database_path)

    assert result.parsed_documents == 1
    assert result.failed_documents == 1
    with sqlite3.connect(database_path) as connection:
        rows = connection.execute(
            "SELECT filename, status, parser_name, error_message FROM documents ORDER BY filename"
        ).fetchall()
    assert rows[0][0] == "bad.pdf"
    assert rows[0][1] == "failed"
    assert rows[0][2] == "office"
    assert "Not a PDF" in rows[0][3]
    assert rows[1] == ("ok.docx", "indexed", "office", None)
