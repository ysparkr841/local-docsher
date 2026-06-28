from __future__ import annotations

import sqlite3
from pathlib import Path

from docsher.cli import main
from docsher.config import default_config
from docsher.indexer import index_pending_documents
from docsher.ocr import (
    FakeOCRBackend,
    OCR_STATUS_COMPLETED,
    OCR_STATUS_FAILED,
    OCR_STATUS_QUEUED,
    process_next_ocr_job,
)
from docsher.scanner import scan


def make_config(tmp_path: Path, root: Path) -> dict:
    config = default_config()
    config["documents"]["roots"] = [str(root)]
    config["storage"]["database_path"] = str(tmp_path / "docsher.sqlite3")
    return config


def make_text_pdf(path: Path) -> None:
    path.write_bytes(
        b"%PDF-1.4\n"
        b"1 0 obj << /Type /Catalog /Pages 2 0 R >> endobj\n"
        b"2 0 obj << /Type /Pages /Kids [3 0 R] /Count 1 >> endobj\n"
        b"3 0 obj << /Type /Page /Parent 2 0 R /Contents 4 0 R >> endobj\n"
        b"4 0 obj << /Length 44 >> stream\n"
        b"BT /F1 12 Tf 72 720 Td (Pdf text layer) Tj ET\n"
        b"endstream endobj\n%%EOF\n"
    )


def make_scanned_pdf(path: Path) -> None:
    path.write_bytes(
        b"%PDF-1.4\n"
        b"1 0 obj << /Type /Catalog /Pages 2 0 R >> endobj\n"
        b"2 0 obj << /Type /Pages /Kids [3 0 R] /Count 1 >> endobj\n"
        b"3 0 obj << /Type /Page /Parent 2 0 R /Contents 4 0 R >> endobj\n"
        b"4 0 obj << /Length 23 >> stream\n"
        b"q 100 0 0 100 0 0 cm /Im1 Do Q\n"
        b"endstream endobj\n%%EOF\n"
    )


def make_two_page_scanned_pdf(path: Path) -> None:
    path.write_bytes(
        b"%PDF-1.4\n"
        b"1 0 obj << /Type /Catalog /Pages 2 0 R >> endobj\n"
        b"2 0 obj << /Type /Pages /Kids [3 0 R 4 0 R] /Count 2 >> endobj\n"
        b"3 0 obj << /Type /Page /Parent 2 0 R /Contents 5 0 R >> endobj\n"
        b"4 0 obj << /Type /Page /Parent 2 0 R /Contents 6 0 R >> endobj\n"
        b"5 0 obj << /Length 23 >> stream\nq 100 0 0 100 0 0 cm /Im1 Do Q\nendstream endobj\n"
        b"6 0 obj << /Length 23 >> stream\nq 100 0 0 100 0 0 cm /Im2 Do Q\nendstream endobj\n"
        b"%%EOF\n"
    )


def test_text_layer_pdf_is_indexed_without_ocr_queue(tmp_path: Path) -> None:
    root = tmp_path / "docs"
    root.mkdir()
    make_text_pdf(root / "text.pdf")
    config = make_config(tmp_path, root)
    database_path = Path(config["storage"]["database_path"])

    scan(config)
    result = index_pending_documents(database_path)

    assert result.parsed_documents == 1
    assert result.queued_ocr_documents == 0
    with sqlite3.connect(database_path) as connection:
        document = connection.execute(
            "SELECT status, parser_name, ocr_status FROM documents WHERE filename = 'text.pdf'"
        ).fetchone()
        jobs_count = connection.execute("SELECT COUNT(*) FROM ocr_jobs").fetchone()[0]
    assert document == ("indexed", "office", "not_required")
    assert jobs_count == 0


def test_textless_pdf_is_queued_for_ocr_instead_of_failed(tmp_path: Path) -> None:
    root = tmp_path / "docs"
    root.mkdir()
    make_scanned_pdf(root / "scanned.pdf")
    config = make_config(tmp_path, root)
    database_path = Path(config["storage"]["database_path"])

    scan(config)
    result = index_pending_documents(database_path)

    assert result.parsed_documents == 0
    assert result.failed_documents == 0
    assert result.queued_ocr_documents == 1
    with sqlite3.connect(database_path) as connection:
        row = connection.execute(
            """
            SELECT documents.status, documents.ocr_status, ocr_jobs.status, ocr_jobs.input_path, ocr_jobs.page_number
            FROM documents JOIN ocr_jobs ON ocr_jobs.document_id = documents.id
            WHERE documents.filename = 'scanned.pdf'
            """
        ).fetchone()
    assert row[0:3] == ("pending", OCR_STATUS_QUEUED, OCR_STATUS_QUEUED)
    assert row[3].endswith(".pgm")
    assert Path(row[3]).exists()
    assert row[4] == 1


def test_image_file_is_discovered_and_queued_for_ocr(tmp_path: Path) -> None:
    root = tmp_path / "docs"
    root.mkdir()
    (root / "scan.png").write_bytes(b"\x89PNG\r\n\x1a\n")
    config = make_config(tmp_path, root)
    database_path = Path(config["storage"]["database_path"])

    scan_result = scan(config)
    index_result = index_pending_documents(database_path)

    assert len(scan_result.new_files) == 1
    assert index_result.queued_ocr_documents == 1
    with sqlite3.connect(database_path) as connection:
        row = connection.execute(
            "SELECT filename, extension, ocr_status FROM documents"
        ).fetchone()
    assert row == ("scan.png", ".png", OCR_STATUS_QUEUED)


def test_ocr_test_cli_reports_scanned_pdf_needs_ocr(tmp_path: Path, capsys) -> None:
    scanned_pdf = tmp_path / "scanned.pdf"
    make_scanned_pdf(scanned_pdf)

    exit_code = main(["ocr-test", str(scanned_pdf)])

    captured = capsys.readouterr()
    assert exit_code == 0
    assert '"ocr_required": true' in captured.out
    assert '"page_number": 1' in captured.out


def test_reindex_does_not_reset_completed_page_ocr_jobs(tmp_path: Path) -> None:
    root = tmp_path / "docs"
    root.mkdir()
    make_two_page_scanned_pdf(root / "scanned.pdf")
    config = make_config(tmp_path, root)
    database_path = Path(config["storage"]["database_path"])

    scan(config)
    index_pending_documents(database_path)
    first_job = process_next_ocr_job(database_path, FakeOCRBackend("page one text"))
    assert first_job is not None
    assert first_job.page_number == 1

    index_pending_documents(database_path)

    with sqlite3.connect(database_path) as connection:
        jobs = connection.execute(
            "SELECT page_number, status, result_text FROM ocr_jobs ORDER BY page_number"
        ).fetchall()
        chunks = connection.execute(
            "SELECT text, page_number FROM chunks ORDER BY page_number"
        ).fetchall()
    assert jobs == [
        (1, OCR_STATUS_COMPLETED, "page one text"),
        (2, OCR_STATUS_QUEUED, None),
    ]
    assert chunks == [("page one text", 1)]


def test_failed_page_keeps_multipage_document_failed_after_later_page_success(tmp_path: Path) -> None:
    root = tmp_path / "docs"
    root.mkdir()
    make_two_page_scanned_pdf(root / "scanned.pdf")
    config = make_config(tmp_path, root)
    database_path = Path(config["storage"]["database_path"])

    scan(config)
    index_pending_documents(database_path)
    failed_job = process_next_ocr_job(database_path, FakeOCRBackend("unused", fail=True))
    succeeded_job = process_next_ocr_job(database_path, FakeOCRBackend("page two text"))

    assert failed_job is not None
    assert succeeded_job is not None
    assert failed_job.page_number == 1
    assert failed_job.status == OCR_STATUS_FAILED
    assert succeeded_job.page_number == 2
    assert succeeded_job.status == OCR_STATUS_COMPLETED
    with sqlite3.connect(database_path) as connection:
        document = connection.execute(
            "SELECT status, parser_name, ocr_status FROM documents WHERE filename = 'scanned.pdf'"
        ).fetchone()
        jobs = connection.execute(
            "SELECT page_number, status FROM ocr_jobs ORDER BY page_number"
        ).fetchall()
        chunks = connection.execute(
            "SELECT text, page_number FROM chunks ORDER BY page_number"
        ).fetchall()
    assert document == ("pending", "ocr:fake", OCR_STATUS_FAILED)
    assert jobs == [(1, OCR_STATUS_FAILED), (2, OCR_STATUS_COMPLETED)]
    assert chunks == [("page two text", 2)]


def test_completed_ocr_result_becomes_searchable_chunk_with_page_number(tmp_path: Path) -> None:
    root = tmp_path / "docs"
    root.mkdir()
    make_scanned_pdf(root / "scanned.pdf")
    config = make_config(tmp_path, root)
    database_path = Path(config["storage"]["database_path"])

    scan(config)
    index_pending_documents(database_path)
    job = process_next_ocr_job(database_path, FakeOCRBackend("인식된 OCR 본문", page_number=1))

    assert job is not None
    assert job.status == OCR_STATUS_COMPLETED
    with sqlite3.connect(database_path) as connection:
        document = connection.execute(
            "SELECT status, parser_name, ocr_status FROM documents WHERE filename = 'scanned.pdf'"
        ).fetchone()
        chunk = connection.execute(
            """
            SELECT chunks.text, chunks.page_number
            FROM chunks JOIN documents ON documents.id = chunks.document_id
            WHERE documents.filename = 'scanned.pdf'
            """
        ).fetchone()
        fts_hit_count = connection.execute(
            "SELECT COUNT(*) FROM chunks_fts WHERE chunks_fts MATCH 'OCR'"
        ).fetchone()[0]
    assert document == ("indexed", "ocr:fake", OCR_STATUS_COMPLETED)
    assert chunk == ("인식된 OCR 본문", 1)
    assert fts_hit_count == 1
