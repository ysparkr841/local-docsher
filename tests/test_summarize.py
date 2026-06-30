from __future__ import annotations

import json
import os
import sqlite3
import subprocess
import sys
from pathlib import Path

from fastapi.testclient import TestClient

from docsher.api import create_app
from docsher.config import ENV_CONFIG_PATH, default_config
from docsher.db import init_database
from docsher.llm import DummyLLMClient
from docsher.summarize import build_summary_prompt, summarize_document


def insert_indexed_document(
    database_path: Path,
    *,
    path: str,
    filename: str,
    text: str,
    content_hash: str = "hash-one",
) -> int:
    init_database(database_path)
    with sqlite3.connect(database_path) as connection:
        cursor = connection.execute(
            """
            INSERT INTO documents(path, filename, extension, size, modified_at, content_hash, status)
            VALUES (?, ?, '.txt', ?, datetime('now'), ?, 'indexed')
            """,
            (path, filename, len(text.encode("utf-8")), content_hash),
        )
        document_id = int(cursor.lastrowid)
        connection.execute(
            """
            INSERT INTO chunks(document_id, chunk_index, text, token_count, section_title)
            VALUES (?, 0, ?, ?, '개요')
            """,
            (document_id, text, len(text.split())),
        )
        return document_id


def make_config(tmp_path: Path, database_path: Path) -> Path:
    config = default_config()
    config["storage"]["database_path"] = str(database_path)
    config["documents"]["roots"] = [str(tmp_path / "docs")]
    config["llm"]["provider"] = "dummy"
    config_path = tmp_path / "config.json"
    config_path.write_text(json.dumps(config), encoding="utf-8")
    return config_path


def summary_json() -> str:
    return json.dumps(
        {
            "summary": "문서는 VPN 신청 절차와 승인 기준을 설명합니다.",
            "keywords": ["VPN", "신청", "승인"],
            "document_type_candidate": "manual",
        },
        ensure_ascii=False,
    )


def test_summarize_document_generates_and_persists_structured_summary(tmp_path: Path) -> None:
    database_path = tmp_path / "docsher.sqlite3"
    document_id = insert_indexed_document(
        database_path,
        path=str(tmp_path / "docs" / "vpn.txt"),
        filename="vpn.txt",
        text="VPN 신청은 보안 포털에서 요청하고 팀장 승인을 받아야 합니다.",
    )
    llm_client = DummyLLMClient(summary_json(), model="local-small")

    summary = summarize_document(document_id, llm_client=llm_client, database_path=database_path)

    assert summary.document_id == document_id
    assert "VPN 신청 절차" in summary.summary
    assert summary.keywords == ("VPN", "신청", "승인")
    assert summary.document_type_candidate == "manual"
    assert summary.model_name == "local-small"
    assert summary.reused is False
    assert len(llm_client.seen_messages) == 2
    assert "Do not use outside knowledge" in llm_client.seen_messages[0].content
    assert "VPN 신청" in llm_client.seen_messages[1].content

    with sqlite3.connect(database_path) as connection:
        row = connection.execute(
            "SELECT summary, keywords, model_name FROM document_summaries WHERE document_id = ?",
            (document_id,),
        ).fetchone()
    assert row is not None
    assert "Document type candidate: manual" in row[0]
    assert json.loads(row[1]) == ["VPN", "신청", "승인"]
    assert row[2] == "local-small"


def test_summarize_reuses_existing_and_same_hash_summaries(tmp_path: Path) -> None:
    database_path = tmp_path / "docsher.sqlite3"
    first_id = insert_indexed_document(
        database_path,
        path=str(tmp_path / "docs" / "a.txt"),
        filename="a.txt",
        text="동일 문서 내용",
        content_hash="same-hash",
    )
    second_id = insert_indexed_document(
        database_path,
        path=str(tmp_path / "docs" / "b.txt"),
        filename="b.txt",
        text="동일 문서 내용",
        content_hash="same-hash",
    )

    first = summarize_document(first_id, llm_client=DummyLLMClient(summary_json()), database_path=database_path)
    existing_client = DummyLLMClient("should not be called")
    existing = summarize_document(first_id, llm_client=existing_client, database_path=database_path)
    same_hash_client = DummyLLMClient("should not be called either")
    copied = summarize_document(second_id, llm_client=same_hash_client, database_path=database_path)

    assert first.reused is False
    assert existing.reused is True
    assert existing.reuse_reason == "existing_document_summary"
    assert existing_client.seen_messages == []
    assert copied.reused is True
    assert copied.reuse_reason == "same_content_hash"
    assert copied.summary == first.summary
    assert same_hash_client.seen_messages == []


def test_build_summary_prompt_includes_metadata_and_bounded_chunks(tmp_path: Path) -> None:
    database_path = tmp_path / "docsher.sqlite3"
    document_id = insert_indexed_document(
        database_path,
        path=str(tmp_path / "docs" / "manual.txt"),
        filename="manual.txt",
        text="alpha beta gamma",
    )
    llm_client = DummyLLMClient(summary_json())
    summarize_document(document_id, llm_client=llm_client, database_path=database_path, force=True)

    prompt = llm_client.seen_messages[1].content
    assert "manual.txt" in prompt
    assert "alpha beta gamma" in prompt
    assert "Return only JSON" in prompt


def test_api_summarize_and_document_detail_include_summary(tmp_path: Path) -> None:
    database_path = tmp_path / "docsher.sqlite3"
    config_path = make_config(tmp_path, database_path)
    document_id = insert_indexed_document(
        database_path,
        path=str(tmp_path / "docs" / "policy.txt"),
        filename="policy.txt",
        text="정책 문서는 휴가 신청 마감일을 설명합니다.",
    )
    app = create_app(config_path=config_path, database_path=database_path)
    app.state.llm_client = DummyLLMClient(summary_json(), model="dummy-summary")
    client = TestClient(app)

    response = client.post(f"/documents/{document_id}/summarize", json={})

    assert response.status_code == 200, response.text
    payload = response.json()
    assert payload["document_id"] == document_id
    assert payload["keywords"] == ["VPN", "신청", "승인"]
    assert payload["document_type_candidate"] == "manual"

    detail = client.get(f"/documents/{document_id}")
    assert detail.status_code == 200
    assert detail.json()["summary"]["model_name"] == "dummy-summary"

    ui = client.get(f"/ui/documents/{document_id}")
    assert ui.status_code == 200
    assert "Summary" in ui.text
    assert "문서는 VPN 신청 절차" in ui.text
    assert "Document type candidate" in ui.text


def test_cli_summarize_outputs_human_and_json_with_dummy_provider(tmp_path: Path) -> None:
    database_path = tmp_path / "docsher.sqlite3"
    config_path = make_config(tmp_path, database_path)
    document_id = insert_indexed_document(
        database_path,
        path=str(tmp_path / "docs" / "cli.txt"),
        filename="cli.txt",
        text="CLI 요약은 로컬 모델 클라이언트를 사용합니다.",
    )
    env = os.environ.copy()
    env[ENV_CONFIG_PATH] = str(config_path)
    cwd = Path(__file__).resolve().parents[1]

    human = subprocess.run(
        [
            sys.executable,
            "-m",
            "docsher",
            "summarize",
            "--document-id",
            str(document_id),
            "--provider",
            "dummy",
            "--database-path",
            str(database_path),
        ],
        cwd=cwd,
        env=env,
        check=False,
        capture_output=True,
        text=True,
    )
    json_output = subprocess.run(
        [
            sys.executable,
            "-m",
            "docsher",
            "summarize",
            "--document-id",
            str(document_id),
            "--provider",
            "dummy",
            "--json",
            "--database-path",
            str(database_path),
        ],
        cwd=cwd,
        env=env,
        check=False,
        capture_output=True,
        text=True,
    )

    assert human.returncode == 0, human.stderr
    assert f"Document ID: {document_id}" in human.stdout
    assert "Summary:" in human.stdout
    assert "Document type candidate" in human.stdout
    assert json_output.returncode == 0, json_output.stderr
    payload = json.loads(json_output.stdout)
    assert payload["document_id"] == document_id
    assert payload["reused"] is True
    assert payload["reuse_reason"] == "existing_document_summary"
