from __future__ import annotations

import json
import os
import sqlite3
import subprocess
import sys
from pathlib import Path

from fastapi.testclient import TestClient

from docsher.api import create_app
from docsher.ask import UNKNOWN_ANSWER, ask_question, build_grounded_prompt
from docsher.config import ENV_CONFIG_PATH, default_config
from docsher.db import init_database
from docsher.llm import DummyLLMClient


def insert_indexed_chunk(
    database_path: Path,
    *,
    path: str,
    filename: str,
    extension: str,
    text: str,
    chunk_index: int = 0,
    page_number: int | None = None,
    slide_number: int | None = None,
    sheet_name: str | None = None,
    section_title: str | None = None,
) -> int:
    init_database(database_path)
    with sqlite3.connect(database_path) as connection:
        cursor = connection.execute(
            """
            INSERT INTO documents(path, filename, extension, size, modified_at, status)
            VALUES (?, ?, ?, ?, datetime('now'), 'indexed')
            """,
            (path, filename, extension, len(text.encode("utf-8"))),
        )
        document_id = int(cursor.lastrowid)
        cursor = connection.execute(
            """
            INSERT INTO chunks(
                document_id, chunk_index, text, page_number, sheet_name,
                slide_number, section_title, token_count
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                document_id,
                chunk_index,
                text,
                page_number,
                sheet_name,
                slide_number,
                section_title,
                len(text.split()),
            ),
        )
        return int(cursor.lastrowid)


def make_config(tmp_path: Path, database_path: Path) -> Path:
    config = default_config()
    config["storage"]["database_path"] = str(database_path)
    config["documents"]["roots"] = [str(tmp_path)]
    config["llm"]["provider"] = "dummy"
    config_path = tmp_path / "config.json"
    config_path.write_text(json.dumps(config), encoding="utf-8")
    return config_path


def test_ask_question_uses_search_evidence_and_returns_sources(tmp_path: Path) -> None:
    database_path = tmp_path / "docsher.sqlite3"
    insert_indexed_chunk(
        database_path,
        path=str(tmp_path / "docs" / "policy.txt"),
        filename="policy.txt",
        extension=".txt",
        text="환불 정책은 구매 후 7일 이내 신청할 수 있습니다.",
        page_number=2,
        section_title="환불",
    )
    llm_client = DummyLLMClient("구매 후 7일 이내 신청할 수 있습니다. [Source 1]")

    response = ask_question("환불 정책은?", llm_client=llm_client, database_path=database_path, top_k=3)

    assert response.question == "환불 정책은?"
    assert "7일" in response.answer
    assert len(response.sources) == 1
    source = response.sources[0]
    assert source.filename == "policy.txt"
    assert source.document_id >= 1
    assert source.path.endswith("policy.txt")
    assert source.chunk_index == 0
    assert source.page_number == 2
    assert source.section_title == "환불"
    assert "환불" in source.snippet
    assert len(llm_client.seen_messages) == 2
    assert "Do not use outside knowledge" in llm_client.seen_messages[0].content
    prompt = llm_client.seen_messages[1].content
    assert "Evidence snippets (the only allowed source of truth)" in prompt
    assert "policy.txt" in prompt
    assert "환불" in prompt


def test_ask_question_does_not_call_llm_without_search_results(tmp_path: Path) -> None:
    database_path = tmp_path / "docsher.sqlite3"
    init_database(database_path)
    llm_client = DummyLLMClient("should not be used")

    response = ask_question("없는 질문", llm_client=llm_client, database_path=database_path)

    assert response.answer == UNKNOWN_ANSWER
    assert response.sources == ()
    assert llm_client.seen_messages == []


def test_build_grounded_prompt_restricts_guessing(tmp_path: Path) -> None:
    database_path = tmp_path / "docsher.sqlite3"
    insert_indexed_chunk(
        database_path,
        path=str(tmp_path / "manual.md"),
        filename="manual.md",
        extension=".md",
        text="alpha evidence",
        slide_number=4,
        sheet_name="Sheet1",
    )
    response = ask_question("alpha", llm_client=DummyLLMClient("ok"), database_path=database_path)
    prompt = build_grounded_prompt(response.question, response.sources)

    assert "Answer only from the evidence snippets above" in prompt
    assert UNKNOWN_ANSWER in prompt
    assert "slide 4" in prompt
    assert "sheet Sheet1" in prompt


def test_api_post_ask_uses_injected_llm_client(tmp_path: Path) -> None:
    database_path = tmp_path / "docsher.sqlite3"
    config_path = make_config(tmp_path, database_path)
    insert_indexed_chunk(
        database_path,
        path=str(tmp_path / "docs" / "faq.txt"),
        filename="faq.txt",
        extension=".txt",
        text="지원 시간은 평일 오전 9시부터 오후 6시까지입니다.",
        chunk_index=1,
    )
    app = create_app(config_path=config_path, database_path=database_path)
    app.state.llm_client = DummyLLMClient("지원 시간은 평일 9시부터 18시까지입니다. [Source 1]")
    client = TestClient(app)

    response = client.post("/ask", json={"question": "지원 시간은?", "top_k": 2})

    assert response.status_code == 200, response.text
    payload = response.json()
    assert payload["question"] == "지원 시간은?"
    assert "9시" in payload["answer"]
    assert payload["sources"][0]["filename"] == "faq.txt"
    assert payload["sources"][0]["chunk_index"] == 1
    assert "snippet" in payload["sources"][0]


def test_cli_ask_outputs_human_and_json_with_dummy_provider(tmp_path: Path) -> None:
    database_path = tmp_path / "docsher.sqlite3"
    config_path = make_config(tmp_path, database_path)
    insert_indexed_chunk(
        database_path,
        path=str(tmp_path / "docs" / "guide.txt"),
        filename="guide.txt",
        extension=".txt",
        text="CLI 질문 답변은 검색 근거를 사용합니다.",
    )
    env = os.environ.copy()
    env[ENV_CONFIG_PATH] = str(config_path)
    cwd = Path(__file__).resolve().parents[1]

    human = subprocess.run(
        [
            sys.executable,
            "-m",
            "docsher",
            "ask",
            "CLI 질문",
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
            "ask",
            "CLI 질문",
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
    assert "Question: CLI 질문" in human.stdout
    assert "Sources:" in human.stdout
    assert "guide.txt" in human.stdout
    assert json_output.returncode == 0, json_output.stderr
    payload = json.loads(json_output.stdout)
    assert payload["question"] == "CLI 질문"
    assert payload["answer"] == "dummy LLM response"
    assert payload["sources"][0]["filename"] == "guide.txt"
