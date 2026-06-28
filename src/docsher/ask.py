"""Search-grounded question answering for Local Docsher."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from docsher.llm import LLMClient, LLMMessage
from docsher.search import SearchResult, search_documents

UNKNOWN_ANSWER = "I do not know based on the indexed documents."
SYSTEM_PROMPT = """You are Local Docsher, a local document question-answering assistant.
Answer using only the evidence snippets provided in this conversation.
Do not use outside knowledge and do not guess or infer facts that are not supported by the evidence.
If the evidence is missing, ambiguous, or insufficient to answer the question, say: "I do not know based on the indexed documents."
Cite the provided source labels when they support your answer."""


@dataclass(frozen=True)
class AskSource:
    """Source chunk metadata returned with grounded answers."""

    document_id: int
    filename: str
    path: str
    chunk_index: int
    snippet: str
    chunk_id: int | None = None
    extension: str | None = None
    page_number: int | None = None
    slide_number: int | None = None
    sheet_name: str | None = None
    section_title: str | None = None
    rank: float | None = None

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


@dataclass(frozen=True)
class AskResponse:
    """Structured answer plus the evidence used to produce it."""

    question: str
    answer: str
    sources: tuple[AskSource, ...]

    def to_dict(self) -> dict[str, object]:
        return {
            "question": self.question,
            "answer": self.answer,
            "sources": [source.to_dict() for source in self.sources],
        }


def source_from_search_result(result: SearchResult) -> AskSource:
    """Convert a search result into public ask source metadata."""

    return AskSource(
        document_id=result.document_id,
        filename=result.filename,
        path=result.path,
        chunk_index=result.chunk_index,
        snippet=result.snippet,
        chunk_id=result.chunk_id,
        extension=result.extension,
        page_number=result.page_number,
        slide_number=result.slide_number,
        sheet_name=result.sheet_name,
        section_title=result.section_title,
        rank=result.rank,
    )


def build_grounded_prompt(question: str, sources: tuple[AskSource, ...]) -> str:
    """Build the user prompt containing only retrieved evidence and metadata."""

    evidence_blocks: list[str] = []
    for index, source in enumerate(sources, start=1):
        location_parts = [f"chunk {source.chunk_index}"]
        if source.page_number is not None:
            location_parts.append(f"page {source.page_number}")
        if source.slide_number is not None:
            location_parts.append(f"slide {source.slide_number}")
        if source.sheet_name:
            location_parts.append(f"sheet {source.sheet_name}")
        if source.section_title:
            location_parts.append(f"section {source.section_title}")
        metadata = {
            "document_id": source.document_id,
            "filename": source.filename,
            "path": source.path,
            "location": ", ".join(location_parts),
        }
        evidence_blocks.append(
            f"[Source {index}]\n"
            f"Metadata: {json.dumps(metadata, ensure_ascii=False, sort_keys=True)}\n"
            f"Snippet:\n{source.snippet}"
        )

    evidence = "\n\n".join(evidence_blocks)
    return (
        f"Question:\n{question}\n\n"
        f"Evidence snippets (the only allowed source of truth):\n{evidence}\n\n"
        "Instructions:\n"
        "- Answer only from the evidence snippets above.\n"
        "- If the evidence is insufficient, answer exactly: "
        f"{UNKNOWN_ANSWER}\n"
        "- Include source labels like [Source 1] for claims supported by evidence."
    )


def ask_question(
    question: str,
    *,
    llm_client: LLMClient,
    database_path: str | Path | None = None,
    top_k: int = 5,
) -> AskResponse:
    """Answer a question using search results as the only evidence."""

    cleaned_question = question.strip()
    if not cleaned_question:
        raise ValueError("Question must not be empty")
    if top_k < 1:
        raise ValueError("top_k must be a positive integer")

    search_results = search_documents(cleaned_question, database_path=database_path, top_k=top_k)
    sources = tuple(source_from_search_result(result) for result in search_results)
    if not sources:
        return AskResponse(question=cleaned_question, answer=UNKNOWN_ANSWER, sources=())

    messages = [
        LLMMessage(role="system", content=SYSTEM_PROMPT),
        LLMMessage(role="user", content=build_grounded_prompt(cleaned_question, sources)),
    ]
    response = llm_client.chat(messages, temperature=0.0)
    answer = response.text.strip() or UNKNOWN_ANSWER
    return AskResponse(question=cleaned_question, answer=answer, sources=sources)


def format_ask_response(response: AskResponse) -> str:
    """Format an ask response for human-readable CLI output."""

    lines = [f"Question: {response.question}", "", response.answer]
    if not response.sources:
        return "\n".join(lines)

    lines.extend(["", "Sources:"])
    for index, source in enumerate(response.sources, start=1):
        lines.append(f"{index}. {source.filename} — {source.path}")
        location_parts = [f"chunk {source.chunk_index}"]
        if source.page_number is not None:
            location_parts.append(f"page {source.page_number}")
        if source.slide_number is not None:
            location_parts.append(f"slide {source.slide_number}")
        if source.sheet_name:
            location_parts.append(f"sheet {source.sheet_name}")
        if source.section_title:
            location_parts.append(f"section {source.section_title}")
        lines.append(f"   Location: {', '.join(location_parts)}")
        lines.append(f"   Snippet: {source.snippet}")
    return "\n".join(lines)
