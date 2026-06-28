"""Deterministic text chunk creation for Local Docsher."""

from __future__ import annotations

import re
from dataclasses import dataclass

DEFAULT_CHUNK_MAX_CHARS = 2_000
DEFAULT_CHUNK_OVERLAP_CHARS = 200


@dataclass(frozen=True)
class TextChunk:
    """A database-ready text chunk."""

    chunk_index: int
    text: str
    token_count: int


def approximate_token_count(text: str) -> int:
    """Return a small stdlib-only token estimate for mixed Korean/technical text."""

    if not text.strip():
        return 0
    word_like = re.findall(r"\S+", text)
    # Whitespace-separated counts work well for code/English; the character
    # floor avoids badly under-counting Korean or other text with few spaces.
    non_space_chars = sum(1 for char in text if not char.isspace())
    char_estimate = max(1, (non_space_chars + 3) // 4)
    return max(len(word_like), char_estimate)


def _split_oversized_text(text: str, max_chars: int, overlap_chars: int) -> list[str]:
    parts: list[str] = []
    start = 0
    text_length = len(text)
    while start < text_length:
        end = min(start + max_chars, text_length)
        parts.append(text[start:end].strip())
        if end >= text_length:
            break
        start = max(end - overlap_chars, start + 1)
    return [part for part in parts if part]


def chunk_text(
    text: str,
    *,
    max_chars: int = DEFAULT_CHUNK_MAX_CHARS,
    overlap_chars: int = DEFAULT_CHUNK_OVERLAP_CHARS,
) -> tuple[TextChunk, ...]:
    """Split document text into deterministic non-empty chunks.

    The MVP chunker prefers paragraph boundaries and falls back to fixed-size
    character windows for paragraphs that exceed ``max_chars``. Empty/whitespace
    documents produce zero chunks.
    """

    if max_chars <= 0:
        raise ValueError("max_chars must be positive")
    if overlap_chars < 0:
        raise ValueError("overlap_chars must be non-negative")
    if overlap_chars >= max_chars:
        raise ValueError("overlap_chars must be smaller than max_chars")

    normalized_text = text.replace("\r\n", "\n").replace("\r", "\n").strip()
    if not normalized_text:
        return ()

    paragraphs = [paragraph.strip() for paragraph in re.split(r"\n\s*\n", normalized_text)]
    chunks: list[str] = []
    current = ""

    def flush_current() -> None:
        nonlocal current
        if current.strip():
            chunks.append(current.strip())
        current = ""

    for paragraph in paragraphs:
        if not paragraph:
            continue
        if len(paragraph) > max_chars:
            flush_current()
            chunks.extend(_split_oversized_text(paragraph, max_chars, overlap_chars))
            continue

        candidate = paragraph if not current else f"{current}\n\n{paragraph}"
        if len(candidate) <= max_chars:
            current = candidate
        else:
            flush_current()
            current = paragraph

    flush_current()

    return tuple(
        TextChunk(
            chunk_index=index,
            text=chunk,
            token_count=approximate_token_count(chunk),
        )
        for index, chunk in enumerate(chunks)
    )
