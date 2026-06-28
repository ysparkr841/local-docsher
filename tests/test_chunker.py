from __future__ import annotations

import pytest

from docsher.chunker import approximate_token_count, chunk_text


def test_empty_or_whitespace_text_produces_no_chunks() -> None:
    assert chunk_text("") == ()
    assert chunk_text("\n  \t\n") == ()


def test_short_text_creates_single_chunk_with_index_and_token_count() -> None:
    chunks = chunk_text("alpha beta gamma")

    assert len(chunks) == 1
    assert chunks[0].chunk_index == 0
    assert chunks[0].text == "alpha beta gamma"
    assert chunks[0].token_count >= 3


def test_chunker_prefers_paragraph_boundaries() -> None:
    text = "first paragraph\n\nsecond paragraph\n\nthird paragraph"

    chunks = chunk_text(text, max_chars=35, overlap_chars=5)

    assert [chunk.chunk_index for chunk in chunks] == [0, 1]
    assert chunks[0].text == "first paragraph\n\nsecond paragraph"
    assert chunks[1].text == "third paragraph"


def test_oversized_paragraph_is_split_with_overlap() -> None:
    text = "abcdefghijklmnopqrstuvwxyz"

    chunks = chunk_text(text, max_chars=10, overlap_chars=2)

    assert [chunk.text for chunk in chunks] == ["abcdefghij", "ijklmnopqr", "qrstuvwxyz"]
    assert [chunk.chunk_index for chunk in chunks] == [0, 1, 2]
    assert all(chunk.token_count > 0 for chunk in chunks)


def test_approximate_token_count_handles_korean_without_spaces() -> None:
    assert approximate_token_count("한글문서검색") >= 2


@pytest.mark.parametrize(
    ("max_chars", "overlap_chars", "message"),
    [
        (0, 0, "max_chars"),
        (10, -1, "overlap_chars"),
        (10, 10, "smaller"),
    ],
)
def test_chunker_validates_size_options(
    max_chars: int,
    overlap_chars: int,
    message: str,
) -> None:
    with pytest.raises(ValueError, match=message):
        chunk_text("text", max_chars=max_chars, overlap_chars=overlap_chars)
