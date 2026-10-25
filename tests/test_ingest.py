import os
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from ingest import chunk_text  # noqa: E402


def reconstruct(chunks, overlap):
    """Rebuild the word stream: first chunk in full, then each subsequent chunk
    minus its ``overlap``-word prefix. Equals the original words iff nothing was
    dropped or duplicated."""
    if not chunks:
        return []
    out = chunks[0].split()
    for c in chunks[1:]:
        out.extend(c.split()[overlap:])
    return out


def test_empty_input():
    assert chunk_text("") == []
    assert chunk_text("   \n  ") == []


def test_shorter_than_one_chunk():
    assert chunk_text("a b c", size=5, overlap=1) == ["a b c"]


def test_no_chunk_exceeds_size():
    text = " ".join(f"w{i}" for i in range(1000))
    assert all(len(c.split()) <= 300 for c in chunk_text(text, size=300, overlap=50))


def test_exact_multiple_of_step_length():
    # size=4, overlap=2 -> step=2; 6 words is exactly 3 steps
    assert chunk_text("w0 w1 w2 w3 w4 w5", size=4, overlap=2) == [
        "w0 w1 w2 w3",
        "w2 w3 w4 w5",
    ]


def test_overlap_between_consecutive_chunks():
    size, overlap = 4, 2
    text = " ".join(f"w{i}" for i in range(11))
    chunks = [c.split() for c in chunk_text(text, size=size, overlap=overlap)]
    assert len(chunks) > 1
    for a, b in zip(chunks, chunks[1:]):
        assert a[-overlap:] == b[:overlap]


def test_no_words_dropped_or_duplicated():
    overlap = 2
    words = [f"w{i}" for i in range(23)]
    chunks = chunk_text(" ".join(words), size=4, overlap=overlap)
    assert reconstruct(chunks, overlap) == words


def test_overlap_ge_size_raises():
    with pytest.raises(ValueError) as e:
        chunk_text("a b c d", size=3, overlap=3)
    msg = str(e.value)
    assert "3" in msg
    with pytest.raises(ValueError):
        chunk_text("a b c d", size=3, overlap=5)
