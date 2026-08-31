"""Deterministic, sentence-aware document chunking."""

from __future__ import annotations

import re
from typing import Iterable, List


SENTENCE_BOUNDARY = re.compile(r"(?<=[.!?])(?:[\"'’”)]*)\s+(?=[A-Z0-9\"'‘“(])")
WHITESPACE = re.compile(r"\s+")


def normalise_text(text: str) -> str:
    """Collapse layout whitespace while preserving readable prose."""
    return WHITESPACE.sub(" ", text).strip()


def word_count(text: str) -> int:
    return len(text.split())


def split_sentences(text: str) -> List[str]:
    """Split prose at conservative sentence boundaries."""
    cleaned = normalise_text(text)
    if not cleaned:
        return []
    return [piece.strip() for piece in SENTENCE_BOUNDARY.split(cleaned) if piece.strip()]


def split_long_unit(text: str, max_words: int) -> List[str]:
    """Word-split a sentence only when it cannot fit within a chunk."""
    words = text.split()
    return [" ".join(words[start : start + max_words]) for start in range(0, len(words), max_words)]


def _rebalance_tail(chunks: List[List[str]], min_words: int, max_words: int) -> None:
    """Move trailing sentences so the final chunk reaches the minimum when possible."""
    if len(chunks) < 2 or word_count(" ".join(chunks[-1])) >= min_words:
        return

    previous = chunks[-2]
    tail = chunks[-1]
    if word_count(" ".join(previous + tail)) <= max_words:
        chunks[-2] = previous + tail
        chunks.pop()
        return

    while len(previous) > 1 and word_count(" ".join(tail)) < min_words:
        candidate = previous[-1]
        if word_count(" ".join([candidate] + tail)) > max_words:
            break
        tail.insert(0, previous.pop())


def chunk_text(text: str, min_words: int = 50, max_words: int = 200) -> List[str]:
    """Return sentence-aware chunks within bounds whenever the document permits."""
    if min_words <= 0 or max_words < min_words:
        raise ValueError("Require 0 < min_words <= max_words")

    units: List[str] = []
    for sentence in split_sentences(text):
        if word_count(sentence) > max_words:
            units.extend(split_long_unit(sentence, max_words))
        else:
            units.append(sentence)

    chunks: List[List[str]] = []
    current: List[str] = []
    current_words = 0

    for unit in units:
        unit_words = word_count(unit)
        if current and current_words + unit_words > max_words:
            chunks.append(current)
            current = []
            current_words = 0
        current.append(unit)
        current_words += unit_words

    if current:
        chunks.append(current)

    _rebalance_tail(chunks, min_words, max_words)
    return [" ".join(chunk).strip() for chunk in chunks]


def valid_chunks(chunks: Iterable[str], min_words: int, max_words: int) -> bool:
    return all(min_words <= word_count(chunk) <= max_words for chunk in chunks)

