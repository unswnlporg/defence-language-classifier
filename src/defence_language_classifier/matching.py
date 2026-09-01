"""Scoring and filtering utilities for topic-matched Wikipedia negatives."""

from __future__ import annotations

import re
from typing import Iterable, Sequence, Set, Tuple

TOKEN = re.compile(r"[a-z0-9]+")
DISAMBIGUATION_MARKERS = (
    "may refer to:",
    "may also refer to:",
    "refers to:",
)


def tokens(text: str) -> list[str]:
    return TOKEN.findall(text.lower())


def token_set(text: str) -> Set[str]:
    return set(tokens(text))


def char_shingles(text: str, n: int = 8) -> Set[str]:
    """Normalised character n-grams, used to catch near-duplicate prose."""
    normalised = re.sub(r"\s+", " ", text.lower()).strip()
    if len(normalised) < n:
        return {normalised} if normalised else set()
    return {normalised[i : i + n] for i in range(len(normalised) - n + 1)}


def jaccard(a: Set[str], b: Set[str]) -> float:
    if not a or not b:
        return 0.0
    intersection = len(a & b)
    union = len(a | b)
    return intersection / union if union else 0.0


def is_near_duplicate(a_text: str, b_text: str, threshold: float = 0.5) -> bool:
    return jaccard(char_shingles(a_text), char_shingles(b_text)) >= threshold


def keyword_overlap(a_text: str, b_text: str) -> float:
    """Cheap lexical overlap between a Defence passage and a candidate passage."""
    return jaccard(token_set(a_text), token_set(b_text))


def length_similarity(a_words: int, b_words: int) -> float:
    if a_words <= 0 or b_words <= 0:
        return 0.0
    return 1.0 - abs(a_words - b_words) / max(a_words, b_words)


def match_score(
    semantic_similarity: float,
    keyword_overlap_score: float,
    length_similarity_score: float,
    weights: Tuple[float, float, float] = (0.7, 0.2, 0.1),
) -> float:
    w_semantic, w_keyword, w_length = weights
    return (
        w_semantic * semantic_similarity
        + w_keyword * keyword_overlap_score
        + w_length * length_similarity_score
    )


def is_list_or_disambiguation(title: str, extract: str) -> bool:
    lowered_title = title.lower()
    if lowered_title.startswith("list of") or lowered_title.startswith("index of"):
        return True
    if lowered_title.endswith("(disambiguation)"):
        return True
    head = extract[:400].lower()
    return any(marker in head for marker in DISAMBIGUATION_MARKERS)


def top_tfidf_terms(feature_names: Sequence[str], row, top_k: int = 3) -> list[str]:
    """Top-weighted TF-IDF terms for one row of a fitted TF-IDF matrix."""
    coo = row.tocoo()
    pairs = sorted(zip(coo.col, coo.data), key=lambda item: item[1], reverse=True)
    seen: list[str] = []
    for col, _weight in pairs:
        term = feature_names[col]
        if any(term in existing or existing in term for existing in seen):
            continue
        seen.append(term)
        if len(seen) >= top_k:
            break
    return seen


def word_tolerance(target_words: int, min_absolute: int = 15, relative: float = 0.10) -> int:
    return max(min_absolute, round(target_words * relative))


def sentence_windows(
    sentences: Sequence[str], sentence_word_counts: Sequence[int], target: int, tolerance: int
) -> Iterable[Tuple[int, int, str, int]]:
    """Contiguous sentence runs whose combined word count lands within tolerance of target.

    Yields (start_index, end_index_exclusive, joined_text, word_count).
    """
    lower, upper = target - tolerance, target + tolerance
    n = len(sentences)
    for start in range(n):
        total = 0
        for end in range(start, n):
            total += sentence_word_counts[end]
            if total > upper:
                break
            if total >= lower:
                yield (start, end + 1, " ".join(sentences[start : end + 1]), total)
