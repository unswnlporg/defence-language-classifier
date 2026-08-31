#!/usr/bin/env python3
"""Convert document-level Defence JSONL into classifier-ready passages."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections import Counter
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from defence_language_classifier.chunking import chunk_text, word_count  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("input", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--min-words", type=int, default=50)
    parser.add_argument("--max-words", type=int, default=200)
    return parser.parse_args()


def stable_id(document_id: str, chunk_index: int, text: str) -> str:
    payload = f"{document_id}\0{chunk_index}\0{text}".encode("utf-8")
    return hashlib.sha256(payload).hexdigest()[:20]


def main() -> None:
    args = parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    stats = Counter()
    word_counts = []
    seen_ids = set()

    with args.input.open(encoding="utf-8") as source, args.output.open("w", encoding="utf-8") as target:
        for line_number, line in enumerate(source, start=1):
            stats["documents_read"] += 1
            row = json.loads(line)
            document_id = row.get("doc_id")
            text = row.get("text")
            if not document_id or not isinstance(text, str) or not text.strip():
                stats["documents_skipped_invalid"] += 1
                continue

            chunks = chunk_text(text, args.min_words, args.max_words)
            if not chunks or word_count(text) < args.min_words:
                stats["documents_skipped_too_short"] += 1
                continue

            for chunk_index, chunk in enumerate(chunks):
                count = word_count(chunk)
                if count < args.min_words or count > args.max_words:
                    stats["chunks_dropped_out_of_bounds"] += 1
                    continue
                example_id = stable_id(document_id, chunk_index, chunk)
                if example_id in seen_ids:
                    stats["chunks_dropped_duplicate_id"] += 1
                    continue
                seen_ids.add(example_id)
                record = {
                    "example_id": example_id,
                    "document_id": document_id,
                    "chunk_index": chunk_index,
                    "text": chunk,
                    "label": 1,
                    "source_group": "defence",
                    "source_name": "forge_war_college",
                    "negative_type": None,
                    "word_count": count,
                    "title": row.get("title"),
                }
                target.write(json.dumps(record, ensure_ascii=False) + "\n")
                word_counts.append(count)
                stats["chunks_written"] += 1

    report = {
        **stats,
        "minimum_chunk_words": min(word_counts) if word_counts else 0,
        "maximum_chunk_words": max(word_counts) if word_counts else 0,
        "average_chunk_words": round(sum(word_counts) / len(word_counts), 2) if word_counts else 0,
        "output": str(args.output.resolve()),
    }
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()

