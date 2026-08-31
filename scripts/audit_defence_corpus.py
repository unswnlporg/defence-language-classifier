#!/usr/bin/env python3
"""Audit a JSONL Defence corpus before constructing classifier passages."""

from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("input", type=Path, help="JSONL file containing Defence documents")
    parser.add_argument("--min-words", type=int, default=50)
    parser.add_argument("--max-words", type=int, default=200)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    records = []
    invalid_json = 0

    with args.input.open(encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, start=1):
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                invalid_json += 1
                continue

            text = row.get("text")
            records.append(
                {
                    "line": line_number,
                    "document_id": row.get("doc_id"),
                    "has_text": isinstance(text, str) and bool(text.strip()),
                    "word_count": len(text.split()) if isinstance(text, str) else 0,
                    "review_status": row.get("review_status"),
                    "quality_flag": row.get("extraction_quality_flag"),
                }
            )

    counts = [row["word_count"] for row in records if row["has_text"]]
    document_ids = [row["document_id"] for row in records if row["document_id"]]
    estimated_chunks = sum(
        max(1, (count + args.max_words - 1) // args.max_words)
        for count in counts
        if count >= args.min_words
    )

    report = {
        "input": str(args.input.resolve()),
        "records": len(records),
        "invalid_json_lines": invalid_json,
        "missing_or_empty_text": sum(not row["has_text"] for row in records),
        "missing_document_id": sum(not row["document_id"] for row in records),
        "duplicate_document_ids": len(document_ids) - len(set(document_ids)),
        "word_counts": {
            "minimum": min(counts) if counts else 0,
            "median": statistics.median(counts) if counts else 0,
            "maximum": max(counts) if counts else 0,
            "total": sum(counts),
        },
        "chunk_policy": {
            "min_words": args.min_words,
            "max_words": args.max_words,
            "rough_upper_bound": estimated_chunks,
            "note": "Final chunk counts require sentence-aware chunking and filtering.",
        },
    }
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()

