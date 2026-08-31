#!/usr/bin/env python3
"""Create and validate frozen document-level train/validation/test splits."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from defence_language_classifier.training import (  # noqa: E402
    apply_splits,
    assign_group_splits,
    file_sha256,
    read_jsonl,
    validate_dataset,
    write_jsonl,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--positive", type=Path, required=True)
    parser.add_argument("--negative", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    records = read_jsonl([args.positive, args.negative])
    ratios = {"train": 0.70, "validation": 0.15, "test": 0.15}
    assignment = assign_group_splits(records, ratios, args.seed)
    split_records = apply_splits(records, assignment)
    summary = validate_dataset(split_records)
    split_records.sort(key=lambda row: (row["split"], row["document_id"], row["chunk_index"]))
    write_jsonl(args.output, split_records)

    manifest = {
        "seed": args.seed,
        "ratios": ratios,
        "sources": {
            str(args.positive): file_sha256(args.positive),
            str(args.negative): file_sha256(args.negative),
        },
        "output_sha256": file_sha256(args.output),
        "summary": summary,
    }
    args.manifest.parent.mkdir(parents=True, exist_ok=True)
    args.manifest.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()

