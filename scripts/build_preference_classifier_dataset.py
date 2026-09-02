#!/usr/bin/env python3
"""Build a classifier dataset from generated preference pairs.

Reads QC-passed (target_answer, dispreferred_answer) pairs from the
prefqa-generation corpus and reshapes them into the same schema as
dataset_splits.jsonl elsewhere in this project, so the existing
train_baselines.py pipeline can be reused unchanged: target_answer becomes a
label=1 example, dispreferred_answer becomes a label=0 example.

Each source `doc_id` in the prefqa-generation corpus corresponds to exactly
one candidate (one chunk per source document was sampled), so a seeded
document-group split has no leakage risk by construction, but this still
goes through the project's standard group-split utility rather than a plain
random split, in case that one-doc-per-candidate property changes upstream.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from defence_language_classifier.chunking import word_count  # noqa: E402
from defence_language_classifier.training import (  # noqa: E402
    apply_splits,
    assign_group_splits,
    file_sha256,
    validate_dataset,
    write_jsonl,
)


def stable_id(*parts: str) -> str:
    import hashlib

    return hashlib.sha256("\0".join(parts).encode("utf-8")).hexdigest()[:20]


def read_shards(paths: list[Path]) -> list[dict]:
    rows = []
    for path in paths:
        with path.open(encoding="utf-8") as stream:
            for line in stream:
                if line.strip():
                    rows.append(json.loads(line))
    return rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--shards", type=Path, nargs="+", required=True)
    parser.add_argument("--output", type=Path, default=Path("data/processed/preference_classifier_dataset_splits.jsonl"))
    parser.add_argument("--manifest", type=Path, default=Path("data/processed/preference_classifier_split_manifest.json"))
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    raw = read_shards(args.shards)
    qc_passed = [r for r in raw if r.get("automatic_qc", {}).get("passed") and "doc_id" in r]
    print(f"{len(raw)} total rows, {len(qc_passed)} pass automatic_qc.")

    records = []
    for row in qc_passed:
        doc_id = row["doc_id"]
        candidate_id = row["candidate_id"]
        for kind, label in (("target_answer", 1), ("dispreferred_answer", 0)):
            text = row[kind]
            records.append(
                {
                    "example_id": stable_id(candidate_id, kind, text),
                    "document_id": doc_id,
                    "chunk_index": 0,
                    "text": text,
                    "label": label,
                    "source_group": "preference_pair",
                    "source_name": "prefqa-generation-qwen3-32b-awq",
                    "negative_type": None if label == 1 else "dispreferred_answer",
                    "word_count": word_count(text),
                    "candidate_id": candidate_id,
                    "chunk_id": row.get("chunk_id"),
                    "question": row.get("question"),
                }
            )

    ratios = {"train": 0.70, "validation": 0.15, "test": 0.15}
    assignment = assign_group_splits(records, ratios, args.seed)
    split_records = apply_splits(records, assignment)
    summary = validate_dataset(split_records)
    split_records.sort(key=lambda row: (row["split"], row["document_id"], row["chunk_index"]))
    write_jsonl(args.output, split_records)

    manifest = {
        "seed": args.seed,
        "ratios": ratios,
        "sources": [str(p) for p in args.shards],
        "total_qc_passed_pairs": len(qc_passed),
        "total_examples": len(records),
        "output_sha256": file_sha256(args.output),
        "summary": summary,
    }
    args.manifest.parent.mkdir(parents=True, exist_ok=True)
    args.manifest.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
