#!/usr/bin/env python3
"""Combine the topic-matched Defence-vs-Wikipedia dataset with the
preference-pair (target vs dispreferred) dataset into one training set, for
the cross-training experiment: does a classifier trained on both tasks
generalize to both, instead of the poor transfer seen training on either
alone?

30 of the 52 Defence source documents also appear as source documents in the
preference-pair generation corpus, so this cannot just concatenate the two
datasets' existing (independently-computed) splits — a document could land
in train under one split and test under the other, which is leakage. Instead
this re-splits the *union* of both datasets' records at the document level,
in one pass, so every record sharing a document_id lands in the same split
regardless of which dataset it came from.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import sys
from collections import defaultdict
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from defence_language_classifier.chunking import word_count  # noqa: E402
from defence_language_classifier.training import file_sha256, validate_dataset, write_jsonl  # noqa: E402

SPLITS = ("train", "validation", "test")


def stable_id(*parts: str) -> str:
    return hashlib.sha256("\0".join(parts).encode("utf-8")).hexdigest()[:20]


def read_jsonl(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as stream:
        return [json.loads(line) for line in stream if line.strip()]


def document_level_split(records: list[dict], ratios: dict[str, float], seed: int) -> dict[str, str]:
    """Single-stratum version of the project's group-split balancer: groups
    purely by document_id (ignoring label), so it can join two otherwise
    independently-labeled datasets without a stratum mismatch letting a
    shared document split across train/val/test."""
    rng = random.Random(seed)
    groups: dict[str, int] = defaultdict(int)
    for row in records:
        groups[row["document_id"]] += 1

    group_sizes = list(groups.items())
    rng.shuffle(group_sizes)
    group_sizes.sort(key=lambda item: item[1], reverse=True)

    total = sum(size for _, size in group_sizes)
    targets = {split: total * ratios[split] for split in SPLITS}
    counts = {split: 0 for split in SPLITS}
    assignment: dict[str, str] = {}

    for document_id, size in group_sizes:
        def cost(split: str) -> tuple[float, float, str]:
            projected = counts[split] + size
            relative_fill = projected / max(targets[split], 1.0)
            overflow = max(0.0, projected - targets[split]) / max(targets[split], 1.0)
            return (overflow * 10.0 + relative_fill, counts[split], split)

        chosen = min(SPLITS, key=cost)
        assignment[document_id] = chosen
        counts[chosen] += size
    return assignment


def load_preference_records(shard_paths: list[Path]) -> list[dict]:
    raw = []
    for path in shard_paths:
        raw.extend(read_jsonl(path))
    qc_passed = [r for r in raw if r.get("automatic_qc", {}).get("passed") and "doc_id" in r]

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
                }
            )
    return records


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--hybrid-dataset", type=Path, default=Path("data/processed/hybrid_topic_dataset_splits.jsonl"))
    parser.add_argument("--preference-shards", type=Path, nargs="+", required=True)
    parser.add_argument("--output", type=Path, default=Path("data/processed/combined_dataset_splits.jsonl"))
    parser.add_argument("--manifest", type=Path, default=Path("data/processed/combined_split_manifest.json"))
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    hybrid_rows = read_jsonl(args.hybrid_dataset)
    hybrid_rows = [{k: v for k, v in row.items() if k != "split"} for row in hybrid_rows]
    preference_rows = load_preference_records(args.preference_shards)

    combined = hybrid_rows + preference_rows
    ratios = {"train": 0.70, "validation": 0.15, "test": 0.15}
    assignment = document_level_split(combined, ratios, args.seed)
    for row in combined:
        row["split"] = assignment[row["document_id"]]

    summary_check = validate_dataset(combined)
    combined.sort(key=lambda row: (row["split"], row["document_id"], row.get("chunk_index", 0)))
    write_jsonl(args.output, combined)

    by_source_split: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))
    for row in combined:
        by_source_split[row["split"]][row["source_group"]] += 1

    manifest = {
        "seed": args.seed,
        "ratios": ratios,
        "sources": {"hybrid_dataset": str(args.hybrid_dataset), "preference_shards": [str(p) for p in args.preference_shards]},
        "documents_shared_between_sources": len(
            {r["document_id"] for r in hybrid_rows} & {r["document_id"] for r in preference_rows}
        ),
        "output_sha256": file_sha256(args.output),
        "total_examples": len(combined),
        "by_split_and_source": {split: dict(counts) for split, counts in by_source_split.items()},
        "validate_dataset_summary": summary_check,
    }
    args.manifest.parent.mkdir(parents=True, exist_ok=True)
    args.manifest.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
