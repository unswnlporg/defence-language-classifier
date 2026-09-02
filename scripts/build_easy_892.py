#!/usr/bin/env python3
"""Verify Easy-892 against the size-controlled experimental standard and
write a manifest referencing the existing frozen dataset (no rewrite)."""
from __future__ import annotations
import json, sys
from pathlib import Path
from collections import Counter

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))
from defence_language_classifier.training import file_sha256, read_jsonl

def main():
    path = Path("data/processed/dataset_splits.jsonl")
    rows = read_jsonl([path])
    pos = [r for r in rows if r["label"] == 1]
    neg = [r for r in rows if r["label"] == 0]
    pos_by_split = Counter(r["split"] for r in pos)
    neg_by_split = Counter(r["split"] for r in neg)

    target = {"train": 622, "validation": 135, "test": 135}
    deviations = []
    for split in ("train", "validation", "test"):
        p, n = pos_by_split[split], neg_by_split[split]
        if p != target[split] or n != target[split]:
            deviations.append({
                "split": split, "target_pairs": target[split],
                "actual_positive": p, "actual_negative": n,
                "note": "existing frozen document-level split does not yield exactly balanced pos/neg counts per split; preserving frozen split rather than moving documents across splits.",
            })

    ids = [r["example_id"] for r in rows]
    texts = [r["text"] for r in rows]
    docsplits = {}
    for r in rows:
        docsplits.setdefault(r["document_id"], set()).add(r["split"])
    leaking = [d for d, s in docsplits.items() if len(s) > 1]

    manifest = {
        "dataset_name": "easy_892",
        "source": str(path),
        "action": "verified_existing_no_rewrite",
        "seed": 42,
        "total_positive": len(pos), "total_negative": len(neg), "total_examples": len(rows),
        "positive_by_split": dict(pos_by_split), "negative_by_split": dict(neg_by_split),
        "target_pairs_per_split": target,
        "deviations_from_target_table": deviations,
        "duplicate_example_ids": len(ids) - len(set(ids)),
        "duplicate_texts": len(texts) - len(set(texts)),
        "documents_leaking_across_splits": len(leaking),
        "sha256": file_sha256(path),
    }
    Path("data/processed/easy_892_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(manifest, indent=2))

if __name__ == "__main__":
    main()
