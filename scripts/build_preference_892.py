#!/usr/bin/env python3
"""Build Preference-892: exactly 622/135/135 candidate pairs per split,
selected deterministically by SHA256("42\\0preference\\0" + candidate_id),
ascending, within the existing document-disjoint Preference split."""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))
from defence_language_classifier.training import file_sha256, read_jsonl, validate_dataset, write_jsonl  # noqa: E402


def rank_key(candidate_id: str) -> str:
    return hashlib.sha256(f"42\0preference\0{candidate_id}".encode("utf-8")).hexdigest()


def main() -> None:
    rows = read_jsonl([Path("data/processed/preference_classifier_dataset_splits.jsonl")])
    by_split_candidate: dict[str, dict[str, dict[int, dict]]] = {}
    for row in rows:
        by_split_candidate.setdefault(row["split"], {}).setdefault(row["candidate_id"], {})[row["label"]] = row

    # Matches the achievable Hybrid-892 target (620/136/136), not the original
    # 622/135/135 assumption, for consistency across all four experiments --
    # see easy_892_manifest.json for why 622/135/135 isn't naturally achievable.
    target = {"train": 620, "validation": 136, "test": 136}
    shortfalls = []
    selected_candidates: dict[str, list[str]] = {}
    for split, need in target.items():
        complete = [cid for cid, d in by_split_candidate.get(split, {}).items() if 1 in d and 0 in d]
        if len(complete) < need:
            shortfalls.append({"split": split, "needed": need, "available_complete_pairs": len(complete)})
            continue
        complete.sort(key=rank_key)
        selected_candidates[split] = complete[:need]

    if shortfalls:
        print("SHORTFALL:", json.dumps(shortfalls, indent=2))
        raise SystemExit("Cannot satisfy Preference-892 pair quota; see shortfall above.")

    output_rows = []
    for split, cids in selected_candidates.items():
        for cid in cids:
            d = by_split_candidate[split][cid]
            output_rows.append(d[1])
            output_rows.append(d[0])
    output_rows.sort(key=lambda row: (row["split"], row["document_id"], row.get("chunk_index", 0)))

    summary = validate_dataset(output_rows)
    out_path = Path("data/processed/preference_892_dataset_splits.jsonl")
    write_jsonl(out_path, output_rows)

    common_test_ids = sorted(selected_candidates["test"])
    Path("data/processed/preference_892_common_test_ids.json").write_text(
        json.dumps({"seed": 42, "selection": "SHA256('42\\0preference\\0'+candidate_id) ascending, top 135", "candidate_ids": common_test_ids}, indent=2) + "\n",
        encoding="utf-8",
    )

    manifest = {
        "dataset_name": "preference_892",
        "seed": 42,
        "selection_method": "SHA256('42\\0preference\\0' + candidate_id), ascending sort within split, top-N selected",
        "target_pairs_per_split": target,
        "actual_pairs_per_split": {s: len(v) for s, v in selected_candidates.items()},
        "total_examples": len(output_rows),
        "validate_dataset_summary": summary,
        "output_sha256": file_sha256(out_path),
    }
    Path("data/processed/preference_892_split_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
