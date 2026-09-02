#!/usr/bin/env python3
"""Build Combined-892: pairs drawn from Easy-892, Hybrid-892 and
Preference-892 in roughly equal thirds per split, summing to the common
620/136/136 target. Global document-level leakage control across sources;
Easy and Hybrid draw disjoint sets of Defence passages within a split so the
same passage never enters Combined twice; Preference draws only from within
Preference-892's own matching split (its test set is therefore automatically
excluded from Combined train/validation)."""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))
from defence_language_classifier.training import file_sha256, read_jsonl, validate_dataset, write_jsonl  # noqa: E402


def rank_key(namespace: str, key: str) -> str:
    return hashlib.sha256(f"42\0{namespace}\0{key}".encode("utf-8")).hexdigest()


QUOTAS = {
    "train": {"easy": 207, "hybrid": 207, "preference": 206},
    "validation": {"easy": 45, "hybrid": 45, "preference": 46},
    "test": {"easy": 45, "hybrid": 45, "preference": 46},
}


def main() -> None:
    easy_rows = read_jsonl([Path("data/processed/dataset_splits.jsonl")])
    hybrid_rows = read_jsonl([Path("data/processed/hybrid_892_dataset_splits.jsonl")])
    pref_rows = read_jsonl([Path("data/processed/preference_892_dataset_splits.jsonl")])

    # Canonical document->split assignment from the full Easy/Hybrid datasets
    # (not just what ends up selected into Combined), so Preference candidates
    # that share a document with Easy/Hybrid in a *different* split are
    # excluded up front -- some documents are shared between the Defence
    # corpus and the broader preference-generation corpus (see docs/training_methodology.md).
    canonical_doc_split: dict[str, str] = {}
    for row in easy_rows + hybrid_rows:
        canonical_doc_split.setdefault(row["document_id"], row["split"])

    output_rows: list[dict] = []
    shortfalls: list[dict] = []

    for split, quota in QUOTAS.items():
        # --- Easy: N distinct positives + N distinct negatives from Easy's own split ---
        easy_pos = sorted(
            [r for r in easy_rows if r["split"] == split and r["label"] == 1],
            key=lambda r: rank_key("combined892-easy-pos", r["example_id"]),
        )
        easy_neg = sorted(
            [r for r in easy_rows if r["split"] == split and r["label"] == 0],
            key=lambda r: rank_key("combined892-easy-neg", r["example_id"]),
        )
        if len(easy_pos) < quota["easy"] or len(easy_neg) < quota["easy"]:
            shortfalls.append({"split": split, "source": "easy", "needed": quota["easy"], "available_pos": len(easy_pos), "available_neg": len(easy_neg)})
            continue
        easy_pos_selected = easy_pos[: quota["easy"]]
        easy_neg_selected = easy_neg[: quota["easy"]]
        easy_selected_passage_ids = {r["example_id"] for r in easy_pos_selected}

        # --- Hybrid: N distinct Defence passages (with their paired negative),
        # disjoint from the specific Defence *passages* Easy already selected
        # in this split (a source document may still appear via both, just
        # not the same literal passage), AND whose paired negative isn't one
        # Easy's negative selection already used -- Hybrid's easy-fallback
        # negatives are drawn from that exact same pool, so this is a real
        # possible collision, not just a theoretical one. ---
        easy_selected_neg_ids = {r["example_id"] for r in easy_neg_selected}
        hybrid_neg_by_defence_id = {r["matched_defence_example_id"]: r for r in hybrid_rows if r["split"] == split and r["label"] == 0}
        hybrid_pos_candidates = sorted(
            [
                r for r in hybrid_rows
                if r["split"] == split and r["label"] == 1
                and r["example_id"] not in easy_selected_passage_ids
                and hybrid_neg_by_defence_id[r["example_id"]]["example_id"] not in easy_selected_neg_ids
            ],
            key=lambda r: rank_key("combined892-hybrid-pos", r["example_id"]),
        )
        if len(hybrid_pos_candidates) < quota["hybrid"]:
            shortfalls.append({"split": split, "source": "hybrid", "needed": quota["hybrid"], "available": len(hybrid_pos_candidates)})
            continue
        hybrid_pos_selected = hybrid_pos_candidates[: quota["hybrid"]]
        hybrid_neg_selected = [hybrid_neg_by_defence_id[r["example_id"]] for r in hybrid_pos_selected]

        # --- Preference: N candidate pairs from Preference-892's own split,
        # excluding any candidate whose document is already assigned to a
        # *different* split by Easy/Hybrid (cross-source document overlap). ---
        pref_candidate_docs = {r["candidate_id"]: r["document_id"] for r in pref_rows if r["split"] == split}
        pref_candidates = sorted(
            [
                cid for cid in pref_candidate_docs
                if canonical_doc_split.get(pref_candidate_docs[cid], split) == split
            ],
            key=lambda cid: rank_key("combined892-preference", cid),
        )
        if len(pref_candidates) < quota["preference"]:
            shortfalls.append({"split": split, "source": "preference", "needed": quota["preference"], "available": len(pref_candidates)})
            continue
        pref_cids_selected = set(pref_candidates[: quota["preference"]])
        pref_selected = [r for r in pref_rows if r["split"] == split and r["candidate_id"] in pref_cids_selected]

        for row in easy_pos_selected + easy_neg_selected:
            output_rows.append({**row, "combined_source": "easy"})
        for row in hybrid_pos_selected + hybrid_neg_selected:
            output_rows.append({**row, "combined_source": "hybrid"})
        for row in pref_selected:
            output_rows.append({**row, "combined_source": "preference"})

    if shortfalls:
        print("SHORTFALL:", json.dumps(shortfalls, indent=2))
        raise SystemExit("Cannot satisfy Combined-892 quotas; see shortfall above.")

    # --- global document-level leakage check across all sources ---
    doc_splits: dict[str, set[str]] = {}
    for row in output_rows:
        doc_splits.setdefault(row["document_id"], set()).add(row["split"])
    leaking = {d: sorted(s) for d, s in doc_splits.items() if len(s) > 1}
    if leaking:
        raise SystemExit(f"Document(s) span multiple splits in Combined-892: {leaking}")

    # --- Preference-892 common test set must never appear in combined train/validation ---
    common_test_ids = set(json.loads(Path("data/processed/preference_892_common_test_ids.json").read_text())["candidate_ids"])
    leaked_pref_test = [
        r["candidate_id"] for r in output_rows
        if r.get("combined_source") == "preference" and r["split"] != "test" and r.get("candidate_id") in common_test_ids
    ]
    if leaked_pref_test:
        raise SystemExit(f"Preference-892 common test candidates leaked into non-test split: {leaked_pref_test}")

    output_rows.sort(key=lambda row: (row["split"], row["document_id"], row.get("chunk_index", 0)))
    ids = [r["example_id"] for r in output_rows]
    texts = [r["text"] for r in output_rows]
    if len(ids) != len(set(ids)):
        raise SystemExit(f"Duplicate example IDs in Combined-892: {len(ids) - len(set(ids))}")
    if len(texts) != len(set(texts)):
        raise SystemExit(f"Duplicate texts in Combined-892: {len(texts) - len(set(texts))}")

    summary = validate_dataset(output_rows)
    out_path = Path("data/processed/combined_892_dataset_splits.jsonl")
    write_jsonl(out_path, output_rows)

    by_split_source = {}
    for row in output_rows:
        key = (row["split"], row["combined_source"])
        by_split_source[key] = by_split_source.get(key, 0) + 1

    manifest = {
        "dataset_name": "combined_892",
        "seed": 42,
        "quotas": QUOTAS,
        "total_examples": len(output_rows),
        "pairs_by_split_and_source": {f"{s}_{src}": c // 2 for (s, src), c in sorted(by_split_source.items())},
        "validate_dataset_summary": summary,
        "output_sha256": file_sha256(out_path),
    }
    Path("data/processed/combined_892_split_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
