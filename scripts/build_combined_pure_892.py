#!/usr/bin/env python3
"""Build Combined-Pure-892: Easy + Pure-Topic-Matched + Preference pairs,
NO Hybrid-892 data (unlike the earlier Combined-892, which mixed
topic-matched with easy-fallback negatives via Hybrid-892).

Global leakage control: a document appearing in ANY of the three external
test sets (Easy test, Pure-Topic test, Preference test) is excluded from
Combined-Pure's train/validation regardless of which source it would be
drawn from -- these three source corpora share some Defence documents, so
this catches cross-source contamination a single-source check would miss.
The Combined-Pure "reserved" split is accounting-only and deliberately
drawn from each source's own test pool; it is never evaluated as a
headline result and is exempt from the exclusion (that's what it draws from).
"""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))
from defence_language_classifier.training import file_sha256, read_jsonl, validate_dataset, write_jsonl  # noqa: E402

QUOTAS = {
    "train": {"easy": 207, "pure_topic": 207, "preference": 206},
    "validation": {"easy": 45, "pure_topic": 45, "preference": 46},
    "test": {"easy": 46, "pure_topic": 45, "preference": 45},
}
SOURCE_PATHS = {
    "easy": Path("data/processed/dataset_splits.jsonl"),
    "pure_topic": Path("data/processed/pure_topic_892_dataset_splits.jsonl"),
    "preference": Path("data/processed/preference_892_dataset_splits.jsonl"),
}


def rank_key(source: str, pair_id: str) -> str:
    return hashlib.sha256(f"42\0combined_pure\0{source}\0{pair_id}".encode("utf-8")).hexdigest()


def form_pairs(source: str, rows: list[dict]) -> list[tuple[str, dict, dict]]:
    """Returns list of (pair_id, positive_row, negative_row)."""
    if source == "preference":
        by_cid: dict[str, dict[int, dict]] = {}
        for r in rows:
            by_cid.setdefault(r["candidate_id"], {})[r["label"]] = r
        return [(cid, d[1], d[0]) for cid, d in by_cid.items() if 1 in d and 0 in d]
    if source == "pure_topic":
        positives = {r["example_id"]: r for r in rows if r["label"] == 1}
        pairs = []
        for neg in [r for r in rows if r["label"] == 0]:
            pos = positives.get(neg.get("matched_defence_example_id"))
            if pos is not None:
                pairs.append((neg["matched_defence_example_id"], pos, neg))
        return pairs
    # easy: no ground-truth pairing recorded at build time -- form a
    # deterministic synthetic pairing (SHA256-sorted positives zipped with
    # SHA256-sorted negatives), itself seeded and reproducible, purely so
    # positive/negative examples can be selected and tracked together as
    # "one pair" per the required procedure. Documented, not claimed as a
    # real content link (see the pairwise-evaluation stage, which refuses
    # to compute Easy pairwise stats for exactly this reason).
    positives = sorted([r for r in rows if r["label"] == 1], key=lambda r: rank_key("easy-pos-order", r["example_id"]))
    negatives = sorted([r for r in rows if r["label"] == 0], key=lambda r: rank_key("easy-neg-order", r["example_id"]))
    n = min(len(positives), len(negatives))
    return [(f"{p['example_id']}::{q['example_id']}", p, q) for p, q in zip(positives[:n], negatives[:n])]


def main() -> None:
    all_rows = {name: read_jsonl([path]) for name, path in SOURCE_PATHS.items()}

    global_test_docs: set[str] = set()
    for name, rows in all_rows.items():
        for r in rows:
            if r["split"] == "test":
                global_test_docs.add(r["document_id"])
    print(f"Global test-document exclusion set: {len(global_test_docs)} documents")

    # Easy and Pure-Topic share the same frozen Defence-document split (Pure-Topic
    # was derived from Hybrid-892, which inherits it exactly). Preference-892 was
    # split independently and can disagree for documents the two corpora share.
    # Without this check, a Preference candidate could land in a different
    # Combined-Pure split than an Easy/Pure-Topic candidate for the same document.
    canonical_doc_split: dict[str, str] = {}
    for source in ("easy", "pure_topic"):
        for row in all_rows[source]:
            canonical_doc_split.setdefault(row["document_id"], row["split"])

    output_rows: list[dict] = []
    shortfalls: list[dict] = []

    for split in ("train", "validation", "test"):
        used_defence_passage_ids: set[str] = set()  # Easy positive vs Pure-topic positive exclusivity
        used_example_ids: set[str] = set()
        used_texts: set[str] = set()

        # pure_topic's 439-passage pool is a strict subset of easy's 892-passage
        # pool, so it must claim its Defence-passage-exclusivity share first --
        # processing easy first would let its arbitrary picks starve pure_topic
        # of candidates it structurally cannot get elsewhere.
        for source in ("pure_topic", "easy", "preference"):
            source_rows = [r for r in all_rows[source] if r["split"] == split]
            pairs = form_pairs(source, source_rows)

            candidates = []
            for pair_id, pos, neg in pairs:
                if split != "test":  # train/validation: apply exclusion sets; reserved/test draws from the source's own test pool by definition
                    if pos["document_id"] in global_test_docs or neg["document_id"] in global_test_docs:
                        continue
                if source == "preference":
                    canonical = canonical_doc_split.get(pos["document_id"])
                    if canonical is not None and canonical != split:
                        continue
                if source in ("easy", "pure_topic") and pos["example_id"] in used_defence_passage_ids:
                    continue
                if pos["example_id"] in used_example_ids or neg["example_id"] in used_example_ids:
                    continue
                if pos["text"] in used_texts or neg["text"] in used_texts:
                    continue
                candidates.append((pair_id, pos, neg))

            candidates.sort(key=lambda item: rank_key(source, item[0]))
            need = QUOTAS[split][source]
            if len(candidates) < need:
                shortfalls.append({"split": split, "source": source, "needed": need, "available": len(candidates)})
                continue
            selected = candidates[:need]
            for pair_id, pos, neg in selected:
                if source in ("easy", "pure_topic"):
                    used_defence_passage_ids.add(pos["example_id"])
                used_example_ids.add(pos["example_id"])
                used_example_ids.add(neg["example_id"])
                used_texts.add(pos["text"])
                used_texts.add(neg["text"])
                output_rows.append({**pos, "split": split, "combined_source": source, "combined_pair_id": pair_id})
                output_rows.append({**neg, "split": split, "combined_source": source, "combined_pair_id": pair_id})

    if shortfalls:
        print("SHORTFALL:", json.dumps(shortfalls, indent=2))
        raise SystemExit("Cannot satisfy Combined-Pure-892 quotas without leakage or duplication; see shortfall above.")

    # --- global checks ---
    ids = [r["example_id"] for r in output_rows]
    texts = [r["text"] for r in output_rows]
    if len(ids) != len(set(ids)):
        raise SystemExit(f"Duplicate example IDs: {len(ids) - len(set(ids))}")
    if len(texts) != len(set(texts)):
        raise SystemExit(f"Duplicate texts: {len(texts) - len(set(texts))}")

    doc_splits: dict[str, set[str]] = {}
    for r in output_rows:
        doc_splits.setdefault(r["document_id"], set()).add(r["split"])
    leaking = {d: sorted(s) for d, s in doc_splits.items() if len(s) > 1}
    if leaking:
        raise SystemExit(f"Document(s) span multiple Combined-Pure-892 splits: {leaking}")

    train_val_docs = {r["document_id"] for r in output_rows if r["split"] in ("train", "validation")}
    leaked_into_external_test = train_val_docs & global_test_docs
    if leaked_into_external_test:
        raise SystemExit(f"{len(leaked_into_external_test)} train/validation document(s) also appear in an external test set: {sorted(leaked_into_external_test)[:5]}")

    pref_test_cids = {r["candidate_id"] for r in all_rows["preference"] if r["split"] == "test"}
    used_pref_cids_train_val = {
        r["combined_pair_id"] for r in output_rows if r["combined_source"] == "preference" and r["split"] in ("train", "validation")
    }
    reused_pref_pairs = used_pref_cids_train_val & pref_test_cids
    if reused_pref_pairs:
        raise SystemExit(f"Preference-892 test candidate(s) leaked into Combined-Pure train/validation: {reused_pref_pairs}")

    output_rows.sort(key=lambda row: (row["split"], row["document_id"], row.get("chunk_index", 0)))
    summary = validate_dataset(output_rows)
    out_path = Path("data/processed/combined_pure_892_dataset_splits.jsonl")
    write_jsonl(out_path, output_rows)

    by_split_source: dict[tuple[str, str], int] = {}
    for r in output_rows:
        key = (r["split"], r["combined_source"])
        by_split_source[key] = by_split_source.get(key, 0) + 1

    manifest = {
        "dataset_name": "combined_pure_892",
        "seed": 42,
        "replaces": "combined_892 (which used Hybrid-892, mixing topic-matched with easy-fallback negatives); this version uses only genuine topic-matched negatives",
        "quotas": QUOTAS,
        "global_test_document_exclusion_count": len(global_test_docs),
        "total_examples": len(output_rows),
        "pairs_by_split_and_source": {f"{s}_{src}": c // 2 for (s, src), c in sorted(by_split_source.items())},
        "easy_pairing_note": "Easy source has no recorded ground-truth positive/negative link; pairs were formed by deterministic SHA256-sorted zipping (seed 42, documented, not a real content pairing). Pairwise Easy evaluation is reported as unavailable for this reason, per spec.",
        "validate_dataset_summary": summary,
        "output_sha256": file_sha256(out_path),
    }
    Path("data/processed/combined_pure_892_split_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
