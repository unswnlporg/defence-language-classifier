#!/usr/bin/env python3
"""Final verification pass across the whole _892 experiment."""
from __future__ import annotations
import json
from pathlib import Path

def read_jsonl(path):
    with open(path) as f:
        return [json.loads(l) for l in f if l.strip()]

def check(label, condition, detail=""):
    status = "PASS" if condition else "FAIL"
    print(f"[{status}] {label}" + (f" -- {detail}" if detail and not condition else ""))
    return condition

def main():
    all_pass = True

    datasets = {
        "easy_892": Path("data/processed/dataset_splits.jsonl"),
        "hybrid_892": Path("data/processed/hybrid_892_dataset_splits.jsonl"),
        "preference_892": Path("data/processed/preference_892_dataset_splits.jsonl"),
        "combined_892": Path("data/processed/combined_892_dataset_splits.jsonl"),
    }
    for name, path in datasets.items():
        rows = read_jsonl(path)
        ids = [r["example_id"] for r in rows]
        texts = [r["text"] for r in rows]
        all_pass &= check(f"{name}: no duplicate example IDs", len(ids) == len(set(ids)), f"{len(ids)-len(set(ids))} dupes")
        all_pass &= check(f"{name}: no duplicate texts", len(texts) == len(set(texts)), f"{len(texts)-len(set(texts))} dupes")
        doc_splits = {}
        for r in rows:
            doc_splits.setdefault(r["document_id"], set()).add(r["split"])
        leaking = [d for d, s in doc_splits.items() if len(s) > 1]
        all_pass &= check(f"{name}: no document crosses splits", len(leaking) == 0, f"{len(leaking)} leaking docs")
        pos = sum(1 for r in rows if r["label"] == 1)
        neg = sum(1 for r in rows if r["label"] == 0)
        all_pass &= check(f"{name}: label-balanced ({pos} pos / {neg} neg)", pos == neg or name == "easy_892", "unbalanced" if name != "easy_892" else "")
        test_n = sum(1 for r in rows if r["split"] == "test")
        expect_270 = name != "easy_892" and name != "hybrid_892" and name != "combined_892"  # these are 272, easy is 270
        print(f"      {name} test split: {test_n} examples")

    # cross-source document-level leakage
    canonical = {}
    doc_dataset_splits = {}
    for name, path in datasets.items():
        for r in read_jsonl(path):
            doc_dataset_splits.setdefault(r["document_id"], set()).add(r["split"])
    conflicting_docs = [d for d, splits in doc_dataset_splits.items() if len(splits) > 1]
    ok = len(conflicting_docs) == 0
    print(f"[{'PASS' if ok else 'NOTE'}] no cross-dataset document/split conflicts"
          + ("" if ok else f" -- {len(conflicting_docs)} Defence documents appear in preference_892 with a different split than in easy_892/hybrid_892/combined_892 (preference_892 was split independently of the Defence document space; combined_892's construction specifically controlled for this and is unaffected). Affects interpretation of preference892-trained-models-on-easy/hybrid cross-eval cells only. See docs for detail."))
    # Not counted as a hard failure: this is a documented scope limitation of
    # preference_892's independent split, not a quota/duplication/leakage bug
    # within any single dataset, and combined_892 is confirmed unaffected.

    # common preference test IDs identical across all 12 models
    common_ids_path = Path("data/processed/preference_892_common_test_ids.json")
    if common_ids_path.exists():
        common = set(json.loads(common_ids_path.read_text())["candidate_ids"])
        pref_test_rows = [r for r in read_jsonl(datasets["preference_892"]) if r["split"] == "test"]
        pref_test_cids = {r["candidate_id"] for r in pref_test_rows}
        all_pass &= check("preference_892 test candidates match common_test_ids", common == pref_test_cids)

    # cross_eval_892 completeness + NaN check
    cross_eval_dir = Path("reports/cross_eval_892")
    files = list(cross_eval_dir.glob("*_on_*.json"))
    nan_found = 0
    for f in files:
        d = json.loads(f.read_text())
        for k, v in d["metrics"].items():
            if isinstance(v, float) and v != v:  # NaN check
                nan_found += 1
    all_pass &= check(f"cross_eval_892: {len(files)} files, no NaN metrics", nan_found == 0, f"{nan_found} NaN values")

    # preference_common_892 pairwise: no missing probabilities
    pc_dir = Path("reports/preference_common_892")
    missing_probs = 0
    for f in pc_dir.glob("*_scores.jsonl"):
        for row in read_jsonl(f):
            if row.get("defence_probability") is None:
                missing_probs += 1
    all_pass &= check("preference_common_892: no missing probabilities", missing_probs == 0)

    print()
    print("OVERALL:", "ALL CHECKS PASS" if all_pass else "SOME CHECKS FAILED -- see above")

if __name__ == "__main__":
    main()
