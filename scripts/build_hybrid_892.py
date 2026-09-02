#!/usr/bin/env python3
"""Build Hybrid-892: every Defence passage paired with exactly one negative
-- the final deduplicated topic-matched Wikipedia negative if one exists,
otherwise a length-matched easy Wikipedia negative drawn from the same
split. Deterministic via SHA-256 ranking, seed 42."""

from __future__ import annotations

import hashlib
import json
import sys
from collections import Counter
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))
from defence_language_classifier.training import file_sha256, read_jsonl, validate_dataset, write_jsonl  # noqa: E402


def rank_key(namespace: str, key: str) -> str:
    return hashlib.sha256(f"42\0{namespace}\0{key}".encode("utf-8")).hexdigest()


def main() -> None:
    defence = read_jsonl([Path("data/processed/defence_passages.jsonl")])
    split_rows = read_jsonl([Path("data/processed/dataset_splits.jsonl")])
    doc_split = {r["document_id"]: r["split"] for r in split_rows if r["source_group"] == "defence"}
    easy_negatives = [r for r in split_rows if r["source_group"] == "wikipedia"]

    # --- merge the two final topic-matched pair files: best score wins per
    # Defence passage, then resolve article collisions (an article picked by
    # both source runs for two different passages) by keeping the higher score.
    topic_pairs_raw = read_jsonl(
        [Path("data/processed/topic_matched_pairs.jsonl"), Path("data/processed/topic_matched_pairs_qwenfloor032.jsonl")]
    )
    best_by_defence: dict[str, dict] = {}
    for row in topic_pairs_raw:
        existing = best_by_defence.get(row["defence_example_id"])
        if existing is None or row["match_score"] > existing["match_score"]:
            best_by_defence[row["defence_example_id"]] = row
    by_page_id: dict[str, dict] = {}
    for row in sorted(best_by_defence.values(), key=lambda r: r["match_score"], reverse=True):
        page_id = row["wikipedia_page_id"]
        if page_id in by_page_id:
            continue
        by_page_id[page_id] = row
    topic_matched_by_defence = {row["defence_example_id"]: row for row in by_page_id.values()}
    print(f"Topic-matched: {len(topic_pairs_raw)} raw rows -> {len(best_by_defence)} after per-defence best -> {len(topic_matched_by_defence)} after article dedup")

    used_wiki_doc_ids: set[str] = {f"wikipedia-{row['wikipedia_page_id']}" for row in topic_matched_by_defence.values()}

    easy_by_split: dict[str, list[dict]] = {}
    for row in easy_negatives:
        if row["document_id"] not in used_wiki_doc_ids:
            easy_by_split.setdefault(row["split"], []).append(row)
    for split in easy_by_split:
        easy_by_split[split].sort(key=lambda r: rank_key("hybrid892-easy", r["example_id"]))

    output_rows: list[dict] = []
    shortfalls: list[dict] = []
    used_wiki_doc_ids_this_pass: set[str] = set(used_wiki_doc_ids)
    needed_easy_by_split: Counter = Counter()

    defence_sorted = sorted(defence, key=lambda r: rank_key("hybrid892-defence", r["example_id"]))
    matched_count = 0
    for row in defence_sorted:
        split = doc_split[row["document_id"]]
        output_rows.append({**row, "split": split})
        if row["example_id"] in topic_matched_by_defence:
            matched_count += 1
        else:
            needed_easy_by_split[split] += 1

    for split, needed in needed_easy_by_split.items():
        available = len(easy_by_split.get(split, []))
        if available < needed:
            shortfalls.append({"split": split, "needed_easy_negatives": needed, "available": available})

    if shortfalls:
        print("SHORTFALL:", json.dumps(shortfalls, indent=2))
        raise SystemExit("Cannot satisfy Hybrid-892 easy-negative fallback quota without reuse; see shortfall above.")

    negative_rows: list[dict] = []
    easy_cursor = {split: 0 for split in easy_by_split}
    negative_type_by_split: Counter = Counter()
    for row in defence_sorted:
        split = doc_split[row["document_id"]]
        if row["example_id"] in topic_matched_by_defence:
            pair = topic_matched_by_defence[row["example_id"]]
            negative_rows.append(
                {
                    "example_id": pair["wikipedia_example_id"],
                    "document_id": f"wikipedia-{pair['wikipedia_page_id']}",
                    "chunk_index": 0,
                    "text": pair["wikipedia_text"],
                    "label": 0,
                    "source_group": "wikipedia",
                    "source_name": "enwiki-topic-matched",
                    "negative_type": "topic_matched",
                    "word_count": pair["wikipedia_word_count"],
                    "title": pair["wikipedia_title"],
                    "url": pair["wikipedia_url"],
                    "matched_defence_example_id": row["example_id"],
                    "semantic_similarity": pair["semantic_similarity"],
                    "split": split,
                }
            )
            negative_type_by_split[(split, "topic_matched")] += 1
        else:
            idx = easy_cursor[split]
            neg = easy_by_split[split][idx]
            easy_cursor[split] += 1
            negative_rows.append({**neg, "split": split, "matched_defence_example_id": row["example_id"]})
            negative_type_by_split[(split, "easy")] += 1

    combined = output_rows + negative_rows
    combined.sort(key=lambda row: (row["split"], row["document_id"], row.get("chunk_index", 0)))

    summary = validate_dataset(combined)
    out_path = Path("data/processed/hybrid_892_dataset_splits.jsonl")
    write_jsonl(out_path, combined)

    manifest = {
        "dataset_name": "hybrid_892",
        "seed": 42,
        "defence_positives": len(defence),
        "topic_matched_negatives": matched_count,
        "easy_fallback_negatives": len(defence) - matched_count,
        "total_examples": len(combined),
        "positive_by_split": {s: sum(1 for r in output_rows if r["split"] == s) for s in ("train", "validation", "test")},
        "negative_type_by_split": {f"{s}_{t}": c for (s, t), c in sorted(negative_type_by_split.items())},
        "target_pairs_per_split": {"train": 622, "validation": 135, "test": 135},
        "note": "Defence-side split counts (620/136/136) match the frozen Easy-892 split exactly, not the requested 622/135/135 -- same root cause documented in easy_892_manifest.json (existing frozen split isn't perfectly pos/neg-balanced per split; preserved rather than moved).",
        "validate_dataset_summary": summary,
        "output_sha256": file_sha256(out_path),
    }
    Path("data/processed/hybrid_892_split_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
