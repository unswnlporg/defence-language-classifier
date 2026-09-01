#!/usr/bin/env python3
"""Build a smaller, fully-clean 50/50 hybrid dataset from what topic-matching
already produced, instead of waiting on full 892-passage topic-match coverage.

For every Defence passage that has an accepted topic-matched negative (158 of
892), include both:
  - the topic-matched (hard) negative
  - one easy, length-matched-only negative drawn from the same split

so the dataset is exactly balanced between hard and easy negatives, and every
row still respects the frozen document-level splits (no cross-split reuse).
No network calls; everything here is already on disk.
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path


def read_jsonl(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as stream:
        return [json.loads(line) for line in stream if line.strip()]


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as stream:
        for row in rows:
            stream.write(json.dumps(row, ensure_ascii=False) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--topic-pairs",
        type=Path,
        nargs="+",
        default=[Path("data/processed/topic_matched_pairs.jsonl")],
        help="One or more pairs files; if a Defence passage appears in more than one, the highest match_score wins.",
    )
    parser.add_argument("--dataset-splits", type=Path, default=Path("data/processed/dataset_splits.jsonl"))
    parser.add_argument("--output", type=Path, default=Path("data/processed/hybrid_topic_dataset_splits.jsonl"))
    parser.add_argument("--manifest", type=Path, default=Path("data/processed/hybrid_topic_split_manifest.json"))
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    best_by_defence_id: dict[str, dict] = {}
    for path in args.topic_pairs:
        for row in read_jsonl(path):
            existing = best_by_defence_id.get(row["defence_example_id"])
            if existing is None or row["match_score"] > existing["match_score"]:
                best_by_defence_id[row["defence_example_id"]] = row

    # Different topic-matching runs each enforce "one Wikipedia article per
    # pair" only within their own run, so merging runs can independently pick
    # the same Wikipedia article for two different Defence passages. Break
    # those collisions by keeping the higher-scoring pair and dropping the
    # other passage from this merge entirely, so every Wikipedia article is
    # still used at most once across the whole merged dataset.
    by_page_id: dict[str, dict] = {}
    for row in sorted(best_by_defence_id.values(), key=lambda r: r["match_score"], reverse=True):
        page_id = row["wikipedia_page_id"]
        if page_id in by_page_id:
            continue
        by_page_id[page_id] = row
    pairs = list(by_page_id.values())
    dropped = len(best_by_defence_id) - len(pairs)
    if dropped:
        print(f"Dropped {dropped} passage(s) whose matched Wikipedia article collided with a higher-scoring pair.")

    all_rows = read_jsonl(args.dataset_splits)
    defence_by_id = {r["example_id"]: r for r in all_rows if r["source_group"] == "defence"}
    old_negatives_by_split: dict[str, list[dict]] = {}
    for row in all_rows:
        if row["source_group"] == "wikipedia":
            old_negatives_by_split.setdefault(row["split"], []).append(row)

    rng = random.Random(args.seed)
    for split_rows in old_negatives_by_split.values():
        rng.shuffle(split_rows)

    used_old_negative_ids: set[str] = set()
    output_rows: list[dict] = []
    used_defence_ids: set[str] = set()

    for pair in pairs:
        defence_id = pair["defence_example_id"]
        if defence_id in used_defence_ids or defence_id not in defence_by_id:
            continue
        defence_row = defence_by_id[defence_id]
        split = pair["split"]

        pool = old_negatives_by_split.get(split, [])
        easy_negative = next((r for r in pool if r["example_id"] not in used_old_negative_ids), None)
        if easy_negative is None:
            continue  # exhausted the easy-negative pool for this split; skip rather than reuse

        used_defence_ids.add(defence_id)
        used_old_negative_ids.add(easy_negative["example_id"])

        output_rows.append(defence_row)
        output_rows.append(
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
                "matched_defence_example_id": defence_id,
                "semantic_similarity": pair["semantic_similarity"],
                "split": split,
            }
        )
        output_rows.append(easy_negative)

    output_rows.sort(key=lambda row: (row["split"], row["document_id"], row.get("chunk_index", 0)))
    write_jsonl(args.output, output_rows)

    summary = {}
    for split in ("train", "validation", "test"):
        split_rows = [r for r in output_rows if r["split"] == split]
        summary[split] = {
            "positive": sum(1 for r in split_rows if r["label"] == 1),
            "topic_matched_negative": sum(1 for r in split_rows if r.get("negative_type") == "topic_matched"),
            "easy_negative": sum(1 for r in split_rows if r["source_group"] == "wikipedia" and r.get("negative_type") != "topic_matched"),
        }

    manifest = {
        "seed": args.seed,
        "source_topic_pairs": str(args.topic_pairs),
        "defence_passages_used": len(used_defence_ids),
        "total_rows": len(output_rows),
        "composition": "1 positive : 1 topic-matched negative : 1 easy negative per Defence passage",
        "summary": summary,
    }
    args.manifest.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
