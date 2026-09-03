#!/usr/bin/env python3
"""Build combined_preference_heavy: nearly all of Easy + all 439 pure
topic-matched + nearly all 2,327 QC-passed Preference pairs, with the three
existing common test sets fully excluded (not just from train/validation,
but from every split of this new dataset -- its own test split is drawn
fresh from what remains, so it is a genuinely novel benchmark, not a
recycled copy of the common test sets).

Unlike combined_892/combined_pure_892 (fixed small quotas from each
source's existing splits), this dataset uses nearly the FULL eligible pool
of each source and re-splits each source independently by document group
(70/15/15), so the combined mixture's split proportions track each
source's overall size.
"""

from __future__ import annotations

import hashlib
import json
import re
import sys
from pathlib import Path
from collections import defaultdict

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))
from defence_language_classifier.training import file_sha256, read_jsonl, validate_dataset, write_jsonl  # noqa: E402

SEED = 42
RATIOS = {"train": 0.70, "validation": 0.15, "test": 0.15}
PREFERENCE_SHARDS = [Path(f"/srv/scratch/z5703460/outputs/prefqa-generation/qwen3-32b-awq-corpus/shard-0{i}.jsonl") for i in range(4)]


def normalise_text(text: str) -> str:
    return re.sub(r"\s+", " ", text.strip().lower())


def rank_key(namespace: str, key: str) -> str:
    return hashlib.sha256(f"42\0{namespace}\0{key}".encode("utf-8")).hexdigest()


def stable_id(*parts: str) -> str:
    return hashlib.sha256("\0".join(parts).encode("utf-8")).hexdigest()[:20]


class UnionFind:
    def __init__(self) -> None:
        self.parent: dict[str, str] = {}

    def find(self, x: str) -> str:
        self.parent.setdefault(x, x)
        while self.parent[x] != x:
            self.parent[x] = self.parent[self.parent[x]]
            x = self.parent[x]
        return x

    def union(self, a: str, b: str) -> None:
        ra, rb = self.find(a), self.find(b)
        if ra != rb:
            self.parent[ra] = rb


def global_component_split(components_weight: dict[str, int], namespace: str) -> dict[str, str]:
    """70/15/15 split over document-graph connected components, weighted by
    the number of pairs each component hosts, greedy bin-packing, seed 42."""
    groups = list(components_weight.items())
    groups.sort(key=lambda item: rank_key(namespace, item[0]))
    groups.sort(key=lambda item: item[1], reverse=True)
    total = sum(w for _, w in groups)
    targets = {s: total * r for s, r in RATIOS.items()}
    counts = {s: 0 for s in RATIOS}
    assignment: dict[str, str] = {}
    for root, weight in groups:
        def cost(split):
            projected = counts[split] + weight
            relative_fill = projected / max(targets[split], 1.0)
            overflow = max(0.0, projected - targets[split]) / max(targets[split], 1.0)
            return (overflow * 10.0 + relative_fill, counts[split], split)
        chosen = min(RATIOS, key=cost)
        assignment[root] = chosen
        counts[chosen] += weight
    return assignment


def main() -> None:
    # --- Common test sets to exclude entirely -------------------------------
    easy_rows = read_jsonl([Path("data/processed/dataset_splits.jsonl")])
    pure_rows = read_jsonl([Path("data/processed/pure_topic_892_dataset_splits.jsonl")])
    pref_892_rows = read_jsonl([Path("data/processed/preference_892_dataset_splits.jsonl")])
    common_test_cids = set(json.loads(Path("data/processed/preference_892_common_test_ids.json").read_text())["candidate_ids"])

    exclude_doc_ids: set[str] = set()
    exclude_texts: set[str] = set()
    exclude_pair_ids: set[str] = set()

    for r in easy_rows:
        if r["split"] == "test":
            exclude_doc_ids.add(r["document_id"])
            exclude_texts.add(normalise_text(r["text"]))
    for r in pure_rows:
        if r["split"] == "test":
            exclude_doc_ids.add(r["document_id"])
            exclude_texts.add(normalise_text(r["text"]))
            exclude_pair_ids.add(r.get("matched_defence_example_id") or r["example_id"])
    for r in pref_892_rows:
        if r["candidate_id"] in common_test_cids:
            exclude_doc_ids.add(r["document_id"])
            exclude_texts.add(normalise_text(r["text"]))
            exclude_pair_ids.add(r["candidate_id"])

    print(f"Exclusion set: {len(exclude_doc_ids)} documents, {len(exclude_texts)} normalised texts, {len(exclude_pair_ids)} pair IDs")

    # --- Pure topic-matched FIRST: real pairing, exclude. Processed before
    # Easy because Pure-Topic's 439-passage pool is a strict subset of Easy's
    # 892-passage pool -- giving the smaller, structurally-constrained source
    # priority avoids starving it (same lesson as combined_pure_892). ---
    pure_pos = {r["example_id"]: r for r in pure_rows if r["label"] == 1}
    pure_pairs_all = []
    for neg in [r for r in pure_rows if r["label"] == 0]:
        pos = pure_pos.get(neg.get("matched_defence_example_id"))
        if pos is not None:
            pure_pairs_all.append((neg["matched_defence_example_id"], pos, neg))
    pure_pairs = [
        (pid, pos, neg) for pid, pos, neg in pure_pairs_all
        if pos["document_id"] not in exclude_doc_ids and neg["document_id"] not in exclude_doc_ids
        and normalise_text(pos["text"]) not in exclude_texts and normalise_text(neg["text"]) not in exclude_texts
    ]
    print(f"Pure topic-matched: {len(pure_pairs_all)} total pairs -> {len(pure_pairs)} eligible (excluded {len(pure_pairs_all) - len(pure_pairs)})")
    pure_topic_defence_ids = {pos["example_id"] for _pid, pos, _neg in pure_pairs}

    # --- Easy: form synthetic pairs (documented as such), exclude ----------
    easy_pos = [r for r in easy_rows if r["label"] == 1]
    easy_neg = [r for r in easy_rows if r["label"] == 0]
    easy_pos_eligible = [r for r in easy_pos if r["document_id"] not in exclude_doc_ids and r["example_id"] not in pure_topic_defence_ids]
    easy_neg_eligible = [r for r in easy_neg if r["document_id"] not in exclude_doc_ids]
    easy_pos_eligible.sort(key=lambda r: rank_key("pref-heavy-easy-pos", r["example_id"]))
    easy_neg_eligible.sort(key=lambda r: rank_key("pref-heavy-easy-neg", r["example_id"]))
    # Easy's negative pool reuses the same Wikipedia article across multiple
    # passages (892 rows, only 326 distinct articles) -- harmless when
    # positives/negatives are independent populations (the original design),
    # but this script links each synthetic pair's two documents together, so
    # a shared article would transitively chain unrelated Defence documents
    # into one giant component. Dedupe to one passage per article first.
    seen_wiki_docs: set[str] = set()
    easy_neg_deduped = []
    for r in easy_neg_eligible:
        if r["document_id"] in seen_wiki_docs:
            continue
        seen_wiki_docs.add(r["document_id"])
        easy_neg_deduped.append(r)
    easy_neg_eligible = easy_neg_deduped
    n_easy = min(len(easy_pos_eligible), len(easy_neg_eligible))
    easy_pairs = [
        (f"{p['example_id']}::{q['example_id']}", p, q)
        for p, q in zip(easy_pos_eligible[:n_easy], easy_neg_eligible[:n_easy])
    ]
    easy_excluded_by_common_test = len(easy_pos) - len([r for r in easy_pos if r["document_id"] not in exclude_doc_ids])
    easy_excluded_by_pure_topic_claim = len([r for r in easy_pos if r["document_id"] not in exclude_doc_ids]) - len(easy_pos_eligible)
    print(f"Easy: {len(easy_pos)} total pairs -> {len(easy_pairs)} eligible "
          f"(excluded {easy_excluded_by_common_test} via common-test overlap, {easy_excluded_by_pure_topic_claim} via Pure-Topic passage claim)")

    # --- Preference: full 2,327 QC-passed corpus, real pairing, exclude ----
    raw = []
    for p in PREFERENCE_SHARDS:
        raw.extend(read_jsonl([p]))
    qc_passed = [r for r in raw if r.get("automatic_qc", {}).get("passed") and "doc_id" in r]
    pref_pairs_all = []
    for row in qc_passed:
        pref_pairs_all.append((row["candidate_id"], row))
    pref_pairs = []
    for cid, row in pref_pairs_all:
        if cid in exclude_pair_ids or row["doc_id"] in exclude_doc_ids:
            continue
        target_text, everyday_text = normalise_text(row["target_answer"]), normalise_text(row["dispreferred_answer"])
        if target_text in exclude_texts or everyday_text in exclude_texts:
            continue
        pref_pairs.append((cid, row))
    print(f"Preference: {len(pref_pairs_all)} QC-passed pairs -> {len(pref_pairs)} eligible (excluded {len(pref_pairs_all) - len(pref_pairs)})")

    # --- Global document graph: any two documents linked by a pair (a Defence
    # passage and its Wikipedia negative, for Easy/Pure-Topic) must land in
    # the same split, transitively, across ALL THREE sources -- documents
    # (both Defence passages and Wikipedia articles) are shared across
    # sources far more at this scale than in the smaller 892-pair
    # experiments, so independent per-source splitting is not safe here.
    uf = UnionFind()
    for _pid, pos, neg in easy_pairs:
        uf.union(pos["document_id"], neg["document_id"])
    for _pid, pos, neg in pure_pairs:
        uf.union(pos["document_id"], neg["document_id"])
    for _cid, row in pref_pairs:
        uf.find(row["doc_id"])  # single-document pair (target+everyday share doc_id); ensure it's registered

    component_weight: dict[str, int] = defaultdict(int)
    for _pid, pos, _neg in easy_pairs:
        component_weight[uf.find(pos["document_id"])] += 1
    for _pid, pos, _neg in pure_pairs:
        component_weight[uf.find(pos["document_id"])] += 1
    for _cid, row in pref_pairs:
        component_weight[uf.find(row["doc_id"])] += 1

    print(f"Document graph: {len(component_weight)} connected components across all three sources")
    component_split = global_component_split(dict(component_weight), "pref-heavy-global-split")

    # --- Assemble output rows ------------------------------------------------
    output_rows: list[dict] = []
    counts: dict[str, dict[str, int]] = {s: {"easy": 0, "pure_topic": 0, "preference": 0} for s in RATIOS}

    for pid, pos, neg in easy_pairs:
        split = component_split[uf.find(pos["document_id"])]
        counts[split]["easy"] += 1
        output_rows.append({**pos, "split": split, "combined_source": "easy", "combined_pair_id": pid})
        output_rows.append({**neg, "split": split, "combined_source": "easy", "combined_pair_id": pid})

    for pid, pos, neg in pure_pairs:
        split = component_split[uf.find(pos["document_id"])]
        counts[split]["pure_topic"] += 1
        output_rows.append({**pos, "split": split, "combined_source": "pure_topic", "combined_pair_id": pid})
        output_rows.append({**neg, "split": split, "combined_source": "pure_topic", "combined_pair_id": pid})

    from defence_language_classifier.chunking import word_count
    for cid, row in pref_pairs:
        split = component_split[uf.find(row["doc_id"])]
        counts[split]["preference"] += 1
        for kind, label in (("target_answer", 1), ("dispreferred_answer", 0)):
            text = row[kind]
            output_rows.append({
                "example_id": stable_id(cid, kind, text),
                "document_id": row["doc_id"],
                "chunk_index": 0,
                "text": text,
                "label": label,
                "source_group": "preference_pair",
                "source_name": "prefqa-generation-qwen3-32b-awq",
                "negative_type": None if label == 1 else "dispreferred_answer",
                "word_count": word_count(text),
                "candidate_id": cid,
                "question": row.get("question"),
                "split": split,
                "combined_source": "preference",
                "combined_pair_id": cid,
            })

    print()
    print("=== PROPOSED FINAL COUNTS (before any training) ===")
    total_pairs = sum(sum(c.values()) for c in counts.values())
    for split in ("train", "validation", "test"):
        c = counts[split]
        print(f"  {split}: easy={c['easy']} pure_topic={c['pure_topic']} preference={c['preference']} total_pairs={sum(c.values())}")
    print(f"  TOTAL PAIRS: {total_pairs}  TOTAL EXAMPLES: {total_pairs * 2}")

    # --- Validate ------------------------------------------------------------
    ids = [r["example_id"] for r in output_rows]
    texts = [r["text"] for r in output_rows]
    norm_texts = [normalise_text(r["text"]) for r in output_rows]
    assert len(ids) == len(set(ids)), f"duplicate example IDs: {len(ids) - len(set(ids))}"
    assert len(texts) == len(set(texts)), f"duplicate texts: {len(texts) - len(set(texts))}"

    doc_splits: dict[str, set[str]] = defaultdict(set)
    for r in output_rows:
        doc_splits[r["document_id"]].add(r["split"])
    leaking = {d: sorted(s) for d, s in doc_splits.items() if len(s) > 1}
    assert not leaking, f"document(s) span multiple splits: {leaking}"

    norm_by_split: dict[str, set[str]] = defaultdict(set)
    for r, nt in zip(output_rows, norm_texts):
        norm_by_split[r["split"]].add(nt)
    for a in ("train", "validation", "test"):
        for b in ("train", "validation", "test"):
            if a < b:
                overlap = norm_by_split[a] & norm_by_split[b]
                assert not overlap, f"normalised text overlap between {a} and {b}: {len(overlap)}"

    used_docs = {r["document_id"] for r in output_rows}
    leaked_common = used_docs & exclude_doc_ids
    assert not leaked_common, f"common-test documents leaked into combined_preference_heavy: {len(leaked_common)}"
    used_texts_norm = set(norm_texts)
    leaked_texts = used_texts_norm & exclude_texts
    assert not leaked_texts, f"common-test texts leaked in: {len(leaked_texts)}"

    for split in ("train", "validation", "test"):
        for source in ("easy", "pure_topic", "preference"):
            rows = [r for r in output_rows if r["split"] == split and r["combined_source"] == source]
            pos = sum(1 for r in rows if r["label"] == 1)
            neg = sum(1 for r in rows if r["label"] == 0)
            assert pos == neg, f"{split}/{source} not balanced: {pos} pos vs {neg} neg"

    output_rows.sort(key=lambda row: (row["split"], row["document_id"], row.get("chunk_index", 0)))
    out_path = Path("data/processed/combined_preference_heavy_dataset_splits.jsonl")
    write_jsonl(out_path, output_rows)
    summary = validate_dataset(output_rows)

    manifest = {
        "dataset_name": "combined_preference_heavy",
        "seed": SEED,
        "exclusion": {
            "excluded_document_ids": len(exclude_doc_ids),
            "excluded_normalised_texts": len(exclude_texts),
            "excluded_pair_ids": len(exclude_pair_ids),
            "easy_pairs_excluded": len(easy_pos) - len(easy_pairs),
            "pure_topic_pairs_excluded": len(pure_pairs_all) - len(pure_pairs),
            "preference_pairs_excluded": len(pref_pairs_all) - len(pref_pairs),
        },
        "eligible_pairs": {"easy": len(easy_pairs), "pure_topic": len(pure_pairs), "preference": len(pref_pairs)},
        "final_counts_by_split_and_source": counts,
        "total_pairs": total_pairs,
        "total_examples": len(output_rows),
        "easy_pairing_note": "Easy pairs are deterministic SHA256-sorted synthetic pairings, not genuine source-linked pairs (Easy negatives were only length-matched at build time). Pure-Topic and Preference pairs are genuine (matched_defence_example_id and candidate_id respectively).",
        "validate_dataset_summary": summary,
        "output_sha256": file_sha256(out_path),
    }
    Path("data/processed/combined_preference_heavy_split_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print()
    print("All validation checks passed.")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
