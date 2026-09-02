#!/usr/bin/env python3
"""Pairwise (positive vs. negative) scoring for a trained model against one
of the three test sets that have -- or can be given -- a 1:1 pairing:

  preference : target_answer vs dispreferred_answer, paired by candidate_id (real pairing)
  pure_topic : Defence passage vs its topic-matched Wikipedia negative,
               paired by matched_defence_example_id (real pairing)
  easy       : Defence passage vs an easy Wikipedia negative -- the original
               easy dataset has no ground-truth link between a specific
               passage and a specific negative (negatives were only
               length-matched at build time, not id-linked), so pairs here
               are formed by deterministic SHA256-sorted position (documented
               in the output as arbitrary-but-deterministic, not a real
               content pairing).

Does not retrain or retune; scores with the model's own saved threshold and
preprocessing only.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import joblib
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))
from defence_language_classifier.training import read_jsonl  # noqa: E402

DATASETS = {
    "preference": Path("data/processed/preference_892_dataset_splits.jsonl"),
    "pure_topic": Path("data/processed/topic_matched_dataset_splits_final439.jsonl"),
    "easy": Path("data/processed/dataset_splits.jsonl"),
}


def rank_key(namespace: str, key: str) -> str:
    return hashlib.sha256(f"42\0{namespace}\0{key}".encode("utf-8")).hexdigest()


def bootstrap_mean_ci(values: np.ndarray, seed: int = 42, samples: int = 10_000) -> list[float]:
    rng = np.random.default_rng(seed)
    means = np.empty(samples)
    for index in range(samples):
        means[index] = rng.choice(values, size=len(values), replace=True).mean()
    return [float(v) for v in np.quantile(means, [0.025, 0.975])]


def score(model_dir: Path, model_type: str, texts: list[str], batch_size: int) -> tuple[np.ndarray, float]:
    metadata = json.loads((model_dir / "model_metadata.json").read_text(encoding="utf-8"))
    config = metadata[f"{model_type}_logistic_regression"]
    threshold = float(config["threshold"])
    if model_type == "tfidf":
        pipeline = joblib.load(model_dir / config["artifact"])
        probabilities = pipeline.predict_proba(texts)[:, 1]
    else:
        from sentence_transformers import SentenceTransformer

        encoder = SentenceTransformer(config["checkpoint"])
        embeddings = encoder.encode(texts, batch_size=batch_size, normalize_embeddings=config["normalize_embeddings"], convert_to_numpy=True, show_progress_bar=False)
        classifier = joblib.load(model_dir / config["artifact"])
        probabilities = classifier.predict_proba(embeddings)[:, 1]
    return probabilities, threshold


def build_pairs(test_name: str, rows: list[dict]) -> list[tuple[dict, dict]]:
    if test_name in ("preference", "pure_topic"):
        key_field = "candidate_id" if test_name == "preference" else None
        by_key: dict[str, dict[int, dict]] = {}
        if test_name == "preference":
            for row in rows:
                by_key.setdefault(row["candidate_id"], {})[row["label"]] = row
        else:  # pure_topic: negatives carry matched_defence_example_id -> positive's example_id
            positives = {r["example_id"]: r for r in rows if r["label"] == 1}
            for neg in [r for r in rows if r["label"] == 0]:
                pos = positives.get(neg.get("matched_defence_example_id"))
                if pos is not None:
                    by_key.setdefault(neg["matched_defence_example_id"], {})[1] = pos
                    by_key.setdefault(neg["matched_defence_example_id"], {})[0] = neg
        pairs = [(d[1], d[0]) for d in by_key.values() if 1 in d and 0 in d]
        return pairs
    else:  # easy: no ground-truth link, deterministic positional pairing
        positives = sorted([r for r in rows if r["label"] == 1], key=lambda r: rank_key("pairwise892-easy-pos", r["example_id"]))
        negatives = sorted([r for r in rows if r["label"] == 0], key=lambda r: rank_key("pairwise892-easy-neg", r["example_id"]))
        n = min(len(positives), len(negatives))
        return list(zip(positives[:n], negatives[:n]))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--model-type", choices=["tfidf", "sbert"], required=True)
    parser.add_argument("--test-name", choices=list(DATASETS), required=True)
    parser.add_argument("--scores-output", type=Path, required=True)
    parser.add_argument("--summary-output", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    dataset_path = DATASETS[args.test_name]
    all_rows = read_jsonl([dataset_path])
    rows = [r for r in all_rows if r["split"] == "test"]

    pairs = build_pairs(args.test_name, rows)
    if not pairs:
        raise ValueError(f"No pairs formed for test-name={args.test_name}")

    id_to_row: dict[str, dict] = {}
    texts: list[str] = []
    for pos, neg in pairs:
        for row in (pos, neg):
            if row["example_id"] not in id_to_row:
                id_to_row[row["example_id"]] = row
                texts.append(row["text"])
    ordered_ids = list(id_to_row.keys())

    probabilities, threshold = score(args.model_dir, args.model_type, texts, args.batch_size)
    prob_by_id = dict(zip(ordered_ids, probabilities))

    if np.isnan(probabilities).any():
        raise ValueError("Missing probabilities (NaN) in scored output")

    args.scores_output.parent.mkdir(parents=True, exist_ok=True)
    with args.scores_output.open("w", encoding="utf-8") as stream:
        for pos, neg in pairs:
            stream.write(json.dumps({
                "positive_example_id": pos["example_id"],
                "negative_example_id": neg["example_id"],
                "positive_score": float(prob_by_id[pos["example_id"]]),
                "negative_score": float(prob_by_id[neg["example_id"]]),
            }, ensure_ascii=False) + "\n")

    pos_scores = np.asarray([prob_by_id[pos["example_id"]] for pos, _neg in pairs])
    neg_scores = np.asarray([prob_by_id[neg["example_id"]] for _pos, neg in pairs])
    margins = pos_scores - neg_scores
    epsilon = 1e-9
    wins = int(np.sum(margins > epsilon))
    losses = int(np.sum(margins < -epsilon))
    ties = len(pairs) - wins - losses

    summary = {
        "model_dir": str(args.model_dir),
        "model_type": args.model_type,
        "test_name": args.test_name,
        "pairing_method": (
            "candidate_id (real pairing)" if args.test_name == "preference"
            else "matched_defence_example_id (real pairing)" if args.test_name == "pure_topic"
            else "deterministic SHA256-sorted position (NO ground-truth content link -- original easy negatives were only length-matched, not id-linked to a specific passage)"
        ),
        "threshold": threshold,
        "n_pairs": len(pairs),
        "pairwise": {
            "positive_greater_than_negative": wins,
            "negative_wins": losses,
            "ties": ties,
            "win_rate": wins / len(pairs),
        },
        "scores": {
            "positive_mean": float(pos_scores.mean()),
            "negative_mean": float(neg_scores.mean()),
            "mean_margin": float(margins.mean()),
            "median_margin": float(np.median(margins)),
            "mean_margin_95_percent_bootstrap_ci": bootstrap_mean_ci(margins, seed=args.seed),
        },
    }
    args.summary_output.parent.mkdir(parents=True, exist_ok=True)
    args.summary_output.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
