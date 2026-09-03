#!/usr/bin/env python3
"""Evaluate a combined_preference_heavy-trained model on its OWN held-out
test set: overall metrics, a per-source breakdown, and pairwise statistics
using the real combined_pair_id (synthetic for Easy, genuine for the other
two sources -- labeled accordingly)."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import joblib
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))
from defence_language_classifier.training import classification_metrics, read_jsonl  # noqa: E402


def bootstrap_mean_ci(values: np.ndarray, seed: int = 42, samples: int = 10_000) -> list[float]:
    rng = np.random.default_rng(seed)
    means = np.empty(samples)
    for i in range(samples):
        means[i] = rng.choice(values, size=len(values), replace=True).mean()
    return [float(v) for v in np.quantile(means, [0.025, 0.975])]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--model-type", choices=["tfidf", "sbert"], required=True)
    parser.add_argument("--dataset", type=Path, default=Path("data/processed/combined_preference_heavy_dataset_splits.jsonl"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    rows = [r for r in read_jsonl([args.dataset]) if r["split"] == "test"]
    metadata = json.loads((args.model_dir / "model_metadata.json").read_text())
    config = metadata[f"{args.model_type}_logistic_regression"]
    threshold = float(config["threshold"])
    texts = [r["text"] for r in rows]

    if args.model_type == "tfidf":
        pipeline = joblib.load(args.model_dir / config["artifact"])
        probs = pipeline.predict_proba(texts)[:, 1]
    else:
        from sentence_transformers import SentenceTransformer
        encoder = SentenceTransformer(config["checkpoint"])
        embeddings = encoder.encode(texts, batch_size=args.batch_size, normalize_embeddings=config["normalize_embeddings"], convert_to_numpy=True, show_progress_bar=False)
        classifier = joblib.load(args.model_dir / config["artifact"])
        probs = classifier.predict_proba(embeddings)[:, 1]

    labels = np.asarray([r["label"] for r in rows])
    overall = classification_metrics(labels, probs, threshold)

    by_source = {}
    for source in ("easy", "pure_topic", "preference"):
        idx = [i for i, r in enumerate(rows) if r["combined_source"] == source]
        if not idx:
            continue
        by_source[source] = classification_metrics(labels[idx], probs[idx], threshold)

    # pairwise via combined_pair_id
    by_pair: dict[str, dict[int, dict]] = {}
    for r, p in zip(rows, probs):
        by_pair.setdefault(r["combined_pair_id"], {})[r["label"]] = {"row": r, "prob": float(p)}
    pairwise_by_source = {}
    for source in ("easy", "pure_topic", "preference"):
        pairs = [(pid, d) for pid, d in by_pair.items() if 1 in d and 0 in d and d[1]["row"]["combined_source"] == source]
        if not pairs:
            continue
        pos = np.asarray([d[1]["prob"] for _pid, d in pairs])
        neg = np.asarray([d[0]["prob"] for _pid, d in pairs])
        margins = pos - neg
        eps = 1e-9
        wins = int(np.sum(margins > eps))
        losses = int(np.sum(margins < -eps))
        pairwise_by_source[source] = {
            "pairing_type": "synthetic (SHA256-sorted, no ground-truth link)" if source == "easy" else "genuine",
            "n_pairs": len(pairs),
            "positive_greater_than_negative": wins,
            "negative_wins": losses,
            "ties": len(pairs) - wins - losses,
            "win_rate": wins / len(pairs),
            "positive_mean": float(pos.mean()),
            "negative_mean": float(neg.mean()),
            "mean_margin": float(margins.mean()),
            "median_margin": float(np.median(margins)),
            "mean_margin_95_percent_bootstrap_ci": bootstrap_mean_ci(margins, seed=args.seed),
        }

    all_pairs = [(pid, d) for pid, d in by_pair.items() if 1 in d and 0 in d]
    pos_all = np.asarray([d[1]["prob"] for _pid, d in all_pairs])
    neg_all = np.asarray([d[0]["prob"] for _pid, d in all_pairs])
    margins_all = pos_all - neg_all
    eps = 1e-9
    wins_all = int(np.sum(margins_all > eps))
    losses_all = int(np.sum(margins_all < -eps))

    result = {
        "model_dir": str(args.model_dir),
        "model_type": args.model_type,
        "threshold": threshold,
        "n_examples": len(rows),
        "n_pairs": len(all_pairs),
        "overall": overall,
        "by_source_classification": by_source,
        "overall_pairwise": {
            "positive_greater_than_negative": wins_all,
            "negative_wins": losses_all,
            "ties": len(all_pairs) - wins_all - losses_all,
            "win_rate": wins_all / len(all_pairs),
            "positive_mean": float(pos_all.mean()),
            "negative_mean": float(neg_all.mean()),
            "mean_margin": float(margins_all.mean()),
            "median_margin": float(np.median(margins_all)),
        },
        "pairwise_by_source": pairwise_by_source,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
