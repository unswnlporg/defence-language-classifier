#!/usr/bin/env python3
"""Evaluate an EASY-dataset-trained classifier on the held-out preference
test set, pairing target vs everyday (dispreferred) answers by candidate_id.

Does not retrain or retune anything: loads the already-saved classifier and
its original preprocessing (TF-IDF vectorizer, or the SentenceTransformer
checkpoint named in the model's own metadata) and scores the frozen text.
"""

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
    for index in range(samples):
        means[index] = rng.choice(values, size=len(values), replace=True).mean()
    return [float(v) for v in np.quantile(means, [0.025, 0.975])]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--model-type", choices=["tfidf", "sbert"], required=True)
    parser.add_argument("--dataset", type=Path, default=Path("data/processed/preference_classifier_dataset_splits.jsonl"))
    parser.add_argument("--scores-output", type=Path, required=True)
    parser.add_argument("--summary-output", type=Path, required=True)
    parser.add_argument("--samples-output", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--expected-examples", type=int, default=None)
    parser.add_argument("--expected-pairs", type=int, default=None)
    args = parser.parse_args()

    rows = [r for r in read_jsonl([args.dataset]) if r["split"] == "test"]
    if args.expected_examples is not None and len(rows) != args.expected_examples:
        raise ValueError(f"Expected {args.expected_examples} test examples, got {len(rows)}")

    example_ids = [r["example_id"] for r in rows]
    if len(example_ids) != len(set(example_ids)):
        raise ValueError("Duplicate example IDs in test split")

    metadata = json.loads((args.model_dir / "model_metadata.json").read_text(encoding="utf-8"))
    config = metadata[f"{args.model_type}_logistic_regression"]
    threshold = float(config["threshold"])

    texts = [r["text"] for r in rows]
    labels = np.asarray([r["label"] for r in rows], dtype=np.int64)

    if args.model_type == "tfidf":
        pipeline = joblib.load(args.model_dir / config["artifact"])
        probabilities = pipeline.predict_proba(texts)[:, 1]
    else:
        from sentence_transformers import SentenceTransformer

        encoder = SentenceTransformer(config["checkpoint"])
        embeddings = encoder.encode(
            texts, batch_size=args.batch_size, normalize_embeddings=config["normalize_embeddings"], convert_to_numpy=True, show_progress_bar=False
        )
        classifier = joblib.load(args.model_dir / config["artifact"])
        probabilities = classifier.predict_proba(embeddings)[:, 1]

    if np.isnan(probabilities).any():
        raise ValueError("Missing probabilities (NaN) in scored output")

    # --- aggregate metrics, must reproduce reports/cross_eval/easy_*_on_pref.json ---
    metrics = classification_metrics(labels, probabilities, threshold)

    # --- per-example scores ---
    scored_rows = []
    for row, probability in zip(rows, probabilities):
        scored_rows.append(
            {
                "example_id": row["example_id"],
                "candidate_id": row["candidate_id"],
                "document_id": row["document_id"],
                "label": row["label"],
                "defence_probability": float(probability),
                "prediction": int(probability >= threshold),
            }
        )
    args.scores_output.parent.mkdir(parents=True, exist_ok=True)
    with args.scores_output.open("w", encoding="utf-8") as stream:
        for record in scored_rows:
            stream.write(json.dumps(record, ensure_ascii=False) + "\n")

    # --- pair by candidate_id: one label=1 (target) + one label=0 (everyday) ---
    by_candidate: dict[str, dict[int, dict]] = {}
    for row, probability in zip(rows, probabilities):
        cid = row["candidate_id"]
        by_candidate.setdefault(cid, {})[row["label"]] = {"row": row, "prob": float(probability)}

    incomplete = [cid for cid, d in by_candidate.items() if not (1 in d and 0 in d)]
    if incomplete:
        raise ValueError(f"{len(incomplete)} candidate_id(s) missing a label-1 or label-0 example: {incomplete[:5]}")
    pairs = [(cid, d) for cid, d in by_candidate.items() if 1 in d and 0 in d]
    if args.expected_pairs is not None and len(pairs) != args.expected_pairs:
        raise ValueError(f"Expected {args.expected_pairs} complete pairs, got {len(pairs)}")

    target_probs = np.asarray([d[1]["prob"] for _cid, d in pairs])
    everyday_probs = np.asarray([d[0]["prob"] for _cid, d in pairs])
    margins = target_probs - everyday_probs
    epsilon = 1e-9
    wins = int(np.sum(margins > epsilon))
    losses = int(np.sum(margins < -epsilon))
    ties = len(pairs) - wins - losses

    summary = {
        "model_dir": str(args.model_dir),
        "model_type": args.model_type,
        "n_examples": len(rows),
        "n_pairs": len(pairs),
        "threshold": threshold,
        "pairwise": {
            "target_greater_than_everyday": wins,
            "everyday_wins": losses,
            "ties": ties,
            "target_win_rate": wins / len(pairs),
        },
        "scores": {
            "target_mean": float(target_probs.mean()),
            "everyday_mean": float(everyday_probs.mean()),
            "mean_margin": float(margins.mean()),
            "median_margin": float(np.median(margins)),
            "mean_margin_95_percent_bootstrap_ci": bootstrap_mean_ci(margins, seed=args.seed),
        },
        "aggregate_metrics_all_698": metrics,
    }
    args.summary_output.parent.mkdir(parents=True, exist_ok=True)
    args.summary_output.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")

    # --- qualitative samples ---
    def sample_record(cid: str, d: dict) -> dict:
        target, everyday = d[1], d[0]
        margin = target["prob"] - everyday["prob"]
        return {
            "candidate_id": cid,
            "question": target["row"].get("question"),
            "target_answer": target["row"]["text"],
            "everyday_answer": everyday["row"]["text"],
            "target_probability": target["prob"],
            "everyday_probability": everyday["prob"],
            "margin": margin,
            "winner": "target" if margin > epsilon else "everyday" if margin < -epsilon else "tie",
        }

    by_margin_desc = sorted(pairs, key=lambda item: item[1][1]["prob"] - item[1][0]["prob"], reverse=True)
    by_abs_margin = sorted(pairs, key=lambda item: abs(item[1][1]["prob"] - item[1][0]["prob"]))

    samples = {
        "highest_positive_margin": [sample_record(cid, d) for cid, d in by_margin_desc[:3]],
        "lowest_negative_margin": [sample_record(cid, d) for cid, d in by_margin_desc[-3:]],
        "closest_to_tie": [sample_record(cid, d) for cid, d in by_abs_margin[:2]],
    }
    args.samples_output.write_text(json.dumps(samples, indent=2) + "\n", encoding="utf-8")

    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
