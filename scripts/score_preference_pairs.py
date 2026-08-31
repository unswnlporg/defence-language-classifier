#!/usr/bin/env python3
"""Score target/dispreferred answer pairs with a frozen Defence classifier."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib
import numpy as np
from scipy.stats import binomtest, wilcoxon
from sentence_transformers import SentenceTransformer


def read_jsonl(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as stream:
        return [json.loads(line) for line in stream if line.strip()]


def bootstrap_mean_ci(values: np.ndarray, seed: int = 42, samples: int = 10_000) -> list[float]:
    rng = np.random.default_rng(seed)
    means = np.empty(samples)
    for index in range(samples):
        means[index] = rng.choice(values, size=len(values), replace=True).mean()
    return [float(value) for value in np.quantile(means, [0.025, 0.975])]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pairs", type=Path, required=True)
    parser.add_argument("--model-dir", type=Path, default=Path("models/pilot"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--summary", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=32)
    args = parser.parse_args()

    rows = read_jsonl(args.pairs)
    if not rows:
        raise ValueError("No preference pairs found")
    required = {"candidate_id", "target_answer", "dispreferred_answer"}
    for index, row in enumerate(rows):
        missing = required - row.keys()
        if missing:
            raise ValueError(f"Pair {index} is missing fields: {sorted(missing)}")

    metadata = json.loads((args.model_dir / "model_metadata.json").read_text(encoding="utf-8"))
    config = metadata["sbert_logistic_regression"]
    encoder = SentenceTransformer(config["checkpoint"])
    classifier = joblib.load(args.model_dir / config["artifact"])

    texts = []
    for row in rows:
        texts.extend((row["target_answer"], row["dispreferred_answer"]))
    embeddings = encoder.encode(
        texts,
        batch_size=args.batch_size,
        show_progress_bar=True,
        normalize_embeddings=config["normalize_embeddings"],
        convert_to_numpy=True,
    )
    probabilities = classifier.predict_proba(embeddings)[:, 1]
    target_scores = probabilities[0::2]
    dispreferred_scores = probabilities[1::2]
    margins = target_scores - dispreferred_scores
    threshold = float(config["threshold"])
    epsilon = 1e-9

    scored = []
    for row, target_score, dispreferred_score, margin in zip(
        rows, target_scores, dispreferred_scores, margins
    ):
        winner = "target_answer" if margin > epsilon else "dispreferred_answer" if margin < -epsilon else "tie"
        scored.append(
            {
                "candidate_id": row["candidate_id"],
                "doc_id": row.get("doc_id"),
                "chunk_id": row.get("chunk_id"),
                "question": row.get("question"),
                "target_answer": row["target_answer"],
                "dispreferred_answer": row["dispreferred_answer"],
                "target_defence_probability": float(target_score),
                "dispreferred_defence_probability": float(dispreferred_score),
                "margin": float(margin),
                "winner": winner,
                "target_prediction": int(target_score >= threshold),
                "dispreferred_prediction": int(dispreferred_score >= threshold),
                "target_word_count": len(row["target_answer"].split()),
                "dispreferred_word_count": len(row["dispreferred_answer"].split()),
                "automatic_qc_passed": row.get("automatic_qc", {}).get("passed"),
            }
        )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as stream:
        for row in scored:
            stream.write(json.dumps(row, ensure_ascii=False) + "\n")

    target_wins = int(np.sum(margins > epsilon))
    dispreferred_wins = int(np.sum(margins < -epsilon))
    ties = len(rows) - target_wins - dispreferred_wins
    non_ties = target_wins + dispreferred_wins
    sign_test = binomtest(target_wins, non_ties, 0.5, alternative="greater") if non_ties else None
    signed_rank = wilcoxon(margins, alternative="greater", zero_method="wilcox")
    target_lengths = np.asarray([row["target_word_count"] for row in scored])
    dispreferred_lengths = np.asarray([row["dispreferred_word_count"] for row in scored])

    summary = {
        "pairs": len(rows),
        "model": "sbert_logistic_regression",
        "checkpoint": config["checkpoint"],
        "classification_threshold": threshold,
        "pairwise": {
            "target_wins": target_wins,
            "dispreferred_wins": dispreferred_wins,
            "ties": ties,
            "target_win_rate": target_wins / len(rows),
            "sign_test_p_value": float(sign_test.pvalue) if sign_test else None,
            "wilcoxon_p_value": float(signed_rank.pvalue),
        },
        "scores": {
            "target_mean": float(target_scores.mean()),
            "target_median": float(np.median(target_scores)),
            "dispreferred_mean": float(dispreferred_scores.mean()),
            "dispreferred_median": float(np.median(dispreferred_scores)),
            "mean_margin": float(margins.mean()),
            "median_margin": float(np.median(margins)),
            "mean_margin_95_percent_bootstrap_ci": bootstrap_mean_ci(margins),
        },
        "threshold_outcomes": {
            "both_defence": int(np.sum((target_scores >= threshold) & (dispreferred_scores >= threshold))),
            "target_only": int(np.sum((target_scores >= threshold) & (dispreferred_scores < threshold))),
            "dispreferred_only": int(np.sum((target_scores < threshold) & (dispreferred_scores >= threshold))),
            "neither": int(np.sum((target_scores < threshold) & (dispreferred_scores < threshold))),
        },
        "answer_lengths": {
            "training_range_words": [50, 200],
            "target_mean_words": float(target_lengths.mean()),
            "dispreferred_mean_words": float(dispreferred_lengths.mean()),
            "target_below_training_range": int(np.sum(target_lengths < 50)),
            "dispreferred_below_training_range": int(np.sum(dispreferred_lengths < 50)),
        },
    }
    args.summary.parent.mkdir(parents=True, exist_ok=True)
    args.summary.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()

