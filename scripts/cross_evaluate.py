#!/usr/bin/env python3
"""Evaluate a trained classifier against a *different* dataset's test split
than the one it was trained on, using the model's own threshold (no
re-tuning) — a transfer/generalization check.
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


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--model-type", choices=["tfidf", "sbert"], required=True)
    parser.add_argument("--target-dataset", type=Path, required=True)
    parser.add_argument("--split", default="test")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=16)
    args = parser.parse_args()

    metadata = json.loads((args.model_dir / "model_metadata.json").read_text(encoding="utf-8"))
    config = metadata[f"{args.model_type}_logistic_regression"]
    threshold = float(config["threshold"])

    rows = [r for r in read_jsonl([args.target_dataset]) if r["split"] == args.split]
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

    metrics = classification_metrics(labels, probabilities, threshold)
    result = {
        "source_model_dir": str(args.model_dir),
        "model_type": args.model_type,
        "target_dataset": str(args.target_dataset),
        "split": args.split,
        "threshold_used": threshold,
        "metrics": metrics,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
