#!/usr/bin/env python3
"""Score one or more text strings with a trained Defence-language classifier."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib
from sentence_transformers import SentenceTransformer


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("text", nargs="+")
    parser.add_argument("--model-dir", type=Path, default=Path("models/pilot"))
    parser.add_argument("--model", choices=("tfidf", "sbert"), default="sbert")
    args = parser.parse_args()

    metadata = json.loads((args.model_dir / "model_metadata.json").read_text(encoding="utf-8"))
    if args.model == "tfidf":
        config = metadata["tfidf_logistic_regression"]
        model = joblib.load(args.model_dir / config["artifact"])
        probabilities = model.predict_proba(args.text)[:, 1]
    else:
        config = metadata["sbert_logistic_regression"]
        encoder = SentenceTransformer(config["checkpoint"])
        embeddings = encoder.encode(args.text, normalize_embeddings=config["normalize_embeddings"])
        model = joblib.load(args.model_dir / config["artifact"])
        probabilities = model.predict_proba(embeddings)[:, 1]

    for text, probability in zip(args.text, probabilities):
        print(
            json.dumps(
                {
                    "text": text,
                    "model": args.model,
                    "defence_probability": float(probability),
                    "threshold": config["threshold"],
                    "prediction": int(probability >= config["threshold"]),
                },
                ensure_ascii=False,
            )
        )


if __name__ == "__main__":
    main()

