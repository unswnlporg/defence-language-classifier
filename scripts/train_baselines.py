#!/usr/bin/env python3
"""Train TF-IDF and frozen-SBERT logistic-regression baselines."""

from __future__ import annotations

import argparse
import json
import platform
import sys
from pathlib import Path

import joblib
import numpy as np
import sklearn
from sentence_transformers import SentenceTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from defence_language_classifier.training import (  # noqa: E402
    classification_metrics,
    evaluate_with_slices,
    file_sha256,
    read_jsonl,
    select_threshold,
    validate_dataset,
    write_jsonl,
)


C_GRID = (0.01, 0.1, 1.0, 10.0)


def split_rows(records: list[dict], split: str) -> list[dict]:
    return [row for row in records if row["split"] == split]


def labels(rows: list[dict]) -> np.ndarray:
    return np.asarray([row["label"] for row in rows], dtype=np.int64)


def tune_logistic(train_x, train_y, validation_x, validation_y) -> tuple[LogisticRegression, dict]:
    candidates = []
    best = None
    for c_value in C_GRID:
        model = LogisticRegression(C=c_value, max_iter=5000, class_weight="balanced", random_state=42)
        model.fit(train_x, train_y)
        probabilities = model.predict_proba(validation_x)[:, 1]
        threshold = select_threshold(validation_y, probabilities)
        metrics = classification_metrics(validation_y, probabilities, threshold)
        candidate = {"C": c_value, "threshold": threshold, "validation": metrics}
        candidates.append(candidate)
        key = (metrics["roc_auc"], metrics["f1"], -c_value)
        if best is None or key > best[0]:
            best = (key, model, candidate)
    return best[1], {"selected": best[2], "candidates": candidates}


def prediction_rows(rows: list[dict], probabilities: np.ndarray, threshold: float, model_name: str):
    output = []
    for row, probability in zip(rows, probabilities):
        output.append(
            {
                "example_id": row["example_id"],
                "document_id": row["document_id"],
                "label": row["label"],
                "negative_type": row.get("negative_type"),
                "model": model_name,
                "defence_probability": float(probability),
                "prediction": int(probability >= threshold),
            }
        )
    return output


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--report-dir", type=Path, required=True)
    parser.add_argument("--cache-dir", type=Path, required=True)
    parser.add_argument("--sbert-checkpoint", default="sentence-transformers/all-MiniLM-L6-v2")
    parser.add_argument("--batch-size", type=int, default=32)
    args = parser.parse_args()

    args.model_dir.mkdir(parents=True, exist_ok=True)
    args.report_dir.mkdir(parents=True, exist_ok=True)
    args.cache_dir.mkdir(parents=True, exist_ok=True)

    records = read_jsonl([args.dataset])
    split_summary = validate_dataset(records)
    train = split_rows(records, "train")
    validation = split_rows(records, "validation")
    test = split_rows(records, "test")
    train_y, validation_y, test_y = labels(train), labels(validation), labels(test)

    tfidf = TfidfVectorizer(
        lowercase=True,
        ngram_range=(1, 2),
        min_df=2,
        max_df=0.98,
        max_features=50_000,
        sublinear_tf=True,
        strip_accents="unicode",
    )
    train_tfidf = tfidf.fit_transform([row["text"] for row in train])
    validation_tfidf = tfidf.transform([row["text"] for row in validation])
    test_tfidf = tfidf.transform([row["text"] for row in test])
    tfidf_lr, tfidf_selection = tune_logistic(train_tfidf, train_y, validation_tfidf, validation_y)
    tfidf_threshold = tfidf_selection["selected"]["threshold"]
    tfidf_probabilities = tfidf_lr.predict_proba(test_tfidf)[:, 1]
    tfidf_metrics = evaluate_with_slices(test, tfidf_probabilities, tfidf_threshold)
    tfidf_pipeline = Pipeline([("tfidf", tfidf), ("classifier", tfidf_lr)])
    joblib.dump(tfidf_pipeline, args.model_dir / "tfidf_logistic_regression.joblib")

    cache_key = file_sha256(args.dataset)[:16]
    embedding_cache = args.cache_dir / f"sbert_{cache_key}.npz"
    if embedding_cache.exists():
        cached = np.load(embedding_cache)
        embeddings = cached["embeddings"]
        cached_ids = cached["example_ids"].tolist()
        if cached_ids != [row["example_id"] for row in records]:
            raise ValueError("Embedding cache IDs do not match dataset order")
    else:
        encoder = SentenceTransformer(args.sbert_checkpoint)
        embeddings = encoder.encode(
            [row["text"] for row in records],
            batch_size=args.batch_size,
            show_progress_bar=True,
            normalize_embeddings=True,
            convert_to_numpy=True,
        )
        np.savez_compressed(
            embedding_cache,
            embeddings=embeddings,
            example_ids=np.asarray([row["example_id"] for row in records]),
        )

    indices = {split: [index for index, row in enumerate(records) if row["split"] == split] for split in ("train", "validation", "test")}
    train_sbert = embeddings[indices["train"]]
    validation_sbert = embeddings[indices["validation"]]
    test_sbert = embeddings[indices["test"]]
    sbert_lr, sbert_selection = tune_logistic(train_sbert, train_y, validation_sbert, validation_y)
    sbert_threshold = sbert_selection["selected"]["threshold"]
    sbert_probabilities = sbert_lr.predict_proba(test_sbert)[:, 1]
    sbert_metrics = evaluate_with_slices(test, sbert_probabilities, sbert_threshold)
    joblib.dump(sbert_lr, args.model_dir / "sbert_logistic_regression.joblib")

    predictions = prediction_rows(test, tfidf_probabilities, tfidf_threshold, "tfidf_lr")
    predictions.extend(prediction_rows(test, sbert_probabilities, sbert_threshold, "sbert_lr"))
    write_jsonl(args.report_dir / "test_predictions.jsonl", predictions)

    test_by_id = {row["example_id"]: row for row in test}
    errors = []
    for prediction in predictions:
        if prediction["label"] == prediction["prediction"]:
            continue
        source = test_by_id[prediction["example_id"]]
        errors.append(
            {
                **prediction,
                "title": source.get("title"),
                "text": source["text"],
            }
        )
    write_jsonl(args.report_dir / "misclassifications.jsonl", errors)

    feature_names = tfidf.get_feature_names_out()
    coefficients = tfidf_lr.coef_[0]
    top_n = 30
    feature_report = {
        "positive_defence": [
            {"feature": feature_names[index], "coefficient": float(coefficients[index])}
            for index in coefficients.argsort()[-top_n:][::-1]
        ],
        "negative_wikipedia": [
            {"feature": feature_names[index], "coefficient": float(coefficients[index])}
            for index in coefficients.argsort()[:top_n]
        ],
    }
    (args.report_dir / "tfidf_feature_signals.json").write_text(
        json.dumps(feature_report, indent=2) + "\n", encoding="utf-8"
    )

    report = {
        "dataset": {"path": str(args.dataset), "sha256": file_sha256(args.dataset), "splits": split_summary},
        "environment": {
            "python": platform.python_version(),
            "scikit_learn": sklearn.__version__,
            "sentence_transformers": __import__("sentence_transformers").__version__,
        },
        "tfidf_logistic_regression": {"selection": tfidf_selection, "test": tfidf_metrics},
        "sbert_logistic_regression": {
            "checkpoint": args.sbert_checkpoint,
            "embedding_dimension": int(embeddings.shape[1]),
            "selection": sbert_selection,
            "test": sbert_metrics,
        },
    }
    (args.report_dir / "metrics.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    model_metadata = {
        "dataset_sha256": file_sha256(args.dataset),
        "tfidf_logistic_regression": {
            "artifact": "tfidf_logistic_regression.joblib",
            "threshold": tfidf_threshold,
        },
        "sbert_logistic_regression": {
            "artifact": "sbert_logistic_regression.joblib",
            "checkpoint": args.sbert_checkpoint,
            "normalize_embeddings": True,
            "embedding_dimension": int(embeddings.shape[1]),
            "threshold": sbert_threshold,
        },
    }
    (args.model_dir / "model_metadata.json").write_text(
        json.dumps(model_metadata, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
