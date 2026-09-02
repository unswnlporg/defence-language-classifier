#!/usr/bin/env python3
"""Aggregate reports/cross_eval_892/*.json into one matrix (JSON + CSV).

Combined-892 models are restricted to exactly three test sets (easy,
pure_topic, preference) per explicit correction -- their own constructed
test set, and Hybrid-892's, are excluded as targets for Combined models.
"""
from __future__ import annotations
import csv, json
from pathlib import Path

TRAIN_NAMES = {"easy892": "Easy-892", "hybrid892": "Hybrid-892", "preference892": "Preference-892", "combined892": "Combined-892"}
MODEL_NAMES = {"tfidf": "TF-IDF", "minilm": "MiniLM", "qwen": "Qwen3-8B"}
TEST_NAMES = {"easy": "Easy", "hybrid": "Hybrid-892", "preference": "Preference-892", "combined": "Combined-892", "pure_topic": "Pure Topic-Matched"}

COMBINED_ALLOWED_TESTS = {"easy", "pure_topic", "preference"}


def main() -> None:
    src_dir = Path("reports/cross_eval_892")
    rows = []
    for path in sorted(src_dir.glob("*_on_*.json")):
        stem = path.stem  # e.g. easy892_tfidf_on_easy
        train_model, test_key = stem.split("_on_")
        train_key, model_key = train_model.rsplit("_", 1)
        if train_key == "combined892" and test_key not in COMBINED_ALLOWED_TESTS:
            continue
        data = json.loads(path.read_text())
        m = data["metrics"]
        rows.append({
            "training_dataset": TRAIN_NAMES.get(train_key, train_key),
            "model": MODEL_NAMES.get(model_key, model_key),
            "test_dataset": TEST_NAMES.get(test_key, test_key),
            "n": m["n"],
            "accuracy": m["accuracy"],
            "precision": m["precision"],
            "recall": m["recall"],
            "f1": m["f1"],
            "roc_auc": m["roc_auc"],
            "average_precision": m["average_precision"],
            "brier_score": m["brier_score"],
            "confusion_matrix": m["confusion_matrix"],
            "threshold": m["threshold"],
            "source_file": str(path),
        })

    rows.sort(key=lambda r: (r["training_dataset"], r["model"], r["test_dataset"]))

    Path("reports/cross_eval_892/full_matrix.json").write_text(json.dumps({"rows": rows, "n_rows": len(rows)}, indent=2) + "\n")

    with open("reports/cross_eval_892/full_matrix.csv", "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["Training dataset", "Model", "Test dataset", "N", "Accuracy", "F1", "AUROC"])
        for r in rows:
            writer.writerow([r["training_dataset"], r["model"], r["test_dataset"], r["n"], f"{r['accuracy']:.4f}", f"{r['f1']:.4f}", f"{r['roc_auc']:.4f}"])

    print(f"{len(rows)} rows written to full_matrix.json / full_matrix.csv")
    combined_rows = [r for r in rows if r["training_dataset"] == "Combined-892"]
    print(f"Combined-892 rows: {len(combined_rows)} (expect 9)")


if __name__ == "__main__":
    main()
