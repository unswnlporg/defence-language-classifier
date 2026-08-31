"""Leakage-safe splitting, model selection, and evaluation utilities."""

from __future__ import annotations

import hashlib
import json
import random
from collections import defaultdict
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    brier_score_loss,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
)


SPLITS = ("train", "validation", "test")


def read_jsonl(paths: Sequence[Path]) -> List[dict]:
    records = []
    for path in paths:
        with path.open(encoding="utf-8") as stream:
            records.extend(json.loads(line) for line in stream if line.strip())
    return records


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def stratum(row: dict) -> str:
    if row["label"] == 1:
        return "defence"
    return row.get("negative_type") or "wikipedia"


def assign_group_splits(records: Sequence[dict], ratios: Dict[str, float], seed: int) -> Dict[str, str]:
    """Assign whole documents while approximately preserving passage-level strata."""
    rng = random.Random(seed)
    groups: Dict[Tuple[str, str], List[dict]] = defaultdict(list)
    for row in records:
        groups[(stratum(row), row["document_id"])].append(row)

    assignment: Dict[str, str] = {}
    by_stratum: Dict[str, List[Tuple[str, int]]] = defaultdict(list)
    for (group_stratum, document_id), rows in groups.items():
        by_stratum[group_stratum].append((document_id, len(rows)))

    for group_stratum, group_sizes in sorted(by_stratum.items()):
        rng.shuffle(group_sizes)
        group_sizes.sort(key=lambda item: item[1], reverse=True)
        total = sum(size for _, size in group_sizes)
        targets = {split: total * ratios[split] for split in SPLITS}
        counts = {split: 0 for split in SPLITS}

        for document_id, size in group_sizes:
            def cost(split: str) -> Tuple[float, float, str]:
                projected = counts[split] + size
                relative_fill = projected / max(targets[split], 1.0)
                overflow = max(0.0, projected - targets[split]) / max(targets[split], 1.0)
                return (overflow * 10.0 + relative_fill, counts[split], split)

            chosen = min(SPLITS, key=cost)
            assignment[document_id] = chosen
            counts[chosen] += size

    return assignment


def apply_splits(records: Sequence[dict], assignment: Dict[str, str]) -> List[dict]:
    output = []
    for row in records:
        item = dict(row)
        item["split"] = assignment[row["document_id"]]
        output.append(item)
    return output


def validate_dataset(records: Sequence[dict]) -> dict:
    ids = [row["example_id"] for row in records]
    texts = [row["text"] for row in records]
    if len(ids) != len(set(ids)):
        raise ValueError("Duplicate example IDs detected")
    if len(texts) != len(set(texts)):
        raise ValueError("Exact duplicate texts detected")
    if any(row.get("split") not in SPLITS for row in records):
        raise ValueError("Missing or invalid split")

    document_splits: Dict[str, set] = defaultdict(set)
    for row in records:
        document_splits[row["document_id"]].add(row["split"])
    leaking = [document_id for document_id, splits in document_splits.items() if len(splits) > 1]
    if leaking:
        raise ValueError(f"Document leakage across splits: {leaking[:5]}")

    summary = {}
    for split in SPLITS:
        subset = [row for row in records if row["split"] == split]
        summary[split] = {
            "passages": len(subset),
            "documents": len({row["document_id"] for row in subset}),
            "positive": sum(row["label"] == 1 for row in subset),
            "negative": sum(row["label"] == 0 for row in subset),
            "general_wikipedia": sum(row.get("negative_type") == "general_wikipedia" for row in subset),
            "military_wikipedia": sum(row.get("negative_type") == "military_wikipedia" for row in subset),
        }
    return summary


def select_threshold(y_true: np.ndarray, probabilities: np.ndarray) -> float:
    precision, recall, thresholds = precision_recall_curve(y_true, probabilities)
    if not len(thresholds):
        return 0.5
    scores = 2 * precision[:-1] * recall[:-1] / np.maximum(precision[:-1] + recall[:-1], 1e-12)
    return float(thresholds[int(np.argmax(scores))])


def classification_metrics(y_true: np.ndarray, probabilities: np.ndarray, threshold: float) -> dict:
    predictions = (probabilities >= threshold).astype(int)
    matrix = confusion_matrix(y_true, predictions, labels=[0, 1])
    return {
        "n": int(len(y_true)),
        "threshold": float(threshold),
        "accuracy": float(accuracy_score(y_true, predictions)),
        "precision": float(precision_score(y_true, predictions, zero_division=0)),
        "recall": float(recall_score(y_true, predictions, zero_division=0)),
        "f1": float(f1_score(y_true, predictions, zero_division=0)),
        "roc_auc": float(roc_auc_score(y_true, probabilities)),
        "average_precision": float(average_precision_score(y_true, probabilities)),
        "brier_score": float(brier_score_loss(y_true, probabilities)),
        "confusion_matrix_labels": [0, 1],
        "confusion_matrix": matrix.tolist(),
    }


def evaluate_with_slices(records: Sequence[dict], probabilities: np.ndarray, threshold: float) -> dict:
    labels = np.asarray([row["label"] for row in records])
    result = {"overall": classification_metrics(labels, probabilities, threshold), "negative_slices": {}}
    for negative_type in ("general_wikipedia", "military_wikipedia"):
        indices = [
            index
            for index, row in enumerate(records)
            if row["label"] == 1 or row.get("negative_type") == negative_type
        ]
        slice_labels = labels[indices]
        slice_probabilities = probabilities[indices]
        result["negative_slices"][negative_type] = classification_metrics(
            slice_labels, slice_probabilities, threshold
        )
    return result


def write_jsonl(path: Path, records: Iterable[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as stream:
        for row in records:
            stream.write(json.dumps(row, ensure_ascii=False) + "\n")

