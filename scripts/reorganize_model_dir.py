#!/usr/bin/env python3
"""Split a train_baselines.py output dir (which always saves tfidf+sbert
together) into separate per-representation dirs matching the requested
models/<dataset>_<repr>/ layout, each with its own self-contained
model_metadata.json."""
from __future__ import annotations
import json, shutil, sys
from pathlib import Path

def main():
    src_dir, dataset_name, repr_name, dest_root = sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4]
    src = Path(src_dir)
    meta = json.loads((src / "model_metadata.json").read_text())

    dest_tfidf = Path(dest_root) / f"{dataset_name}_tfidf"
    dest_tfidf.mkdir(parents=True, exist_ok=True)
    shutil.copy(src / meta["tfidf_logistic_regression"]["artifact"], dest_tfidf / meta["tfidf_logistic_regression"]["artifact"])
    (dest_tfidf / "model_metadata.json").write_text(json.dumps({
        "dataset_sha256": meta["dataset_sha256"],
        "tfidf_logistic_regression": meta["tfidf_logistic_regression"],
    }, indent=2) + "\n")

    dest_sbert = Path(dest_root) / f"{dataset_name}_{repr_name}"
    dest_sbert.mkdir(parents=True, exist_ok=True)
    shutil.copy(src / meta["sbert_logistic_regression"]["artifact"], dest_sbert / meta["sbert_logistic_regression"]["artifact"])
    (dest_sbert / "model_metadata.json").write_text(json.dumps({
        "dataset_sha256": meta["dataset_sha256"],
        "sbert_logistic_regression": meta["sbert_logistic_regression"],
    }, indent=2) + "\n")
    print(f"reorganized {src_dir} -> {dest_tfidf}, {dest_sbert}")

if __name__ == "__main__":
    main()
