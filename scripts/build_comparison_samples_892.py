#!/usr/bin/env python3
"""Build reports/preference_common_892/comparison_samples.md: 2 representative
successes, 2 representative failures, 1 near-tie, each scored by all 12
models trained across the four _892 datasets, on the same candidate pairs."""
from __future__ import annotations
import json
from pathlib import Path

MODELS = [
    ("easy892", "tfidf"), ("easy892", "minilm"), ("easy892", "qwen"),
    ("hybrid892", "tfidf"), ("hybrid892", "minilm"), ("hybrid892", "qwen"),
    ("preference892", "tfidf"), ("preference892", "minilm"), ("preference892", "qwen"),
    ("combined892", "tfidf"), ("combined892", "minilm"), ("combined892", "qwen"),
]
LABELS = {"easy892": "Easy-892", "hybrid892": "Hybrid-892", "preference892": "Preference-892", "combined892": "Combined-892",
          "tfidf": "TF-IDF", "minilm": "MiniLM", "qwen": "Qwen3-8B"}


def main():
    by_model = {}
    text_by_cid = {}
    for name, repr_ in MODELS:
        key = f"{name}_{repr_}"
        rows = []
        with open(f"reports/preference_common_892/{key}_scores.jsonl") as f:
            for line in f:
                rows.append(json.loads(line))
        by_cid = {}
        for r in rows:
            by_cid.setdefault(r["candidate_id"], {})[r["label"]] = r["defence_probability"]
        by_model[key] = by_cid

    pref_rows = []
    with open("data/processed/preference_892_dataset_splits.jsonl") as f:
        for line in f:
            r = json.loads(line)
            if r["split"] == "test":
                pref_rows.append(r)
    for r in pref_rows:
        d = text_by_cid.setdefault(r["candidate_id"], {"question": r.get("question")})
        d["target_answer" if r["label"] == 1 else "everyday_answer"] = r["text"]

    model_keys = [f"{n}_{r}" for n, r in MODELS]
    common_cids = set.intersection(*[set(by_model[k]) for k in model_keys])
    print(f"common candidate_ids across all 12 models: {len(common_cids)}")

    records = []
    for cid in common_cids:
        margins = {}
        for k in model_keys:
            d = by_model[k][cid]
            if 1 not in d or 0 not in d:
                break
            margins[k] = d[1] - d[0]
        else:
            records.append((cid, margins))

    def avg_margin(m):
        return sum(m.values()) / len(m)

    all_positive = [r for r in records if all(v > 0 for v in r[1].values())]
    all_negative = [r for r in records if all(v < 0 for v in r[1].values())]
    print(f"all-12-success: {len(all_positive)}, all-12-failure: {len(all_negative)}")

    all_positive.sort(key=lambda r: avg_margin(r[1]))
    all_negative.sort(key=lambda r: avg_margin(r[1]))
    success_picks = all_positive[len(all_positive) // 2 - 1: len(all_positive) // 2 + 1] if len(all_positive) >= 2 else all_positive[:2]
    failure_picks = all_negative[len(all_negative) // 2 - 1: len(all_negative) // 2 + 1] if len(all_negative) >= 2 else all_negative[:2]

    records.sort(key=lambda r: sum(abs(v) for v in r[1].values()))
    near_tie = records[0]

    def fmt_pair(cid, margins, label):
        t = text_by_cid[cid]
        lines = [f"### {label} -- `{cid}`", "", f"**Question:** {t.get('question')}", "",
                 f"**Defence-register target answer:**", f"> {t['target_answer']}", "",
                 f"**Everyday-language answer:**", f"> {t['everyday_answer']}", "",
                 "| Training dataset | Model | Target P(defence) | Everyday P(defence) | Margin |",
                 "|---|---|---:|---:|---:|"]
        for name, repr_ in MODELS:
            key = f"{name}_{repr_}"
            d = by_model[key][cid]
            m = d[1] - d[0]
            lines.append(f"| {LABELS[name]} | {LABELS[repr_]} | {d[1]:.3f} | {d[0]:.3f} | {m:+.3f} |")
        lines.append("")
        return "\n".join(lines)

    out = ["# All 12 models on the common Preference-892 test set: representative examples", "",
           "Every model below (4 training datasets x 3 representations) scored on the exact "
           "same candidate pairs from the Preference-892 common test set, using each model's "
           "own saved threshold. Selected to be typical of model behaviour, not the most extreme cases.", "",
           "## Representative successes (all 12 models rank target higher)", ""]
    for i, (cid, margins) in enumerate(success_picks):
        out.append(fmt_pair(cid, margins, f"Success {chr(65+i)}"))
    out.append("## Representative failures (all 12 models rank everyday higher)")
    out.append("")
    for i, (cid, margins) in enumerate(failure_picks):
        out.append(fmt_pair(cid, margins, f"Failure {chr(65+i)}"))
    out.append("## Near-tie example")
    out.append("")
    out.append(fmt_pair(near_tie[0], near_tie[1], "Near tie"))

    Path("reports/preference_common_892/comparison_samples.md").write_text("\n".join(out) + "\n")
    print("wrote reports/preference_common_892/comparison_samples.md")


if __name__ == "__main__":
    main()
