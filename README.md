# Defence Language Classifier

Train a small, reproducible classifier that estimates whether a passage resembles
the language in the Defence corpus.

This project adapts the broad setup from *In Vino Veritas and Vulnerabilities*:
frozen Sentence-BERT embeddings followed by logistic regression. It does not
attempt to reproduce details that the paper does not report.

## Research question

Can a lightweight classifier distinguish Defence-corpus prose from matched
non-Defence prose, including military-related Wikipedia text?

The model output is:

```text
P(defence-corpus language | text)
```

This is a domain-language score, not a factual-correctness or overall-quality
score.

## Experimental design

1. Clean and chunk Defence documents into 50-200 word passages.
2. Build length-matched negative passages from:
   - general Wikipedia;
   - military-related Wikipedia as hard negatives.
3. Split by source document, never by passage.
4. Train TF-IDF + logistic regression as a lexical baseline.
5. Train frozen SBERT embeddings + logistic regression.
6. Evaluate both models on the same held-out test set.
7. Score the existing preference-QA answers only after the classifier passes
   held-out and hard-negative checks.

## Success criteria for the pilot

- No document leakage across splits.
- Report accuracy, precision, recall, F1, AUROC, calibration, and confusion
  matrices.
- Report performance separately for general and military Wikipedia negatives.
- Inspect false positives and false negatives.
- Demonstrate that results are not explained only by passage length, metadata,
  or a small set of military keywords.

## Repository layout

```text
config/       Experiment configuration
data/         Local and generated datasets
docs/         Method and decision records
models/       Saved encoders/classifiers
reports/      Metrics and error analysis
scripts/      Command-line pipeline stages
src/          Reusable Python package
tests/        Dataset and leakage checks
```

## Dataset record

```json
{
  "example_id": "stable-hash",
  "document_id": "source-document-id",
  "text": "A clean standalone passage.",
  "label": 1,
  "source_group": "defence",
  "source_name": "forge_war_college",
  "negative_type": null,
  "word_count": 84,
  "split": "train"
}
```

Labels are `1` for Defence-corpus passages and `0` for non-Defence passages.

## Current status

- [x] Define the task and limitations.
- [x] Define the dataset schema and leakage policy.
- [x] Add an initial Defence-corpus audit tool.
- [x] Audit usable positive passages.
- [x] Select and acquire Wikipedia negatives.
- [x] Build frozen dataset splits.
- [x] Train TF-IDF baseline.
- [x] Train SBERT + logistic regression.
- [x] Evaluate and conduct error analysis.
- [x] Score the 300 QA candidates.
- [x] Re-run frozen-embedding + logistic-regression training with
      `Qwen/Qwen3-Embedding-8B` on Katana.
- [x] Construct topic-matched Wikipedia negatives for a harder evaluation
      (439/892 passages matched; see "Topic-matched dataset" in
      `docs/training_methodology.md` for the yield ceiling and why it wasn't
      forced to 892).
- [x] Compare Qwen3-Embedding-8B against the MiniLM and TF-IDF baselines on
      the topic-matched hybrid dataset.
- [ ] Consider whether topic-matching yield can be raised past ~49% with a
      different approach (e.g. a local full-Wikipedia index), or whether 439
      is the practical ceiling for this corpus.

## Katana handoff

The repository includes the processed pilot passages, frozen dataset split,
evaluation reports, and preference-pair scores. Local caches, virtual
environments, raw source material, and serialized model binaries remain
excluded.

For the next experiment, keep the embedding model frozen and replace
`sentence-transformers/all-MiniLM-L6-v2` with
`Qwen/Qwen3-Embedding-8B`. Train the same logistic-regression classifier on the
resulting embeddings, tune only on the validation split, and evaluate once on
the held-out test split. The topic-matched dataset should be constructed and
frozen before comparing encoders.

The executed pilot methodology and results are documented in
[`docs/training_methodology.md`](docs/training_methodology.md).
