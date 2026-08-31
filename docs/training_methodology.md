# Defence-Language Classifier: Pilot Methodology

## Objective

The pilot trains a lightweight binary classifier to estimate whether a standalone
piece of text resembles the language observed in the Defence corpus.

```text
Input:  standalone text passage or answer
Output: P(defence-corpus language | text)
```

The classifier measures corpus resemblance. It does not directly measure factual
correctness, doctrinal correctness, answer relevance, usefulness, or expert
preference.

## Motivation

The earlier preference-data pipeline depended on generation prompts and an LLM
judge without a sufficiently constrained expert rubric. This introduced prompt
sensitivity, answer-order effects, and interpretive uncertainty.

This pilot replaces the style component of that process with a fixed and
repeatable statistical signal. It adapts the broad classifier design reported in
*In Vino Veritas and Vulnerabilities*: frozen Sentence-BERT features followed by
logistic regression. The cited paper does not report its negative-class
construction, exact SBERT checkpoint, classifier split, hyperparameters, or
classification metrics. The decisions below are therefore our own explicit
experimental design rather than an exact reproduction.

## Data construction

### Positive class

The positive class came from the cleaned Forge War College Defence corpus:

- 53 source documents read;
- 52 documents retained;
- one 36-word document excluded as too short;
- 161,819 source words before chunking;
- 892 unique Defence passages produced.

Each source document was split using a deterministic, sentence-aware chunker.
Passages contain 50-200 words whenever the document permits. The final positive
passages range from 53 to 200 words, with an average of 181.37 words.

Every passage retains a stable `document_id`, `chunk_index`, and content-derived
`example_id`. Titles are retained as metadata but are not classifier inputs.

### Negative class

The negative class came from the English Wikipedia dataset hosted by Hugging Face
(`wikimedia/wikipedia`, configuration `20231101.en`). It contains:

- 446 general Wikipedia passages;
- 446 military-related Wikipedia passages;
- 892 unique negative passages;
- 326 source articles in total;
- 207 general source articles;
- 119 military-related source articles.

Military hard negatives were selected using explicit military signals in article
titles, such as battles, wars, armed services, units, weapons, or military
personnel. This subset prevents military subject matter alone from acting as the
positive label.

### What “matched” means

Positive and negative passages were not paired by topic or meaning. Matching was
limited to observable dataset properties:

- equal class sizes;
- the same 50-200-word range;
- closely matched passage-length distributions;
- clean standalone English prose;
- preserved document provenance.

The negative passage selected for each target length differed by 0.08 words on
average, with a maximum difference of four words.

The final dataset contains 1,784 passages:

```text
Defence:    892 (label 1)
Wikipedia:  892 (label 0)
```

## Data splitting and leakage control

Splits were assigned at source-document level using seed `42`. Every passage from
one document remains in exactly one split. Exact duplicate IDs and texts were
rejected, and the completed dataset passed the document-leakage check.

| Split | Passages | Documents | Defence | General Wiki | Military Wiki |
|---|---:|---:|---:|---:|---:|
| Train | 1,244 | 260 | 620 | 312 | 312 |
| Validation | 270 | 59 | 136 | 67 | 67 |
| Test | 270 | 59 | 136 | 67 | 67 |

The validation set was used for model and threshold selection. The held-out test
set was evaluated after those decisions were fixed.

The frozen split artifact has SHA-256 digest:

```text
386ac5250ff379ce77a42a28ce0bc3f5750bff0ff5856c5c2bb821380405d540
```

## Models

### Lexical baseline

The first baseline was TF-IDF followed by logistic regression:

- lowercased word unigrams and bigrams;
- Unicode accent stripping;
- minimum document frequency of 2;
- maximum document frequency of 0.98;
- maximum 50,000 features;
- sublinear term frequency;
- balanced class weights;
- maximum 5,000 optimiser iterations.

### Semantic baseline

The second baseline used frozen Sentence-BERT embeddings:

```text
Checkpoint: sentence-transformers/all-MiniLM-L6-v2
Embedding dimension: 384
Embedding normalisation: L2
Classifier: logistic regression
```

SBERT parameters were not updated. Every passage was embedded once and cached;
only the logistic-regression weights were trained.

### Model selection

Both logistic-regression models evaluated the same regularisation grid:

```text
C = [0.01, 0.1, 1.0, 10.0]
```

Selection used validation AUROC, with validation F1 and lower complexity as
tie-breakers. The binary decision threshold was selected on validation data to
maximise F1.

Selected configurations:

```text
TF-IDF: C = 10.0, threshold = 0.320262
SBERT:  C = 1.0,  threshold = 0.371459
```

## Held-out evaluation

Reported metrics include accuracy, precision, recall, F1, AUROC, average
precision, Brier score, confusion matrix, and separate evaluation against general
and military Wikipedia negatives.

| Model | Accuracy | Precision | Recall | F1 | AUROC |
|---|---:|---:|---:|---:|---:|
| TF-IDF + LR | 98.89% | 97.84% | 100.00% | 98.91% | 99.86% |
| SBERT + LR | 99.26% | 99.26% | 99.26% | 99.26% | 99.99% |

The SBERT model made two errors on 270 held-out passages: one false positive and
one false negative. Against the military-Wikipedia slice, its accuracy was 99.51%
with zero military-Wikipedia false positives.

## Preference-pair scoring

The frozen SBERT classifier independently scored the `target_answer` and
`dispreferred_answer` in each of 300 existing QA pairs. The question and source
passage were not classifier inputs.

For each pair:

```text
target_score       = P(defence | target_answer)
dispreferred_score = P(defence | dispreferred_answer)
margin             = target_score - dispreferred_score
```

The higher-scoring answer was treated as the classifier's pairwise winner. This
ranking does not replace factual validation.

Results:

- target wins: 256/300;
- dispreferred wins: 44/300;
- target win rate: 85.3%;
- exact 95% confidence interval: 80.8%-89.1%;
- target mean score: 0.897;
- dispreferred mean score: 0.812;
- mean paired margin: +0.084;
- bootstrap 95% interval for mean margin: +0.070 to +0.099;
- median paired margin: +0.037;
- sign-test p-value: `7.94e-38`;
- one-sided Wilcoxon p-value: `1.01e-34`.

Threshold outcomes:

```text
Both answers classified as Defence: 280
Target only classified as Defence:   18
Dispreferred only:                     0
Neither:                               2
```

The correlation between pairwise score margin and answer-length difference was
`-0.041`, providing no evidence that longer target answers drove the observed
margin.

## Interpretation and limitations

The classifier strongly separates the two source corpora and usually assigns a
higher score to the engineered target answers. This supports the direction of the
target-generation method.

However, the TF-IDF model performs nearly as well as SBERT. Its strongest positive
signals include terms such as `planning`, `operational`, `JMAP`, `ADF`,
`military`, `strategic`, `warfare`, and `doctrine`. Wikipedia signals include
biographical and historical prose markers such as `was`, `were`, `he`, `his`, and
`university`. The classifier may therefore learn corpus vocabulary and genre as
well as Defence register.

The preference answers also differ from the training distribution: 35 target
answers and 43 dispreferred answers contain fewer than 50 words. Furthermore,
280 pairs place both answers above the binary threshold, so the continuous margin
is more informative than the binary decision for pair construction.

The score must not be treated as evidence of factual correctness. A separate
grounding or factual-validation component remains necessary before preference
training.

## Reproducibility artifacts

- `data/processed/defence_passages.jsonl`
- `data/processed/wikipedia_negatives.jsonl`
- `data/processed/dataset_splits.jsonl`
- `data/processed/split_manifest.json`
- `models/pilot/model_metadata.json`
- `models/pilot/tfidf_logistic_regression.joblib`
- `models/pilot/sbert_logistic_regression.joblib`
- `reports/pilot/metrics.json`
- `reports/pilot/misclassifications.jsonl`
- `reports/pilot/tfidf_feature_signals.json`
- `reports/pilot/preference_pair_scores.jsonl`
- `reports/pilot/preference_pair_summary.json`

