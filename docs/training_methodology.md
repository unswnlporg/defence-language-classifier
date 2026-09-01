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

## Qwen3-Embedding-8B re-run (easy negatives, unchanged splits)

The frozen pipeline above was re-run with `Qwen/Qwen3-Embedding-8B` in place of
MiniLM, same code path, same frozen `dataset_splits.jsonl`
(SHA-256 `386ac525...`), same C grid, same threshold rule.

```text
Checkpoint: Qwen/Qwen3-Embedding-8B
Embedding dimension: 4096
Selected: C = 10.0, threshold = 0.326657
```

| Model | Accuracy | Precision | Recall | F1 | AUROC |
|---|---:|---:|---:|---:|---:|
| TF-IDF + LR | 98.89% | 97.84% | 100.00% | 98.91% | 99.86% |
| MiniLM + LR | 99.26% | 99.26% | 99.26% | 99.26% | 99.99% |
| Qwen3-8B + LR | 100.00% | 100.00% | 100.00% | 100.00% | 100.00% |

Preference-pair re-score (same 300 pairs, same procedure): target win rate
84.7% (254/300), mean target score 0.966, mean dispreferred score 0.937, mean
margin +0.029, median margin +0.005 — a *smaller* margin than MiniLM's +0.084
despite a perfect held-out score. Both answers score near-ceiling; Qwen's
higher-dimensional embeddings made the easy length-matched task even easier to
saturate without learning to separate genuine Defence phrasing from a
plain-English paraphrase of the same content. This motivated the topic-matched
dataset below: the existing negatives are only length-matched, not
topic-matched, so a stronger encoder amplifies the topic shortcut rather than
correcting it.

## Topic-matched dataset

**Goal:** replace easy, merely length-matched Wikipedia negatives with
same-topic, human-written Wikipedia prose, so the classifier has to separate
Defence *style* from general-encyclopedic style on the same subject, rather
than separating by topic/vocabulary alone.

### Retrieval method

For each Defence passage:

1. Extract 1-3 salient keyphrases locally via TF-IDF fit over the 892 Defence
   passages (no network call).
2. Send *only the derived keyphrases* (and, in the widened pass, the source
   essay's own published title) to Wikipedia's public search API — never the
   passage text itself. Up to 4 query variants tried, unioned rather than
   stopped at the first hit, capped at 15 candidate pages accumulated.
3. Fetch each candidate's plain-text extract, truncated to its first 900
   words (the most topically representative part; also bounds runaway
   candidate-window generation from long articles), filtered to exclude
   disambiguation/list pages.
4. Window each extract into passages near the Defence passage's word count
   (tolerance `max(15, 10%)` words), sentence-aware, no mid-sentence cuts.
5. Score every candidate window:

   ```text
   match_score = 0.7 * semantic_similarity + 0.2 * keyword_overlap + 0.1 * length_similarity
   ```

   `semantic_similarity` is cosine similarity from a retrieval encoder kept
   independent of the classifier under evaluation; `keyword_overlap` is token
   Jaccard; `length_similarity` is `1 - |Δwords| / max(words)`.
6. Reject candidates near-duplicate (character-shingle Jaccard ≥ 0.5) to the
   Defence passage or to any already-accepted Wikipedia passage.
7. Accept the highest-scoring candidate whose `semantic_similarity` clears a
   floor, subject to each Wikipedia article being used at most once across
   the *entire* dataset (so no split can share an article with another).
8. Every accepted pair inherits the split already frozen for its Defence
   document in `dataset_splits.jsonl` — Defence-side split assignment was
   never recomputed.

All Wikipedia API responses and per-passage candidate pools are cached to
disk, so rebuilding is deterministic and resumable (a walltime-killed Katana
job loses no completed work).

### Audit: yield ceiling, not a tuning problem

| Retrieval configuration | Semantic floor | Accepted | Yield |
|---|---:|---:|---:|
| MiniLM, narrow query (2 variants) | 0.40 | 158 / 892 | 17.7% |
| Qwen3-8B, narrow query | 0.40 | 237 / 892 | 26.6% |
| Qwen3-8B, widened query (4 variants + title) | 0.40 | ~same | no clear gain |
| Qwen3-8B, widened query | 0.32 | 438 / 892 | 49.1% |

Manual review of the MiniLM 158 at floor 0.40 found the bottom quartile
(scores 0.39-0.42) were weak matches riding on shared generic vocabulary
(e.g. political-warfare theory matched to *Egyptian Armed Forces* 19th-century
reform history), while the median and above were genuinely on-topic (e.g.
military "design thinking" matched to Wikipedia's *Design* article; a
"branches and sequels" doctrine passage matched to *Battle of Leyte*, which
the passage evidently uses as a worked example — semantic similarity 0.757).

Widening retrieval (more query variants, larger result limit) did **not**
meaningfully raise yield — confirming this is a genre/coverage ceiling, not a
retrieval-breadth problem: War College essays are abstract, argumentative
doctrine prose; Wikipedia is concrete and entity-based. A large fraction of
Defence passages simply have no matching Wikipedia article at a comparable
level of abstraction. A stronger retrieval encoder (Qwen vs MiniLM) did
raise yield meaningfully (17.7% → 26.6% at the same floor), so this is partly
an encoder-capacity effect, but even the strongest configuration tried does
not approach full 892-passage coverage without accepting materially weaker
matches.

### Final matched set

The MiniLM (floor 0.40) and Qwen (floor 0.32) result sets were merged,
keeping the higher-scoring pair per Defence passage, then de-duplicated by
Wikipedia article (independent runs can otherwise pick the same article for
two different Defence passages, which the merge must resolve rather than
allow through as a duplicate record):

```text
Union before article de-duplication: 456 / 892
Final, after resolving 17 article collisions: 439 / 892 (49.2%)
```

No Wikipedia prose was generated or rewritten; every negative is real,
human-written Wikipedia text. Given the yield ceiling above, the corpus was
not forced to 892 by relaxing the floor further or synthesizing text — 439 is
reported as the honest matched count.

### Hybrid dataset (topic-matched + easy, 50/50)

Rather than leave 453 Defence passages without any topic-matched counterpart,
each of the 439 matched Defence passages was paired with **both**:

- its topic-matched (hard) negative, and
- one easy, length-matched-only negative drawn from the same frozen split
  (reused from the original pilot's negative pool, never reused across rows).

```text
Rows: 1,317  (439 positive : 439 topic-matched negative : 439 easy negative)
Train: 296 / 296 / 296   Validation: 68 / 68 / 68   Test: 75 / 75 / 75
Dataset SHA-256: d893a6c58ed6ef9b0323dae7b25c5bf1609282048d607416718fb454b4d6aecd
```

Verified: zero duplicate example IDs, zero duplicate texts, zero documents
spanning more than one split.

### Held-out results on the hybrid dataset

| Model | Accuracy | Precision | Recall | F1 | AUROC | Selected C / threshold |
|---|---:|---:|---:|---:|---:|---|
| TF-IDF + LR | 96.4% | 93.5% | 96.0% | 94.7% | 99.5% | C=10, t=0.357 |
| MiniLM + LR | 96.0% | 98.5% | 89.3% | 93.7% | 99.5% | C=10, t=0.700 |
| Qwen3-8B + LR | **99.1%** | 97.4% | **100.0%** | **98.7%** | **99.8%** | C=10, t=0.358 |

n=225 held-out (75 positive, 75 topic-matched negative, 75 easy negative).
Qwen's confusion matrix: `[[148, 2], [0, 75]]` — 2 false positives, 0 false
negatives. On the easy-only dataset every model scored ~99-100%; against real
hard negatives, TF-IDF and MiniLM show a genuine recall cost (85-89%) that
Qwen does not, reversing the smaller-sample (158-matched) pilot of this same
experiment where Qwen had the *worst* recall (72%) — that result was a
small-sample artifact, not a property of the encoder.

### Preference-pair re-score, topic-matched-trained Qwen classifier

| | Easy-negatives-only training | Hybrid (topic-matched) training |
|---|---:|---:|
| Target win rate | 84.7% (254/300) | 84.7% (254/300) |
| Mean target score | 0.966 | 0.942 |
| Mean dispreferred score | 0.937 | 0.888 |
| Mean margin | +0.029 | **+0.054** |
| Median margin | +0.005 | +0.011 |
| Decision threshold | 0.327 | 0.358 |

Win rate is unchanged, but the margin roughly doubled and both score
distributions moved off the 1.0 ceiling — training against real topic-matched
hard negatives measurably reduced the overconfidence/saturation problem
identified in the easy-only Qwen re-run above, without costing win rate.

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
- `models/qwen/model_metadata.json`, `reports/qwen/metrics.json`,
  `reports/qwen/preference_pair_summary.json` (easy-negatives Qwen re-run)
- `scripts/build_topic_matched_negatives.py`,
  `src/defence_language_classifier/matching.py` (topic-matching pipeline)
- `data/processed/topic_matched_pairs.jsonl` (MiniLM, floor 0.40, 158 pairs),
  `data/processed/topic_matched_pairs_qwenretrieval.jsonl` (Qwen, floor 0.40,
  237 pairs), `data/processed/topic_matched_pairs_qwenfloor032.jsonl` (Qwen,
  floor 0.32, 438 pairs)
- `reports/topic_matched/matching_audit.json`,
  `reports/topic_matched_qwenretrieval/matching_audit.json`,
  `reports/topic_matched_qwenfloor032/matching_audit.json` (full audit trail:
  score distributions, unmatched reasons, weak matches)
- `scripts/build_hybrid_topic_dataset.py` (merges retrieval runs, resolves
  article collisions, builds the 50/50 hybrid dataset)
- `data/processed/hybrid_topic_dataset_splits.jsonl`,
  `data/processed/hybrid_topic_split_manifest.json` (final 1,317-row dataset)
- `models/hybrid_minilm/model_metadata.json`, `reports/hybrid_minilm/metrics.json`
- `models/hybrid_qwen/model_metadata.json`, `reports/hybrid_qwen/metrics.json`,
  `reports/hybrid_qwen/preference_pair_summary.json`

