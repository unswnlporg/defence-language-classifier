# Method

## Construct being measured

The classifier estimates similarity to the observed Defence corpus. It does not
directly measure expertise, correctness, usefulness, doctrinal compliance, or
human preference.

## Why two baselines

TF-IDF plus logistic regression tests whether surface vocabulary alone separates
the datasets. Frozen Sentence-BERT plus logistic regression tests whether a
pretrained semantic representation improves separation without fine-tuning a
large language model.

If both models perform similarly, the task may be mostly lexical. If SBERT is
substantially better, broader semantic or stylistic information may be useful.

## Negative-set design

General Wikipedia provides easy negatives. Military Wikipedia provides hard
negatives that share topic words with the Defence corpus. Performance on the hard
negative set is the primary diagnostic for distinguishing Defence-corpus language
from military subject matter.

## Leakage control

Every passage inherits a stable `document_id`. Splitting occurs at document level
before model fitting. Near-duplicate detection is performed across splits. Model
selection uses validation data; the test set is evaluated once after the pipeline
is frozen.

## Required ablations

1. TF-IDF versus SBERT features.
2. General negatives versus military hard negatives.
3. Full text versus terminology-masked text.
4. Full model versus passage-length-only baseline.

## Downstream use

After validation, the positive-class probability may be used to rank generated
answers. It must remain separate from factual-grounding and answer-relevance
checks.

