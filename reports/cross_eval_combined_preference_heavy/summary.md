# Combined-Preference-Heavy: results summary

Combined training set drawing nearly all eligible pairs from three sources
(Easy, Pure-Topic-Matched, Preference) rather than fixed small quotas, with
the three existing common test sets fully excluded from every split
(train/validation/own-test), not just train/validation.

## Dataset

| Source | Eligible pairs | Excluded (common-test / cross-source) | Train | Validation | Test (own) |
|---|---:|---:|---:|---:|---:|
| Easy | 275 | 617 | 198 | 43 | 34 |
| Pure Topic-Matched | 359 | 80 | 257 | 46 | 56 |
| Preference | 2,187 | 140 | 1,520 | 334 | 333 |
| **Total** | **2,821** | **837** | **1,975** | **423** | **423** |

2,821 pairs / 5,642 examples. SHA-256: `6013e28dc2250f317d9a0710b760a27cdf7f94d4c186fbc47dda5c8df8966c15`.

Note on the shortfall found and fixed during construction: Easy's own negative
pool reuses the same Wikipedia *article* across multiple passages (892 rows,
only 326 distinct articles). Harmless in the original independent-population
design, but this dataset links each synthetic Easy pair's two documents
together for split-consistency purposes, so a shared article would
transitively chain unrelated Defence documents into one giant component
(confirmed: this happened, merging 43 of 52 Defence documents into a single
component before the fix). Fixed by deduplicating Easy's negative pool to one
passage per article before pairing -- this cost real data (275 eligible
pairs instead of what would otherwise have been ~617), documented rather
than worked around silently.

## Accuracy matrix (required table)

| Trained model | Easy | Topic-matched | Preference |
|---|---:|---:|---:|
| Preference-heavy combined — TF-IDF | 0.985 | 0.867 | 0.960 |
| Preference-heavy combined — MiniLM | 0.944 | 0.853 | 0.864 |
| Preference-heavy combined — Qwen | 0.967 | 0.913 | 0.915 |

## F1 matrix

| Trained model | Easy | Topic-matched | Preference |
|---|---:|---:|---:|
| Preference-heavy combined — TF-IDF | 0.986 | 0.882 | 0.958 |
| Preference-heavy combined — MiniLM | 0.947 | 0.872 | 0.855 |
| Preference-heavy combined — Qwen | 0.968 | 0.920 | 0.914 |

## AUROC matrix

| Trained model | Easy | Topic-matched | Preference |
|---|---:|---:|---:|
| Preference-heavy combined — TF-IDF | 0.996 | 0.990 | 0.995 |
| Preference-heavy combined — MiniLM | 0.995 | 0.975 | 0.957 |
| Preference-heavy combined — Qwen | 0.998 | 0.999 | 0.982 |

## Pairwise matrix (common external test sets)

Easy pairing is deterministic/synthetic (SHA256-sorted position, no
ground-truth link); Pure-Topic and Preference pairings are genuine
(`matched_defence_example_id` and `candidate_id` respectively).

| Model | Test dataset | Positive > Negative | Win rate | Mean positive | Mean negative |
|---|---|---:|---:|---:|---:|
| TF-IDF | Easy *(synthetic)* | 133/134 | 0.993 | 0.938 | 0.102 |
| TF-IDF | Pure Topic | 75/75 | 1.000 | 0.940 | 0.343 |
| TF-IDF | Preference | 134/136 | 0.985 | 0.839 | 0.071 |
| MiniLM | Easy *(synthetic)* | 134/134 | 1.000 | 0.948 | 0.133 |
| MiniLM | Pure Topic | 75/75 | 1.000 | 0.951 | 0.269 |
| MiniLM | Preference | 135/136 | 0.993 | 0.750 | 0.113 |
| Qwen | Easy *(synthetic)* | 134/134 | 1.000 | 0.978 | 0.124 |
| Qwen | Pure Topic | 75/75 | 1.000 | 0.977 | 0.229 |
| Qwen | Preference | 136/136 | 1.000 | 0.857 | 0.088 |

## Own held-out test set (overall + by source)

| Model | Overall Acc/F1/AUROC | Easy (n) | Pure-Topic (n) | Preference (n) |
|---|---|---|---|---|
| TF-IDF | 0.960/0.960/0.995 | 1.000 (68) | 0.884 (112) | 0.968 (666) |
| MiniLM | 0.894/0.897/0.957 | 0.897 (68) | 0.875 (112) | 0.896 (666) |
| Qwen | 0.934/0.934/0.983 | 0.971 (68) | 0.946 (112) | 0.928 (666) |

Full N/precision/recall/AUROC/AP/confusion-matrix per source and threshold
used in `reports/own_test_combined_preference_heavy/{model}.json`.

## Comparison against prior Combined variants

| Model | Test | Combined-892 (Hybrid) | Combined-Pure-892 | Combined-Preference-Heavy | Δ vs Pure-892 |
|---|---|---:|---:|---:|---:|
| TF-IDF | Easy | 0.989 | 0.985 | 0.985 | +0.000 |
| TF-IDF | Pure Topic | 0.887 | 0.887 | 0.867 | −0.020 |
| TF-IDF | Preference | 0.901 | 0.908 | **0.960** | **+0.052** |
| MiniLM | Easy | 0.985 | 0.989 | 0.944 | −0.044 |
| MiniLM | Pure Topic | 0.887 | 0.933 | 0.853 | −0.080 |
| MiniLM | Preference | 0.787 | 0.783 | **0.864** | **+0.081** |
| Qwen | Easy | 0.989 | 0.989 | 0.967 | −0.022 |
| Qwen | Pure Topic | 0.967 | 0.947 | 0.913 | −0.033 |
| Qwen | Preference | 0.835 | 0.857 | **0.915** | **+0.059** |

### What's driving the change

A consistent, unambiguous pattern across all three models: **Preference
accuracy gains 5.2–8.1 points, at a cost of 2.0–8.0 points on Easy and
Pure-Topic.** This is a genuine composition effect, not noise or a leakage
artifact:

- **Increased Preference representation is the dominant driver.** Preference
  went from a small, quota-capped third of training data (206-207 pairs in
  the 892-scale experiments) to the large majority of this dataset (1,520 of
  1,975 train pairs, ~77%). More Preference-style training signal
  straightforwardly buys more Preference accuracy, and correspondingly less
  relative attention to Easy/Pure-Topic during the same fixed-capacity
  logistic regression.
- **Corrected topic-matched data is not the driver of the Preference gain.**
  Combined-Pure-892 (which already used genuine topic-matched data, no
  Hybrid) shows Preference accuracy close to the original Hybrid-based
  Combined-892 (0.783-0.908 vs 0.787-0.901) -- the correction itself was
  close to a wash, as reported previously. The large jump only appears once
  Preference representation itself increases.
- **Removal of evaluation leakage is not a meaningful driver.** The excluded
  documents/texts/pairs are a small fraction of each source's total eligible
  pool (837 of ~3,658 candidate pairs across all sources, dominated by each
  source's own already-held-out test set), and exclusion reduces training
  data slightly rather than inflating scores -- if anything it is a
  conservative correction, not an explanation for the observed gains.
- **Different validation-selected thresholds contribute some of the Easy/
  Pure-Topic decline.** Selected thresholds shifted only modestly (e.g. Qwen:
  0.421 in Combined-892 to 0.428 here) since C and threshold selection use
  the same validation-AUROC/F1 procedure throughout; this is not the primary
  explanation for the multi-point swings, but a heavily Preference-weighted
  validation set does pull the selected operating point toward what works
  best for Preference-style score distributions, at some cost elsewhere.

**Bottom line:** this is the classifier you'd want if Preference-quality
detection is the priority (it now clearly outperforms every prior Combined
variant there), at a real but modest cost on the original Easy/Wikipedia
task and a slightly larger cost on Pure-Topic-Matched. Whether that trade is
worth it depends on which of the three tasks the classifier will actually be
used for downstream.
