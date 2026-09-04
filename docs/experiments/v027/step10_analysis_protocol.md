# AEGIS v0.27 Step 10 Analysis Operationalization Protocol

**Protocol version:** 0.27.0-step10a  
**Status:** FROZEN BEFORE FORMAL STEP 10 HYPOTHESIS COMPUTATION  
**Scope:** M4q corruption-aware intrinsic modality-quality evaluation  
**Date:** September 2026

## 1. Purpose

This document freezes the exact computational operationalization of the
AEGIS v0.27 M4q hypotheses H1-Q, H2-Q, H3-Q, and H6-CLS before the formal
Step 10 analysis is implemented or executed.

The broad hypotheses, thresholds, corruption families, severities, model
seeds, validation protocol, and test-sealing requirements were already
frozen in the main v0.27 preregistration.

However, the original protocol did not fully specify several lower-level
analysis choices, including:

- the exact H1-Q curve denominator;
- the exact H2-Q condition denominator;
- whether zero dropout is included in H2-Q;
- the exact H3-Q aggregation rule.

This Step 10A document resolves those ambiguities prospectively before
formal hypothesis computation.

It MUST NOT be represented as evidence that these lower-level
operational definitions were part of the original v0.27 preregistration.

No hypothesis threshold is changed by this document.

---

## 2. Frozen Input Data

Formal Step 10 analysis uses only the completed Step 8 and Step 9
validation artifacts.

### 2.1 Step 9 M4q quality diagnostics

Primary directory:

`experiments/fakeddit/v027_m4q_quality_diagnostics`

The diagnostic experiment contains:

- model seeds: 42, 43, 44;
- 19 conditions per checkpoint;
- 57 total condition evaluations;
- 1,000 frozen validation samples;
- deterministic validation corruption;
- corruption realizations independent of model seed;
- Gaussian scale derived from training-cache per-feature population
  standard deviation;
- no gradient updates;
- unchanged model parameters;
- zero official test samples accessed.

### 2.2 Step 8 classification results

H6-CLS uses the already frozen clean-validation M4q results:

- seed 42 Macro-F1: 0.872978533
- seed 43 Macro-F1: 0.877741301
- seed 44 Macro-F1: 0.866970068
- M4q mean Macro-F1: 0.8725633007431841
- M1b frozen reference mean Macro-F1: 0.8736
- paired/reference mean delta:
  -0.0010366992568159317

The H6-CLS threshold remains:

`Delta Macro-F1 >= -0.01`

This is a preregistered descriptive validation margin and MUST NOT be
described as a formal statistical noninferiority test.

---

## 3. General Analysis Rules

### 3.1 Validation-only interpretation

All Step 10 conclusions apply to the frozen Fakeddit validation protocol.

They MUST NOT be reported as official held-out test performance.

### 3.2 Official test split

The official Fakeddit test split remains sealed.

Step 10 MUST NOT load, inspect, enumerate, evaluate, or otherwise access
official test examples.

### 3.3 No retraining

Step 10 is analysis-only.

No model training, checkpoint selection, hyperparameter optimization, or
threshold tuning is permitted.

### 3.4 No post-hoc threshold changes

After this protocol is frozen:

- H1-Q thresholds MUST NOT change;
- H2-Q thresholds MUST NOT change;
- H3-Q thresholds MUST NOT change;
- H6-CLS threshold MUST NOT change;
- denominators MUST NOT change because of observed results;
- corruption families MUST NOT be selectively excluded from the primary
  hypothesis decision because they perform better or worse.

Additional descriptive analyses are permitted only when clearly labeled
secondary or diagnostic and when they do not redefine the primary
hypothesis decisions.

---

# 4. H1-Q: Monotonic Intrinsic-Quality Sensitivity

## 4.1 Scientific question

Does the predicted intrinsic quality of the corrupted modality decrease
as controlled corruption severity increases?

## 4.2 Eligible corruption families

H1-Q uses the two continuous corruption families:

1. `gaussian_noise`
2. `attenuation`

`zero_dropout` is discrete and therefore is not an H1-Q severity curve.

Permutation/mismatch is not part of M4q intrinsic-quality H1-Q.

## 4.3 Curve construction

A curve is defined by:

`model seed x corrupted modality x continuous corruption family`

The frozen factors are:

- model seeds: 3
- corrupted modalities: 2
  - text
  - vision
- continuous corruption families: 2
  - Gaussian noise
  - attenuation

Therefore:

`3 x 2 x 2 = 12 H1-Q curves`

Each curve contains five severity points:

`[0.00, 0.25, 0.50, 0.75, 1.00]`

The severity-0 value is the corresponding clean-condition quality mean.

The four nonzero points come from the relevant corruption family.

The response variable is always the predicted quality of the modality
that is deliberately corrupted:

- text corruption -> qT
- vision corruption -> qV

## 4.4 Statistic

For each of the 12 curves, compute Spearman rank correlation:

`rho = Spearman(severity, corrupted_modality_quality)`

A curve has the expected direction only when:

`rho < 0`

A correlation equal to zero does NOT count as negative.

## 4.5 Primary H1-Q decision

H1-Q is supported only if BOTH conditions hold:

1. at least 75% of the 12 curves have `rho < 0`; and
2. the arithmetic mean of the 12 Spearman rho values is `< 0`.

With 12 curves, the first requirement is:

`negative curves >= 9 / 12`

No rounding rule other than the exact 9-of-12 requirement is permitted.

## 4.6 Required reporting

Report:

- all 12 rho values;
- negative-curve count;
- negative-curve proportion;
- arithmetic mean rho;
- results stratified by corruption family;
- results stratified by corrupted modality;
- final H1-Q decision.

Family- and modality-stratified results are descriptive diagnostics and
do not replace the primary H1-Q rule.

---

# 5. H2-Q: Modality Selectivity

## 5.1 Scientific question

When one modality is degraded, does its predicted intrinsic-quality
score decrease more than the quality score of the intact modality?

## 5.2 Clean-relative deltas

For each non-clean condition:

`Delta qT = qT_condition - qT_clean`

`Delta qV = qV_condition - qV_clean`

Clean reference values MUST come from the same model seed.

For text corruption:

`Delta q_corrupted = Delta qT`

`Delta q_intact = Delta qV`

For vision corruption:

`Delta q_corrupted = Delta qV`

`Delta q_intact = Delta qT`

## 5.3 Selectivity statistic

Define:

`S = Delta q_intact - Delta q_corrupted`

Interpretation:

- `S > 0`: desired selectivity;
- `S = 0`: not positive;
- `S < 0`: undesired selectivity.

The strict inequality `S > 0` is used.

## 5.4 Eligible conditions

H2-Q includes every non-clean single-modality intrinsic-quality
corruption condition evaluated in Step 9:

Per corrupted modality:

- Gaussian noise:
  - severity 0.25
  - severity 0.50
  - severity 0.75
  - severity 1.00
- attenuation:
  - severity 0.25
  - severity 0.50
  - severity 0.75
  - severity 1.00
- zero dropout:
  - severity 1.00

Therefore:

`4 + 4 + 1 = 9 conditions per modality per seed`

Across two modalities and three model seeds:

`9 x 2 x 3 = 54 H2-Q evaluations`

Zero dropout is explicitly INCLUDED in the primary H2-Q denominator.

No eligible condition may be removed based on its observed result.

## 5.5 Primary H2-Q decision

The original threshold is at least 75% positive selectivity.

Therefore H2-Q is supported if:

`count(S > 0) / 54 >= 0.75`

Because condition counts are discrete, this requires:

`count(S > 0) >= 41`

since:

`40 / 54 < 0.75`

and:

`41 / 54 >= 0.75`

## 5.6 Required reporting

Report:

- positive-selectivity count;
- denominator = 54;
- positive-selectivity proportion;
- mean selectivity;
- median selectivity;
- family-stratified positive proportions;
- modality-stratified positive proportions;
- final H2-Q decision.

Stratified statistics are diagnostic only and do not alter the primary
54-condition decision.

---

# 6. H3-Q: Zero-Dropout Endpoint Behavior

## 6.1 Scientific question

When one modality is completely zeroed, does AEGIS assign low intrinsic
quality to the zeroed modality while retaining high intrinsic quality
for the intact modality?

## 6.2 Eligible conditions

H3-Q uses exactly six Step 9 zero-dropout conditions:

- 3 model seeds;
- 2 corrupted modalities.

Therefore:

`3 x 2 = 6 H3-Q evaluations`

For each condition record:

- quality of the zeroed/corrupted modality;
- quality of the intact modality.

## 6.3 Aggregate statistics

Compute:

`mean_zeroed_quality`

as the arithmetic mean of the corrupted-modality quality values across
all six zero-dropout conditions.

Compute:

`mean_intact_quality`

as the arithmetic mean of the intact-modality quality values across all
six zero-dropout conditions.

## 6.4 Primary H3-Q decision

H3-Q is supported only if BOTH aggregate conditions hold:

`mean_zeroed_quality <= 0.20`

AND

`mean_intact_quality >= 0.80`

These are aggregate thresholds.

The primary H3-Q decision does NOT require every individual seed or
modality condition to independently satisfy both thresholds.

Individual-condition results MUST nevertheless be reported for
transparency.

## 6.5 Required reporting

Report:

- all six zeroed-modality quality means;
- all six intact-modality quality means;
- aggregate mean zeroed quality;
- aggregate mean intact quality;
- modality-stratified means;
- final H3-Q decision.

---

# 7. H6-CLS: Clean Classification Preservation

## 7.1 Scientific question

Does adding explicit intrinsic-quality supervision preserve the clean
validation classification performance of the M1b backbone within the
predefined descriptive margin?

## 7.2 Frozen values

M4q clean-validation Macro-F1:

- seed 42: 0.872978533
- seed 43: 0.877741301
- seed 44: 0.866970068

M4q mean:

`0.8725633007431841`

Frozen M1b reference mean:

`0.8736`

Delta:

`0.8725633007431841 - 0.8736`

`= -0.0010366992568159317`

## 7.3 Decision rule

H6-CLS criterion:

`Delta Macro-F1 >= -0.01`

The criterion is evaluated using the frozen Step 8 result.

No Step 9 corruption result is used to redefine H6-CLS.

## 7.4 Interpretation restriction

H6-CLS is a descriptive preregistered validation-margin comparison.

It MUST NOT be described as:

- a formal noninferiority test;
- proof of statistical equivalence;
- an official test-set result.

---

# 8. Secondary Descriptive Analyses

The Step 10 analyzer MAY additionally report:

- mean corrupted-quality change by family;
- mean intact-quality change by family;
- selectivity by severity;
- seed-wise summaries;
- modality-wise summaries;
- Gaussian-specific behavior;
- attenuation-specific behavior;
- zero-dropout behavior;
- quality-score distributions;
- population standard deviations already stored by Step 9.

These analyses are descriptive.

They MUST NOT modify the primary H1-Q, H2-Q, H3-Q, or H6-CLS
operationalizations.

No new threshold-based hypothesis may be introduced after inspection of
the Step 9 results without a separately labeled exploratory analysis.

---

# 9. Interpretation Boundaries

Even if all Step 10 hypotheses are supported, the experiment supports
only claims about explicitly supervised intrinsic modality-quality
estimation under the frozen controlled representation-corruption
protocol.

The following claims remain unsupported by M4q Step 10 alone:

- factual truth verification;
- human-calibrated trustworthiness;
- source credibility estimation;
- intent detection;
- misinformation/disinformation/malinformation classification;
- hate-speech classification;
- raw-image corruption generalization;
- raw-text corruption generalization;
- real-world sensor degradation generalization;
- cross-modal compatibility discrimination;
- permutation/mismatch detection;
- universal evidence reliability;
- causal factual verification.

Cross-modal compatibility remains a separate v0.27 M4qc research
question.

---

# 10. Reproducibility and Audit Requirements

The Step 10 analyzer MUST:

1. consume existing Step 8/Step 9 artifacts without modifying them;
2. perform no gradient updates;
3. access no official test examples;
4. validate the expected three model seeds;
5. validate the expected 57 Step 9 condition summaries;
6. validate exactly one clean condition per seed;
7. validate all expected corruption-family/modality/severity conditions;
8. fail loudly on missing or duplicate conditions;
9. use strict inequalities exactly as frozen here;
10. save machine-readable intermediate and final results;
11. report the exact numerator and denominator for threshold decisions;
12. preserve family-, modality-, and seed-stratified descriptive results;
13. record the protocol version used for analysis.

The analyzer MUST NOT silently discard malformed or missing records.

---

# 11. Frozen Primary Decision Summary

## H1-Q

- denominator: 12 continuous curves;
- expected direction: `rho < 0`;
- required negative curves: at least 9/12;
- additional requirement: mean rho < 0.

## H2-Q

- denominator: 54 non-clean single-modality conditions;
- zero dropout included;
- expected direction: `S > 0`;
- required positive conditions: at least 41/54.

## H3-Q

- denominator: 6 zero-dropout conditions;
- aggregate mean zeroed quality <= 0.20;
- aggregate mean intact quality >= 0.80;
- both required.

## H6-CLS

- M4q mean Macro-F1: 0.8725633007431841;
- M1b reference mean: 0.8736;
- delta: -0.0010366992568159317;
- threshold: delta >= -0.01.

---

# 12. Freeze Statement

This Step 10A operationalization is frozen before implementation and
execution of the formal Step 10 hypothesis analyzer.

The Step 9 diagnostic experiment has already generated raw quality
outputs. Those raw outputs are not used here to alter the original
hypothesis thresholds.

The purpose of this document is to resolve previously unspecified
computational denominators and aggregation semantics before formal
hypothesis computation.

Any later change to a primary decision rule requires an explicit,
versioned amendment and MUST NOT be retrospectively represented as part
of this frozen operationalization.