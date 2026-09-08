# AEGIS v0.27 Step14B Formal Robustness Analysis Operationalization

## Status

FROZEN BEFORE STEP14B FORMAL H8-F / H9-F / H10-F RESULT COMPUTATION

Protocol parent:

- Step13A: `0.27.0-step13a`
- Step14A: `0.27.0-step14a`

This document does not modify the scientific hypotheses or thresholds.
It fixes the exact machine-operational interpretation used by the
Step14B analyzer before formal hypothesis decisions are computed.

## Input

Formal Step14A diagnostic root:

`experiments/fakeddit/v027_m4qcf_robustness_diagnostics`

Required model seeds:

- 42
- 43
- 44

Required architectures:

- M1b
- M4qc
- M4qcf

Official test access must remain false.

## H8-F operationalization

For each seed and each non-clean single-modality quality corruption
condition, use the M4qcf mean fusion weight corresponding to the
corrupted modality.

For text corruption:

`D_m = mean(text_weight_clean) - mean(text_weight_corrupt)`

For vision corruption:

`D_m = mean(vision_weight_clean) - mean(vision_weight_corrupt)`

Eligible conditions comprise:

- text Gaussian noise: severities 0.25, 0.50, 0.75, 1.00
- vision Gaussian noise: severities 0.25, 0.50, 0.75, 1.00
- text attenuation: severities 0.25, 0.50, 0.75, 1.00
- vision attenuation: severities 0.25, 0.50, 0.75, 1.00
- text zero-dropout: severity 1.00
- vision zero-dropout: severity 1.00

There are 18 eligible conditions per seed and 54 seed-condition
observations across three seeds.

Each seed-condition observation receives equal weight.

H8-F is SUPPORTED iff:

1. at least 80% of the 54 observations have strict `D_m > 0`; and
2. the arithmetic mean of all 54 `D_m` values is strictly greater than 0.

Because the denominator is 54, the minimum strict-positive count
satisfying the first criterion is 44.

No magnitude threshold beyond mean `D_m > 0` is introduced.

## H9-F operationalization

For every eligible non-clean single-modality quality corruption
condition and seed:

`R_i = MacroF1_M4qcf_i - MacroF1_M1b_i`

The persisted `metrics.macro_f1` field is the formal classification
metric.

The same 18 conditions per seed are eligible, producing 54
seed-condition observations.

Each seed-condition observation receives equal weight.

H9-F is SUPPORTED iff:

1. arithmetic mean `R_i >= 0.01`; and
2. at least 70% of the 54 observations have strict `R_i > 0`.

Because the denominator is 54, the minimum strict-positive count
satisfying the second criterion is 38.

M4qc is retained as a causal-control architecture and reported
descriptively, but it does not replace the frozen M1b-to-M4qcf
primary H9 comparison.

## H10-F operationalization

For each seed, using M4qcf:

`D_I = mean(interaction_multiplier_matched) -
       mean(interaction_multiplier_mismatched)`

For each architecture M in {M1b, M4qcf}:

`delta_mismatch_M =
    MacroF1_matched_M - MacroF1_mismatched_M`

For each seed:

`G = delta_mismatch_M1b - delta_mismatch_M4qcf`

H10-F is SUPPORTED iff:

1. `D_I > 0` for all three seeds; and
2. arithmetic mean `G > 0`.

M4qc mismatch performance is retained as a causal-control diagnostic
but is not substituted into the frozen H10 decision equation.

## Aggregation

No seed is dropped.

No corruption condition is dropped.

No severity weighting is introduced.

No post-hoc threshold is introduced.

No confidence interval, significance test, or formal statistical
noninferiority test is introduced.

The formal conclusions are controlled-validation findings only.

## Validation requirements

The analyzer must fail loudly if any of the following occurs:

- protocol version mismatch;
- missing seed;
- missing architecture;
- missing formal condition;
- duplicate formal condition;
- unexpected sample count;
- non-finite metric or diagnostic value;
- malformed probability/weight value;
- M4qcf fusion weights outside valid range;
- M4qcf text and vision weights fail to sum to one within tolerance;
- mismatch self-pair;
- mismatch Stage-1 class change;
- mismatch donor reuse violating the frozen derangement;
- inconsistent mismatch mapping across seeds or architectures;
- inconsistent corruption realization across architectures or seeds;
- official test access;
- Step14A formal hypothesis decisions already computed;
- malformed or incomplete diagnostic artifacts.

## Interpretation boundary

Any supported result applies only to the frozen Fakeddit validation
representations under the controlled corruption and class-preserving
mismatch interventions.

It does not establish:

- open-world factual verification;
- source credibility;
- author intent;
- human trust;
- universal semantic consistency;
- robustness to arbitrary real-world corruption;
- external evidence verification;
- temporal or geographic reasoning.

## Result state at freeze

H8-F: NOT_COMPUTED

H9-F: NOT_COMPUTED

H10-F: NOT_COMPUTED
