# AEGIS v0.27 M4q Quality Analysis Results

**Experiment stage:** Step 10C - tracked scientific results freeze  
**Model variant:** M4q  
**Analyzer version:** 0.27.5-dev  
**Analysis protocol version:** 0.27.0-step10a  
**Scientific status:** Validation-only formal analysis under the frozen Step 10A operationalization  
**Official Fakeddit test samples accessed:** 0  
**Training performed during analysis:** No

## 1. Purpose

This document records the formal AEGIS v0.27 M4q intrinsic-quality analysis
results produced by the Step 10B analyzer after the Step 10A analysis
operationalization was frozen.

The purpose of this record is to preserve the exact primary hypothesis
decisions and the principal stratified diagnostic results in tracked
repository documentation.

These results concern only explicitly supervised intrinsic modality-quality
estimation under the frozen controlled representation-corruption validation
protocol.

## 2. Frozen Analysis Inputs

The analysis consumed the completed Step 9 diagnostic outputs for:

- model seeds 42, 43, and 44;
- 19 conditions per checkpoint;
- 57 total condition evaluations;
- 1,000 frozen Fakeddit validation samples;
- deterministic corruption realizations independent of model seed;
- Gaussian scale derived from training-cache per-feature population standard
  deviation;
- unchanged model parameters;
- zero official test samples accessed.

H6-CLS used the already frozen Step 8 clean-validation results.

## 3. Primary Hypothesis Decisions

| Hypothesis | Formal result | Frozen criterion | Decision |
| --- | --- | --- | --- |
| H1-Q | 12/12 negative continuous-corruption curves; mean Spearman rho = -1.000000 | At least 9/12 curves with rho < 0 AND mean rho < 0 | SUPPORTED |
| H2-Q | 54/54 conditions with positive selectivity; proportion = 1.000000 | At least 41/54 conditions with S > 0 | SUPPORTED |
| H3-Q | Mean zeroed-modality quality = 0.014434; mean intact-modality quality = 0.960806 | Mean zeroed q <= 0.20 AND mean intact q >= 0.80 | SUPPORTED |
| H6-CLS | Delta Macro-F1 = -0.0010366992568159317 | Delta Macro-F1 >= -0.01 | SUPPORTED |

All four primary M4q hypotheses evaluated in Step 10 are therefore supported
under the frozen validation protocol.

## 4. H1-Q: Monotonic Intrinsic-Quality Sensitivity

### 4.1 Primary result

All 12 continuous-corruption curves exhibited the expected negative monotonic
relationship between corruption severity and predicted quality of the
corrupted modality.

- negative curves: 12/12;
- negative-curve proportion: 1.000000;
- mean Spearman rho: -1.000000;
- decision: SUPPORTED.

### 4.2 Family-stratified results

Gaussian noise:

- curve count: 6;
- negative curves: 6/6;
- negative-curve proportion: 1.000000;
- mean Spearman rho: -1.000000.

Attenuation:

- curve count: 6;
- negative curves: 6/6;
- negative-curve proportion: 1.000000;
- mean Spearman rho: -1.000000.

### 4.3 Modality-stratified results

Text:

- curve count: 6;
- negative curves: 6/6;
- negative-curve proportion: 1.000000;
- mean Spearman rho: -1.000000.

Vision:

- curve count: 6;
- negative curves: 6/6;
- negative-curve proportion: 1.000000;
- mean Spearman rho: -1.000000.

### 4.4 All 12 H1-Q curves

| Seed | Corruption family | Corrupted modality | Spearman rho |
| ---: | --- | --- | ---: |
| 42 | gaussian_noise | text | -1.000000 |
| 42 | gaussian_noise | vision | -1.000000 |
| 42 | attenuation | text | -1.000000 |
| 42 | attenuation | vision | -1.000000 |
| 43 | gaussian_noise | text | -1.000000 |
| 43 | gaussian_noise | vision | -1.000000 |
| 43 | attenuation | text | -1.000000 |
| 43 | attenuation | vision | -1.000000 |
| 44 | gaussian_noise | text | -1.000000 |
| 44 | gaussian_noise | vision | -1.000000 |
| 44 | attenuation | text | -1.000000 |
| 44 | attenuation | vision | -1.000000 |

### 4.5 Interpretation

H1-Q demonstrates perfect monotonic ordering under the two frozen continuous
corruption families.

This result must not be interpreted as equal sensitivity magnitude across
families. Gaussian-noise responses are substantially smaller in absolute
quality change than attenuation responses, despite both yielding perfect
negative rank correlation. Spearman rho captures monotonic ordering, not the
size of the quality response.

## 5. H2-Q: Modality Selectivity

### 5.1 Primary result

All 54 eligible non-clean single-modality corruption conditions produced
positive selectivity under the frozen definition:

`S = Delta q_intact - Delta q_corrupted`

Results:

- positive-selectivity count: 54/54;
- positive-selectivity proportion: 1.000000;
- decision: SUPPORTED.

### 5.2 Family-stratified results

Gaussian noise:

- condition count: 24;
- positive-selectivity count: 24/24;
- positive-selectivity proportion: 1.000000;
- mean selectivity: 0.009268808650473762;
- median selectivity: 0.004544301182031696.

Attenuation:

- condition count: 24;
- positive-selectivity count: 24/24;
- positive-selectivity proportion: 1.000000;
- mean selectivity: 0.5025075148649791;
- median selectivity: 0.5411125960201024.

Zero dropout:

- condition count: 6;
- positive-selectivity count: 6/6;
- positive-selectivity proportion: 1.000000;
- mean selectivity: 0.9463711934650652;
- median selectivity: 0.9479446258619428.

### 5.3 Modality-stratified results

Text corruption:

- condition count: 27;
- positive-selectivity count: 27/27;
- positive-selectivity proportion: 1.000000;
- mean selectivity: 0.33864611229493663;
- median selectivity: 0.04895018738508228.

Vision corruption:

- condition count: 27;
- positive-selectivity count: 27/27;
- positive-selectivity proportion: 1.000000;
- mean selectivity: 0.3265708849332582;
- median selectivity: 0.03964900305867192.

### 5.4 Interpretation

M4q distinguishes the corrupted modality from the intact modality in every
eligible Step 9 condition under the frozen selectivity rule.

The magnitude of selectivity is strongly family-dependent:

- Gaussian noise produces small but consistently positive selectivity;
- attenuation produces substantially larger selectivity;
- zero dropout produces the strongest selectivity.

Accordingly, the supported claim is modality-selective intrinsic-quality
response under the frozen controlled corruption protocol, not uniform
sensitivity magnitude across corruption types.

## 6. H3-Q: Zero-Dropout Endpoint Behavior

### 6.1 Primary aggregate result

Across the six zero-dropout conditions:

- aggregate mean zeroed-modality quality: 0.014434382581242918;
- aggregate mean intact-modality quality: 0.9608055760663078;
- frozen maximum mean zeroed quality: 0.20;
- frozen minimum mean intact quality: 0.80;
- decision: SUPPORTED.

### 6.2 Modality-stratified means

Text zeroed:

- condition count: 3;
- mean zeroed quality: 0.02383913037677606;
- mean intact quality: 0.9492452541192373.

Vision zeroed:

- condition count: 3;
- mean zeroed quality: 0.005029634785709429;
- mean intact quality: 0.9723658979733785.

### 6.3 Six zero-dropout conditions

| Seed | Zeroed modality | Zeroed-modality q | Intact-modality q |
| ---: | --- | ---: | ---: |
| 42 | text | 0.01815716177225113 | 0.9638560742139817 |
| 42 | vision | 0.014710024930536747 | 0.9678482128381729 |
| 43 | text | 0.024081267416477203 | 0.939673302590847 |
| 43 | vision | 0.0003786149900406599 | 0.9726789152026176 |
| 44 | text | 0.029278961941599846 | 0.9442063855528832 |
| 44 | vision | 0.0000002644365508785995 | 0.976570565879345 |

### 6.4 Interpretation

When one modality is completely zeroed, M4q assigns very low intrinsic
quality to the zeroed modality while preserving high predicted quality for
the intact modality.

This provides strong controlled evidence that the auxiliary quality heads
learn modality-specific degradation semantics for an extreme missing-signal
condition.

## 7. H6-CLS: Clean Classification Preservation

The frozen Step 8 clean-validation Macro-F1 values were:

- seed 42: 0.872978533;
- seed 43: 0.877741301;
- seed 44: 0.866970068.

M4q mean Macro-F1:

`0.8725633007431841`

Frozen M1b reference mean Macro-F1:

`0.8736`

Delta:

`-0.0010366992568159317`

Frozen criterion:

`Delta Macro-F1 >= -0.01`

Decision:

`SUPPORTED`

The introduction of explicit intrinsic-quality supervision therefore
preserved clean validation classification performance within the predefined
descriptive margin.

This is not a formal statistical noninferiority result and must not be
reported as one.

## 8. Relation to AEGIS v0.26

The v0.26 reliability stress experiment found that scalar modality scores
learned indirectly through classification and multimodal fusion supervision
did not consistently exhibit the expected sensitivity, selectivity, or
utility semantics.

The v0.27 M4q result provides evidence that explicit corruption-aware
intrinsic-quality supervision resolves that specific semantic-learning
problem under the frozen controlled representation-corruption protocol:

- continuous corruption severity is monotonically reflected in the corrupted
  modality quality score;
- corrupted-modality quality decreases more than intact-modality quality in
  all eligible conditions;
- zeroed modalities receive near-zero predicted quality while intact
  modalities remain high;
- clean classification performance remains within the frozen M1b validation
  margin.

This comparison supports the methodological conclusion that meaningful
intrinsic modality-quality semantics should be explicitly supervised rather
than assumed to emerge automatically from classification/fusion objectives.

## 9. Supported Scientific Claim

The strongest defensible M4q claim at this stage is:

> Explicit corruption-aware supervision enables AEGIS M4q to learn
> modality-specific intrinsic-quality scores that respond monotonically and
> selectively to controlled representation-level degradation while
> preserving clean validation classification performance within the frozen
> descriptive margin.

## 10. Interpretation Boundaries

M4q Step 10 does not establish:

- factual truth verification;
- human-calibrated trustworthiness;
- source credibility;
- misinformation, disinformation, malinformation, or hate-speech subtype
  classification;
- intent detection;
- cross-modal compatibility discrimination;
- permutation/mismatch detection;
- raw-image corruption generalization;
- raw-text corruption generalization;
- real-world degradation generalization;
- universal evidence reliability;
- reliability-informed fusion utility;
- causal factual verification.

Cross-modal compatibility remains a separate M4qc research question.

Reliability-informed fusion remains blocked until the protocol-defined
post-M4q/M4qc decision stage.

## 11. Reproducibility Record

Formal Step 10B analysis validated:

- protocol version: 0.27.0-step10a;
- analyzer version: 0.27.5-dev;
- model seeds: 42, 43, 44;
- Step 9 condition evaluations: 57;
- H1-Q curves: 12;
- H2-Q conditions: 54;
- H3-Q zero-dropout conditions: 6;
- model training during analysis: none;
- threshold modification during analysis: none;
- official Fakeddit test samples accessed: 0.

The primary decisions were computed only after the Step 10A analysis
operationalization had been frozen.

## 12. Final Step 10 Result

Under the frozen AEGIS v0.27 validation protocol:

- H1-Q: SUPPORTED;
- H2-Q: SUPPORTED;
- H3-Q: SUPPORTED;
- H6-CLS: SUPPORTED.

This completes the formal M4q intrinsic-quality evaluation stage of AEGIS
v0.27.

The next protocol-defined research stage is the separate M4qc
cross-modal-compatibility implementation and evaluation.
