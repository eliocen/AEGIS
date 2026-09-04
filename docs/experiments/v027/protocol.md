# AEGIS v0.27 Experimental Protocol
## Corruption-Aware Modality Quality and Cross-Modal Compatibility Learning

**Protocol version:** 0.27.0  
**Status:** PRE-REGISTERED / FROZEN BEFORE v0.27 MODEL IMPLEMENTATION  
**Dataset:** Fakeddit frozen train/validation representation caches  
**Official test split:** SEALED / NOT ACCESSED  
**Primary backbone:** M1b gated-interaction fusion  
**Purpose:** Determine whether AEGIS can learn meaningful modality-quality and cross-modal compatibility scores under explicit supervision before allowing those scores to influence primary fusion.

---

## 1. Motivation

AEGIS v0.26 stress testing showed that the scalar quantities previously interpreted as modality reliability did not behave consistently as evidence-quality estimates.

The v0.26 frozen analysis produced the following diagnostic outcomes:

- H1 sensitivity: only 6/24 continuous corruption curves (25.0%) showed the expected negative relationship between corruption severity and corrupted-modality reliability.
- Mean Spearman correlation between corruption severity and corrupted-modality reliability: +0.3125.
- M2b positive reliability selectivity: 46.7%.
- M3 positive reliability selectivity: 38.3%.
- M2b correct weight-shift rate: 46.7%.
- M3 correct weight-shift rate: 38.3%.
- Overall Spearman correlation between reliability selectivity and classification preservation: 0.0125.
- M2b and M3 showed slightly stronger normalized corruption AUC than M1b, but this robustness was not associated with meaningful reliability selectivity.

Therefore, v0.27 must not assume that reliability semantics will emerge from classification supervision alone.

The central v0.27 research question is:

> **Can AEGIS learn modality-quality and cross-modal compatibility scores that respond correctly to controlled evidence degradation and mismatch when these concepts are explicitly supervised?**

---

## 2. Core Scientific Distinction

v0.27 separates three concepts that must not be conflated:

### 2.1 Intrinsic modality quality

For each modality:

\[
q_T \in [0,1], \qquad q_V \in [0,1]
\]

where:

- \(q_T\) estimates the quality of text evidence.
- \(q_V\) estimates the quality of visual evidence.

Quality concerns degradation or loss of the modality representation itself.

Examples:

- clean representation → high quality;
- attenuated representation → reduced quality;
- noisy representation → reduced quality;
- zeroed representation → zero quality.

### 2.2 Cross-modal compatibility

\[
c_{TV} \in [0,1]
\]

Compatibility estimates whether the text and image evidence belong together semantically.

Examples:

- original matched pair → high compatibility;
- permuted/mismatched pair → low compatibility.

A permuted image can still be intrinsically high-quality while being incompatible with the accompanying text.

### 2.3 External factual verification

External factual verification is outside the scope of the v0.27 Fakeddit corruption experiment.

Quality and compatibility do **not** establish factual truth.

The later verification layer must independently handle external evidence retrieval, provenance, temporal/geographic consistency, claim support and contradiction.

---

## 3. Frozen Architectural Progression

The v0.27 ablation sequence is fixed before implementation:

| Variant | Backbone | Quality Supervision | Compatibility Supervision | Quality/Compatibility Affect Primary Fusion |
|---|---|---:|---:|---:|
| M1b | Existing gated-interaction fusion | No | No | No |
| M4q | M1b | Yes | No | No |
| M4qc | M1b | Yes | Yes | No |
| M4qcf | M1b | Yes | Yes | Yes |

The variants must be implemented and evaluated in this order.

### 3.1 M1b

Frozen v0.25 gated-interaction baseline.

No architecture changes.

### 3.2 M4q

M1b backbone plus:

- text quality estimator \(Q_T\);
- vision quality estimator \(Q_V\);
- explicit quality supervision.

Quality scores are **diagnostic-only**.

They must not influence the primary fusion representation.

### 3.3 M4qc

M4q plus:

- cross-modal compatibility estimator \(C_{TV}\);
- explicit compatibility supervision.

Quality and compatibility remain **diagnostic-only**.

They must not influence the primary fusion representation.

### 3.4 M4qcf

Only after M4q and M4qc pass their behavioral tests, quality and compatibility may influence fusion.

M4qcf must preserve M1b's expressive dimension-wise gated fusion and introduce quality/compatibility as an auxiliary modulation pathway rather than replacing the gate with two scalar weights.

---

## 4. Representation Level

The first v0.27 study uses the same frozen Fakeddit encoder-representation caches as v0.25/v0.26.

Corruption occurs:

> **after frozen pretrained encoders and before learned AEGIS projection heads.**

This isolates the behavior of the AEGIS evidence-integration subsystem and keeps the experiment computationally controlled.

This experiment does not claim equivalence to real-world raw-text or raw-image corruption.

A later follow-up may use semantic/raw-input corruptions.

---

## 5. Training Data

The v0.27 experiment must use the frozen training representation cache already used in controlled v0.25 experiments.

Expected training cache:

`data/processed/fakeddit/frozen_embeddings/train_n5000_seed42`

Expected sample count:

`5000`

The existing frozen validation cache is used only for validation/evaluation:

`data/processed/fakeddit/frozen_embeddings/validation_n1000_seed42`

Expected sample count:

`1000`

The official Fakeddit test split must remain sealed.

---

## 6. Corruption Families

Training-time synthetic corruption is restricted to the following pre-specified families.

### 6.1 Gaussian representation noise

For modality \(m\):

\[
\tilde{x}_m = x_m + \lambda \sigma_m \epsilon,
\qquad
\epsilon \sim \mathcal{N}(0,I)
\]

where \(\sigma_m\) is an empirical per-feature scale computed from the **training cache only**.

Training severities:

\[
\lambda \in \{0.25, 0.50, 0.75, 1.00\}
\]

Clean examples correspond to:

\[
\lambda = 0
\]

### 6.2 Attenuation

\[
\tilde{x}_m = (1-\lambda)x_m
\]

with:

\[
\lambda \in \{0.25, 0.50, 0.75, 1.00\}
\]

### 6.3 Zero dropout

\[
\tilde{x}_m = 0
\]

This represents complete modality loss.

### 6.4 Permutation mismatch

A modality representation is replaced with another sample's representation using a deterministic derangement.

Permutation is treated as a **compatibility corruption**, not an intrinsic-quality corruption.

For a permuted pair:

- quality target of the substituted modality remains high;
- compatibility target is low.

---

## 7. Corruption Sampling During Training

Each training example is assigned one of four corruption states:

1. clean;
2. text-quality corruption;
3. vision-quality corruption;
4. cross-modal mismatch.

Initial fixed sampling probabilities:

\[
P(clean)=0.40
\]

\[
P(text\ quality)=0.20
\]

\[
P(vision\ quality)=0.20
\]

\[
P(mismatch)=0.20
\]

Within text-quality or vision-quality corruption:

- Gaussian noise: 40%
- attenuation: 40%
- zero dropout: 20%

For Gaussian noise and attenuation, severity is sampled uniformly from:

\[
\{0.25,0.50,0.75,1.00\}
\]

No simultaneous corruption of both modalities is used in the first v0.27 experiment.

This restriction is deliberate to preserve causal interpretability.

---

## 8. Quality Targets

### 8.1 Clean modality

\[
y_m^q = 1.0
\]

### 8.2 Gaussian noise and attenuation

Initial target:

\[
y_m^q = 1-\lambda
\]

Therefore:

| Severity | Quality target |
|---:|---:|
| 0.00 | 1.00 |
| 0.25 | 0.75 |
| 0.50 | 0.50 |
| 0.75 | 0.25 |
| 1.00 | 0.00 |

This linear mapping is a controlled synthetic target and is not claimed to represent human-perceived quality.

### 8.3 Zero dropout

\[
y_m^q = 0
\]

### 8.4 Intact modality during single-modality corruption

\[
y_{\bar m}^q = 1
\]

### 8.5 Permutation mismatch

For a permuted but otherwise unmodified modality:

\[
y_T^q = 1,\qquad y_V^q = 1
\]

Quality remains high because each individual representation is intact.

Compatibility is supervised separately.

---

## 9. Compatibility Targets

For original matched pairs:

\[
y^{compat}=1
\]

For deterministic permuted pairs:

\[
y^{compat}=0
\]

Quality corruption alone does not automatically imply incompatibility.

Therefore, for Gaussian noise, attenuation and zero dropout, compatibility targets are not used for the primary compatibility loss in the first implementation unless the pair itself is permuted.

This avoids teaching the compatibility estimator to equate low quality with semantic mismatch.

---

## 10. Estimator Architecture Constraints

### 10.1 Quality estimators

M4q/M4qc/M4qcf use separate estimators:

\[
q_T = Q_T(h_T)
\]

\[
q_V = Q_V(h_V)
\]

The first implementation must estimate intrinsic quality primarily from the corresponding modality representation.

The quality estimator must not depend directly on the other modality in M4q.

This prevents compatibility information from leaking into the intrinsic-quality target.

A small MLP is sufficient for the first controlled experiment.

Recommended structure:

\[
d \rightarrow 256 \rightarrow 64 \rightarrow 1
\]

with:

- GELU activation;
- dropout 0.1;
- final sigmoid.

### 10.2 Compatibility estimator

M4qc/M4qcf uses:

\[
c_{TV}=C([h_T;h_V;|h_T-h_V|;h_T\odot h_V;s_{TV}])
\]

where:

\[
s_{TV}=\cos(h_T,h_V)
\]

Recommended structure:

\[
(4d+1)\rightarrow256\rightarrow64\rightarrow1
\]

with:

- GELU;
- dropout 0.1;
- final sigmoid.

The architecture may be adjusted only for implementation correctness before any comparative v0.27 result is observed.

---

## 11. Loss Functions

### 11.1 Classification loss

Existing binary classification objective:

\[
\mathcal{L}_{cls}
\]

must remain unchanged.

### 11.2 Alignment loss

Existing alignment objective:

\[
\mathcal{L}_{align}
\]

must remain unchanged unless a later separate ablation explicitly studies it.

### 11.3 Quality loss

For each example:

\[
\mathcal{L}_{quality}
=
\frac{1}{2}
\left[
\operatorname{MSE}(q_T,y_T^q)
+
\operatorname{MSE}(q_V,y_V^q)
\right]
\]

### 11.4 Compatibility loss

For M4qc/M4qcf:

\[
\mathcal{L}_{compat}
=
\operatorname{BCE}(c_{TV},y^{compat})
\]

computed only on examples for which a compatibility target is defined.

### 11.5 Total loss

M4q:

\[
\mathcal{L}
=
\mathcal{L}_{cls}
+
\lambda_a\mathcal{L}_{align}
+
\lambda_q\mathcal{L}_{quality}
\]

M4qc:

\[
\mathcal{L}
=
\mathcal{L}_{cls}
+
\lambda_a\mathcal{L}_{align}
+
\lambda_q\mathcal{L}_{quality}
+
\lambda_c\mathcal{L}_{compat}
\]

Initial fixed coefficients:

\[
\lambda_q = 1.0
\]

\[
\lambda_c = 1.0
\]

The existing alignment coefficient must remain whatever is used by the controlled M1b training protocol.

No tuning of \(\lambda_q\) or \(\lambda_c\) is permitted based on validation classification performance in the first controlled run.

If loss-scale imbalance is severe, a later explicitly named coefficient ablation may be performed.

---

## 12. M4qcf Fusion Rule

M4qcf must not replace M1b's gated fusion.

Let the frozen M1b-style gated-interaction representation be:

\[
z_{M1b}
\]

Define effective modality confidence terms only after M4q/M4qc behavioral validation.

A conservative first rule is:

\[
\hat q_T = q_T \cdot c_{TV}
\]

\[
\hat q_V = q_V \cdot c_{TV}
\]

However, because a single pairwise compatibility scalar penalizes both modalities equally and cannot identify which modality caused mismatch, this rule must be treated as provisional.

Therefore the first M4qcf implementation should use quality primarily for modality-specific modulation and compatibility as a bounded residual confidence term.

The exact M4qcf fusion equation must be frozen in a separate protocol amendment **after M4q and M4qc results are complete**.

M4qcf must not be implemented before that amendment.

---

## 13. Determinism and Randomness

All corruption generation must be deterministic under explicit seeds.

Training corruption must use a dedicated corruption RNG separate from:

- model initialization RNG;
- minibatch/shuffle RNG;
- dropout RNG where feasible.

The corruption realization should be reproducible from:

- training seed;
- epoch;
- batch index or sample identifier;
- corruption family;
- modality;
- severity.

For the v0.27 validation stress test, corruption realizations must be identical across architectures and trained model seeds.

Therefore validation corruption seeds must be **decoupled from model seed**.

This corrects the v0.26 limitation in which corruption realization could vary by model seed.

---

## 14. Training Seeds

Controlled training seeds:

\[
\{42,43,44\}
\]

The same seed set is used for:

- M1b reference;
- M4q;
- M4qc;
- later M4qcf.

Common-component initialization must be paired wherever architectures share parameters.

The M1b backbone initialization protocol must be preserved.

---

## 15. Validation Protocol

Validation has two distinct roles.

### 15.1 Standard clean validation

Used for:

- checkpoint selection;
- clean Macro-F1;
- clean accuracy;
- standard classification metrics.

### 15.2 Reliability diagnostic validation

Used for:

- quality score sensitivity;
- quality target calibration;
- compatibility discrimination;
- corruption robustness;
- selectivity;
- weight shift only when fusion uses quality.

The validation diagnostic protocol must reuse the frozen v0.26 corruption families:

- Gaussian noise;
- attenuation;
- permutation;
- zero dropout.

Continuous severities:

\[
\{0,0.25,0.50,0.75,1.00\}
\]

The official test split remains sealed.

---

## 16. Primary v0.27 Hypotheses

### H1-Q — Quality sensitivity

For corruption of modality \(m\):

\[
\rho(\lambda,q_m) < 0
\]

for Gaussian noise and attenuation.

Success criterion:

- at least 75% of architecture-seed-family-modality curves show the expected negative direction;
- mean Spearman correlation is negative.

### H2-Q — Quality selectivity

When modality \(m\) is corrupted:

\[
\Delta q_m < \Delta q_{\bar m}
\]

equivalently:

\[
S_m^q = \Delta q_{\bar m} - \Delta q_m > 0
\]

Success criterion:

- positive selectivity in at least 75% of evaluated corruption conditions.

### H3-Q — Zero-dropout recognition

For complete modality removal:

\[
q_m \rightarrow 0
\]

Operational criterion:

- mean predicted quality for the zeroed modality \(\le 0.20\);
- intact modality mean predicted quality \(\ge 0.80\).

### H4-C — Compatibility discrimination

Matched pairs should have higher compatibility than permuted pairs:

\[
c_{TV}^{matched} > c_{TV}^{permuted}
\]

Primary metrics:

- ROC-AUC;
- PR-AUC;
- matched/permuted mean separation;
- binary accuracy at threshold 0.5 as secondary.

Success criterion:

\[
ROC\text{-}AUC \ge 0.80
\]

on the frozen validation diagnostic set.

### H5-C — Quality/compatibility disentanglement

Permutation mismatch should reduce compatibility without causing large reductions in intrinsic modality-quality estimates.

Operational expectation:

\[
|\Delta q_T| < 0.10
\]

and

\[
|\Delta q_V| < 0.10
\]

on average under permutation, while compatibility decreases substantially.

### H6-CLS — No destructive auxiliary supervision

M4q and M4qc should not substantially degrade clean classification performance relative to the paired M1b baseline.

Non-inferiority diagnostic margin:

\[
\Delta MacroF1 \ge -0.01
\]

relative to paired M1b mean.

This is a diagnostic margin, not a formal statistical non-inferiority test.

---

## 17. Secondary Metrics

For quality estimation:

- MSE;
- MAE;
- Pearson correlation with synthetic quality target;
- Spearman correlation with corruption severity;
- calibration by severity bin;
- clean-vs-corrupt separation.

For compatibility estimation:

- ROC-AUC;
- PR-AUC;
- BCE;
- matched mean;
- permuted mean;
- separation effect size.

For classification:

- Macro-F1;
- accuracy;
- class-wise precision/recall/F1;
- confusion matrix;
- corruption AUC;
- \(\Delta MacroF1\) from clean.

For later M4qcf:

- quality-weight shift;
- compatibility response;
- robustness;
- clean-performance retention.

---

## 18. Checkpoint Selection

Checkpoint selection must use clean validation classification Macro-F1 as the primary criterion to preserve comparability with prior experiments.

Quality/compatibility diagnostic metrics must be logged but must not determine the selected checkpoint in the first controlled experiment.

This prevents selecting a checkpoint specifically to optimize the new diagnostic metrics.

A later multi-objective checkpoint-selection study may be performed separately.

---

## 19. Required Logging

Each training run must save:

- architecture;
- seed;
- git commit;
- resolved configuration;
- training cache path and sample count;
- validation cache path and sample count;
- corruption sampling probabilities;
- corruption RNG protocol;
- quality target rules;
- compatibility target rules;
- loss coefficients;
- parameter counts;
- initialization fingerprints where applicable;
- epoch metrics;
- checkpoint-selection criterion;
- final checkpoint path;
- test_split_accessed=false.

Each validation diagnostic condition must save:

- corruption family;
- modality;
- severity;
- corruption seed;
- sample-level label/prediction;
- \(q_T\);
- \(q_V\);
- \(c_{TV}\) where available;
- classification logits/probabilities;
- clean-vs-corrupt deltas;
- parameter fingerprint before/after evaluation.

---

## 20. Required Experiment Directory

```text
experiments/
└── fakeddit/
    └── v027_corruption_aware_reliability/
        ├── protocol.json
        ├── m4q/
        │   ├── seed42/
        │   ├── seed43/
        │   └── seed44/
        ├── m4qc/
        │   ├── seed42/
        │   ├── seed43/
        │   └── seed44/
        ├── diagnostics/
        └── aggregate/
```

M4qcf must not be added until its fusion-rule protocol amendment is frozen.

---

## 21. Implementation Order

The implementation sequence is fixed:

1. Freeze protocol.
2. Add corruption-generation utilities with unit tests.
3. Add quality-target generation with unit tests.
4. Add deterministic permutation/compatibility-target utilities.
5. Add modality-quality estimator module.
6. Integrate M4q without affecting M1b behavior.
7. Add M4q tests.
8. Train M4q seeds 42/43/44.
9. Run clean + corruption diagnostics.
10. Analyze H1-Q/H2-Q/H3-Q/H6-CLS.
11. Add compatibility estimator.
12. Integrate M4qc.
13. Add M4qc tests.
14. Train M4qc seeds 42/43/44.
15. Run compatibility diagnostics.
16. Analyze H4-C/H5-C/H6-CLS.
17. Freeze M4q/M4qc findings.
18. Decide whether M4qcf is scientifically justified.
19. If justified, write and freeze a separate M4qcf fusion protocol amendment.
20. Only then implement M4qcf.

---

## 22. What Must Not Be Changed After Results Are Observed

Without creating a new explicitly versioned protocol amendment, do not change:

- corruption families;
- severity set;
- corruption sampling probabilities;
- quality target mapping;
- compatibility target definition;
- training seeds;
- primary hypotheses;
- primary success criteria;
- checkpoint-selection criterion;
- validation corruption grid;
- M4q/M4qc diagnostic-only constraint.

Do not tune these elements after seeing v0.27 comparative results.

---

## 23. Claims Allowed if Successful

If M4q/M4qc satisfy the pre-specified behavioral criteria, AEGIS may claim:

- explicitly supervised modality-quality estimation under controlled representation corruption;
- cross-modal compatibility discrimination under deterministic mismatching;
- empirical separation of intrinsic evidence quality and cross-modal compatibility;
- corruption-aware auxiliary evidence assessment.

AEGIS must still not claim:

- factual verification from these scores alone;
- human-calibrated trustworthiness;
- real-world corruption generalization without raw-input experiments;
- intent detection;
- universal reliability across modalities/datasets.

---

## 24. Falsification Conditions

The v0.27 hypothesis must be considered unsupported if any of the following broadly occur:

- quality scores remain insensitive or move in the wrong direction under increasing corruption;
- quality estimators collapse to modality-global constants;
- permutation substantially changes quality scores instead of compatibility;
- compatibility cannot discriminate matched from permuted pairs;
- auxiliary supervision causes substantial clean classification degradation;
- apparent improvement is confined to one seed and is not directionally reproducible.

Negative results must be preserved and reported.

---

## 25. Current Scientific Position

The v0.27 design is intentionally conservative.

AEGIS will not allow a mechanism to control primary multimodal fusion merely because it has been named "reliability."

The mechanism must first demonstrate that:

1. its outputs respond to known degradation;
2. the response is modality-selective;
3. semantic mismatch is distinguished from intrinsic quality;
4. the learned scores remain compatible with strong classification performance.

Only after these conditions are demonstrated can quality/compatibility-informed fusion be scientifically justified.
