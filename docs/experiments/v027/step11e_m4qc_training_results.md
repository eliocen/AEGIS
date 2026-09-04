# AEGIS v0.27 — Step11E M4qc Formal Training Results

**Status:** COMPLETE AND FROZEN

**Record version:** `0.27.0-step11e`

**Frozen source protocol:** `0.27.0-step11a`

## 1. Purpose

Step11E records the formal multi-seed training results for M4qc before deterministic compatibility diagnostics are implemented or exposed.

M4qc retains the M1b gated-interaction primary fusion pathway and adds diagnostic intrinsic-quality heads qT/qV plus the diagnostic cross-modal compatibility head cTV. The diagnostic outputs do not alter the primary fused representation.

## 2. Formal training results

| Seed | Best epoch | Epochs completed | Stop reason | Accuracy | Macro-F1 | Classification loss |
|---:|---:|---:|---|---:|---:|---:|
| 42 | 5 | 13 | early_stopping | 0.863000000 | 0.862746217757 | 0.352464729786 |
| 43 | 6 | 14 | early_stopping | 0.866000000 | 0.865986598660 | 0.338411127806 |
| 44 | 5 | 13 | early_stopping | 0.866000000 | 0.865785256410 | 0.338070027828 |

All three formal runs:

- used seeds 42, 43, and 44;
- used λQ = 1.0 and λC = 1.0;
- selected checkpoints using clean validation Macro-F1 with validation classification loss as the tie-breaker;
- retained qT, qV, and cTV as diagnostic-only outputs;
- persisted finite quality and compatibility training losses;
- populated clean, text-quality, vision-quality, and mismatch training conditions;
- updated all expected trainable M4qc components; and
- accessed zero official Fakeddit test samples.

## 3. Classification-preservation gate

M1b reference Macro-F1: `0.873600000000`

M4qc formal seed Macro-F1 values:

- Seed 42: `0.862746217757`
- Seed 43: `0.865986598660`
- Seed 44: `0.865785256410`

M4qc three-seed mean Macro-F1: `0.864839357609`

Delta versus M1b: `-0.008760642391`

Frozen criterion: `Δ Macro-F1 >= -0.01`

**Classification-preservation gate: PASS**

This is a descriptive held-out validation-margin criterion. It is not a formal statistical noninferiority test.

## 4. Hypothesis status

**H4-C: PENDING / NOT COMPUTED.** Step11E does not evaluate matched-versus-class-preserving-mismatched compatibility discrimination.

**H5-C: PENDING / NOT COMPUTED.** Step11E does not evaluate intrinsic-quality stability under mismatch or the formal compatibility-drop criterion.

No H4-C or H5-C result is inferred from training losses, validation classification performance, or the Step11D smoke run.

## 5. Scientific interpretation

Step11E establishes that M4qc can be trained under the frozen v0.27 protocol across the three formal seeds while preserving clean validation classification performance within the preregistered descriptive margin.

This result does not yet establish that the compatibility estimator discriminates matched from mismatched pairs, nor that qT/qV remain stable under controlled mismatch. Those questions are reserved for the frozen H4-C/H5-C diagnostics.

## 6. Interpretation boundaries

- No claim of factual truth or open-world factual verification is made.
- No claim of source credibility or human trust is made.
- No claim of intent inference is made.
- No claim of universal semantic consistency is made.
- Compatibility is a controlled learned pairwise diagnostic under the v0.27 protocol.
- The official Fakeddit test split remains sealed and was not accessed.

## 7. Freeze and next stage

The Step11E formal training results are frozen before Step12B deterministic compatibility diagnostics.

Step12B may:

- load the frozen best checkpoints for seeds 42/43/44;
- construct the frozen deterministic class-preserving validation derangement;
- emit matched and mismatched qT/qV/cTV diagnostics; and
- persist the required machine-readable mapping.

Step12B must not:

- retrain M4qc;
- reselect checkpoints;
- tune thresholds from observed results;
- access the official test split;
- compute M4qcf behavior; or
- change the frozen Step11A H4-C/H5-C operationalization.

Formal H4-C/H5-C analysis remains reserved for Step12C.
