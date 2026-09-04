# AEGIS v0.27 Step12D — M4qc Formal Compatibility Results

**Record version:** 0.27.0-step12d  
**Status:** COMPLETE AND FROZEN  
**Source protocol:** 0.27.0-step11a  
**Formal analysis implementation commit:** `2f7560c`  
**Step12B diagnostics implementation commit:** `56dcf45`  
**Step11E training-results record commit:** `dfcbea9`

## 1. Scientific scope

This record freezes the formal M4qc compatibility results obtained from the frozen Fakeddit validation protocol using model seeds 42, 43, and 44.

M4qc is diagnostic-only. Intrinsic text quality (`qT`), intrinsic vision quality (`qV`), and pairwise text–vision compatibility (`cTV`) do not alter the primary M1b fusion representation or classifier pathway in this experiment.

The official Fakeddit test split remained sealed. No checkpoint reselection, retraining, threshold tuning, or M4qcf computation was performed during Step12C.

## 2. H4-C — controlled compatibility discrimination

**Hypothesis:** the learned compatibility estimator should discriminate matched text–vision pairs from deterministic class-preserving mismatched pairs.

The frozen primary decision rule is:

`mean ROC-AUC across seeds >= 0.80`

| Seed | ROC-AUC |
|---:|---:|
| 42 | 0.810778 |
| 43 | 0.802873 |
| 44 | 0.800905 |
| **Mean** | **0.804852** |

**Formal decision: SUPPORTED.**

The result supports controlled pairwise compatibility discrimination under the frozen Fakeddit validation protocol. The margin above the preregistered threshold is modest (`0.804852 - 0.800000 = 0.004852`), so the result must not be described as strong or near-perfect discrimination.

## 3. H5-C — separation of intrinsic modality quality from pairwise compatibility

The frozen H5-C decision requires all three conditions:

- mean absolute text-quality change `< 0.10`;
- mean absolute vision-quality change `< 0.10`;
- mean compatibility drop `> 0`.

Quality comparisons are aligned by **representation identity**, not pair-row identity.

| Metric | Seed 42 | Seed 43 | Seed 44 | Aggregate |
|---|---:|---:|---:|---:|
| mean `|ΔqT|` | 0.000000 | 0.000000 | 0.000000 | **0.000000** |
| mean `|ΔqV|` | 0.000000 | 0.000000 | 0.000000 | **0.000000** |
| compatibility drop `Δc` | 0.267254 | 0.361748 | 0.299678 | **0.309560** |

**Formal decision: SUPPORTED.**

Under the controlled mismatch intervention, the intrinsic quality outputs are exactly invariant when the same underlying text or vision representation is compared, while pairwise compatibility decreases by an average of approximately `0.30956`. This provides evidence that M4qc has learned a compatibility signal that is functionally distinct from the separately supervised intrinsic modality-quality signals under this protocol.

The zero quality deltas are an architectural/diagnostic consequence consistent with the estimator separation: qT is computed from the text representation alone and qV from the vision representation alone, while cTV is computed jointly. They must not be interpreted as evidence that real-world modality quality is perfectly invariant.

## 4. Classification-preservation gate

Frozen Step11E result:

- M1b reference validation Macro-F1: `0.873600`
- M4qc three-seed mean validation Macro-F1: `0.864839357609`
- Delta: `-0.008760642391`
- Frozen margin: `-0.010000`
- Criterion: `delta >= -0.01`

**Decision: PASS.**

This is a descriptive held-out validation-margin preservation criterion. It is **not** a formal statistical noninferiority test.

## 5. Combined M4qc conclusion

M4qc satisfies all frozen gates required at this stage:

1. H4-C is supported: mean controlled compatibility ROC-AUC is `0.804852`, exceeding the frozen `0.80` threshold.
2. H5-C is supported: intrinsic text and vision quality remain stable under representation-identity comparison (`0.0` aggregate mean absolute change for both), while compatibility decreases (`0.309560` mean drop).
3. Clean validation classification is preserved within the frozen descriptive margin (`Δ Macro-F1 = -0.008760642391 >= -0.01`).
4. The official Fakeddit test split remained sealed.

Therefore, the M4qc phase is scientifically complete under the frozen controlled-validation protocol and may be closed without modifying the completed hypotheses or thresholds.

## 6. Interpretation boundaries

These results establish only the behavior measured by the frozen controlled Fakeddit protocol. They do **not** establish:

- factual truth or factual verification;
- source credibility;
- misinformation/disinformation intent;
- human trustworthiness;
- universal semantic consistency;
- external-world evidence support;
- raw-world or cross-dataset generalization;
- temporal or geographic verification;
- benefit from using qT/qV/cTV to alter fusion.

In particular, H4-C should be described as **threshold-supporting but modest** controlled compatibility discrimination, not as proof of general semantic verification.

## 7. Decision gate for the next phase

**M4qc status: COMPLETE AND FROZEN.**

The previous block on considering M4qcf is now cleared at the protocol level because formal M4qc training, deterministic diagnostics, H4-C/H5-C analysis, classification-preservation evaluation, and tracked results freeze have been completed successfully.

This record does **not** authorize immediate implementation of M4qcf. Any M4qcf experiment must first receive a separate pre-result protocol/amendment that freezes:

- the reliability-informed fusion equation;
- how qT, qV, and cTV influence modality/evidence weights;
- safeguards against trivial score collapse or classifier leakage;
- training objectives and coefficients;
- checkpoint-selection rule;
- formal comparison baselines;
- seeds and data splits;
- hypotheses and quantitative success/failure thresholds;
- test-split policy and interpretation boundaries.

No M4qcf result exists at this freeze point.
