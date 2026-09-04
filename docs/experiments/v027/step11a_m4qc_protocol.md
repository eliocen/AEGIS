# AEGIS v0.27 Step 11A — M4qc Compatibility Implementation and Evaluation Protocol

**Protocol version:** `0.27.0-step11a`  
**Status:** `FROZEN BEFORE M4qc IMPLEMENTATION AND RESULT EXPOSURE`  
**Model stage:** `M4qc`  
**Purpose:** Exact operationalization of diagnostic cross-modal compatibility learning and evaluation after completion of M4q intrinsic-quality analysis.

## 1. Scientific question

M4q established explicitly supervised intrinsic modality-quality estimation:

q_T = f_T(h_T), q_V = f_V(h_V)

M4qc introduces a distinct pairwise compatibility estimator:

c_TV = f_C(h_T, h_V), with c_TV in [0,1].

The scientific question is whether AEGIS can explicitly learn whether two individually usable modality representations form a compatible text-image pair, while preserving the intrinsic-quality semantics established by M4q and without altering the existing classification/fusion pathway.

Intrinsic quality and cross-modal compatibility are distinct. A clean text representation and a clean image representation may both have high intrinsic quality while being mutually incompatible:

q_T ~= 1, q_V ~= 1, c_TV ~= 0.

## 2. Architectural boundary

M4qc MUST preserve the M1b/M4q classification and fusion architecture. The compatibility head is diagnostic-only.

The following are forbidden in M4qc:

- feeding q_T, q_V, or c_TV into fusion weights;
- gating classification features with compatibility;
- changing M1b fusion topology;
- changing the classifier based on compatibility;
- threshold tuning against validation H4/H5 outcomes;
- checkpoint selection using compatibility metrics;
- accessing the official test split;
- implementing M4qcf reliability-informed fusion.

M4qcf remains blocked until a later explicit decision gate.

## 3. Compatibility representation

For projected text and vision representations h_T,h_V in R^d, the compatibility feature is frozen as:

z_C = [h_T; h_V; |h_T-h_V|; h_T*h_V; cosine(h_T,h_V)]

with dimensionality 4d+1.

The compatibility estimator is frozen as:

(4d+1) -> Linear(256) -> GELU -> Dropout(0.1) -> Linear(64) -> GELU -> Dropout(0.1) -> Linear(1) -> Sigmoid.

No other pairwise features may be added during M4qc without a protocol amendment.

## 4. Compatibility supervision

Matched pair:
(h_T^(i), h_V^(i)) -> c* = 1.

Mismatched pair:
(h_T^(i), h_V^(pi(i))) with pi(i) != i -> c* = 0.

Mismatch does not imply intrinsic corruption. For clean representations participating in mismatch:

q_T* = 1, q_V* = 1.

Compatibility loss is binary cross-entropy.

## 5. Total M4qc training objective

M4qc extends the frozen M4q objective:

L = lambda_A L_A + lambda_CLS L_CLS + lambda_Q L_Q + lambda_C L_C.

Frozen values:

lambda_Q = 1
lambda_C = 1

Existing M1b/M4q alignment and classification coefficients MUST remain unchanged. No coefficient may be tuned after viewing first M4qc validation results.

## 6. Training corruption mixture

The existing v0.27 mixture remains unchanged:

- clean: 0.40
- text quality corruption: 0.20
- vision quality corruption: 0.20
- mismatch: 0.20

Quality-family probabilities remain:

- Gaussian noise: 0.40
- attenuation: 0.40
- zero dropout: 0.20

Continuous severities remain {0.25, 0.50, 0.75, 1.00}, uniformly sampled.

Quality corruption alone does not automatically imply incompatibility.

## 7. Deterministic training mismatch construction

For a training batch of size B > 1, mismatch construction MUST use a deterministic cyclic derangement.

For each mismatch batch, sample an offset k in {1,...,B-1} using the already frozen deterministic batch-corruption RNG stream and define:

pi(j) = (j + k) mod B.

This guarantees no self-pair. Text remains in original batch order and vision is cyclically permuted.

If a mismatch batch has B < 2, execution MUST fail loudly.

Training mismatch generation is not class-constrained.

## 8. Validation data and test sealing

Formal M4qc compatibility evaluation MUST use only the frozen Fakeddit validation cache:

- validation samples: 1,000
- model seeds: 42, 43, 44
- official test samples accessed: 0

The official test split may not be used for checkpoint selection, compatibility analysis, threshold selection, permutation construction, hyperparameter tuning, or architecture selection.

## 9. Primary validation mismatch construction

Primary validation mismatch evaluation MUST use a deterministic class-preserving derangement.

Reason: global random mismatch may create an avoidable Stage-1 class-label shortcut. The primary compatibility test should discriminate matched from mismatched pairs even when the Stage-1 label is preserved.

For each validation class independently:

1. collect validation indices belonging to that class;
2. sort indices in ascending original validation order;
3. initialize deterministic RNG with seed 27110000 + class_id;
4. generate permutations until a full derangement is obtained;
5. map every text sample to a different vision sample from the same Stage-1 class.

Required invariants:

- no self-pair;
- same Stage-1 label in each mismatched pair;
- exactly one mismatched partner per validation text sample;
- every validation vision sample used exactly once within its class;
- identical mapping for seeds 42,43,44;
- mapping independent of model-training RNG.

If a class has fewer than two samples, evaluation MUST fail loudly.

The mapping MUST be persisted as a machine-readable artifact.

## 10. H4-C — compatibility discrimination

Per checkpoint evaluate:

- 1,000 clean matched pairs, target 1;
- 1,000 deterministic class-preserving mismatched pairs, target 0.

Total = 2,000 compatibility observations per model seed.

Compute ROC-AUC separately for seeds 42,43,44.

Primary aggregate:

mean_AUC = arithmetic mean of the three seed-level AUCs.

H4-C is supported iff:

mean_AUC >= 0.80.

All seed-level AUCs MUST be reported. No per-seed minimum is imposed on the primary decision.

## 11. H5-C — quality/compatibility disentanglement

Quality scores MUST be compared by representation identity, not by pair row.

For each text representation h_T^(i):

Delta q_T^(i) = q_T^mismatch(h_T^(i)) - q_T^matched(h_T^(i)).

For each permuted vision representation h_V^(pi(i)):

Delta q_V^(pi(i)) = q_V^mismatch(h_V^(pi(i))) - q_V^matched(h_V^(pi(i))).

Per seed compute mean absolute text-quality change M_T^(s) and mean absolute vision-quality change M_V^(s).

Primary aggregates are arithmetic means across seeds:

mean_M_T = mean_s M_T^(s)
mean_M_V = mean_s M_V^(s)

Frozen quality criteria:

mean_M_T < 0.10
mean_M_V < 0.10

Strict inequality is required.

For compatibility drop, per seed define:

D_C^(s) = mean(c_TV matched) - mean(c_TV mismatched).

Primary aggregate:

mean_D_C = arithmetic mean across seeds.

The phrase "compatibility drops" is operationalized as:

mean_D_C > 0.

Strict positivity is required. No additional drop-magnitude threshold is introduced because H4-C already supplies the discrimination-strength criterion.

H5-C is supported iff all three conditions hold:

mean_M_T < 0.10 AND mean_M_V < 0.10 AND mean_D_C > 0.

Seed-level values MUST be reported descriptively but do not alter the primary aggregate decision.

## 12. Classification-preservation gate for M4qc

M4qc adds an auxiliary compatibility loss, so clean classification must be checked again.

This is a preservation gate and does not overwrite the completed M4q H6-CLS result.

Primary M4qc Macro-F1 is the arithmetic mean across seeds 42,43,44.

Frozen M1b reference mean Macro-F1 = 0.8736.

Delta_F1_M4qc = M4qc_mean - 0.8736.

The preservation gate passes iff:

Delta_F1_M4qc >= -0.01.

This is a descriptive validation margin, not a formal statistical noninferiority test.

## 13. Checkpoint selection

Checkpoint selection MUST continue to use clean validation Macro-F1 only.

Compatibility AUC, compatibility loss, H4-C, H5-C, corruption diagnostics, and official test data MUST NOT influence checkpoint selection.

Early stopping remains otherwise unchanged from M4q.

## 14. Required implementation/evaluation sequence

Step 11A: freeze this protocol.
Step 11B: implement compatibility estimator and unit tests.
Step 11C: integrate diagnostic-only M4qc training.
Step 11D: run smoke validation.
Step 11E: train seeds 42/43/44.
Step 12B: run deterministic compatibility diagnostics.
Step 12C: compute formal H4-C, H5-C, and classification-preservation decisions.
Step 12D: freeze tracked scientific results.
Then conduct a separate M4qcf decision gate.

Formal H4/H5 decisions MUST NOT be computed during model implementation or training.

## 15. Required machine-readable outputs

The M4qc diagnostic stage MUST persist enough information to independently recompute all primary decisions, including:

- protocol manifest;
- validation derangement mapping;
- per-sample matched compatibility scores;
- per-sample mismatched compatibility scores;
- matched and mismatched q_T/q_V;
- representation/source identity needed for H5 alignment;
- per-seed summaries;
- aggregate summaries;
- explicit official_test_split_accessed=false.

The formal analyzer MUST fail loudly on missing/duplicate records, self-pairs, class-changing primary mismatches, unexpected sample counts, unexpected seeds, malformed/nonfinite scores, scores outside [0,1], official test access, or protocol-version mismatch.

## 16. Interpretation boundaries

Successful M4qc results would establish only explicit cross-modal compatibility discrimination and its separation from intrinsic modality-quality estimation under the frozen controlled Fakeddit validation protocol.

M4qc does NOT establish factual truth, factual verification, source credibility, misinformation/disinformation/malinformation intent, human-calibrated trust, universal semantic consistency, external-world evidence support, raw-world generalization, temporal/geographic verification, or reliability-informed fusion benefit.

A high compatibility score means only that the learned pairwise representation is compatible under the training/evaluation construction. It does not prove factual truth.

## 17. M4qcf decision boundary

M4qcf remains blocked.

No reliability-informed fusion implementation may begin until M4qc training, formal H4-C evaluation, formal H5-C evaluation, the classification-preservation gate, and tracked result freezing are complete, followed by a separate explicit M4qcf protocol/amendment.

Negative M4qc results MUST be preserved and analyzed rather than hidden by post-hoc threshold or architecture changes.

## 18. Frozen Step 11A summary

Frozen before M4qc implementation:

- diagnostic-only compatibility architecture;
- pairwise feature construction;
- BCE compatibility loss;
- lambda_C = 1;
- existing M4q coefficients unchanged;
- existing v0.27 corruption mixture unchanged;
- deterministic cyclic training derangement;
- deterministic class-preserving validation derangement;
- validation mapping independent of model seed;
- 1,000 matched + 1,000 mismatched observations per seed;
- H4-C primary aggregate = mean ROC-AUC across seeds;
- H4-C threshold = 0.80;
- H5-C representation-identity alignment;
- H5-C mean absolute text-quality change < 0.10;
- H5-C mean absolute vision-quality change < 0.10;
- H5-C mean compatibility drop > 0;
- M4qc clean classification preservation margin >= -0.01 versus M1b;
- checkpoint selection by clean validation Macro-F1 only;
- official test split remains sealed;
- M4qcf remains blocked.
