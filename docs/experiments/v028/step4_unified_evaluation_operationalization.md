# AEGIS v0.28 Step4 Unified Evaluation Operationalization

**Status:** FROZEN BEFORE UNIFIED v0.28 EVALUATION

This record operationalizes the already frozen Step1 evaluation contract. It
introduces no new hypothesis, architecture, threshold, seed, condition, or
checkpoint choice. Evidence remains controlled validation-development evidence.

Official-test samples accessed: **0**.

## Frozen checkpoint matrix

| Architecture | Fusion implementation | Frozen checkpoint pattern |
|---|---|---|
| M1b | `gated_interaction` | `experiments/fakeddit/v025_m1b_gated_interaction_seed{seed}/best_model.pt` |
| M4qc | `quality_compatibility_supervised` | `experiments/fakeddit/v027_m4qc_seed{seed}/best_model.pt` |
| M4qcf | `quality_compatibility_fusion` | `experiments/fakeddit/v027_m4qcf_seed{seed}/best_model.pt` |
| M4qcs-w | `quality_compatibility_selective_weights` | `experiments/fakeddit/v028_m4qcs_w_seed{seed}/best_model.pt` |
| M4qcs-i | `quality_compatibility_selective_interaction` | `experiments/fakeddit/v028_m4qcs_i_seed{seed}/best_model.pt` |
| M4qcs | `quality_compatibility_selective_fusion` | `experiments/fakeddit/v028_m4qcs_seed{seed}/best_model.pt` |

All 18 architecture-seed checkpoints are restricted to `best_model.pt`.
Retraining and checkpoint reselection are prohibited.

## Unified evaluation matrix

- Architectures: M1b, M4qc, M4qcf, M4qcs-w, M4qcs-i, and M4qcs.
- Seeds: 42, 43, and 44.
- Quality conditions: 19 per architecture-seed, including clean and 18 frozen
  single-modality corruption conditions.
- Quality output: 342 summaries and 342,000 sample rows.
- Mismatch states: matched and deterministic class-preserving mismatched.
- Mismatch output: 36 summaries and 36,000 sample rows.
- Gaussian scales come only from the frozen training cache.
- Corruption tensors and mismatch mappings are shared across all architectures
  and model seeds.

## Controller invariants

- M4qcs-w uses selective weights and fixes the interaction multiplier to one.
- M4qcs-i fixes both modality weights to one half and uses compatibility-based
  interaction suppression.
- M4qcs combines selective weights with independent interaction suppression.
- Controller inputs remain stop-gradient and the controller remains parameter-free.
- Evaluation must not mutate any model state.

## Scientific boundary

- Evaluation only; no training or checkpoint reselection.
- No tuning, architecture modification, or condition/seed exclusion.
- V28-H1, V28-H2, V28-H3, and V28-H4 remain **NOT COMPUTED**.
- Step4 records measurements only. Step5 alone computes formal decisions.
- The official Fakeddit test split remains sealed.

## Next step

Step4B implements, tests, freezes, and runs the unified evaluator. Formal
hypothesis decisions remain prohibited until Step5.
