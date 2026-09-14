# AEGIS v0.33 Step4 鈥?Implementation Plan and Source Allowlist

## Audit conclusion

The Step3 read-only audit verified that the v0.31 code already contains
`p_ref_true`, `p_candidate_true`, `delta_u`, and `u_target`, together with the
counterfactual candidate embedding path. The old target transform is
`sigmoid(delta_u / 0.05)`, and the old utility loss uses BCE on the already
sigmoided selector output.

v0.33 will therefore **reuse the existing counterfactual and candidate-action
machinery but introduce a separate v0.33 selector-learning path**. It will not
rewrite the v0.31 mechanism in place.

## Exact Step5 source allowlist

Existing files that may be modified:

1. `aegis/reliability/reliability_controller.py`
2. `aegis/alignment/model.py`
3. `scripts/run_fakeddit_ablation.py`

New test files that may be created:

4. `tests/test_v033_utility_supervision_selector.py`
5. `tests/test_v033_training_path.py`

No other tracked file may change during Step5 implementation.

## Critical implementation requirements

- new primary architecture: `M4qusli`
- v0.31 behavior must remain unchanged
- inherited candidate action must match v0.31 numerically
- v0.33 target: `clip(0.5 + delta_u / 0.20, 0, 1)`
- selector exposes raw `selector_logit`
- utility probability is `sigmoid(selector_logit)`
- utility objective is magnitude-weighted **BCEWithLogits**
- effective utility-loss weight is exactly `0.50`
- transition objective weight is exactly `0.0`
- active threshold is `0.60`
- maximum intervention strength is `0.15`
- below threshold means exact zero intervention

## Step5 hard gates

Implementation cannot be frozen unless the controlled positive/negative target
smoke and the real one-batch training path demonstrate a finite utility loss,
non-zero selector optimization signal, measurable selector parameter movement,
the exact runtime-effective loss weight, and persisted machine-readable
diagnostics.

Formal training remains prohibited even after implementation success. A later
step must separately freeze formal-training operationalization.

Official test samples accessed: **0**.
