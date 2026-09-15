# AEGIS v0.34 Step8A 鈥?Formal Entry-Point Observability Activation Correction Plan

**Status:** `CORRECTION_PLAN_REMOTELY_FREEZABLE`

The Step8 AST audit established that `train_one_epoch` supports `observability_root` but the single ordinary formal call does not pass it. Step8A therefore freezes a narrow operationalization-only correction.

## Exact implementation allowlist
1. `scripts/run_fakeddit_ablation.py`
2. `tests/test_v034_observability_training_path.py`

No controller, model, evaluator, utility target, selector mechanism, intervention mapping, checkpoint-selection logic, or dataset/evaluation semantics may change.

## Required implementation
The runner must derive the prospective observability location from the existing per-run output/provenance boundary, explicitly pass it to the formal `train_one_epoch` call, and activate the existing v0.34 observability gate only for the authorized primary v0.34 formal path. Diagnostics must come from the live training path; no second forward or post-hoc reconstruction is allowed.

The regression must exercise the ordinary entry-point/main-loop wiring, verify creation of the mandatory persisted diagnostic artifacts, verify independent recoverability, and verify inherited/non-authorized paths remain disabled by default.

## Scientific boundary
This is not a mechanism redesign. Q1鈥換6 remain `NOT_YET_EVALUATED`. No formal training or formal evaluation is authorized by Step8A. Official test access remains **0**.

## Next
Step8B: implement and engineering-verify this exact correction under the frozen two-file allowlist.