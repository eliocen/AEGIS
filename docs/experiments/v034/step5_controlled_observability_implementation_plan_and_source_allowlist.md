# AEGIS v0.34 Step5 鈥?Controlled Observability Implementation Plan & Exact Source Allowlist

**Status:** `CONTROLLED_OBSERVABILITY_IMPLEMENTATION_PLAN_ALLOWLIST_FROZEN`

## Purpose
The next implementation is diagnostic-only. It closes GAP1-GAP4 and creates engineering evidence for GAP5 without changing the inherited v0.33 scientific mechanism.

## Exact Step6 allowlist
1. `scripts/run_fakeddit_ablation.py`
2. `tests/test_v034_observability_training_path.py`
3. `tests/test_v034_observability_recoverability.py`

The controller, model, v0.33 evaluator, and inherited v0.33 regression tests are protected and are not allowlisted.

## Mechanism invariants
Candidate action, stable-reference path, utility target transform, selector architecture/inputs, selector loss, effective utility weight, intervention mapping, inactive-default identity behavior, checkpoint selection and formal evaluator semantics remain unchanged.

## Required new evidence
The runner must persist per-training-sample counterfactual records and per-optimization-step target/gradient/update records. Epoch summaries must include exact target-distribution statistics required by Step1. Effective runtime weights must be persisted from the live training computation.

Diagnostics must come from the live forward/training path. A second model forward pass may not be introduced merely to reconstruct them.

## Engineering verification
Step6 must verify positive/negative and non-neutral target examples; exact counterfactual arithmetic; inherited target-transform parity; same-step non-neutral target and selector gradient evidence; selector parameter update/movement; effective runtime weight; and exact inactive/default identity.

A separate recoverability verifier must recompute the planned statistics from persisted engineering artifacts alone.

## Formal boundary
Passing Step6 does **not** authorize formal training. It proves observability and mechanism parity only. A later frozen scientific mechanism decision and explicit training authorization remain required.

Official test access remains **0**.

Next: **v0.34 Step6 鈥?controlled observability implementation and engineering verification**.
