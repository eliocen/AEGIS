# AEGIS v0.35 Step5A 鈥?Corrective-Design Dataflow Feasibility & Source-Allowlist Amendment

**Status:** DATAFLOW_FEASIBILITY_RESOLVED_ALLOWLIST_AMENDED_IMPLEMENTATION_NOT_PERFORMED

## Exact parent

- Step5 commit: `c658a33ab79b0ddb03533eccc1059b7b26a2b84d`
- Step5 JSON SHA256: `9E2A6CAD37A2E3903D28C0236C909B356C794DBA00A8A4D0CE00E4F5B318CFAA`
- Step5 Markdown SHA256: `2C69D80D1F3836180B68161F56C46A8EA7E6AE0304DC26BABAE4FE71E2361E20`

## Preserved Step5 scientific correction

Step5A does **not** replace or redesign the frozen correction.

The selected correction remains:

**V035_C1_CLASSIFIER_POSTERIOR_CONTEXT 鈥?Inference-Available Classifier-Posterior Context Augmentation**

The selector remains prospectively defined as 7 existing quality/compatibility
features plus three detached, label-free posterior-context features:

- reference class-1 posterior;
- candidate class-1 posterior;
- candidate-minus-reference class-1 posterior difference.

The 7 -> 10 selector-input contract, anti-leakage boundary, BCEWithLogits
objective, selector sigmoid, intervention mapping, M3 criterion and M4 criterion
are preserved.

## Feasibility finding

**FEASIBLE_WITH_EXPLICIT_TWO_STAGE_DATAFLOW_AMENDMENT**

Exact source inspection establishes that the current v0.33 selector is invoked
inside the alignment model before the runner computes classifier logits for the
stable-reference and candidate counterfactual embeddings.

Therefore, the frozen Step5 posterior context does not exist at the current
selector call site. Merely widening the controller from 7 to 10 inputs would be
insufficient.

This is an implementation/dataflow gap, not a refutation of the Step3 mechanism
finding or Step5 information correction.

## Amended two-stage dataflow

### Stage A 鈥?counterfactual context preparation

The alignment path computes aligned modalities, quality, compatibility and
interaction state. Using the **existing deterministic reference/candidate action
equations**, it constructs the stable-reference and candidate counterfactual
fused embeddings without requiring the learned selector output.

These counterfactual embeddings are returned for classification.

### Stage B 鈥?posterior-conditioned selector finalization

The runner reuses the existing classification model to obtain reference and
candidate logits.

Without labels it computes:

`p_ref_pos = softmax(ref_logits, dim=1)[:,1]`

`p_cand_pos = softmax(cand_logits, dim=1)[:,1]`

`delta_pos = p_cand_pos - p_ref_pos`

These three tensors are detached and supplied through an explicit v0.33
finalization interface.

The controller then constructs the frozen 10-D selector input, computes the
selector logit/probability/intervention gate, and the alignment path constructs
the final learned-intervention fused embedding. The runner then classifies that
final embedding normally.

The existing label-indexed reference/candidate logits remain usable for
training-time `u_target` construction only. Ground truth is not exposed to the
selector.

## Why this resolves the apparent circularity

The stable-reference and candidate counterfactual actions are deterministic
functions of the quality/compatibility evidence. They do not require the learned
selector gate.

They can therefore be constructed and classified **before** final selector
gating. The resulting label-free posterior context can then condition the
selector that determines how strongly to move from the stable reference toward
the candidate action.

No second classifier head is introduced.

## Amended source allowlist

The controlled implementation allowlist is amended to exactly:

1. `aegis/reliability/reliability_controller.py`
2. `aegis/alignment/model.py`
3. `scripts/run_fakeddit_ablation.py`
4. `tests/test_v033_utility_supervision_selector.py`
5. `tests/test_v034_observability_training_path.py`

The addition of `aegis/alignment/model.py` and promotion of the runner from
observability-only to explicit v0.33 dataflow orchestration are required by the
frozen feasibility finding.

This amendment does not authorize unrelated changes in those files.

## Preserved boundaries

Step5A does not authorize:

- implementation;
- training or retraining;
- new seeds;
- checkpoint reselection;
- threshold search/tuning;
- loss-family or utility-loss-weight changes;
- formal robustness evaluation;
- official-test access.

Official-test samples accessed: **0**.

## Next

**v0.35 Step6 鈥?Controlled Corrective Implementation**

Step6 may proceed only against the exact combined Step5 + Step5A contract and
must freeze implementation evidence and regression results before any corrective
training is authorized.
