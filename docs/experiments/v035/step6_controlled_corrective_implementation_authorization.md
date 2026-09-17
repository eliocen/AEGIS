# AEGIS v0.35 Step6 鈥?Controlled Corrective Implementation Authorization

**Status:** CONTROLLED_IMPLEMENTATION_AUTHORIZED_NOT_YET_EXECUTED

This freeze authorizes the exact controlled source modification required by the
combined Step5 + Step5A contract. It does not itself modify implementation code.

## Frozen correction

**V035_C1_CLASSIFIER_POSTERIOR_CONTEXT**

Two-stage dataflow:

**counterfactual context -> classifier posteriors -> detached posterior context -> selector finalization**

The selector input changes only from **7 to 10** features.

## Exact source allowlist

1. `aegis/reliability/reliability_controller.py`
2. `aegis/alignment/model.py`
3. `scripts/run_fakeddit_ablation.py`
4. `tests/test_v033_utility_supervision_selector.py`
5. `tests/test_v034_observability_training_path.py`

No other tracked implementation file is authorized.

## Controller implementation contract

The v0.33 selector must receive the frozen three posterior features in addition
to its existing seven quality/compatibility features. The posterior features are
label-free, detached and validated. Hidden width, output logit, sigmoid,
counterfactual target construction and intervention mapping remain unchanged.

## Two-stage alignment contract

Stage A constructs the stable-reference and candidate counterfactual fused
embeddings without learned selector finalization.

The existing classifier then provides the reference/candidate logits needed for
label-free posterior context.

Stage B receives only the detached posterior context and finalizes the 10-D
selector intervention and final fused embedding.

## Runner contract

The runner orchestrates Stage A -> counterfactual classification -> posterior
context -> Stage B -> final classification for M4qusli only.

The same reference/candidate logits are reused for the existing training-time
label-indexed `u_target`. No additional classifier is introduced.

## Fail-closed verification

The implementation must not be committed unless:

- all five files compile;
- posterior semantics and detachment tests pass;
- the [B,10] selector contract passes;
- two-stage integration tests pass;
- existing selector/observability focused tests pass;
- only the frozen allowlist is modified.

## Boundary

- Controlled implementation: **AUTHORIZED**
- Implementation performed by this authorization freeze: **NO**
- Corrective training: **NOT AUTHORIZED**
- New seeds: **NOT AUTHORIZED**
- Threshold tuning: **NOT AUTHORIZED**
- Formal evaluation: **NOT AUTHORIZED**
- Official-test samples accessed: **0**

## Next

**v0.35 Step6B 鈥?Execute and Verify Controlled Corrective Implementation**
