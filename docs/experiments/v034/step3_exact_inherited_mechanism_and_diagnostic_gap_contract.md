# AEGIS v0.34 Step3 鈥?Exact Inherited-Mechanism & Diagnostic-Gap Contract

**Status:** `EXACT_INHERITED_MECHANISM_DIAGNOSTIC_GAP_CONTRACT_FROZEN`

## Inheritance boundary
The v0.33 stable-reference/no-intervention behavior is preserved as a hard successor constraint because the frozen clean safeguard was supported. Counterfactual stable-versus-candidate observability and existing machine-readable utility/gate diagnostics are preserved and extended rather than redesigned from scratch.

## What is not yet selected
Step3 selects no new architecture, utility-target equation, selector objective, evidence representation, intervention threshold/strength/mapping, or candidate action.

Target transform, selector supervision/calibration, evidence representation and intervention mapping are scientifically eligible for later modification only under an exact prospective contract. Candidate action is only conditionally eligible: it must not be changed merely because v0.33 H3/H5 failed.

## Exact diagnostic gaps
1. **Training-sample counterfactuals:** persist complete per-occurrence `p_ref_true`, `p_candidate_true`, `delta_u`, `u_target`, `utility_probability`, gates and activation with epoch/optimization-step/sample identity.
2. **Target-gradient alignment:** persist non-neutral-target count/fraction and selector gradient on the same optimization step.
3. **Parameter trajectory:** persist selector update-this-step and movement-from-initial at optimization-step granularity.
4. **Runtime-weight provenance:** assert effective loss weights from the actual training path.
5. **Recoverability:** prove Q1-Q6 statistics can be computed from engineering artifacts without rerunning training.

## Causal boundary
Target informativeness, selector optimization, selector learning/calibration, evidence representation and intervention mapping remain unresolved. H3/H5 failure does not uniquely identify any one of them. The clean-reference path remains `PRESERVE`.

## Next operation
**v0.34 Step4 鈥?exact inherited mechanism source extraction and gradient/diagnostic-path audit.**

Step4 remains read-only. It should extract the precise inherited M4qusli code regions for candidate action, target transform, selector loss, utility probability, intervention mapping, optimizer/gradient boundaries and diagnostics before any successor mechanism is selected.
