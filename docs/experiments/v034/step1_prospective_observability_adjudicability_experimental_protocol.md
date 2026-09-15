# AEGIS v0.34 Step1 鈥?Prospective Observability, Adjudicability & Experimental Protocol Freeze

**Status:** `PROSPECTIVE_OBSERVABILITY_ADJUDICABILITY_PROTOCOL_FROZEN`

## Purpose
v0.34 must distinguish target informativeness, selector optimization/learning, utility discrimination/calibration, intervention mapping, and clean-performance preservation without repeating the v0.33 H1/H2 evidence-granularity failure.

## Formal mechanism questions
- **Q1 Target informativeness:** is counterfactual utility supervision non-neutral, heterogeneous and condition-sensitive?
- **Q2 Selector optimization:** does the selector receive non-trivial gradient signal and move when non-neutral supervision exists?
- **Q3 Selector learning/calibration:** does predicted utility track the target and separate beneficial from harmful intervention opportunities?
- **Q4 Condition discrimination:** does utility order matched, degraded, mismatched and catastrophic regimes in the intended direction?
- **Q5 Intervention mapping:** conditional on informative utility, does intervention activate selectively while retaining exact inactive/default behavior?
- **Q6 Clean safeguard:** is the inherited clean-performance safeguard retained?

## Prospective evidence requirement
Formal training evidence must be persisted at **sample**, **optimization-step**, **epoch-summary**, **validation-condition**, and **primary validation-sample** granularity. Terminal output is not evidence.

Training sample records must include `p_ref_true`, `p_candidate_true`, `delta_u`, `u_target`, `utility_probability`, gates and activation. Optimization-step records must align non-neutral-target counts with selector gradient norm, utility loss, selector parameter movement and effective runtime weights.

## Frozen aggregation principles
For target-signal adjudication, sample-level training records across all authorized primary formal runs form the formal population. Epoch standard deviations may not be averaged as a substitute. A non-neutral target is prospectively defined by `abs(delta_u) >= 0.01`. The neutral target interval is `u_target in [0.45,0.55]`.

Selector-gradient adjudication uses only optimization steps with at least one non-neutral target. Parameter movement is measured from exact selector initialization. Formal comparisons use full stored precision.

## Missingness rule
Any criterion missing evidence at its required granularity is `NOT_ADJUDICABLE_FROM_FROZEN_EVIDENCE`. Missingness may not be converted to `NOT_SUPPORTED` and may not be repaired by retraining after formal results are exposed.

## Hard pre-training gates
1. Diagnostic schema contract.
2. Target observability and exact counterfactual consistency.
3. Optimization-step target/gradient/parameter-update alignment.
4. Effective runtime-weight assertion.
5. Exact inactive/default intervention assertion.
6. Recoverability: all planned formal mechanism statistics must be computable from persisted engineering artifacts alone.

Failure of any gate prohibits formal v0.34 training.

## Boundary
Step1 selects no successor architecture, candidate action, target equation, selector loss, intervention threshold/strength, comparator roster or formal numerical support thresholds. Those items must be prospectively frozen before formal training.

No implementation, training, formal evaluation, checkpoint reselection, threshold search or official-test access occurs at Step1. Official test samples accessed remain **0**.

Next: **v0.34 Step2 鈥?inherited-source and failure-mechanism evidence audit prior to exact mechanism selection**.
