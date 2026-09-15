# AEGIS v0.34 Step0 鈥?Successor Scientific Question & Failure-Boundary Inheritance

**Status:** `SUCCESSOR_QUESTION_AND_FAILURE_BOUNDARY_FROZEN`

## Parent boundary
v0.34 begins only from the scientifically closed `v0.33.0` release at commit `93e0b376ce383856f7f0353ab6500ad9d9d621ee`.

Inherited v0.33 results:
- H1 target signal 鈥?`NOT_ADJUDICABLE_FROM_FROZEN_EVIDENCE`
- H2 selector learning 鈥?`NOT_ADJUDICABLE_FROM_FROZEN_EVIDENCE`
- H3 utility discrimination 鈥?`NOT_SUPPORTED`
- H4 clean safeguard 鈥?`SUPPORTED`
- H5 graded robustness 鈥?`NOT_SUPPORTED`

## Primary scientific question
Why did prospectively defined utility supervision fail to produce discriminative and graded-robust intervention behavior while preserving clean performance, and which observable mechanism boundary鈥攖arget informativeness, selector learnability/calibration, evidence representation, or intervention mapping鈥攁ccounts for that failure?

## Failure boundary inherited from v0.33
v0.34 must preserve the distinction between observed failure and causal attribution. v0.33 establishes failure of utility discrimination and graded robustness, but does not identify target informativeness, selector optimization, calibration, evidence representation, or intervention mapping as the unique cause.

The clean safeguard is a positive inherited constraint: a successor must not discard the stable clean-reference behavior without prospective justification.

## Mandatory observability inheritance
Before any formal training, v0.34 must prospectively guarantee machine-readable evidence at the granularity required for later adjudication, including sample-level counterfactual targets; optimization-step selector gradients aligned with target neutrality; selector parameter movement; utility probability/calibration; intervention activation; effective runtime weights; and validation-condition diagnostics.

Missing required formal-run evidence remains `NOT_AVAILABLE`; it may not be reconstructed by retraining after formal results are exposed.

## Step0 prohibition
Step0 selects no architecture, loss, target transform, or intervention threshold. It performs no implementation, training, formal evaluation, checkpoint reselection, threshold search, or official-test access.

Next: **v0.34 Step1 鈥?prospective observability, adjudicability, and experimental protocol freeze**.
