# AEGIS v0.32 Step1 鈥?Prospective Failure-Characterization Protocol

## Scope

This protocol prospectively freezes the analysis of the v0.31 utility-learning
failure. It uses only already-frozen training and validation evidence.

## Frozen questions

1. Does delta_u/u_target contain sufficient learning signal?
2. Is the candidate intervention beneficial often enough to learn utility?
3. Does the fixed utility-target transform compress targets near 0.5?
4. Did the utility selector actually move away from neutral initialization?
5. Does learned utility discriminate meaningful corruption/mismatch states?
6. Which failure category best explains the mechanism?

## Governance

No training, checkpoint reselection, new formal evaluation, threshold tuning,
post-exposure subset construction, or official-test access is permitted.

If a required diagnostic was not recorded in frozen artifacts, it is reported
NOT_AVAILABLE rather than regenerated.

The next step is a read-only evidence/schema preflight.
