# AEGIS v0.34 Step11 鈥?Post-Training Observability & Mechanism-Adjudication Protocol Freeze

**Status:** PROTOCOL_FROZEN_NO_RESULTS_COMPUTED

This step prospectively freezes the analysis contract for the already-completed v0.34 formal
training evidence. It computes no mechanism result and does not authorize formal robustness
evaluation.

## Evidence boundary

Only the prospectively persisted Step10 M4qusli seed 42/43/44 training and `v034_observability`
artifacts are admissible. Missing required diagnostics are reported as
`NOT_ADJUDICABLE_FROM_FROZEN_EVIDENCE`; they may not be reconstructed after results are known.

## Frozen mechanism questions

### M1 鈥?Counterfactual target signal
Support requires, for **every seed**, `delta_u_abs_ge_0p01_fraction >= 0.10`,
`u_target_ge_0p60_fraction + u_target_le_0p40_fraction >= 0.10`, and
`u_target_std >= 0.05`.

### M2 鈥?Selector optimization signal
Support requires, for **every seed**, positive median selector gradient norm on non-neutral
steps and positive selector parameter L2 movement from initialization.

### M3 鈥?Selector output discrimination
Support requires, for **every seed**, `utility_probability_std >= 0.05`.
Utility-probability mean is descriptive for centering and does not replace the discrimination
criterion.

### M4 鈥?Intervention activation
Support requires, for **every seed**, `active_intervention_rate > 0`.

### M5 鈥?Failure localization
The frozen ordered localization is:

1. M1 not supported -> `INSUFFICIENT_TARGET_SIGNAL`
2. M1 supported and M2 not supported -> `SELECTOR_OPTIMIZATION_FAILURE`
3. M1/M2 supported and M3 not supported -> `SELECTOR_DISCRIMINATION_FAILURE`
4. M1/M2/M3 supported and M4 not supported -> `ACTIVATION_MAPPING_FAILURE`
5. M1-M4 supported -> `MECHANISM_ACTIVE_NO_PRE_EVAL_FAILURE_LOCALIZED`
6. Any prerequisite not adjudicable -> `NOT_ADJUDICABLE_FROM_FROZEN_EVIDENCE`

Localization is constrained mechanistic attribution, not proof of sole causal responsibility.

## Aggregation and prohibitions

Criteria are applied per seed using full stored precision. Cross-seed arithmetic means are
descriptive only and cannot override the frozen per-seed criteria. No post-hoc threshold changes
are allowed.

No retraining, additional seeds, checkpoint reselection, threshold search, architecture/loss
redesign, new quality conditions, formal robustness evaluation, official-test access, or
reconstruction of missing prospective diagnostics is permitted by this step.

## Evaluation authorization boundary

Step11 does **not** authorize formal evaluation. After the one-time Step12 mechanism computation
is frozen, any formal robustness evaluation requires a separate explicit authorization freeze.

**Official test samples accessed:** 0

**Next:** Step12 one-time post-training observability and mechanism-adjudication computation.
