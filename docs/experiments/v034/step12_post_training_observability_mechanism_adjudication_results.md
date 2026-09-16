# AEGIS v0.34 Step12 鈥?Post-Training Observability & Mechanism-Adjudication Results

**Status:** MECHANISM_ADJUDICATION_RESULTS_FROZEN

## Frozen decisions

- M1 鈥?Counterfactual target signal: **SUPPORTED**
- M2 鈥?Selector optimization signal: **SUPPORTED**
- M3 鈥?Selector output discrimination: **NOT_SUPPORTED**
- M4 鈥?Intervention activation: **NOT_SUPPORTED**
- M5 鈥?Failure localization: **SELECTOR_DISCRIMINATION_FAILURE**

## Per-seed evidence

### Seed 42

- `delta_u_abs_ge_0p01_fraction`: `0.38705263157894737`
- `delta_u_std`: `0.02431767644313861`
- `u_target_ge_0p60_fraction`: `0.1118842105263158`
- `u_target_le_0p40_fraction`: `0.12216842105263158`
- `u_target_neutral_0p45_0p55_fraction`: `0.6129473684210527`
- `u_target_std`: `0.11756215892347124`
- `selector_gradient_norm_median_non_neutral_steps`: `0.046311032663833344`
- `selector_parameter_l2_movement_from_initial`: `0.08252905305988768`
- `utility_probability_mean`: `0.49477014608791`
- `utility_probability_std`: `0.012056356727677615`
- `active_intervention_rate`: `0.0`

### Seed 43

- `delta_u_abs_ge_0p01_fraction`: `0.39482105263157896`
- `delta_u_std`: `0.024703236270153136`
- `u_target_ge_0p60_fraction`: `0.11518947368421052`
- `u_target_le_0p40_fraction`: `0.12314736842105263`
- `u_target_neutral_0p45_0p55_fraction`: `0.605178947368421`
- `u_target_std`: `0.11793867800924909`
- `selector_gradient_norm_median_non_neutral_steps`: `0.02864261777577873`
- `selector_parameter_l2_movement_from_initial`: `0.14235170802782893`
- `utility_probability_mean`: `0.4960478912974659`
- `utility_probability_std`: `0.010378722941974613`
- `active_intervention_rate`: `0.0`

### Seed 44

- `delta_u_abs_ge_0p01_fraction`: `0.39084`
- `delta_u_std`: `0.026278570536445185`
- `u_target_ge_0p60_fraction`: `0.11438`
- `u_target_le_0p40_fraction`: `0.12905`
- `u_target_neutral_0p45_0p55_fraction`: `0.60916`
- `u_target_std`: `0.12496533131221244`
- `selector_gradient_norm_median_non_neutral_steps`: `0.035981796893984515`
- `selector_parameter_l2_movement_from_initial`: `0.22143169428344184`
- `utility_probability_mean`: `0.4930366360729933`
- `utility_probability_std`: `0.01315188747884486`
- `active_intervention_rate`: `0.0`

## Scientific boundary

These results apply the exact Step11 frozen criteria to the prospectively persisted Step10
training observability. Cross-seed means are descriptive only and do not override per-seed
criteria. M5 is constrained mechanistic localization and is not proof of sole causal
responsibility.

No formal robustness evaluation, retraining, checkpoint reselection, threshold tuning, or
official-test access occurred. Step12 does not authorize formal evaluation. A separate explicit
scientific-disposition/authorization step is required.

**Official test samples accessed:** 0
