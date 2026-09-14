# AEGIS v0.33 Step2 鈥?Exact Utility-Supervision and Selector-Learning Contract

## Primary architecture

`M4qusli` 鈥?**M4 Quality Utility-Supervised Learned Intervention**

The candidate intervention action is inherited unchanged from frozen v0.31
`M4qcesi`. v0.33 changes the utility-supervision, selector-learning, and gate
pathway first; it does not alter the candidate action space at Step2.

## Utility target

For each sample:

`delta_u = p_candidate_true - p_ref_true`

`u_target = clip(0.5 + delta_u / 0.20, 0, 1)`

This freezes 0.5 as neutral. A counterfactual gain of +0.01 maps to 0.55 and a
loss of -0.01 maps to 0.45.

## Selector objective

The selector uses BCE-with-logits against `u_target`, with a frozen magnitude
weight:

`w = 1 + 4 * min(abs(delta_u)/0.10, 1)`

`L_utility = mean(w * BCEWithLogits(selector_logit, u_target))`

The effective utility-loss weight is frozen at **0.50**. The transition
objective remains **0.0**.

## Intervention gate

- active threshold: **0.60**
- soft gate: `clip((utility_probability - 0.60)/0.40, 0, 1)`
- maximum intervention strength: **0.15**
- applied strength: `0.15 * soft_gate`
- below 0.60: exact no-intervention behavior
- no calibrated-risk multiplier in the v0.33 primary gate
- threshold search is prohibited

## Mandatory pre-training gates

Formal training is prohibited until the tensor contract, controlled
positive/negative target smoke, real one-batch learning path, runtime-effective
weight assertion, and inactive-default gate all pass.

## Formal hypotheses

v0.33 prospectively freezes hypotheses for target signal, selector learning,
utility discrimination, clean preservation, and graded robustness. These may
not be changed after formal evidence is exposed.

## Current boundary

No source modification, training, formal evaluation, checkpoint reselection,
threshold tuning, or official-test access is authorized by Step2.

The next action is **Step3 鈥?exact source-layout and inherited candidate-action
audit**.

Official test samples accessed: **0**.
