# AEGIS v0.35 Step5 鈥?Prospective Minimal Corrective-Design Freeze

**Status:** MINIMAL_CORRECTIVE_DESIGN_FROZEN_IMPLEMENTATION_NOT_AUTHORIZED

## Exact parent

- Step4 commit: `6032bccfea367078fda4e93225c08ab4662cf264`
- Step4 JSON SHA256: `E28FBC0920E56B377197E72E8B29DB412828615FB5847D4B54E7A94E4A8C0556`
- Step4 Markdown SHA256: `87665A36636615FDD4C99C123D0F244586A6D98D236F64DA6B5775B6E715F962`

## Frozen problem

- D1: **NO_EXPLANATORY_DEFECT_IDENTIFIED**
- D2: **COUPLING_DEFECT_IDENTIFIED**
- Mechanism: **SELECTOR_INFORMATION_TARGET_MISMATCH**
- Primary problem: **SELECTOR_OUTPUT_DISCRIMINATION**
- Secondary downstream question: **INTERVENTION_ACTIVATION_AFTER_DISCRIMINATION_IS_RESTORED**

## Prospectively selected correction

**V035_C1_CLASSIFIER_POSTERIOR_CONTEXT 鈥?Inference-Available Classifier-Posterior Context Augmentation**

The selector currently receives seven quality/compatibility-derived features, while
its supervision target is defined from candidate-versus-reference classifier
behavior. Step5 prospectively selects the minimum non-label bridge: add detached
classifier-posterior context already computable from the reference and candidate
logits available to the controller.

The three added features are:

1. `reference_positive_probability = softmax(reference_logits, dim=1)[:,1]`
2. `candidate_positive_probability = softmax(candidate_logits, dim=1)[:,1]`
3. `candidate_minus_reference_positive_probability = candidate_positive_probability - reference_positive_probability`

All three are detached before selector feature concatenation.

The selector input therefore changes prospectively from **7 to 10 dimensions**.
Its hidden width remains 16 and its output remains one scalar logit per sample.

## Why this is admissible

The added quantities are model-derived and computable without the ground-truth
label. They therefore respect the intended inference information boundary.

The correction does **not** expose:

- ground-truth class identity;
- ground-truth-indexed `delta_u`;
- `u_target`;
- official-test labels or label-derived test quantities.

Detachment is mandatory so selector supervision does not create a new gradient
route into the classifier through the added posterior context.

This is an anti-leakage/inference-availability argument only. It does not claim
that the correction will restore selector discrimination.

## Frozen tensor contract

- reference logits: `[B,2]`
- candidate logits: `[B,2]`
- reference positive probability: `[B]`
- candidate positive probability: `[B]`
- signed positive-probability delta: `[B]`
- existing selector features: `[B,7]`
- corrected selector features: `[B,10]`
- selector logit: `[B]`
- utility probability: `[B]`
- utility target: `[B]`

## Preserved mechanism

Step5 does not change:

- multimodal encoders;
- classifier heads;
- counterfactual `u_target` construction;
- BCEWithLogits selector objective;
- selector sigmoid output semantics;
- intervention threshold/mapping;
- inherited M3 or M4 criteria.

The inherited M3 reference remains `utility_probability_std >= 0.05` per seed.
The inherited M4 reference remains `active_intervention_rate > 0` per seed.

M4 remains downstream of M3.

## Prospective source allowlist

Implementation, when separately authorized, is limited to:

- `aegis/reliability/reliability_controller.py` 鈥?posterior-context construction and selector input width 7 -> 10;
- `tests/test_v033_utility_supervision_selector.py` 鈥?tensor, anti-leakage and selector regression tests;
- `tests/test_v034_observability_training_path.py` 鈥?observability/training-path regression tests.

`scripts/run_fakeddit_ablation.py` may be changed only if later required for observability or assertion
wiring. Step5 does not authorize changing loss family, utility-loss weight,
threshold, seeds, training schedule or formal evaluation.

## Required regression contract

The implementation must prove the 10-D selector contract, exact posterior-feature
semantics, label independence, gradient detachment, absence of `u_target` and
ground-truth class identity from selector inputs, and preservation of existing
target/loss/sigmoid/intervention semantics.

## Current boundary

- Exact correction selected prospectively: **YES**
- Implementation: **NOT AUTHORIZED**
- Corrective training: **NOT AUTHORIZED**
- New seeds: **NOT AUTHORIZED**
- Checkpoint reselection: **NOT AUTHORIZED**
- Threshold tuning: **NOT AUTHORIZED**
- Formal evaluation: **NOT AUTHORIZED**
- Official-test samples accessed: **0**

## Next

**v0.35 Step6 鈥?Minimal Corrective Implementation Authorization and Controlled Implementation**

Step6 must verify this exact design and source boundary before modifying any
implementation file.
