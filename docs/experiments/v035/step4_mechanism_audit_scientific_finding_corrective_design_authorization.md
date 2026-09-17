# AEGIS v0.35 Step4 鈥?Mechanism-Audit Scientific Finding and Corrective-Design Authorization Boundary

**Status:** SCIENTIFIC_FINDING_FROZEN_CORRECTIVE_DESIGN_AUTHORIZED

## Exact parent

- Step3 commit: `45e70ec0c007a67395a94892607b29ce32430eb6`
- Step3 source-manifest SHA256: `779232EB59E2265F78E56A4337403CBC5BBB1A768D1CBF63464A44D16E7F5E62`
- Step3 JSON SHA256: `A73E5BFB6EE15F41FA8F897129FCAF1D41162A14F376D33EEBD52B3106642732`
- Step3 Markdown SHA256: `02BEAABF74F1765BC7B9F0F082F79C4EA088F87D66D1FA7163C529AE716C4253`

## Frozen inherited mechanism state

- M1 鈥?counterfactual target signal: **SUPPORTED**
- M2 鈥?selector optimization signal: **SUPPORTED**
- M3 鈥?selector output discrimination: **NOT_SUPPORTED**
- M4 鈥?intervention activation: **NOT_SUPPORTED**
- M5 鈥?failure localization: **SELECTOR_DISCRIMINATION_FAILURE**

## Frozen Step3 mechanism-audit finding

- D1 selector output path: **NO_EXPLANATORY_DEFECT_IDENTIFIED**
- D2 objective-to-output coupling: **COUPLING_DEFECT_IDENTIFIED**
- D2 mechanism: **SELECTOR_INFORMATION_TARGET_MISMATCH**
- Joint outcome: **D1_NO_EXPLANATORY_DEFECT__D2_COUPLING_DEFECT**

The source audit therefore localizes the next corrective-design question to
**target-information availability/coupling**. Current evidence does not require
redesign of the selector's sigmoid output path or the downstream intervention
mapping.

This is an explanatory localization, not sole causal proof, and Step4 does not
claim that any particular correction will restore M3 or M4.

## D3 authorization

Step4 authorizes **prospective minimal corrective design only**.

The design activity may determine the minimum inference-available representation
needed to reduce the frozen target-information mismatch, compare bounded design
alternatives, define exact tensor/interface semantics, define regression and
observability requirements, and prospectively freeze one correction before
implementation.

Step4 does **not** authorize implementation, training, new seeds, checkpoint
reselection, threshold search/tuning, formal robustness evaluation, official-test
access, or post-correction M3/M4 adjudication.

## Anti-leakage boundary

The correction must respect the intended inference information boundary.

The following may not be supplied directly as inference-time selector features:

- ground-truth class identity;
- ground-truth-indexed `delta_u`;
- `u_target`;
- any target quantity requiring the ground-truth label;
- official-test labels or official-test-derived quantities.

Model-derived quantities such as classifier predictions are **not automatically
authorized** by this statement. Their exact admissibility, semantics, gradient
behavior and inference availability must be prospectively justified and frozen
in the minimal corrective-design step.

## Minimality boundary

The correction must address the frozen D2 mechanism without broad unrelated
redesign. Unless the prospective design demonstrates necessity, it preserves the
multimodal encoders, classifier architecture, selector sigmoid semantics,
intervention mapping and unrelated reliability mechanisms.

The inherited M3 reference criterion
`utility_probability_std >= 0.05` per seed remains preserved and is not
weakened because v0.34 failed. The inherited M4 reference
`active_intervention_rate > 0` per seed is likewise preserved.

M4 remains downstream of M3. The v0.34 evidence did not independently establish
an activation-mapping failure.

## Current authorization boundary

- Corrective design: **AUTHORIZED**
- Exact correction selected: **NO**
- Implementation modification: **NO**
- Corrective training: **NO**
- Formal evaluation: **NO**
- Official-test samples accessed: **0**

## Next

**v0.35 Step5 鈥?Prospective Minimal Corrective-Design Freeze**

Before implementation, Step5 must freeze one exact minimal correction, its
mechanism-linked rationale, inference-availability/anti-leakage proof, source
allowlist, tensor/interface contract, regression-test contract and
post-correction observability contract.
