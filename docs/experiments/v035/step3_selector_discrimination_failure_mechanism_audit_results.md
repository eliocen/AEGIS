# AEGIS v0.35 Step3 鈥?Read-Only Selector Discrimination Failure-Mechanism Audit Results

**Status:** SOURCE_AUDIT_COMPLETED_RESULTS_FROZEN

## Parent

- Step2 commit: `7939ad117002a38be2f7983d2906ffb960e27604`
- Step2 JSON SHA256: `D7D916F91684D9A5F4A4467FB8C12CE801F0D1AECB7367C383CC09FE9F53C9D7`
- Step2 Markdown SHA256: `BB40C35421786181465BFD15A4E15EC39BEE96E2D0A6619A0D73FDE2A3D59DE3`
- Substantive source-manifest SHA256: `779232EB59E2265F78E56A4337403CBC5BBB1A768D1CBF63464A44D16E7F5E62`

## D1 鈥?Selector Output-Path Audit

**Adjudication: NO_EXPLANATORY_DEFECT_IDENTIFIED**

The frozen path is:

`7-D quality/compatibility-derived features -> Linear(7,16) -> ReLU -> Linear(16,1) -> selector_logit -> sigmoid -> utility_probability -> intervention threshold/gate`.

No additional selector-output temperature, batch/sample reduction, detach/no-grad,
or probability clamp was identified between the learned selector logit and
per-sample utility probability. The sigmoid mapping is semantically consistent
with the BCE-with-logits objective applied to the same selector logit.

This does not prove that no runtime or learnability mechanism exists; it means
the prospectively enumerated D1 static output-path defects were not identified.

## D2 鈥?Objective-to-Output Coupling Audit

**Adjudication: COUPLING_DEFECT_IDENTIFIED**

**Mechanism: SELECTOR_INFORMATION_TARGET_MISMATCH**

The target and predictor do not have equivalent information boundaries.

The utility target is constructed from the difference between candidate and
reference classifier probability for the ground-truth class and is transformed
into `u_target`. In contrast, the selector receives seven
quality/compatibility-derived features: text quality, vision quality,
compatibility, quality deficit, compatibility deficit, quality disagreement and
risk. It does not receive the reference/candidate classifier logits,
ground-truth class identity, `delta_u`, or an equivalent
candidate-versus-reference utility signal.

The objective wiring itself is intact: BCEWithLogits directly supervises the
same `selector_logit` that is sigmoid-mapped downstream, the weighted utility
loss enters total loss, gradients are backpropagated, and the optimizer steps.

The defect therefore concerns **information-to-target coupling**, not a broken
gradient path. The selector can receive distinct or opposing counterfactual
utility targets without receiving the discriminative variables from which those
targets were computed. Under BCE supervision, that information bottleneck is
directionally capable of producing conditional-mean predictions near the
neutral region while gradients remain nonzero and parameters move.

This is a structurally plausible explanatory mechanism, not sole causal proof,
and it does not authorize a particular correction.

## Joint outcome

`D1_NO_EXPLANATORY_DEFECT__D2_COUPLING_DEFECT`

The next scientific question is therefore how to make the **minimum
prospectively justified correction to target-information availability/coupling**
without changing unrelated architecture or weakening the inherited M3
criterion.

## Boundary

- Corrective implementation selected: **NO**
- Implementation modified: **NO**
- Training: **NO**
- Formal evaluation: **NO**
- Official-test samples accessed: **0**

## Next

**v0.35 Step4 鈥?mechanism-audit scientific finding and corrective-design authorization boundary.**
