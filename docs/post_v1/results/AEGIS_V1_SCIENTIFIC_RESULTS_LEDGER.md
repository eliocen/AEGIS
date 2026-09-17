# AEGIS Research Core v1.0 鈥?Scientific Results Ledger

## Interpretation rule

This ledger consolidates frozen results. `SUPPORTED` means a prospectively defined criterion was satisfied. `NOT_SUPPORTED` means it was evaluated but not satisfied. `NOT_COMPUTED` means the result was not computed. `CHARACTERIZED` is descriptive/mechanistic evidence within a bounded scope. None of these labels may be strengthened during post-v1 packaging.

| Version / stage | Question / hypothesis | Prospective criterion | Observed frozen result | Status | Authorized interpretation |
|---|---|---|---|---|---|
| v0.34 Step12 | M1 | Frozen v0.34 criterion | Criterion satisfied | SUPPORTED | M1 is supported within the frozen v0.34 protocol. |
| v0.34 Step12 | M2 | Frozen v0.34 criterion | Criterion satisfied | SUPPORTED | M2 is supported within the frozen v0.34 protocol. |
| v0.34 Step12 | M3 selector-output discrimination | `utility_probability_std >= 0.05` per seed | Criterion not satisfied | NOT_SUPPORTED | Required selector-output discrimination was not demonstrated. |
| v0.34 Step12 | M4 active intervention | `active_intervention_rate > 0` per seed | Criterion not satisfied | NOT_SUPPORTED | Required active intervention was not demonstrated. |
| v0.34 Step12/13 | M5 / failure localization | Frozen diagnostic decision | `SELECTOR_DISCRIMINATION_FAILURE`; pre-evaluation mechanism failure confirmed | CHARACTERIZED | Failure was localized to selector discrimination sufficiently to block formal evaluation. |
| v0.35 Steps 1鈥? | Corrective mechanism design | Prospectively frozen minimal correction | `V035_C1_CLASSIFIER_POSTERIOR_CONTEXT` selected and implemented | IMPLEMENTED / EVALUATED | Posterior-context augmentation was the frozen corrective intervention. |
| v0.35 Step9 | Corrective training validation performance | Frozen checkpoint rule: highest validation Macro-F1, tie-break lowest validation classification loss | seed42 0.8669; seed43 0.8716; seed44 0.8610 best validation Macro-F1 | CHARACTERIZED | Validation performance may be reported with dataset/protocol qualification; it is not formal robustness evidence. |
| v0.35 Step9 | M3 after correction | `utility_probability_std >= 0.05` per seed | seed42 0.016633; seed43 0.012812; seed44 0.009565 | NOT_SUPPORTED | Posterior context did not produce the required selector-output discrimination. |
| v0.35 Step9 | M4 after correction | `active_intervention_rate > 0` per seed | seed42 0; seed43 0; seed44 0 | NOT_SUPPORTED | No qualifying active intervention was observed under the frozen criterion. |
| v0.35 Step9 | Was optimization signal absent? | Frozen diagnostic evidence | Nonzero median selector gradient norms and selector parameter L2 movement approximately 0.0885, 0.1508, 0.1488 | CHARACTERIZED | Optimization signal existed despite failure to achieve required output discrimination/activation. |
| v0.35 Step10 | Corrective mechanism conclusion | M3/M4 support required | `CORRECTION_NOT_SUPPORTED` | NOT_SUPPORTED | The v0.35 posterior-context correction is not scientifically supported as resolving the selector failure. |
| v0.35 Step10 | Failure disposition | Frozen closure decision | `PERSISTENT_SELECTOR_OUTPUT_DISCRIMINATION_FAILURE_AFTER_POSTERIOR_CONTEXT_CORRECTION` | CHARACTERIZED | Persistent selector-output discrimination failure remained after correction. |
| v0.36 Step2 | Q1 selector activation eligibility | v0.35 M3 AND M4 supported for every formal seed | Prerequisite failed | NOT_SUPPORTED / INELIGIBLE | Active-intervention robustness evaluation was not eligible to proceed. |
| v0.36 Step2 | Q2 optimization/activation relationship | Frozen characterization question | `OBSERVED_OPTIMIZATION_ACTIVATION_DECOUPLING` | CHARACTERIZED | Optimization occurred without qualifying selector activation. |
| v0.36 Step2 | Q3 seed consistency | Frozen characterization question | `CONSISTENT_FAILURE_ACROSS_ALL_THREE_SEEDS` | CHARACTERIZED | Failure pattern was consistent across the three formal corrective-training seeds. |
| v0.36 Step2 | Q4 requested summary quantity | Must exist as unambiguous frozen summary field | `NOT_AVAILABLE_FROM_UNAMBIGUOUS_FROZEN_SUMMARY_FIELD` | NOT_COMPUTED | No value should be reconstructed or inferred post hoc. |
| v0.36 Step2 | Q5 formal active-intervention robustness | Requires eligible active selector | `FORMAL_ACTIVE_INTERVENTION_ROBUSTNESS_EVALUATION_INADMISSIBLE` | NOT_COMPUTED | No formal active-intervention robustness result exists. |
| v0.36 closure | Robustness disposition | Eligibility boundary | `ROBUSTNESS_EVALUATION_BLOCKED_BY_INACTIVE_SELECTOR_PREREQUISITE` | NOT_COMPUTED | This is a blocked evaluation, not evidence of classifier non-robustness. |
| v0.37 | Threat-intelligence/scientific posture | Frozen characterization protocol T1鈥揟6 | `ANALYST DECISION-SUPPORT RESEARCH CHARACTERIZATION` | CHARACTERIZED | AEGIS may be presented as bounded analyst decision-support research. |
| v0.38 | Integration/reproducibility/research release | R1鈥揜10 release-readiness gates | R1鈥揜10 PASS | SUPPORTED | Frozen repository/evidence package satisfied its defined research-release readiness gates. |
| v1.0 | Research release | Exact release provenance | Immutable `v1.0.0-research-core`; official-test samples accessed = 0 | CHARACTERIZED | v1.0 is the immutable reproducible scientific baseline, not a production certification. |

## Global result boundary

The strongest supported post-v1 description is a reproducible, scientifically bounded research core with characterized analyst decision-support relevance. The frozen record does not establish active adaptive robustness, production readiness, autonomous attribution, or completed future-modality capabilities.
