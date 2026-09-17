# AEGIS Research Core v1.0 鈥?Scientific Experiment Timeline

## v0.34 鈥?Observability and selector-discrimination failure

Step12 froze the key hypothesis outcomes: M1 `SUPPORTED`, M2 `SUPPORTED`, M3 `NOT_SUPPORTED`, M4 `NOT_SUPPORTED`, and M5 `SELECTOR_DISCRIMINATION_FAILURE`. The frozen M3 criterion required `utility_probability_std >= 0.05` per seed; M4 required `active_intervention_rate > 0` per seed.

Step13 closed the stage as `PRE_EVALUATION_MECHANISM_FAILURE_CONFIRMED`; formal evaluation was not authorized.

## v0.35 鈥?Prospective corrective mechanism

Steps1鈥? localized the problem and selected `V035_C1_CLASSIFIER_POSTERIOR_CONTEXT`, a minimal inference-available classifier-posterior context augmentation. Step5A froze the two-stage dataflow. Step6B implemented and verified the correction without training.

Steps7鈥? froze provenance, training protocol, and authorization. Step9 performed the authorized corrective training for seeds 42, 43, and 44. Best validation Macro-F1 was 0.8669, 0.8716, and 0.8610 respectively.

M3 remained `NOT_SUPPORTED`: utility-probability standard deviation was 0.016633, 0.012812, and 0.009565. M4 remained `NOT_SUPPORTED`: active-intervention rate was 0 for all three seeds. Selector parameter movement and gradient signal were observed.

Step10 closed v0.35 with `CORRECTION_NOT_SUPPORTED` and `PERSISTENT_SELECTOR_OUTPUT_DISCRIMINATION_FAILURE_AFTER_POSTERIOR_CONTEXT_CORRECTION`.

## v0.36 鈥?Failure-aware robustness characterization

Step1 prospectively required v0.35 M3 and M4 support for every formal seed before active-intervention robustness evaluation.

Step2 found Q1 `INELIGIBLE`, Q2 `OBSERVED_OPTIMIZATION_ACTIVATION_DECOUPLING`, Q3 `CONSISTENT_FAILURE_ACROSS_ALL_THREE_SEEDS`, Q4 `NOT_AVAILABLE_FROM_UNAMBIGUOUS_FROZEN_SUMMARY_FIELD`, and Q5 `FORMAL_ACTIVE_INTERVENTION_ROBUSTNESS_EVALUATION_INADMISSIBLE`.

Step3 closed v0.36 with robustness `NOT_COMPUTED` and disposition `ROBUSTNESS_EVALUATION_BLOCKED_BY_INACTIVE_SELECTOR_PREREQUISITE`. Classifier non-robustness and active adaptive robustness were not claimed.

## v0.37 鈥?Threat-intelligence scientific characterization

v0.37 characterized the evidence for analyst-oriented interpretation while preserving the distinction among classifier, reliability, selector, and intervention mechanisms. It closed with the supported posture `ANALYST DECISION-SUPPORT RESEARCH CHARACTERIZATION`.

Production capability, autonomous attribution, and national-security operational validation were not established.

## v0.38 鈥?Integration, reproducibility, and research release

v0.38 prospectively audited repository identity, evidence provenance, environment/dependencies, workflow reproducibility, tests/invariants, architecture/documentation consistency, claims/limitations, release manifest, official-test/data boundary, and v1 readiness.

R1鈥揜10 passed. v0.38 closed and authorized the AEGIS Research Core v1 freeze without additional training or scientific evaluation.

## AEGIS Research Core v1.0 鈥?Immutable scientific baseline

Release commit: `39492cf6d93987fdf9ebee29b95fdb826b8391e4`.

Annotated tag: `v1.0.0-research-core`.

The release preserves official-test samples accessed = **0** and production deployment as **NOT AUTHORIZED / NOT ESTABLISHED**.

Post-v1 work is packaging, documentation, reproducibility, figures, manuscript, demonstration, repository presentation, and separately governed future research. It does not mutate the v1.0 scientific baseline.
