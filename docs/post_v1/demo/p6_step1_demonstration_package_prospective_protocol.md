# AEGIS Post-v1 P6 Step1 鈥?Demonstration Package Prospective Protocol

## Status

**POST_V1_P6_STEP1_DEMONSTRATION_PACKAGE_PROSPECTIVE_PROTOCOL_FROZEN**

## Purpose

P6 will package AEGIS Research Core v1.0 for a bounded research demonstration. The demonstration is intended to make the architecture, evidence model, frozen results, provenance, failure analysis, and supported research behavior understandable without converting the research core into a production or autonomous intelligence claim.

The authoritative immutable release remains `v1.0.0-research-core` at commit `39492cf6d93987fdf9ebee29b95fdb826b8391e4`.

## Demonstration modes

**D1 鈥?Architecture and dataflow walkthrough.** Explain multimodal input processing, fusion/classification, quality and compatibility evidence, reliability, selector behavior, intervention mapping, and evidence outputs.

**D2 鈥?Frozen evidence and result exploration.** Present only already-frozen scientific results. Validation Macro-F1 is `0.8669`, `0.8716`, and `0.8610` for seeds 42鈥?4. Utility-probability standard deviation is `0.016633`, `0.012812`, and `0.009565` against M3 `utility_probability_std >= 0.05`, therefore M3 is `NOT_SUPPORTED`. Active intervention rate is `0`, `0`, and `0` against M4 `active_intervention_rate > 0`, therefore M4 is `NOT_SUPPORTED`. The v0.35 correction `V035_C1_CLASSIFIER_POSTERIOR_CONTEXT` is `CORRECTION_NOT_SUPPORTED`. v0.36 active-intervention robustness is `NOT_COMPUTED`.

**D3 鈥?Reproducibility and provenance walkthrough.** Demonstrate release identity, evidence hashes, frozen workflow records, claim traceability, and the distinction between identity/workflow reconstruction and a separately authorized scientific rerun.

**D4 鈥?Bounded inference demonstration, conditional.** A live or recorded inference demonstration may be specified only if an already-existing v1.0 interface and frozen artifacts support it. Such a demonstration must not access the official test set, train a model, change weights, create a new benchmark result, or be presented as production validation. If no admissible existing interface is established, D4 must be recorded as `NOT_INCLUDED`.

**D5 鈥?Failure and limitation explanation.** The demonstration must explicitly show that classifier performance and selector/intervention behavior are separate scientific questions and must preserve the selector-discrimination failure, inactive intervention pathway, and robustness-admissibility boundary.

## Mandatory demonstration surfaces

The package must expose the research-core identity; architecture/dataflow; classifier, reliability, selector, and intervention separation; frozen v0.35 results; v0.36 `NOT_COMPUTED` robustness state; evidence/provenance traceability; and limitations.

## Prohibited interpretations

The demonstration must not claim production readiness, autonomous attribution, autonomous threat determination, demonstrated active adaptive robustness, classifier non-robustness inferred from selector failure, official-test performance, state-of-the-art performance, or comparative superiority.

Future AEGIS-Vision, AEGIS-Audio, AEGIS-Video, multilingual/cross-lingual, and broader multimodal integration remain future research and must not be displayed as completed v1.0 capabilities.

## Execution boundary

P6 Step1 is protocol-only. Demo execution: **NONE**. Model execution: **NONE**. Training: **NONE**. New evaluation: **NONE**. Official test samples: **0**.

A later bounded inference demonstration, if admissible, requires an explicit source/interface and artifact audit before execution.

## Planned P6 progression

P6 Step2 will inventory demonstrable v1.0 evidence and interfaces and draft the non-executing demonstration package. A subsequent step may determine whether D4 bounded inference is admissible. Final P6 closure will audit every demonstrated capability against the immutable v1.0 evidence boundary.
