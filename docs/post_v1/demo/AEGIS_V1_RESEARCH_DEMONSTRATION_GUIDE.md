# AEGIS v1.0 鈥?Research Demonstration Guide

## Demonstration identity

This package demonstrates the immutable **AEGIS Research Core v1.0**, release tag `v1.0.0-research-core`, commit `39492cf6d93987fdf9ebee29b95fdb826b8391e4`.

It is a research demonstration, not a production deployment or autonomous intelligence system.

## Recommended walkthrough

1. **Research problem and architecture (D1).** Present multimodal inputs, fusion/classification, quality and compatibility evidence, reliability, selector behavior, intervention mapping, and evidence outputs. Emphasize that classification and adaptive intervention are separate mechanisms.
2. **Frozen results (D2).** Present validation Macro-F1 `0.8669`, `0.8716`, `0.8610`. Then present selector discrimination: utility-probability standard deviation `0.016633`, `0.012812`, `0.009565` versus M3 `utility_probability_std >= 0.05`, yielding `NOT_SUPPORTED`. Present active intervention `0`, `0`, `0` versus M4 `active_intervention_rate > 0`, yielding `NOT_SUPPORTED`.
3. **Corrective mechanism and failure (D5).** Explain `V035_C1_CLASSIFIER_POSTERIOR_CONTEXT`, its label-free detached posterior context, and the frozen result `CORRECTION_NOT_SUPPORTED`. Preserve the disposition that selector output discrimination remained inadequate after the correction.
4. **Robustness admissibility (D5).** Explain that v0.36 active-intervention robustness is `NOT_COMPUTED`. This is not zero robustness and does not establish classifier non-robustness.
5. **Provenance and reproducibility (D3).** Show the immutable release identity, evidence hashes, P1鈥揚5 package lineage, and the distinction between identity/workflow reconstruction and separately authorized scientific reruns.
6. **Capability boundary.** State explicitly that production readiness, autonomous attribution, autonomous threat determination, demonstrated active adaptive robustness, official-test performance, and comparative superiority are not established.

## Live inference status

D4 bounded inference is **NOT AUTHORIZED PENDING EXPLICIT INTERFACE AND ARTIFACT CONTRACT AUDIT**. Filename-level discovery is insufficient to prove a stable inference contract, compatible frozen checkpoint, valid input contract, and non-evaluative output path.

No live inference should be presented as part of the frozen P6 package unless a later fail-closed audit explicitly authorizes it.

## Future research

AEGIS-Vision, AEGIS-Audio, AEGIS-Video, multilingual/cross-lingual capability, and broader multimodal integration remain post-v1 research directions.
