# AEGIS Post-v1 P6 Step4 鈥?D4 Admissibility Decision and Demonstration Package Closure

## Closure status

**POST_V1_P6_DEMONSTRATION_PACKAGE_CLOSED**

## Included demonstration modes

- **D1 鈥?Architecture and dataflow walkthrough:** included as a documentary research demonstration.
- **D2 鈥?Frozen evidence and result exploration:** included as a documentary research demonstration.
- **D3 鈥?Reproducibility and provenance walkthrough:** included as a documentary research demonstration.
- **D4 鈥?Bounded inference demonstration:** **NOT_INCLUDED**.
- **D5 鈥?Failure and limitation explanation:** included as a documentary research demonstration.

## D4 decision

D4 is `NOT_INCLUDED` because the complete exact frozen inference contract required by the P6 prospective protocol was not established.

The Step3 read-only audit found 52 inference-like candidate source surfaces by a deliberately broad filename heuristic and **0 tracked model-like artifacts**. The candidates include evaluation, ablation, acquisition, test, and related research surfaces. Static token matches do not establish a supported v1.0 user-facing inference entry point, compatible frozen checkpoint, stable input contract, non-evaluative output semantics, and official-test isolation as one exact contract.

This decision does **not** claim that AEGIS cannot perform inference. It means only that live inference is not included in this frozen post-v1 demonstration package under the required fail-closed evidence standard.

## Frozen scientific demonstration boundary

The package preserves validation Macro-F1 `0.8669`, `0.8716`, `0.8610`; utility-probability standard deviation `0.016633`, `0.012812`, `0.009565`; M3 `utility_probability_std >= 0.05` as `NOT_SUPPORTED`; active intervention rate `0`, `0`, `0`; M4 `active_intervention_rate > 0` as `NOT_SUPPORTED`; v0.35 correction result `CORRECTION_NOT_SUPPORTED`; and v0.36 active-intervention robustness `NOT_COMPUTED`.

The demonstration does not establish production readiness, autonomous attribution, autonomous threat determination, active adaptive robustness, classifier non-robustness, official-test performance, state-of-the-art performance, or comparative superiority.

## Release and execution boundary

The authoritative immutable release remains `v1.0.0-research-core` at commit `39492cf6d93987fdf9ebee29b95fdb826b8391e4`.

Demo/model execution: **NONE**. Training/new evaluation: **NONE**. Official test samples: **0**.

AEGIS-Vision, AEGIS-Audio, AEGIS-Video, multilingual/cross-lingual capability, and broader integration remain separate post-v1 research.

## Next package

Proceed to **P7 鈥?Repository Presentation**. P7 may improve repository navigation, research presentation, documentation discoverability, release visibility, and claim-safe public-facing explanation, but must not mutate the immutable v1.0 scientific baseline or expand the demonstrated capability boundary.
