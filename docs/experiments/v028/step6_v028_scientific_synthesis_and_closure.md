# AEGIS v0.28 Scientific Synthesis and Closure

> **FROZEN CONTROLLED VALIDATION-DEVELOPMENT SCIENTIFIC CLOSURE**

## Release identity

- Record version: `0.28.0-step6-closure`
- Pre-closure commit: `0712bec5db59020a11d6eac62976f1f8cd31f33b`
- Target annotated release tag: `v0.28.0`
- Complete regression suite: 1011 passed
- Official-test samples accessed: 0

## Formal decision matrix

| Hypothesis | Decision | Decisive outcome |
|---|---|---|
| V28-H1 | NOT_SUPPORTED | Graded mean delta -0.002855888025; 9/36 positive |
| V28-H2 | NOT_SUPPORTED | Harmful/beneficial ratio 1.246987951807 was not below M1b 1.172549019608 |
| V28-H3 | NOT_SUPPORTED | Combined M4qcs did not exceed M4qcs-w mismatch Macro-F1 or net-harmful balance |
| V28-H4 | NOT_SUPPORTED | Text-Gaussian mean delta -0.002477026819; 3/12 positive |

## Clean-performance safeguard

The clean-performance safeguard **PASSED**. Mean clean deltas were -0.007536053021 versus M1b and -0.001344043845 versus M4qcf. This safeguard result does not override any failed formal hypothesis gate.

## Scientific synthesis

The v0.28 programme tested prospectively specified selective modality-weight allocation, selective interaction suppression, and their combination. The complete M4qcs design preserved clean performance within the frozen tolerance and improved several mismatch measures, including net harmful flips relative to M1b and M4qcf. However, it did not deliver the required graded-corruption coverage, did not improve the harmful-to-beneficial ratio relative to M1b, did not outperform the weight-only ablation on the mandatory mismatch comparisons, and did not meet the text-Gaussian improvement and coverage gates.

Accordingly, none of the four formal hypotheses is supported. The results favor the narrower interpretation that weight-only allocation was the strongest mismatch-control component in this experiment, while combining weight allocation with interaction suppression introduced no prospectively sufficient advantage. This is a controlled descriptive inference, not a causal or universal claim.

## Retained contributions

- A prospectively frozen selective reliability and mismatch-control protocol.
- Tested M4qcs-w, M4qcs-i, and M4qcs implementations with controller invariants.
- Nine locked training runs and 18 frozen evaluated checkpoints.
- A unified 342-row corruption record and 36-row mismatch record.
- A one-time conjunctive formal analysis retaining all unsupported outcomes.
- Evidence that clean preservation alone is insufficient for robustness support.

## Limitations

- Evidence is confined to controlled Fakeddit validation-development representations.
- The official test split remained sealed and provides no v0.28 evidence.
- Corruptions operate on frozen representations rather than raw open-world inputs.
- Three seeds do not establish broad deployment reliability.
- No causal mechanism, factual-verification, source-credibility, or author-intent claim is supported.
- The v0.28 evidence is not independent external confirmation.

## Closure rule

AEGIS v0.28 is scientifically closed. Do not rerun the analyzer, retune thresholds, reselect checkpoints, exclude conditions or seeds, change formal gates, or expand v0.28 after the closure record is frozen. Any redesign must be prospectively specified on a new v0.29 development line.

## Next development line

AEGIS v0.29 — prospectively redesign graded-corruption reliability and mismatch control while preserving the clean-performance safeguard and all frozen v0.28 outcomes.
