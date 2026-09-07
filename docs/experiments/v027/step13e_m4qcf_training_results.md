# AEGIS v0.27 Step13E - M4qcf Formal Training Results

**Record version:** `0.27.0-step13e`
**Protocol:** `0.27.0-step13a`
**Status:** `COMPLETE_AND_FROZEN`
**Official Fakeddit test:** `SEALED_NOT_ACCESSED`

## Purpose

Step13E records the preregistered three-seed formal training results for M4qcf reliability-informed fusion.

The clean-validation hypothesis evaluated at this stage is H7-F. H8-F, H9-F, and H10-F remain uncomputed pending the frozen Step14 robustness diagnostics.

## Formal per-seed results

| Seed | Best epoch | Macro-F1 | Accuracy | F1 | Classification loss | Epochs | Stop |
|---:|---:|---:|---:|---:|---:|---:|---|
| 42 | 14 | 0.8659093547237933 | 0.866 | 0.8624229979466119 | 0.3508504779338837 | 22 | `early_stopping` |
| 43 | 9 | 0.8697702747646849 | 0.87 | 0.8643006263048018 | 0.34784842205047606 | 17 | `early_stopping` |
| 44 | 14 | 0.8665664744755712 | 0.867 | 0.8589607635206787 | 0.37738130235671996 | 22 | `early_stopping` |

## H7-F - Clean classification preservation

- Seed Macro-F1 values: `[0.8659093547237933, 0.8697702747646849, 0.8665664744755712]`
- Mean Macro-F1: `0.8674153679880164`
- Sample SD: `0.002065705941595623`
- Frozen M1b reference Macro-F1: `0.8736`
- Delta versus M1b: `-0.006184632011983604`
- Frozen preservation margin: `-0.01`
- Decision: **SUPPORTED**

H7-F is a descriptive held-out validation-margin criterion. It is not a formal statistical noninferiority test.

M4qcf does not outperform M1b on mean clean Macro-F1 in this experiment. Rather, its clean-validation performance remains within the preregistered 1-percentage-point preservation margin.

## Remaining formal hypotheses

- **H8-F:** `NOT_COMPUTED`
- **H9-F:** `NOT_COMPUTED`
- **H10-F:** `NOT_COMPUTED`

These hypotheses require the separately frozen Step14 robustness and mismatch diagnostic pipeline.

## Protocol integrity

- Architecture: `quality_compatibility_fusion`
- Reliability-controller epsilon: `0.10`
- Interaction multiplier gamma: `1.0`
- Reliability controller: parameter-free
- Reliability signals in fusion: stop-gradient
- Alignment loss weight: `0.5`
- Classification loss weight: `1.0`
- Quality loss weight: `1.0`
- Compatibility loss weight: `1.0`
- Formal seeds: `42, 43, 44`
- Official test samples accessed: `0`

## Interpretation boundary

Step13E supports only clean held-out validation classification preservation under the frozen H7-F criterion. It does not yet establish robustness improvement, corrupted-modality down-weighting, or mismatch-performance benefit.

It also does not establish open-world factual verification, source credibility, intent inference, human trust, or real-world robustness.

## Next phase

Proceed to **Step14A**, the frozen M4qcf robustness diagnostic runner. H8-F, H9-F, and H10-F must remain uncomputed until that diagnostic evidence is generated.
