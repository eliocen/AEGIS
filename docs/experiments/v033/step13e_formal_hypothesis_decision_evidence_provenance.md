# AEGIS v0.33 Step13E 鈥?Formal Hypothesis Decision Evidence & Provenance

**Status:** FORMAL_HYPOTHESIS_DECISIONS_FROZEN

## Formal decisions
- V33_H1_TARGET_SIGNAL: **NOT_ADJUDICABLE_FROM_FROZEN_EVIDENCE**
- V33_H2_SELECTOR_LEARNING: **NOT_ADJUDICABLE_FROM_FROZEN_EVIDENCE**
- V33_H3_UTILITY_DISCRIMINATION: **NOT_SUPPORTED**
- V33_H4_CLEAN_SAFEGUARD: **SUPPORTED**
- V33_H5_GRADED_ROBUSTNESS: **NOT_SUPPORTED**

## Frozen formal statistics
- H3 mismatched-minus-matched utility: `-0.00541312666734062`
- H3 catastrophic-minus-clean utility: `-0.008737900202473`
- H4 clean delta vs M1b: `-0.00911024253801163`
- H4 clean delta vs M4qcs-w: `-5.66868960502065e-05`
- H5 graded mean delta vs M4qcs-w: `-0.00114102123489178`
- H5 positive graded count: `12/36`
- H5 catastrophic delta vs M4qcf: `0.00967330895634733`

Successor performance recommendation eligible: **TRUE**

This freeze consumes the exact Step13D result artifact without recomputation. No training, evaluator rerun, checkpoint reselection, threshold tuning, aggregation change, threshold change, or official-test access occurred.

Next: Step14 scientific synthesis and v0.33 closure.
