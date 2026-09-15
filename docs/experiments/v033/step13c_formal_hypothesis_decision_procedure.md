# AEGIS v0.33 Step13C 鈥?Formal Hypothesis Decision Procedure

**Status:** PROCEDURE_FROZEN_RESULTS_NOT_COMPUTED

This record freezes the exact decision procedure before numerical formal hypothesis analysis. It performs no hypothesis calculation.

## Evidence boundary
- Parent Step12 commit: `0886b7fd13014363b02cadcd1a66f24a8220f892`
- Formal evaluation tree SHA256: `8FCACEA904129574211E3BB63DD25C46D55A2FC218F223577AE01BDF7E777E43`
- Official test access: 0

## Fixed dispositions from Step13B
- V33_H1_TARGET_SIGNAL: `NOT_ADJUDICABLE_FROM_FROZEN_EVIDENCE`
- V33_H2_SELECTOR_LEARNING: `NOT_ADJUDICABLE_FROM_FROZEN_EVIDENCE`
- Missing evidence is not equivalent to `NOT_SUPPORTED`.

## Authorized numerical decisions
### V33_H3 鈥?Utility discrimination
- Mismatch statistic: mean M4qusli `utility_probability.mean` over 3 mismatched rows minus mean over 3 matched rows.
- Catastrophic statistic: mean over 4 frozen catastrophic conditions x 3 seeds minus mean clean over 3 seeds.
- `SUPPORTED` iff both prospectively frozen thresholds pass.

### V33_H4 鈥?Clean safeguard
- Average clean Macro-F1 across seeds 42/43/44 separately for M4qusli, M1b, and M4qcs-w.
- Compute primary-minus-comparator deltas.
- `SUPPORTED` iff both deltas are >= -0.01.
- Failure prohibits successor performance recommendation.

### V33_H5 鈥?Graded robustness
- Consume H4 first.
- Exactly 36 paired comparisons: 12 frozen graded conditions x 3 seeds, paired on `model_seed` and `condition`.
- Mean delta = arithmetic mean of those 36 M4qusli-minus-M4qcs-w deltas.
- Positive count uses strict `> 0`.
- Catastrophic delta uses all 4 frozen catastrophic conditions x 3 seeds for M4qusli versus M4qcf.
- If H4 is supported, H5 is `SUPPORTED` iff all three remaining frozen thresholds pass.
- If H4 is not supported, H5 is `NOT_SUPPORTED_BY_PREREQUISITE`; subordinate statistics may still be reported descriptively.

## Global rules
No training, evaluator rerun, checkpoint reselection, threshold tuning, post-exposure subset changes, alternate aggregation, or official-test access. Persisted machine-readable values are compared without pre-decision rounding.

## Boundary after this freeze
V33_H1..H5 remain `NOT_COMPUTED`. Numerical formal analysis is reserved for Step13D.
