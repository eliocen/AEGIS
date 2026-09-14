# AEGIS v0.30 Step9 鈥?One-Time Formal Hypothesis Analysis

Status: **FORMAL HYPOTHESIS ANALYSIS COMPLETE**

## Formal decisions

- Clean-performance safeguard: **PASSED**
- V30-H1: **NOT_SUPPORTED**
- V30-H2: **NOT_SUPPORTED**
- V30-H3: **NOT_SUPPORTED**
- V30-H4: **NOT_SUPPORTED**

## Clean-performance safeguard

- M4qesri clean Macro-F1: `0.865881516326`
- Delta vs M1b: `-0.007725860838`; frozen floor `-0.01`
- Delta vs M4qcs-w: `0.001327694804`; frozen floor `-0.01`

## V30-H1

- Graded mean delta vs M4qcs-w: `0.000023484596`; threshold `+0.005`
- Positive graded comparisons: `17/36`; threshold `>=22/36`
- Positive fraction: `0.472222`; threshold `>=0.60`
- Catastrophic mean delta vs M4qcf: `0.004777118985`; floor `-0.01`

## V30-H2

- M4qesri harmful/beneficial ratio: `1.050847457627`
- M4qgrt harmful/beneficial ratio: `1.250000000000`
- M4qcs-w harmful/beneficial ratio: `1.028089887640`
- M4qesri net harmful: `9`
- M4qgrt net harmful: `35`
- M4qcs-w net harmful: `5`
- M4qesri mismatch gap: `0.002910863174`
- M4qesri mismatched Macro-F1: `0.862970653152`

## V30-H3

Full primary-vs-ablation comparisons against M4qesri-w and M4qesri-t are preserved in the JSON record.

## V30-H4

- Clean active intervention rate: `0.009666666667`
- Graded active intervention rate: `0.008083333333`
- Catastrophic active intervention rate: `0.000000000000`
- Matched active intervention rate: `0.009666666667`
- Mismatched active intervention rate: `0.204666666667`

Severity-response sequences and frozen tolerance checks are preserved in the JSON record.

## Scientific safety

- Training/retraining: **NO**
- Formal re-evaluation: **NO**
- Checkpoint reselection: **NO**
- Threshold tuning: **NO**
- Hypothesis definitions modified: **NO**
- Official test accessed: **NO**
- Official test samples: **0**

## Next step

Proceed to **v0.30 Step10 mechanism characterization**, preserving these formal Step9 decisions unchanged.
