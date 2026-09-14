# AEGIS v0.30 Step10B 鈥?Mechanism-Characterization Evidence

Status: **MECHANISM CHARACTERIZATION EVIDENCE EXTRACTED**

## Immutable Step9 decisions

- Clean-performance safeguard: **PASSED**
- V30-H1: **NOT_SUPPORTED**
- V30-H2: **NOT_SUPPORTED**
- V30-H3: **NOT_SUPPORTED**
- V30-H4: **NOT_SUPPORTED**

## MC1 鈥?H1 robustness localization

M4qesri vs M4qcs-w graded mean delta:
`0.000023484596`

M4qesri vs M4qcf catastrophic mean delta:
`0.004777118985`

Full family-, modality-, seed-, and condition-level decompositions are preserved in the JSON record.

## MC2 鈥?H2 transition failure decomposition

M4qesri:
- harmful transitions: `186`
- beneficial transitions: `177`
- harmful/beneficial ratio: `1.050847457627`
- net harmful: `9`
- mismatch gap: `0.002910863174`

Comparator decompositions are preserved in JSON.

## MC3 鈥?H3 ablation attribution

Combined-model descriptive labels:
- synergy: `22`
- interference: `7`
- mixed: `28`

## MC4 鈥?H4 gate response

Mean active-intervention rate:
- clean: `0.009666666667`
- graded: `0.008083333333`
- catastrophic: `0.000000000000`

Matched/mismatched activation and severity trajectories are preserved in JSON.

## MC5 鈥?Evidence-performance coupling

Condition-level reliability evidence, intervention intensity, and performance deltas are preserved for descriptive ranking only. No causal inference or post-hoc thresholding is performed.

## Scientific safety

- Training/retraining: **NO**
- Re-evaluation: **NO**
- Checkpoint loading: **NO**
- Checkpoint reselection: **NO**
- Threshold tuning: **NO**
- New hypothesis testing: **NO**
- Step9 decision modification: **NO**
- Official test accessed: **NO**
- Official test samples: **0**
