# AEGIS v0.31 Step1 鈥?Calibrated Selective Intervention Prospective Protocol

Status: **PROSPECTIVE PROTOCOL FROZEN BEFORE IMPLEMENTATION**

## Primary architecture

`M4qcesi` 鈥?Calibrated Evidence Selective Intervention.

Ablations:

- `M4qcesi-c` 鈥?calibration-only
- `M4qcesi-u` 鈥?utility-selector-only
- `M4qcesi` 鈥?combined calibration + intervention utility

## Core design

The stable M4qcs-w-compatible reference remains the default path. v0.31 introduces:

1. a calibrated reliability-risk representation;
2. an intervention-utility selector estimating whether intervention is likely to help;
3. bounded intervention that returns toward the stable reference when estimated utility is low.

Exact equations, tensor contracts, bounds, gradient flow, and reporting thresholds are **not yet implemented** and must be frozen prospectively in Step2.

## Formal hypotheses

- V31-H1 鈥?graded corruption robustness
- V31-H2 鈥?mismatch transition quality
- V31-H3 鈥?combined-mechanism superiority
- V31-H4 鈥?utility selectivity
- V31-H5 鈥?evidence/utility alignment

All remain **NOT_COMPUTED**.

## Safety

- Implementation: NO
- Training: NO
- Formal evaluation: NO
- Official test access: NO
- Official test samples: 0

Next: **v0.31 Step2 鈥?exact calibrated-evidence and intervention-utility contract freeze**.
