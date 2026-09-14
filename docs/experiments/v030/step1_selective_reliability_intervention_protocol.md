# AEGIS v0.30 Step1 ??Prospective Protocol Freeze

Status: **FROZEN PROSPECTIVELY BEFORE IMPLEMENTATION, TRAINING, AND RESULT EXPOSURE**

## Research objective

Test whether an evidence-conditioned selective controller can improve graded
corruption and mismatch robustness while defaulting to a stable path when
intervention is not justified.

## Architecture family

Primary: `M4qesri`

Ablations:

- `M4qesri-w` ??selective modality-weight intervention only.
- `M4qesri-t` ??selective transition intervention only.
- `M4qesri` ??combined selective weighting + transition intervention.

The shared evidence gate is bounded in [0,1]. A value of 0 means no intervention.
The exact tensor equations, intervention bound, margins, and gradient-flow rules
must be frozen in Step2 before formal training.

## Hard clean safeguard

Mean clean validation Macro-F1 across seeds:

- delta versus M1b >= -0.01
- delta versus M4qcs-w >= -0.01

Failure of either gate prevents successor recommendation.

## Formal hypotheses

### V30-H1 ??broad robustness
M4qesri must improve graded corruption over M4qcs-w while preserving clean and
catastrophic-loss safeguards.

### V30-H2 ??transition selectivity
M4qesri must improve harmful/beneficial transition balance over M4qgrt and
match or exceed the M4qcs-w stability reference.

### V30-H3 ??mechanism synergy
The combined controller must outperform each selective ablation on graded and
mismatch metrics without meaningful clean loss.

### V30-H4 ??selective activation
Intervention must remain low on clean inputs and increase prospectively under
graded degradation, catastrophic degradation, and class-preserving mismatch.

## Scientific boundaries

- Seeds: 42, 43, 44.
- New formal runs: 9.
- Validation samples: 1000.
- Frozen quality conditions: 19.
- Checkpoint selection: clean validation Macro-F1, then clean classification loss.
- Robustness/mismatch/intervention metrics cannot select checkpoints.
- No post-result tuning, gate changes, checkpoint reselection, seed dropping,
  or condition dropping.
- Official Fakeddit test remains sealed.
- Formal decisions may be computed only at Step9 after frozen unified evaluation.

Next: **Step2 ??exact selective-intervention operationalization and implementation.**
