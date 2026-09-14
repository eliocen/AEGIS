# AEGIS v0.31 Step8 鈥?Evaluator Adaptation Implementation

The frozen v0.30 unified evaluator was adapted to the prospectively frozen v0.31
formal roster and calibration/utility diagnostics. The model, controller,
training runner, trained checkpoints, corruption matrix, and mismatch
construction remain unchanged.

## Recovery notes

Two engineering-smoke issues were evaluator-schema assumptions only.

First, v0.31 checkpoints encode controller mode through the exact frozen
`fusion_architecture`; no separate top-level controller-mode field is required.
The transition-objective audit uses the actual nested
`transition_objective.effective_weight` field and requires 0.0.

Second, `effective_text_reliability` and `effective_vision_reliability` belong to
the historical deterministic reliability-controller output contract. The v0.31
controller does not expose those fields. The v0.31 evaluator therefore requires
the actual frozen v0.31 outputs: raw/calibrated evidence, final modality weights,
interaction multiplier, calibrated risk, utility, intervention gate, active
indicator, weight intervention magnitude, and interaction suppression. Legacy
architectures retain their historical diagnostic requirements.

Neither correction changes scientific design or evaluation semantics.

## Verification

A non-formal seed-42 engineering smoke covers all seven formal architectures
using two quality conditions plus matched/mismatched evaluation. Formal
evaluation remains unperformed; V31-H1 through V31-H5 remain NOT_COMPUTED; the
official test split remains sealed with zero samples accessed.
