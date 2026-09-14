# AEGIS v0.30 Step7 鈥?Formal Evaluation Operationalization

## Status

- Prospective operationalization: **FROZEN BEFORE FORMAL EVALUATION**
- Formal evaluation executed: **NO**
- Checkpoints loaded for evaluation: **NO**
- Formal hypothesis decisions: **NO**
- Official Fakeddit test samples accessed: **0**

## Evaluation roster

| Architecture | Seeds | Role |
|---|---|---|
| M1b | 42, 43, 44 | Clean-performance / transition baseline |
| M4qcf | 42, 43, 44 | Catastrophic-corruption comparator |
| M4qcs-w | 42, 43, 44 | Stable-reference comparator |
| M4qgrt | 42, 43, 44 | v0.29 transition-control comparator |
| M4qesri-w | 42, 43, 44 | v0.30 selective-weighting ablation |
| M4qesri-t | 42, 43, 44 | v0.30 selective-transition ablation |
| M4qesri | 42, 43, 44 | v0.30 primary combined model |

Total frozen checkpoints: **21**.

## Inherited evaluation semantics

The v0.30 formal evaluation inherits the frozen v0.29 unified-evaluation corruption conditions and mismatch mapping without modification. The architecture roster remains seven models across three seeds and nineteen conditions, yielding **399 quality summaries**. The paired matched/mismatched evaluation yields **42 mismatch summaries**.

## v0.30 evaluator adaptation contract

The v0.29 evaluator is the implementation base. Before Step8 execution it may be adapted only to route the three v0.30 architectures/checkpoints and to emit the intervention diagnostics frozen by the v0.30 protocol. Corruption definitions, mismatch mapping, seeds, conditions, checkpoint selection, and evaluation semantics may not change.

Required intervention diagnostics include gate intensity, active-intervention indicator at the frozen 0.5 reporting threshold, modality-weight intervention magnitude, and interaction-suppression magnitude.

## Prohibitions

- No training or retraining.
- No checkpoint reselection.
- No hyperparameter tuning.
- No condition addition/removal.
- No seed dropping.
- No result-based reweighting.
- No V30-H1鈥揤30-H4 computation in Step7 or Step8 execution.
- No official-test access.

## Next stage

**v0.30 Step8 鈥?Formal Unified Evaluation**
