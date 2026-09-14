# AEGIS v0.33 Step9 鈥?Formal Evaluation Operationalization

Step9 prospectively freezes the v0.33 formal-evaluation roster, inherited validation condition matrix, mismatch construction semantics, expected result cardinalities, hypothesis inputs, and evaluator-adaptation boundary. No checkpoint is loaded and no formal evaluation is performed in this step.

## Formal roster

- M1b
- M4qcf
- M4qcs-w
- M4qusli

Each architecture is evaluated for seeds 42, 43, and 44: 12 frozen checkpoints total.

## Formal evaluation cardinalities

- Frozen quality conditions per architecture-seed: 19
- Expected quality summaries: 228
- Expected mismatch summaries: 24

## Inherited semantics

The 19-condition validation matrix and class-preserving mismatch construction are inherited exactly from the frozen v0.31/v0.30/v0.29 unified-evaluation lineage. Conditions cannot be added, removed, reweighted, redefined, or tuned after exposure.

## v0.33 diagnostics

The evaluator adaptation must add only M4qusli routing and the frozen utility-supervision/selector diagnostics required for V33-H1 through V33-H3. Historical comparators retain their frozen behavior.

## Scientific boundary

No training, checkpoint reselection, seed dropping, result-based tuning, formal hypothesis decision, official-test access, threshold search, or post-exposure rebinning is permitted. V33-H1 through V33-H5 remain NOT_COMPUTED.

Next stage: Step10 v0.33 evaluator adaptation implementation and engineering smoke before formal evaluation.
