# AEGIS v0.31 Step7 鈥?Formal Evaluation Operationalization

Step7 prospectively freezes the v0.31 formal-evaluation roster, inherited validation condition matrix, mismatch construction semantics, expected result cardinalities, and evaluator-adaptation boundary. No checkpoint is loaded and no formal evaluation is performed in this step.

## Formal roster

- M1b
- M4qcf
- M4qcs-w
- M4qgrt
- M4qcesi-c
- M4qcesi-u
- M4qcesi

Each architecture is evaluated for seeds 42, 43, and 44: 21 frozen checkpoints total.

## Formal evaluation cardinalities

- Frozen quality conditions per architecture-seed: 19
- Expected quality summaries: 399
- Expected mismatch summaries: 42

## Inherited semantics

The 19-condition validation matrix and class-preserving mismatch construction are inherited exactly from the frozen v0.30/v0.29 unified-evaluation protocol. Conditions cannot be added, removed, reweighted, or redefined after exposure.

## v0.31 diagnostics

The evaluator adaptation must expose the frozen calibration/utility mechanism diagnostics needed by V31-H4 and V31-H5 while preserving the inherited evaluation semantics. Legacy baselines may report intervention diagnostics as zero or not applicable.

## Scientific boundary

No training, checkpoint reselection, seed dropping, result-based tuning, formal hypothesis decision, official-test access, threshold search, or H5 rebinning is permitted. V31-H1 through V31-H5 and the clean-performance safeguard remain NOT_COMPUTED.

Next stage: Step8 v0.31 evaluator adaptation implementation and verification.
