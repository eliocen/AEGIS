# AEGIS v0.33 Step8 鈥?Formal Training Evidence and Provenance Freeze

Status: **FROZEN**

This step freezes provenance for the three authorized v0.33 formal M4qusli training runs. It performs no training, no formal evaluation, no hypothesis decisions, no checkpoint reselection, no threshold search, and no official-test access.

## Frozen boundary

- Parent Step6A commit: `14b98298fc450ba9e59d794f2f1935b268b5157f`
- Architecture: `M4qusli`
- Runner identifier: `quality_compatibility_utility_supervised_intervention`
- Seeds: `42, 43, 44`
- Effective utility-loss weight: `0.50`
- Transition objective weight: `0.0`
- Official test samples accessed: `0`

## Formal-training evidence

- Seed 42: run root `experiments/fakeddit/v033/formal_training/m4qusli_seed42`; metrics rows 15; best epoch 7; best validation Macro-F1 schema-dependent / recorded in summary.json.
- Seed 43: run root `experiments/fakeddit/v033/formal_training/m4qusli_seed43`; metrics rows 17; best epoch 9; best validation Macro-F1 schema-dependent / recorded in summary.json.
- Seed 44: run root `experiments/fakeddit/v033/formal_training/m4qusli_seed44`; metrics rows 12; best epoch 4; best validation Macro-F1 schema-dependent / recorded in summary.json.

Each run was verified to contain `experiment.json`, `metrics.jsonl`, `best_model.pt`, `final_model.pt`, `best_validation_predictions.json`, and `summary.json`. SHA256 identities are recorded in the Step8 JSON.

## Frozen invariants

- Utility-loss weight remains exactly `0.50`.
- Transition objective remains `0.0`.
- Selector optimization and intervention diagnostics are machine-readable.
- Validation utility/gate/intervention diagnostics are machine-readable.
- Exactly three authorized formal run roots exist.
- Official test split remains sealed and not accessed.
- Formal evaluation remains **NOT AUTHORIZED**.

Next scientific action requires a separate formal-evaluation operationalization/freeze.
