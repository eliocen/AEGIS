# AEGIS v0.28 Step3 Locked Training Operationalization

**Status:** FROZEN BEFORE FORMAL v0.28 TRAINING
**Evidence:** CONTROLLED VALIDATION-DEVELOPMENT
**Source implementation:** `6f48b5df4c7690c49cd53fcf63b1aad7bab3ba4f`

## Boundary

This record freezes the exact nine-run training schedule before formal v0.28
training. The successful seed-42 smoke runs are non-formal implementation
checks and were not used to select architectures or hyperparameters.

Official-test samples accessed: **0**.

## Locked run order

| Run | Architecture | Seed | Output directory | Status |
|---:|---|---:|---|---|
| 1 | M4qcs-w | 42 | `experiments/fakeddit/v028_m4qcs_w_seed42` | LOCKED_NOT_STARTED |
| 2 | M4qcs-w | 43 | `experiments/fakeddit/v028_m4qcs_w_seed43` | LOCKED_NOT_STARTED |
| 3 | M4qcs-w | 44 | `experiments/fakeddit/v028_m4qcs_w_seed44` | LOCKED_NOT_STARTED |
| 4 | M4qcs-i | 42 | `experiments/fakeddit/v028_m4qcs_i_seed42` | LOCKED_NOT_STARTED |
| 5 | M4qcs-i | 43 | `experiments/fakeddit/v028_m4qcs_i_seed43` | LOCKED_NOT_STARTED |
| 6 | M4qcs-i | 44 | `experiments/fakeddit/v028_m4qcs_i_seed44` | LOCKED_NOT_STARTED |
| 7 | M4qcs | 42 | `experiments/fakeddit/v028_m4qcs_seed42` | LOCKED_NOT_STARTED |
| 8 | M4qcs | 43 | `experiments/fakeddit/v028_m4qcs_seed43` | LOCKED_NOT_STARTED |
| 9 | M4qcs | 44 | `experiments/fakeddit/v028_m4qcs_seed44` | LOCKED_NOT_STARTED |

## Locked configuration

- Entrypoint: `python -m scripts.run_fakeddit_ablation`
- Epoch ceiling: 50
- Batch size: 32
- Learning rate: 0.001
- Weight decay: 0.0001
- Shared/hidden dimension: 128/128
- Dropout: 0.1
- Temperature: 0.07
- Alignment/classification weights: 0.5/1.0
- Quality/compatibility weights: 1.0/1.0
- Gradient clip: 1.0
- Patience/minimum epochs/minimum delta: 8/5/0.000001

Checkpoint selection uses clean validation Macro-F1, with clean validation
classification loss as the tie-breaker. Robustness, mismatch, quality, and
compatibility metrics cannot select or reselect checkpoints.

## Prohibitions

- No hyperparameter or architecture search.
- No overwriting completed formal runs.
- No dropping failed runs, conditions, or seeds.
- No robustness or mismatch evaluation during Step3.
- No official-test access.
- No modification of frozen v0.27 records or decisions.

## Completion

Step3 completes only after all nine runs and their provenance, checkpoint,
and scientific-safety audits are frozen. The next action is execution of the
nine locked formal training runs in the recorded order.
