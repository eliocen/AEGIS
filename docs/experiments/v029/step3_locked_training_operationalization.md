# AEGIS v0.29 Step3 Locked Training Operationalization

**Status:** FROZEN BEFORE FORMAL v0.29 TRAINING

This record freezes the exact nine-run GPU training schedule before any formal v0.29 training result is exposed. The Step2 seed-42 smoke runs remain non-evidentiary and cannot select architecture or hyperparameters.

Official-test samples accessed: **0**.

## Locked run order

| Run | Architecture | Seed | Output directory | Status |
|---:|---|---:|---|---|
| 1 | M4qgr | 42 | `experiments/fakeddit/v029_m4qgr_seed42` | LOCKED_NOT_STARTED |
| 2 | M4qgr | 43 | `experiments/fakeddit/v029_m4qgr_seed43` | LOCKED_NOT_STARTED |
| 3 | M4qgr | 44 | `experiments/fakeddit/v029_m4qgr_seed44` | LOCKED_NOT_STARTED |
| 4 | M4qtc | 42 | `experiments/fakeddit/v029_m4qtc_seed42` | LOCKED_NOT_STARTED |
| 5 | M4qtc | 43 | `experiments/fakeddit/v029_m4qtc_seed43` | LOCKED_NOT_STARTED |
| 6 | M4qtc | 44 | `experiments/fakeddit/v029_m4qtc_seed44` | LOCKED_NOT_STARTED |
| 7 | M4qgrt | 42 | `experiments/fakeddit/v029_m4qgrt_seed42` | LOCKED_NOT_STARTED |
| 8 | M4qgrt | 43 | `experiments/fakeddit/v029_m4qgrt_seed43` | LOCKED_NOT_STARTED |
| 9 | M4qgrt | 44 | `experiments/fakeddit/v029_m4qgrt_seed44` | LOCKED_NOT_STARTED |

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
- Transition objective weight: 0.25
- Gradient clip: 1.0
- Patience/minimum epochs/minimum delta: 8/5/0.000001

Checkpoint selection uses clean validation Macro-F1, with clean validation classification loss as the sole tie-breaker. Corruption, mismatch, quality, compatibility, and transition metrics cannot select or reselect checkpoints.

## Scientific boundary

No architecture search, hyperparameter search, overwrite of completed runs, seed dropping, robustness evaluation, mismatch evaluation, formal hypothesis decision, or official-test access is permitted during Step3.

## Next action

Execute all nine locked formal training runs sequentially and freeze their clean-selected checkpoints and training record.
