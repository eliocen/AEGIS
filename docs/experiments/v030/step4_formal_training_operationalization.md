# AEGIS v0.30 Step4 鈥?Formal-Training Operationalization

**Status:** FROZEN BEFORE FORMAL v0.30 TRAINING

Step4 freezes the exact formal-training matrix and does not execute training.

## Scientific boundary

- Formal training performed: **NO**
- Formal evaluation performed: **NO**
- Formal decisions computed: **NO**
- Official test samples accessed: **0**
- Hyperparameter search permitted: **NO**

## Architecture order

1. M4qesri-w 鈥?weights-only intervention
2. M4qesri-t 鈥?transition-only intervention
3. M4qesri 鈥?combined intervention

Each architecture is trained with seeds 42, 43 and 44, producing exactly nine formal runs.

## Locked hyperparameters

- epochs: 50
- batch size: 32
- learning rate: 0.001
- weight decay: 0.0001
- shared dimension: 128
- hidden dimension: 128
- dropout: 0.1
- temperature: 0.07
- alignment weight: 0.5
- classification weight: 1.0
- quality weight: 1.0
- compatibility weight: 1.0
- v0.29 transition-objective weight: **0.0**
- gradient clip: 1.0
- patience: 8
- minimum epochs: 5
- minimum delta: 1e-6

## Formal run matrix

| # | Architecture | Seed | Runner architecture | Output root |
|---:|---|---:|---|---|
| 1 | M4qesri-w | 42 | `quality_compatibility_selective_intervention_weights` | `experiments/fakeddit/v030_m4qesri_w_seed42` |
| 2 | M4qesri-w | 43 | `quality_compatibility_selective_intervention_weights` | `experiments/fakeddit/v030_m4qesri_w_seed43` |
| 3 | M4qesri-w | 44 | `quality_compatibility_selective_intervention_weights` | `experiments/fakeddit/v030_m4qesri_w_seed44` |
| 4 | M4qesri-t | 42 | `quality_compatibility_selective_intervention_transition` | `experiments/fakeddit/v030_m4qesri_t_seed42` |
| 5 | M4qesri-t | 43 | `quality_compatibility_selective_intervention_transition` | `experiments/fakeddit/v030_m4qesri_t_seed43` |
| 6 | M4qesri-t | 44 | `quality_compatibility_selective_intervention_transition` | `experiments/fakeddit/v030_m4qesri_t_seed44` |
| 7 | M4qesri | 42 | `quality_compatibility_selective_intervention` | `experiments/fakeddit/v030_m4qesri_seed42` |
| 8 | M4qesri | 43 | `quality_compatibility_selective_intervention` | `experiments/fakeddit/v030_m4qesri_seed43` |
| 9 | M4qesri | 44 | `quality_compatibility_selective_intervention` | `experiments/fakeddit/v030_m4qesri_seed44` |

## Checkpoint selection

- Primary metric: clean validation Macro-F1
- Tie-breaker: clean validation classification loss
- Robustness, mismatch, quality, compatibility and intervention diagnostics cannot select checkpoints.
- Post-result checkpoint reselection is prohibited.

## Data identities

- Training cache tree SHA256: `5FB8F4DDAC94211B44D5CDFC54ADC824D24D5B4073B2EF8B22BA08D179981CC2`
- Validation cache tree SHA256: `88BD8D27A4C9E48409FDA3E2B08A1A14EBC0A4F0F8F995266B49BA9CAAA7167E`
- Cache replacement or mutation after Step4 is prohibited.

## Failure policy

Completed formal runs may not be overwritten or rerun to seek a better result. Technical failures must be preserved and resumed under the same frozen scientific configuration unless a prospective correction is frozen before training resumes.

## Next gate

v0.30 Step5 鈥?formal training execution in exact frozen order.
