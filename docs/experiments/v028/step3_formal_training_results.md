# AEGIS v0.28 Step3 Formal Training Results

**Status:** FROZEN FORMAL v0.28 TRAINING RESULTS

All nine prospectively locked training runs completed. Checkpoints were selected
using clean validation Macro-F1 only, with clean validation classification loss
as the tie-breaker. These clean metrics document checkpoint selection and do not
compute any v0.28 formal robustness hypothesis.

Official-test samples accessed: **0**.

## Per-run training record

| Architecture | Seed | Best epoch | Epochs completed | Best clean Macro-F1 | Best clean accuracy | Stop reason |
|---|---:|---:|---:|---:|---:|---|
| M4qcs-w | 42 | 8 | 16 | 0.861732313759 | 0.862000000000 | early_stopping |
| M4qcs-w | 43 | 5 | 13 | 0.864977181144 | 0.865000000000 | early_stopping |
| M4qcs-w | 44 | 5 | 13 | 0.866951969661 | 0.867000000000 | early_stopping |
| M4qcs-i | 42 | 7 | 15 | 0.860976505029 | 0.861000000000 | early_stopping |
| M4qcs-i | 43 | 6 | 14 | 0.866625350610 | 0.867000000000 | early_stopping |
| M4qcs-i | 44 | 13 | 21 | 0.864989064114 | 0.865000000000 | early_stopping |
| M4qcs | 42 | 12 | 20 | 0.866653164882 | 0.867000000000 | early_stopping |
| M4qcs | 43 | 9 | 17 | 0.868918073796 | 0.869000000000 | early_stopping |
| M4qcs | 44 | 5 | 13 | 0.862642733750 | 0.863000000000 | early_stopping |

## Descriptive clean-training aggregates

| Architecture | Mean Macro-F1 | Population SD | Mean accuracy |
|---|---:|---:|---:|
| M4qcs-w | 0.864553821521 | 0.002151840601 | 0.864666666667 |
| M4qcs-i | 0.864196973251 | 0.002373172286 | 0.864333333333 |
| M4qcs | 0.866071324143 | 0.002594722484 | 0.866333333333 |

These aggregates are descriptive training records. They do not select an
architecture and do not constitute V28-H1, V28-H2, V28-H3, or V28-H4 decisions.

## Scientific boundary

- All nine locked architecture-seed runs are retained.
- No hyperparameter tuning or architecture modification was performed.
- No robustness or mismatch evaluation was performed during Step3.
- No checkpoint was reselected after robustness exposure.
- The official Fakeddit test split remains sealed.
- Frozen v0.27 records and decisions remain unchanged.

## Next step

Step4 performs the frozen unified clean, corruption, mismatch, ablation, and
text-Gaussian evaluation. Formal hypothesis decisions remain prohibited until
Step5.
