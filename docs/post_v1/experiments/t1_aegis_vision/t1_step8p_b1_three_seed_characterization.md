# T1 Step 8P — B1 Three-Seed Characterization

Status: **Descriptive validation evidence only**. Formal evaluation and sealed-test access remain unauthorized.

Parent frozen commit: `9c0b3416cd72159c4176ce1f54f22cd8db994137`.

## B1 per-seed results

| Seed | Macro-F1 | Balanced accuracy | AUROC | BCE | Best epoch | Epochs | Optimizer updates |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 42 | 0.988483367 | 0.988478216 | 0.999198061 | 0.070422003 | 4 | 8 | 40,479 |
| 43 | 0.990486488 | 0.990485793 | 0.999373976 | 0.075195110 | 7 | 11 | 55,660 |
| 44 | 0.990736716 | 0.990732952 | 0.999557747 | 0.097237993 | 17 | 20 | 101,209 |

## B0 versus B1 aggregate validation results

Sample SD uses n-1 denominator with three seeds.

| Metric | B0 mean | B0 sample SD | B1 mean | B1 sample SD | B1 minus B0 |
|---|---:|---:|---:|---:|---:|
| macro_f1 | 0.919060829 | 0.000487030 | 0.989902190 | 0.001235090 | +0.070841361 |
| balanced_accuracy | 0.919059952 | 0.000487804 | 0.989898987 | 0.001236614 | +0.070839035 |
| auroc | 0.973510126 | 0.000126373 | 0.999376595 | 0.000179857 | +0.025866469 |
| bce | 0.207432348 | 0.000519222 | 0.080951702 | 0.014304828 | -0.126480646 |

## Matched-seed descriptive differences

| Seed | B1 minus B0 Macro-F1 | B1 minus B0 AUROC |
|---:|---:|---:|
| 42 | +0.069617202 | +0.025547802 |
| 43 | +0.071785249 | +0.025898671 |
| 44 | +0.071121634 | +0.026152933 |

## Evidence provenance

The JSON companion records SHA256 for all five run artifacts for each of six runs, per-run provenance, and the Step 8O parent evidence hash. Checkpoints are not committed to Git.

## Interpretation boundaries

- Validation results are selection-exposed because validation Macro-F1 selected checkpoints.
- No official test set was accessed; formal evaluation is not authorized.
- No claim of cross-generator generalization, deployment readiness, or statistical significance is made.
- Dataset-specific generator/compression artifacts may confound image-authenticity performance.
- B1 seed44 reached the 20-epoch limit, with best checkpoint at epoch 17.

## Next gate

Step 8P may be remotely frozen after reviewing the evidence and verifying the GitHub commit. A1 training requires its own preflight and is not authorized by this record.
