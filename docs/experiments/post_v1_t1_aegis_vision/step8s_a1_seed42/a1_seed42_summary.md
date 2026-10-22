# AEGIS-Vision T1 ? A1 Seed 42

## Training evidence

- Repository HEAD: `671d6a8675d9fdb18d9cc485f4be557ca498c424`
- Runner Git blob: `aaa82f0261681cc3036714777bd662f0dd81be57`
- Epochs completed: 20
- Selected checkpoint epoch: 19
- Optimizer updates: 101213
- Validation Macro-F1: 0.990903487
- Validation AUROC: 0.999434481
- Validation BCE: 0.112063079
- Reliability correctness AUROC: 0.967836210
- Reliability Brier score: 0.009050360

## Artifact hashes

- `best.json`: `19ECAE8C886B9E371695E6C5B29954A14A0A28CE014190303BAEFBACDDFEC435`
- `best.pt`: `E90DA739B19DF6D409D078F736EA525276890488D4762DE4B39D9ADE90C55BD6`
- `final_summary.json`: `8492BF92401C62D51E7E1041BF6DB77D4EFBA27D01D227E808064D1508D372CE`
- `history.json`: `C67A0C90FDE53A644F412837AF32D479221407EFA26AA44F3EA4EFDE458AF8BC`
- `provenance.json`: `7B1A42AF13B5391346B7F20AF08DE3AC65C664BF80F80D82C7093A92C314D1CC`

## Scientific interpretation

This record establishes completion and integrity of the A1/seed42 training artifacts. The selected metrics are validation results, not independent test estimates.

The reliability head distinguishes correctness to some extent, but high predicted reliability on incorrect classifications requires further calibration analysis.

Full model-state compatibility and multi-seed A1 characterization remain pending.

## Architecture compatibility verification

- Frozen model: `AegisVisionModel`
- Strict checkpoint loading: PASS
- Missing state keys: 0
- Unexpected state keys: 0
- Synthetic forward pass: PASS
- Output heads: authenticity and reliability
- Independent test evaluation: NOT PERFORMED
