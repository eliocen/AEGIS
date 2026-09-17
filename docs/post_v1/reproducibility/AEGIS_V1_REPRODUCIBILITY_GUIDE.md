# AEGIS Research Core v1.0 鈥?Reproducibility Guide

## Scope

This guide identifies how to reproduce the identity, environment, workflow, provenance, and scientific decision logic of the immutable AEGIS Research Core v1.0 baseline. P3 does not authorize retraining or new scientific evaluation.

## Level A 鈥?Identity reproduction

Verify the annotated tag `v1.0.0-research-core` resolves to commit `39492cf6d93987fdf9ebee29b95fdb826b8391e4`. Verify the v1 release manifest SHA256 is `5BD15FB6FBECBB97A8727DD3E363D399B39B31EB6C8B1A673C238863FA715AAA` and the release-notes SHA256 is `88F100C9B0536981C546C79C20F7CBE608850339C0F551B041106A15258748A5`.

Level A requires no model execution.

## Level B 鈥?Environment and workflow reconstruction

The frozen primary research environment was Windows/PowerShell with Python 3.10.11 at `D:\Research\AEGIS\.venv\Scripts\python.exe`, NVIDIA GeForce GTX 1660 SUPER, PyTorch 2.13.0+cu132, and CUDA available.

These values document the primary environment; they are not all asserted to be universal minimum requirements. Dependency lock/configuration records in the repository remain the authoritative source where present.

The frozen v0.35 corrective-training data boundary used:

- train cache `data/processed/fakeddit/frozen_embeddings/train_n5000_seed42`
- validation cache `data/processed/fakeddit/frozen_embeddings/validation_n1000_seed42`
- train manifest SHA256 `5BC8BF53F988B3DD5299A1D940DB6676F48E37250F52C8B3569150BE153417C5`
- validation manifest SHA256 `C4E95F818C24043DB1DEE9C0CFC0DE85C2E591E9E77B98CB2490046D1BC04F87`

Official-test samples accessed remain **0**.

## Level C 鈥?Scientific rerun

A scientific rerun is outside P3 authorization. If separately authorized, it must preserve the frozen source boundary, data/cache identities, seeds 42/43/44, checkpoint rule, training configuration, M3 criterion `utility_probability_std >= 0.05` per seed, M4 criterion `active_intervention_rate > 0` per seed, and the original decision logic.

The purpose of a rerun would be reproduction of the frozen experiment, not retrospective optimization.

## Expected scientific decision reproduction

The frozen v0.35 correction is `V035_C1_CLASSIFIER_POSTERIOR_CONTEXT`. Its result is `CORRECTION_NOT_SUPPORTED`; M3 and M4 are `NOT_SUPPORTED`. v0.36 formal active-intervention robustness is `NOT_COMPUTED` because the inactive-selector prerequisite blocked formal robustness evaluation. v0.37 supports `ANALYST DECISION-SUPPORT RESEARCH CHARACTERIZATION`.

A reproduction must preserve these meanings. `NOT_COMPUTED` must not be converted into a negative robustness score.

## Reproducibility limitations

Bitwise numerical equality is not guaranteed across hardware, CUDA/cuDNN builds, kernels, dependency builds, or external assets. Repository-verifiable identity, environment reconstruction, and numerical rerun reproducibility are distinct levels of evidence. Validation reproduction is not official-test evaluation.
