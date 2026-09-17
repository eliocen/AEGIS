# AEGIS Post-v1 P3 Step1 鈥?Reproducibility Package Prospective Protocol

## Status

**POST_V1_P3_STEP1_REPRODUCIBILITY_PACKAGE_PROTOCOL_FROZEN**

P3 packages reproducibility for the immutable AEGIS Research Core v1.0 baseline. It is not a new experiment and does not authorize model/source changes, training, new scientific evaluation, checkpoint reselection, threshold changes, or official-test access.

## Governing release

The reproducibility package is anchored to:

- release branch `release/v1.0-research-core`
- release commit `39492cf6d93987fdf9ebee29b95fdb826b8391e4`
- annotated tag `v1.0.0-research-core`
- frozen v1 release manifest and release notes

P3 documentation must preserve the distinction between the immutable v1.0 scientific baseline and the downstream `docs/post-v1-research-packaging` branch.

## Reproducibility axes

P3 covers release identity; source/evidence provenance; environment/dependencies; data/cache boundaries; commands/workflows; tests/invariants; artifact/hash verification; scientific decision reproduction; the official-test boundary; and external limitations.

## Required outputs

1. `AEGIS_V1_REPRODUCIBILITY_GUIDE.md`
2. `AEGIS_V1_ENVIRONMENT_AND_DEPENDENCY_LEDGER.md`
3. `AEGIS_V1_COMMAND_AND_WORKFLOW_LEDGER.md`
4. `AEGIS_V1_PROVENANCE_AND_HASH_LEDGER.md`
5. `AEGIS_V1_REPRODUCIBILITY_CHECKLIST.md`

## Frozen primary environment context

The documented primary environment includes Windows/PowerShell, Python 3.10.11, the `.venv` interpreter under `D:\Research\AEGIS`, NVIDIA GeForce GTX 1660 SUPER, PyTorch 2.13.0+cu132, and CUDA availability. P3 must distinguish observed environment identity from dependencies that are strictly required.

## Frozen data boundary

The v0.35 corrective-training protocol used the frozen Fakeddit train and validation embedding caches:

- train cache: `data/processed/fakeddit/frozen_embeddings/train_n5000_seed42`
- validation cache: `data/processed/fakeddit/frozen_embeddings/validation_n1000_seed42`
- train manifest SHA256: `5BC8BF53F988B3DD5299A1D940DB6676F48E37250F52C8B3569150BE153417C5`
- validation manifest SHA256: `C4E95F818C24043DB1DEE9C0CFC0DE85C2E591E9E77B98CB2490046D1BC04F87`

Official-test samples accessed remain **0**.

## Scientific decision reproduction

A scientific reproduction must preserve seeds 42/43/44, the frozen checkpoint-selection rule, M3 `utility_probability_std >= 0.05` per seed, M4 `active_intervention_rate > 0` per seed, and the frozen interpretation of `CORRECTION_NOT_SUPPORTED` and robustness `NOT_COMPUTED`.

P3 packaging itself does not authorize a rerun. It documents how the frozen workflow and decisions are identified.

## Reproduction levels

**Level A 鈥?identity reproduction:** verify release, tag, commit, hashes, records, and evidence chain without executing the model.

**Level B 鈥?environment/workflow reproduction:** reconstruct the documented environment and workflow without claiming numerical reproduction.

**Level C 鈥?scientific rerun:** execute the frozen scientific workflow only under a separately authorized reproduction exercise. P3 does not itself authorize Level C.

## Limitations

Bitwise numerical reproducibility may depend on hardware, CUDA/cuDNN behavior, dependency builds, nondeterministic kernels, data availability, and third-party assets. P3 must not claim that repository documentation reproduces external assets that are absent or unavailable. Validation reproduction is not official-test evaluation.

## P3 Step2

After this protocol is remotely frozen, P3 Step2 may perform read-only evidence extraction and generate the five required reproducibility documents. No scientific execution is authorized.
