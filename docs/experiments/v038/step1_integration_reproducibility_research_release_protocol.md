# AEGIS v0.38 Step1 鈥?Integration, Reproducibility, and Research-Release Prospective Protocol

**Status:** PROSPECTIVE_RELEASE_READINESS_PROTOCOL_FROZEN_NO_AUDIT_COMPUTED

## Objective

Determine whether the frozen AEGIS Research Core is internally consistent, reproducible, provenance-complete, scientifically bounded, and suitable to freeze as a research-release candidate before v1.0.

## Prospective release-readiness gates

1. **R1 鈥?Repository identity and cleanliness:** exact remote lineage and clean tracked boundary.
2. **R2 鈥?Frozen evidence provenance:** stable identities and complete release-lineage records.
3. **R3 鈥?Environment/dependency reproducibility:** sufficient tracked reconstruction metadata; missing mandatory metadata is a blocker.
4. **R4 鈥?Command/workflow reproducibility:** inventory established runners, scripts, and documented workflows without rerunning scientific experiments.
5. **R5 鈥?Test/invariant readiness:** inventory tracked verification assets; execute only already-established non-scientific verification if safely identifiable.
6. **R6 鈥?Architecture/documentation consistency:** source structure and documentation must support an internally consistent research-core inventory.
7. **R7 鈥?Scientific claim/limitation consistency:** preserve all frozen SUPPORTED, NOT_SUPPORTED, NOT_COMPUTED, NOT_AVAILABLE, and PROHIBITED boundaries.
8. **R8 鈥?Artifact/release manifest completeness:** release candidate must not depend on unidentified untracked local artifacts.
9. **R9 鈥?Official-test/data boundary:** official-test access remains zero.
10. **R10 鈥?v1 readiness:** R1-R9 must pass with no unresolved scientific-integrity, provenance, or reproducibility blocker.

## Step2 authorization

Step2 may perform a repository-wide read-only inventory, stable tracked-file identity collection, environment/documentation/test inventories, release-manifest construction, and release-readiness reporting.

Already-established **non-scientific** verification may run only if it can be identified without changing code or scientific results.

Step2 may not change source behavior or architecture, train models, perform a new scientific evaluation or corruption experiment, tune thresholds, reselect checkpoints, access official tests, or claim production deployment.

## Blocker policy

Scientific failures remain frozen and are not repaired in v0.38. A reproducibility/integration blocker is frozen first; only a later narrow non-scientific correction may be authorized. Missing optional material may be recorded as a limitation; missing mandatory material is a blocker.

## Required release-candidate outputs

The release candidate must cover repository provenance, frozen evidence, environment/dependencies, workflows/commands, tests/invariants, architecture/components, scientific claims/limitations, release artifacts, release-readiness gates, and research-release notes.

## v1 policy

There is no ordinary v0.39/v0.40 feature cycle. Passing R10 may authorize the **AEGIS Research Core v1.0** freeze.

v1.0 means a reproducible, scientifically bounded **research-core** release; it is not a production-deployment-readiness claim.

## Integrity boundary

No audit result is computed in Step1. No source-behavior change, training, new scientific evaluation, official-test access, or production deployment occurs.

**Parent v0.37 closure:** `72807aa43f49c5fa1f7762f57de1dcd3d9915621`

**Official test samples accessed:** `0`

## Next

v0.38 Step2 鈥?one-pass repository-wide release-readiness audit and research-release candidate manifest construction under R1-R10.
