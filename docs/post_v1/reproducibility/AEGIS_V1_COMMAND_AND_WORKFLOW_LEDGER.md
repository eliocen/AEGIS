# AEGIS Research Core v1.0 鈥?Command and Workflow Ledger

## Reproduction workflow

| Phase | Purpose | P3 authorization |
|---|---|---|
| Repository identity | Verify branch/tag/commit and clean tracked state | Authorized, read-only |
| Artifact identity | Verify frozen SHA256 values and Git blob identity | Authorized, read-only |
| Environment reconstruction | Recreate documented Python/dependency environment | Documented; no scientific claim by itself |
| Data-boundary verification | Verify frozen cache paths and manifest identities | Authorized, read-only |
| Test/invariant verification | Identify frozen tests and previously frozen pass evidence | Documentation only in P3 |
| Corrective-training rerun | Re-execute seeds 42/43/44 | NOT AUTHORIZED by P3 |
| New evaluation | Generate new scientific measurements | NOT AUTHORIZED |
| Official-test evaluation | Access official test samples | NOT AUTHORIZED |

## Repository identity commands

The following are non-scientific identity checks:

```powershell
git rev-parse v1.0.0-research-core^{commit}
git show-ref --tags v1.0.0-research-core
git status --porcelain=v1
```

Expected release commit: `39492cf6d93987fdf9ebee29b95fdb826b8391e4`.

## Artifact verification pattern

```powershell
Get-FileHash -Algorithm SHA256 -LiteralPath <artifact>
git diff --quiet -- <tracked-artifact>
```

Use file SHA256 for exact working-file identity where line-ending normalization is controlled. Where Windows checkout normalization differs, use Git tracked/blob identity and `git diff --quiet` to distinguish content changes from checkout EOL representation.

## Scientific workflow boundary

The frozen v0.35 training protocol used seeds 42, 43, and 44 and selected the checkpoint with highest validation Macro-F1, tie-breaking on lowest validation classification loss. The frozen M3 criterion is `utility_probability_std >= 0.05` per seed; M4 is `active_intervention_rate > 0` per seed.

P3 records these commands/workflows but does not execute training, perform checkpoint reselection, alter thresholds, or access official-test data.
