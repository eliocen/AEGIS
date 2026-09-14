# AEGIS v0.30 Step3A ??Runner and Ablation Integration Contract

Status: **FROZEN BEFORE RUNNER INTEGRATION AND FORMAL TRAINING**

## Exact runner mapping

- `quality_compatibility_selective_intervention_weights` -> `M4qesri-w` -> `weights_only`
- `quality_compatibility_selective_intervention_transition` -> `M4qesri-t` -> `transition_only`
- `quality_compatibility_selective_intervention` -> `M4qesri` -> `combined`

All three variants use the same M4qcs-compatible corruption/mismatch training
mixture, quality supervision, compatibility supervision, paired initialization,
and optimizer component policy. Their experimental difference is restricted to
the already-frozen Step2 intervention mode.

The v0.29 harmful-transition auxiliary loss is not active for any v0.30 mode.
`M4qesri-t` refers to the Step2 bounded forward-path transition intervention,
not the v0.29 training objective.

## Step3 verification boundary

Only synthetic forward smoke checks and repository tests are permitted. Step3
must not launch formal training, create formal v0.30 run directories, access the
official test split, perform formal evaluation, or compute V30-H1..H4.
