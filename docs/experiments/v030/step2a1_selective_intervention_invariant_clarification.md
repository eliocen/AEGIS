# AEGIS v0.30 Step2A.1 ??Selective-Intervention Invariant Clarification

Status: **PROSPECTIVE CLARIFICATION BEFORE FORMAL TRAINING/EVALUATION**

A non-evidentiary focused tensor-contract test exposed an internal Step2A
specification conflict.

`M4qesri-t` is required to preserve the exact `M4qcs-w` stable-reference
modality weights because its weight-intervention pathway is disabled. The frozen
M4qcs-w allocation can legitimately produce a modality weight below `0.10` or
above `0.90`.

Therefore the `[0.10,0.90]` final-weight bound applies only to architectures in
which selective weight intervention is enabled:

- `M4qesri-w`: bound applies.
- `M4qesri`: bound applies.
- `M4qesri-t`: bound does not apply; exact stable-reference weights are required.

No tensor equation, risk coefficient, gate threshold, intervention magnitude,
training rule, evaluation rule, hypothesis gate, or scientific result is
changed by this clarification.

No formal training, formal evaluation, formal decisions, or official-test access
occurred before this clarification.
