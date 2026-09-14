# AEGIS v0.30 Step0 ??v0.29 Failure Decomposition

Status: **FROZEN**

Source release: `v0.29.0`
Closure commit: `2551430bea7f0cc8865265519e1d46da1a5c2230`
Evidence class: `CONTROLLED_VALIDATION_DEVELOPMENT_EVIDENCE`

## Frozen v0.29 outcome

- V29-H1: NOT_SUPPORTED
- V29-H2: NOT_SUPPORTED
- V29-H3: NOT_SUPPORTED
- V29-H4: NOT_SUPPORTED
- Clean safeguard: FAILED
- Official test samples accessed: 0

## Failure decomposition

The combined v0.29 controller reduced total prediction-flip rate relative to M1b
but did not achieve a favorable harmful/beneficial transition balance. M4qcs-w
remained the strongest stability reference, with only 5
net harmful flips and mean mismatch gap 0.001640214228.
M4qgrt produced 35 net harmful flips and harmful-to-beneficial
ratio 1.250000000000.

M4qgrt also failed broad graded-corruption improvement: mean delta versus M4qcs
was -0.005069211290, with only
9/36 positive observations.

The clean mean Macro-F1 delta versus M1b was
-0.012784645712, outside the frozen -0.01 safeguard.

## v0.30 design conclusion

v0.30 will test **evidence-conditioned selective reliability intervention**:
preserve a stable reference path by default and intervene only when reliability
evidence prospectively justifies intervention. The goal is not merely to reduce
the number of transitions, but to reduce harmful transitions while preserving
beneficial changes and clean performance.

No v0.30 implementation, training, formal evaluation, or official-test access
has occurred in this Step0 record.
