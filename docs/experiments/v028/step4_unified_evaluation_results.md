# AEGIS v0.28 Step4 Unified Evaluation Results

> **FROZEN CONTROLLED VALIDATION-DEVELOPMENT EVIDENCE — FORMAL DECISIONS DEFERRED**

## Record identity

- Record version: `0.28.0-step4-results`
- Implementation commit: `22a653b9b9f15dd6101fe06a206f660df9a632d1`
- Architectures: 6
- Formal seeds: 42, 43, 44
- Frozen checkpoints: 18
- Quality summaries: 342/342
- Mismatch summaries: 36/36
- Official-test samples accessed: 0

## Scientific boundary

Step4 records the complete locked evaluation measurements. It does not apply the frozen support gates and does not decide V28-H1, V28-H2, V28-H3, or V28-H4. Those decisions remain exclusively reserved for Step5.

| Hypothesis | Step4 status |
|---|---|
| V28-H1 | NOT_COMPUTED |
| V28-H2 | NOT_COMPUTED |
| V28-H3 | NOT_COMPUTED |
| V28-H4 | NOT_COMPUTED |

## Descriptive quality aggregates

These aggregates are descriptive measurements, not formal hypothesis decisions.

| Architecture | Mean clean Macro-F1 | Mean non-clean Macro-F1 | Mean text-Gaussian Macro-F1 |
|---|---:|---:|---:|
| M1b | 0.873607377163 | 0.794148096978 | 0.865626296288 |
| M4qc | 0.864839357609 | 0.846795128588 | 0.855665297966 |
| M4qcf | 0.867415367988 | 0.845259385710 | 0.857283634953 |
| M4qcs-w | 0.864553821521 | 0.841814940629 | 0.854530180992 |
| M4qcs-i | 0.864196973251 | 0.838551564840 | 0.851243784793 |
| M4qcs | 0.866071324143 | 0.842366107598 | 0.854806608134 |

## Descriptive mismatch aggregates

| Architecture | Matched Macro-F1 | Mismatched Macro-F1 | Mean gap | Flips | Harmful | Beneficial | Net harmful | Harmful/beneficial |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| M1b | 0.873607377163 | 0.858900286917 | 0.014707090247 | 554 | 299 | 255 | 44 | 1.172549019608 |
| M4qc | 0.864839357609 | 0.856492349138 | 0.008347008471 | 345 | 185 | 160 | 25 | 1.156250000000 |
| M4qcf | 0.867415367988 | 0.845084338677 | 0.022331029311 | 357 | 212 | 145 | 67 | 1.462068965517 |
| M4qcs-w | 0.864553821521 | 0.862913607293 | 0.001640214228 | 361 | 183 | 178 | 5 | 1.028089887640 |
| M4qcs-i | 0.864196973251 | 0.845134411769 | 0.019062561482 | 327 | 192 | 135 | 57 | 1.422222222222 |
| M4qcs | 0.866071324143 | 0.852401048033 | 0.013670276110 | 373 | 207 | 166 | 41 | 1.246987951807 |

## Integrity and safety

- The same corruption realizations were shared across architectures and seeds.
- The same deterministic class-preserving mismatch mapping was shared across architectures and seeds.
- Every checkpoint retained identical before/after model fingerprints.
- Training, checkpoint reselection, threshold tuning, and official-test access were not performed.
- No architecture, seed, condition, or recorded result was dropped.

## Next step

Step5 — apply the prospectively frozen conjunctive gates to the frozen Step4 record and compute V28-H1 through V28-H4 exactly once.
