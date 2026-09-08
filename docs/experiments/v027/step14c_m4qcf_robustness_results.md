# AEGIS v0.27 Step14C — M4qcf Formal Robustness Results

## Status

**STEP14C_SCIENTIFIC_RESULTS_FROZEN**

This record preserves the first formal Step14B analysis performed after
the Step14B operationalization and analyzer implementation were frozen.

## Provenance

- Step14A protocol commit: `5883d52`
- Step14A diagnostic implementation commit: `8b0fb7e`
- Step14B analysis operationalization commit: `8ac69d5`
- Step14B analyzer implementation commit: `aaba6e9`
- Input protocol: `0.27.0-step14a`
- Analyzer: `0.27.0-step14b`
- Seeds: 42, 43, 44
- Architectures: M1b, M4qc, M4qcf
- Official Fakeddit test accessed: **No**

## Formal decisions

| Hypothesis | Decision |
|---|---|
| H8-F | **SUPPORTED** |
| H9-F | **NOT_SUPPORTED** |
| H10-F | **NOT_SUPPORTED** |

## H8-F — corrupted-modality down-weighting

**Decision: SUPPORTED**

- Eligible seed-condition observations: 54
- Strict-positive observations: 54/54
- Strict-positive proportion: 1.000000000000
- Mean corrupted-modality down-weighting: 0.133367390132
- Frozen requirement: at least 80% strict-positive observations and
  aggregate mean down-weighting greater than zero.

H8-F is supported. Under the controlled representation-corruption
setting, M4qcf consistently shifted fusion weight away from the
corrupted modality.

## H9-F — robustness improvement over M1b

**Decision: NOT_SUPPORTED**

- Eligible seed-condition observations: 54
- Strict-positive M4qcf-minus-M1b observations:
  29/54
- Strict-positive proportion:
  0.537037037037
- Equal-weight mean Macro-F1 difference:
  0.051111288732
- Mean-improvement requirement: at least 0.01 — **PASS**
- Coverage requirement: at least 70% / 38 of 54 positive — **FAIL**

H9-F is therefore not supported. The aggregate mean improvement is
positive and exceeds the preregistered magnitude threshold, but the
improvement is not sufficiently consistent across seed-condition
observations.

## H10-F — mismatch robustness

**Decision: NOT_SUPPORTED**

- Interaction suppression positive seeds:
  3/3
- Mean interaction suppression D_I:
  0.420315261080
- Mean G:
  -0.007623939064
- D_I > 0 for all seeds — **PASS**
- Mean G > 0 — **FAIL**

### Per-seed values

| Seed | D_I | G |
|---:|---:|---:|
| 42 | 0.448590586446 | -0.011817905863 |
| 43 | 0.395744409232 | -0.005975861711 |
| 44 | 0.416610787562 | -0.005078049618 |

H10-F is therefore not supported. Compatibility-informed interaction
suppression behaved in the intended direction for all three seeds, but
the resulting classifier did not reduce mismatch-induced degradation
relative to M1b under the frozen criterion.

## Evidence integrity

- Quality sample rows audited:
  171000
- Mismatch sample rows audited:
  18000
- M4qcf reliability rows audited:
  63000
- Maximum fusion-weight sum error:
  1.4901161193847656e-07
- Step14A evidence audit: **PASS**
- Independent Step14B formal result audit: **PASS**

### Formal artifact fingerprints

- `formal_analysis.json`:
  `82179d5c7eaa60db4f5a188f0625087c2a85479edb78bb5f897dee8bf71dd2b4`
- `summary.json`:
  `2c71db7d513157a580e2d902f12d2461a47e4f725544683b09bb95778a45decd`

The formal analysis artifacts remain excluded from Git by the
repository's experiment-output policy. These hashes identify the exact
artifacts from which this tracked scientific record was produced.

## Scientific conclusion

Within the frozen controlled Fakeddit validation setting, M4qcf
demonstrated consistent reliability-informed fusion adaptation to
single-modality degradation: H8-F is supported.

That mechanistic success did not establish a sufficiently consistent
downstream robustness improvement over M1b: H9-F is not supported.
Likewise, compatibility-driven interaction suppression occurred
reliably, but it did not reduce mismatch-induced classification
degradation relative to M1b under the preregistered H10-F criterion:
H10-F is not supported.

The Step14 results therefore distinguish **successful internal
reliability adaptation** from **demonstrated downstream robustness
benefit**.

## Interpretation boundary

These results apply only to the frozen controlled Fakeddit validation
setting and the defined representation corruptions and deterministic
class-preserving mismatch. They do not establish open-world factual
verification, source credibility, author intent, human trust, universal
semantic consistency, arbitrary real-world corruption robustness,
external evidence verification, or temporal/geographic reasoning.

No post-exposure threshold, criterion, loss, fusion, epsilon, gamma,
checkpoint, or corruption-policy tuning was used to alter these formal
outcomes.
