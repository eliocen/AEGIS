# AEGIS T1 Step 8Q — B0/B1 Comparative Closure and A1 Readiness

**Status:** Evidence prepared; remote freeze pending.
**Frozen parent:** `176197ef9af771cfc69e4115bae93f31c34b5264`
**Source Step8P SHA256:** `85731B5D68BE14D875F4D29FDC507B89B87F2FC8C1374B1D0DDBD8EFD2D81FFE`
**Frozen runner Git blob:** `aaa82f0261681cc3036714777bd662f0dd81be57`

## Descriptive validation results

| Variant | Seed 42 | Seed 43 | Seed 44 | Mean Macro-F1 | Sample SD |
|---|---:|---:|---:|---:|---:|
| B0 | 0.918866165 | 0.918701239 | 0.919615082 | 0.919060829 | 0.000487030 |
| B1 | 0.988483367 | 0.990486488 | 0.990736716 | 0.989902190 | 0.001235090 |

**B1 − B0 mean validation Macro-F1:** `0.070841361470` (about 7.08 percentage points).

## A1 architecture and semantics

- Reliability head: Linear(384,128), GELU, Dropout(0.1), Linear(128,1).
- Reliability input uses `z.detach()`: the reliability loss does not directly update the shared visual encoder.
- Reliability target is the binary correctness of the current authenticity prediction.
- Loss: authenticity BCE-with-logits + 0.25 × reliability BCE-with-logits.
- This review verifies implementation signatures and source semantics; it is not an independent runtime behavior test.

## Readiness and authorization boundary

- All A1 canonical run roots (seeds 42, 43, 44) were empty or absent at verification.
- Existing frozen training authorization permits A1 under the specified runner and seeds.
- This document **does not itself authorize launching A1/seed42**; a separate preflight and remote evidence freeze are required.
- No formal evaluation or official test-set access is authorized.

## Limitations

- Validation results are selection-exposed; they are not independent test estimates.
- Reliability prediction is not automatically calibrated.
- Cross-generator and domain generalization remain unestablished.

## Evidence hashes

- Step8P JSON SHA256: `85731B5D68BE14D875F4D29FDC507B89B87F2FC8C1374B1D0DDBD8EFD2D81FFE`
- Authorization SHA256: `22A351E83AF40EC2982A13DA35F14007A73EAB80543BDF61AFB118AF5B567453`
- Model source SHA256: `7D4398739C28C3D9EFBDFA9F70017E799A85AF0A502A827BDA59C139E3177900`
- Loss source SHA256: `BC01D2752AA60E1CDD1D3C285DC0ECDC015B41984C36D87B54C02A95F13ADC86`

This is a documentary freeze of the observed readiness state, not a claim that training or evaluation was executed.
