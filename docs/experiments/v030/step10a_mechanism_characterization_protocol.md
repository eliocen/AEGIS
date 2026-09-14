# AEGIS v0.30 Step10A 鈥?Prospective Mechanism-Characterization Protocol

Status: **PROTOCOL FROZEN BEFORE MECHANISM EVIDENCE EXTRACTION**

## Purpose

Characterize why v0.30 preserved the clean-performance safeguard but failed all four frozen formal hypotheses. Step10 is descriptive/mechanistic only. It cannot alter or rescue Step9 decisions.

## Immutable Step9 decisions

- Clean-performance safeguard: **PASSED**
- V30-H1: **NOT_SUPPORTED**
- V30-H2: **NOT_SUPPORTED**
- V30-H3: **NOT_SUPPORTED**
- V30-H4: **NOT_SUPPORTED**

## Characterization modules

1. **MC1 鈥?H1 robustness localization:** condition-, modality-, family-, severity-, and seed-level Macro-F1 deltas.
2. **MC2 鈥?H2 transition failure decomposition:** harmful/beneficial transitions, ratio, net harmful, flips, mismatch gap and per-seed behavior.
3. **MC3 鈥?H3 ablation attribution:** combined versus weights-only and transition-only, including descriptive synergy/interference.
4. **MC4 鈥?H4 gate response:** activation strength, separation, saturation/non-monotonicity patterns, mismatch response and severity trajectories.
5. **MC5 鈥?Evidence-performance coupling:** descriptive relationship between reliability evidence, intervention strength, and performance change.

## Scientific restrictions

No training, retraining, checkpoint loading, re-evaluation, threshold tuning, official-test access, new corruption conditions, dropped seeds, hypothesis redefinition, Step9 decision changes, or causal claims.

## Planned next output

Step10B will extract only the predefined evidence from the frozen Step8 artifacts and save:

- `docs/experiments/v030/step10b_mechanism_characterization_evidence.json`
- `docs/experiments/v030/step10b_mechanism_characterization_evidence.md`
