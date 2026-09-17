# AEGIS Research Core v1.0 鈥?Reproducibility Checklist

Use this checklist fail-closed: an unresolved identity, data, environment, or decision-boundary mismatch must be reported rather than silently repaired.

- [ ] Confirm the intended baseline is AEGIS Research Core v1.0.
- [ ] Confirm annotated tag `v1.0.0-research-core`.
- [ ] Confirm tag target `39492cf6d93987fdf9ebee29b95fdb826b8391e4`.
- [ ] Confirm release manifest SHA256 `5BD15FB6FBECBB97A8727DD3E363D399B39B31EB6C8B1A673C238863FA715AAA`.
- [ ] Confirm release notes SHA256 `88F100C9B0536981C546C79C20F7CBE608850339C0F551B041106A15258748A5`.
- [ ] Record OS, shell, Python, PyTorch, CUDA, GPU, driver, and dependency state.
- [ ] Distinguish observed primary environment values from portable requirements.
- [ ] Confirm Fakeddit train-cache boundary and train manifest SHA256 `5BC8BF53F988B3DD5299A1D940DB6676F48E37250F52C8B3569150BE153417C5`.
- [ ] Confirm Fakeddit validation-cache boundary and validation manifest SHA256 `C4E95F818C24043DB1DEE9C0CFC0DE85C2E591E9E77B98CB2490046D1BC04F87`.
- [ ] Confirm official-test samples accessed remain 0 unless a future separately governed protocol explicitly changes that boundary.
- [ ] Confirm seeds 42, 43, and 44 for the frozen v0.35 corrective-training protocol.
- [ ] Confirm checkpoint selection: highest validation Macro-F1; tie-break lowest validation classification loss.
- [ ] Confirm M3 criterion `utility_probability_std >= 0.05` per seed.
- [ ] Confirm M4 criterion `active_intervention_rate > 0` per seed.
- [ ] Preserve v0.35 `CORRECTION_NOT_SUPPORTED`.
- [ ] Preserve v0.36 robustness `NOT_COMPUTED`.
- [ ] Do not infer classifier non-robustness from selector failure.
- [ ] Preserve v0.37 `ANALYST DECISION-SUPPORT RESEARCH CHARACTERIZATION`.
- [ ] Do not claim production readiness or autonomous attribution.
- [ ] Do not treat validation reproduction as official-test evaluation.
- [ ] Do not reselect checkpoints, tune thresholds, or alter prospective criteria during reproduction.
- [ ] Record any missing third-party data/model asset explicitly.
- [ ] Record any hardware/software divergence that could affect numerical reproducibility.
- [ ] Separate identity reproduction, environment/workflow reconstruction, and scientific rerun claims.
- [ ] If any required identity or boundary cannot be verified, stop and report the unresolved discrepancy.
