# AEGIS v0.34 Step10 鈥?Formal Training Evidence and Provenance Freeze

**Status:** FORMAL_TRAINING_EVIDENCE_AND_PROVENANCE_FROZEN

- Parent Step9 commit: `cb7099002b683eff2c9060df8b6af70efd836e72`
- Architecture: `M4qusli`
- Formal RNG seeds: `42, 43, 44`
- Shared frozen cache sampling seed: `42`
- Formal-training evidentiary files: `82`
- Formal-training tree SHA256: `8D035BB46FC909A1AA21A277160C6CEE2F456BB3EDF3D38401F953B0A709BE45`
- Runner SHA256: `08E9A9466A641316280739BB948E13EC35DE408700CB07EF7584863A918D4B61`
- Step9 authorization JSON SHA256: `7F027BE122C1DB74256662317504AD63C1E4F5D916CF369066D328F51078E440`
- Training cache manifest SHA256: `5BC8BF53F988B3DD5299A1D940DB6676F48E37250F52C8B3569150BE153417C5`
- Validation cache manifest SHA256: `C4E95F818C24043DB1DEE9C0CFC0DE85C2E591E9E77B98CB2490046D1BC04F87`
- Prospective observability verification: **PASS for all three seeds**
- Seed42 preservation: **PASS 鈥?completed result was not rerun**
- Effective utility-loss weight: `0.50`
- Transition-objective weight: `0.0`
- Official test samples accessed: `0`
- Formal evaluation executed: **NO**
- Formal hypothesis decisions made: **NO**
- Checkpoint reselection: **NO**
- Threshold search/tuning: **NO**

## Scientific boundary

This step freezes formal-training evidence and provenance only. It does not adjudicate the
v0.34 hypotheses and does not authorize formal evaluation or additional training.
Prospectively persisted observability is retained as evidence for later analysis under a
separately frozen protocol.

## Run evidence

### Seed 42

- Run root: `experiments/fakeddit/v034/step10_formal_training/M4qusli_seed42`
- Metrics rows: `19`
- Active intervention rate: `0.0`
- Utility probability mean: `0.49477014608791`
- Utility probability std: `0.012056356727677615`
- Selector parameter L2 movement: `0.08252905305988768`
- Median selector gradient norm on non-neutral steps: `0.046311032663833344`
- Official test: `sealed_not_accessed`

### Seed 43

- Run root: `experiments/fakeddit/v034/step10_formal_training/M4qusli_seed43`
- Metrics rows: `19`
- Active intervention rate: `0.0`
- Utility probability mean: `0.4960478912974659`
- Utility probability std: `0.010378722941974613`
- Selector parameter L2 movement: `0.14235170802782893`
- Median selector gradient norm on non-neutral steps: `0.02864261777577873`
- Official test: `sealed_not_accessed`

### Seed 44

- Run root: `experiments/fakeddit/v034/step10_formal_training/M4qusli_seed44`
- Metrics rows: `20`
- Active intervention rate: `0.0`
- Utility probability mean: `0.4930366360729933`
- Utility probability std: `0.01315188747884486`
- Selector parameter L2 movement: `0.22143169428344184`
- Median selector gradient norm on non-neutral steps: `0.035981796893984515`
- Official test: `sealed_not_accessed`
