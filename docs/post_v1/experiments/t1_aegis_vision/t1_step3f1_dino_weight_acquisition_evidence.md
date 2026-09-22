# AEGIS T1 Step3F1 鈥?DINO Weight Acquisition Evidence

Parent Step3F commit: `f3a0cc46e597200f384c3e8c563e31eb77971958`.

The authorized DINO ViT-S/16 safetensors artifact is present at `data/models/timm/vit_small_patch16_224.dino/model.safetensors`.

- Observed bytes: `86676340`
- Observed SHA-256: `8D4795B8E4E327C6D841E15A2E3E6D88C45346256A43F8CDAE8C26D92C806F7A`
- Required SHA-256: `8D4795B8E4E327C6D841E15A2E3E6D88C45346256A43F8CDAE8C26D92C806F7A`
- Cryptographic identity: **VERIFIED**
- Git path status: `?? data/models/timm/vit_small_patch16_224.dino/model.safetensors`

The observed byte count is retained as evidence. Acceptance is based on the prospectively frozen SHA-256 identity, which matches exactly. No model instantiation, safetensors tensor load, state-dict load, or forward pass was performed.

Acquisition result: **SUPPORTED**.

The next gate is Step3 final design-closure review; this evidence freeze itself does not authorize implementation, model execution, training, formal evaluation, or official-test access.
