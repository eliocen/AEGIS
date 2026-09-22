# AEGIS T1 Step3 鈥?SD1.4 Baseline and Architecture Design Protocol

Parent dataset-provenance freeze: `630fc4141bec47dbfcb1b93456f20c6b2e78d73c`. Usable population identity: `09CF00DB0B3217677B615DA8E38B706A270B8EFA4FBF39F8A6EA0747C6C7161F` (323,997 train; 11,983 validation).

## Scientific objective
The first AEGIS-Vision experiment addresses binary image authenticity: authentic versus AI-generated. It does not establish source attribution, generator attribution, manipulation localization, domain-general robustness, or production readiness.

## Comparative design
Three controlled model roles are planned: **B0**, a frozen pretrained visual encoder with a linear classification probe; **B1**, a stronger fine-tuned or parameter-efficient visual classifier; and **A1**, the AEGIS-Vision detector using a shared pretrained visual encoder, binary authenticity head, and an explicit reliability/error-risk head. All must use the same frozen population and input contract so gains are interpretable.

## Input and shortcut boundary
Only decoded RGB pixels may enter the visual model. File extension, container format, file size, filename, path, and class-directory tokens are prohibited features. Backbone-native normalization will be used. A 224脳224 input is the current candidate, but the exact resize/crop transform must be frozen with the selected backbone before implementation.

## Compute boundary
The design must remain feasible on the verified NVIDIA GeForce GTX 1660 SUPER. Frozen-backbone, parameter-efficient, mixed-precision, gradient-accumulation, or small-batch strategies may therefore be selected prospectively. No giant foundation model will be trained from scratch.

## Governance
Step3 authorizes design only. No model execution, implementation, training, formal evaluation, or official-test access is authorized. Exact backbone/pretraining provenance, input transform, reliability target/loss, and baseline configuration must be frozen before Step4 implementation authorization.

Next: **Step3A 鈥?backbone/baseline selection and provenance review**.
