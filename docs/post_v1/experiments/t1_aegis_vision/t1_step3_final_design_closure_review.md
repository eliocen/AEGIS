# AEGIS T1 Step3 鈥?Final Design Closure Review

Parent commit: `07bab30b8cb76bfe8aaabedf5cce2caf6f92b8f8`.

## Disposition
**STEP3_SCIENTIFIC_DESIGN_COMPLETE_STEP4_IMPLEMENTATION_AUTHORIZATION_ADMISSIBLE**

The SD1.4 usable population is frozen at 323,997 training and 11,983 native-validation samples under manifest SHA-256 `09CF00DB0B3217677B615DA8E38B706A270B8EFA4FBF39F8A6EA0747C6C7161F`.

The common backbone is `timm/vit_small_patch16_224.dino`. The acquired artifact is `86676340` bytes and its SHA-256 is exactly `8D4795B8E4E327C6D841E15A2E3E6D88C45346256A43F8CDAE8C26D92C806F7A`. Step3F's differing expected byte count is reconciled as metadata error; the prospectively frozen cryptographic acceptance criterion matched exactly.

B0 is a frozen encoder plus linear authenticity head. B1 trains the final two transformer blocks, final norm and authenticity head. A1 uses the same backbone trainability plus the frozen-designed 384鈫?28鈫? reliability head. Reliability is supervised by a detached correctness indicator with BCE-with-logits at loss weight 0.25; optimization activity alone is not evidence of useful reliability.

The compute/input protocol remains RGB 224脳224, AdamW, physical batch 16, accumulation 4, effective batch 64, AMP FP16 and seeds 42/43/44. The verified environment is Python 3.10.11, torch 2.13.0+cu132, torchvision 0.28.0+cu132, timm 1.0.29 and CUDA 13.2 on GTX 1660 SUPER.

## Remaining controlled limitations
The weight is locally untracked and must remain outside Git; Step4 implementation must add or verify an ignore rule before any implementation commit. Residual pixel-level JPEG/PNG provenance artifacts remain a known dataset limitation, so this first experiment cannot support domain-general robustness claims.

Step3 is closed. Step4 implementation authorization is admissible, but training, formal evaluation, official-test access, production claims and domain-general robustness claims remain prohibited.
