# AEGIS T1 Step3A 鈥?SD1.4 Backbone/Baseline Selection and Provenance

Parent Step3 commit: `19aabea1699ba464eae4ca06a7ffbe5e9cb8a3b0`.

## Selected backbone
**`timm/vit_small_patch16_224.dino`** is selected prospectively as the common visual encoder for B0, B1 and A1. It is a ViT-S/16 feature backbone with 21.7M parameters, 384-dimensional features, 224脳224 fixed input, and DINO self-supervised ImageNet-1k pretraining. The recorded safetensors SHA-256 is `8D4795B8E4E327C6D841E15A2E3E6D88C45346256A43F8CDAE8C26D92C806F7A`.

The selection is driven by compute feasibility on the GTX 1660 SUPER, a compact transformer representation, and the value of using a general self-supervised visual backbone rather than importing an existing synthetic-image detector. This does not claim that DINO is intrinsically optimal for forensic detection.

## Controlled comparison
B0 freezes the DINO ViT-S/16 encoder and trains only a linear binary head. B1 uses the same backbone with a prospectively bounded parameter-efficient or partial-fine-tuning policy. A1 uses the same shared encoder with an authenticity head and an explicit AEGIS reliability/error-risk head. This isolates the contribution of the AEGIS reliability design from a backbone change.

## Input contract
Evaluation uses decoded RGB pixels, bicubic backbone-compatible resizing, center crop to 224脳224, tensor conversion, and mean/std normalization `[0.485,0.456,0.406]` / `[0.229,0.224,0.225]`. Training augmentation remains unresolved and is not authorized here.

## Remaining design gates
Before implementation authorization, freeze: local timm compatibility/version; local weight acquisition and cryptographic verification; A1 reliability target/loss and head topology; B1 exact trainable-layer policy; and training augmentation.

No model download, model execution, implementation, training, formal evaluation, or official-test access is authorized by Step3A.
