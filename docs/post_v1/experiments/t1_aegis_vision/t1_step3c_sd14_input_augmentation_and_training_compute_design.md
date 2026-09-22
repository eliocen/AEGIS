# AEGIS T1 Step3C 鈥?SD1.4 Input, Augmentation and Training-Compute Design

Parent Step3B commit: `479bf3fbc77dc1968fbd66da3961cd9fea9e1147`.

Training input is decoded RGB followed by class-independent random resized crop to 224脳224 (scale 0.8鈥?.0, ratio 0.9鈥?.1), horizontal flip p=0.5, tensor conversion and ImageNet normalization. Validation is deterministic: decoded RGB, backbone-compatible resize, center crop 224, tensor conversion and the same normalization.

JPEG recompression, blur, injected noise, color jitter, RandAugment, MixUp and CutMix are excluded from this first protocol. This keeps augmentation minimal and avoids deliberately destroying or manufacturing low-level forensic evidence. Raw container/metadata features remain prohibited.

Training design uses AdamW, backbone LR 1e-5, head LR 1e-4, weight decay 0.05, cosine decay with one warmup epoch, maximum 20 epochs, minimum 5 epochs, early-stopping patience 4, gradient clipping 1.0, CUDA AMP FP16, physical batch 16 and four-step gradient accumulation for effective batch 64. Seeds are 42, 43 and 44.

Checkpoint selection is based on highest validation authenticity Macro-F1, then lowest validation authenticity BCE loss, then earliest epoch. Reliability metrics are reported but do not select the checkpoint, preventing the auxiliary objective from superseding the primary authenticity task.

Any OOM or compatibility failure requires an explicit protocol amendment; batch size, transforms or architecture may not be silently changed. Local dependency compatibility and exact pretrained-weight acquisition/verification remain unresolved. No model download, execution, implementation, training, formal evaluation or official-test access is authorized.
