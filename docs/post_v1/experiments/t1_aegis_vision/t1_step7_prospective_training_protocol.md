# AEGIS T1 Step7 鈥?Prospective Training Protocol

**Disposition:** `TRAINING_PROTOCOL_FROZEN_TRAINING_NOT_YET_AUTHORIZED`

The formal training matrix is B0, B1 and A1 across seeds 42, 43 and 44: **9 runs**. The frozen usable population is 323,997 training and 11,983 native-validation samples. AdamW uses backbone LR 1e-5, head LR 1e-4, weight decay 0.05, cosine decay with one warm-up epoch, maximum 20 epochs, minimum 5 epochs and early-stopping patience 4. Physical batch size is 16 with gradient accumulation 4 (effective batch 64), gradient clipping 1.0 and CUDA AMP FP16.

A1 uses `L_auth + 0.25 * L_rel`. The reliability representation is detached from the shared encoder, so the reliability loss trains the reliability head only by default. Checkpoint selection is lexicographic: highest validation authenticity Macro-F1, then lowest validation authenticity BCE, then earliest epoch. Reliability metrics cannot select the checkpoint.

The native validation population is explicitly **selection-exposed** because it drives epoch metrics, early stopping and checkpoint selection. It must not later be described as an untouched independent formal test set. Step10 therefore requires a separately prospective evaluation population/protocol or an explicitly selection-aware evaluation claim. The official test remains sealed.

Each variant/seed receives an isolated run root containing environment/provenance, epoch history, selected-checkpoint metadata and final summary. Python, NumPy, Torch CPU and CUDA RNGs must be seeded. Exact source commit, usable-manifest SHA256 and pretrained-weight SHA256 must be recorded. OOM, non-finite loss, or an incompatibility requiring a protocol change stops the affected run; no silent adjustment or replacement seed is permitted.

This Step7 freeze does **not** itself authorize an optimizer step. A training runner must implement this protocol and pass bounded dry verification before separate Step8 training authorization.
