# AEGIS T1 Step3B 鈥?Reliability Head, Target/Loss and Trainable-Parameter Design

Parent Step3A commit: `a341a9956465d7b52ba7c90bb0738f45327f4715`.

A1 uses the frozen-selected DINO ViT-S/16 384-dimensional representation. Its authenticity head is linear (384鈫?) with BCE-with-logits. The AEGIS reliability head is a lightweight 384鈫?28鈫? MLP with GELU and 0.1 dropout, also trained with BCE-with-logits.

The reliability supervision target is a detached per-sample correctness indicator: 1 when the current thresholded authenticity prediction matches the ground-truth authenticity label and 0 otherwise. The authenticity logit used to construct this target is detached. By default the reliability loss therefore trains the reliability head rather than providing an indirect route for changing the classifier merely to make reliability prediction easier. Total A1 loss is `L_auth + 0.25 * L_rel`.

For a controlled comparison, B1 and A1 use the same partial-backbone policy: only the final two transformer blocks and final normalization are trainable, together with their task heads. B0 freezes the entire encoder and trains only its linear authenticity head.

Reliability usefulness must be evaluated through output discrimination and calibration, including reliability-score standard deviation, Brier score, AUROC for correct versus incorrect predictions, and mean reliability on correct versus incorrect samples. Non-zero gradients or parameter movement alone are explicitly insufficient evidence of a useful reliability mechanism.

This is still prospective design only. Local weight verification, training augmentation, optimizer/scheduler, batch/accumulation/AMP policy and checkpoint selection remain unresolved. No model download, execution, implementation, training, formal evaluation, or official-test access is authorized.
