# AEGIS T1 Step7D 鈥?Prospective Training Execution Semantics

This record prospectively completes operational details left underspecified by the frozen Step7 protocol. It does not authorize training and does not alter the Step3/Step7 scientific design.

## Scheduler
Scheduler stepping is per optimizer update. The first epoch is linear warmup, progressing from `1 / warmup_updates` to the frozen base learning rate. Remaining updates use cosine decay to zero at the final planned optimizer update of epoch 20, preserving the backbone/head learning-rate ratio.

## Gradient accumulation
Physical batch size remains 16, accumulation remains four microbatches, and `drop_last=False`. Each microbatch mean loss is multiplied by its actual sample count before backward. At an accumulation boundary, AMP gradients are unscaled, divided by the accumulated sample count, clipped to global norm 1.0, and only then stepped. This preserves a sample mean for a final partial accumulation window.

## AdamW
Frozen learning rates and weight decay are unchanged. Weight decay 0.05 applies to all trainable parameters; no bias/norm exemptions are introduced.

## Checkpointing and early stopping
Checkpoint comparison uses exact finite values in the frozen lexicographic order: highest validation authenticity Macro-F1, then lowest validation authenticity BCE, then earliest epoch. Improvement is tracked from epoch 1 and resets the no-improvement counter. Stopping is forbidden before five completed epochs. At or after epoch 5, four consecutive completed epochs without lexicographic improvement trigger stopping. Reliability metrics never select the checkpoint.

## Metrics and reproducibility
Authenticity classification uses logit >= 0. AUROC is deterministic and rank-based with average ranks for ties. BCE is sample-weighted. Train shuffling uses a generator seeded by the formal run seed; validation is not shuffled; worker seeds derive from `torch.initial_seed() mod 2^32`; no samples are dropped.

## Execution boundary
Training remains NOT AUTHORIZED. The eventual runner must remain execution-locked until a separate Step8 authorization is frozen. Step7D performs no optimizer step, no training, no formal evaluation, and no official-test access.
