# AEGIS v0.34 Step9 鈥?Formal Training Operationalization & Authorization Freeze

**Status:** FORMAL_TRAINING_OPERATIONALIZATION_AUTHORIZED_AND_FROZEN

Step9 prospectively authorizes exactly three primary formal M4qusli runs: seeds 42, 43, and 44. No architecture, loss, target-transform, intervention-threshold, or checkpoint-selection redesign is authorized.

## Frozen training configuration

`multimodal / quality_compatibility_utility_supervised_intervention`; 50 maximum epochs; batch size 32; learning rate 0.001; weight decay 0.0001; shared/hidden dimensions 128; dropout 0.1; temperature 0.07; alignment weight 0; classification/quality/compatibility weights 1; effective utility loss weight 0.50; transition objective weight explicitly 0; gradient clip 1; patience 8; minimum epochs 5; minimum delta 1e-6.

Checkpoint selection remains validation Macro-F1 with validation classification loss as tie-breaker. Post-result checkpoint reselection and threshold search are prohibited.

Each formal run must prospectively persist sample-level training observability, optimization-step observability, and per-epoch observability summaries. Missing required diagnostics are reported NOT_AVAILABLE and may not be reconstructed by retraining after formal results are exposed.

Formal evaluation remains unauthorized. Official test access remains zero.

**Next:** Step10 executes exactly the three authorized runs and freezes formal-training evidence/provenance.
