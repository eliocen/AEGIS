# AEGIS v0.31 Step4 鈥?Locked Formal-Training Operationalization

Step4 freezes the exact formal-training execution plan for v0.31. No formal training, formal evaluation, checkpoint loading, hypothesis decision, or official-test access occurs in this step.

## Formal run roster

- Architectures: M4qcesi-c, M4qcesi-u, M4qcesi
- Seeds: 42, 43, 44
- Exact formal run count: 9

## Frozen auxiliary weights

- M4qcesi-c: calibration 0.25; utility 0.00
- M4qcesi-u: calibration 0.00; utility 0.50
- M4qcesi: calibration 0.25; utility 0.50
- v0.29 harmful-transition objective: 0.00 for every v0.31 architecture
- Utility-target temperature: 0.05

## Shared hyperparameters

- epochs 50; batch size 32; learning rate 0.001; weight decay 0.0001
- shared dim 128; hidden dim 128; dropout 0.1; temperature 0.07
- alignment 0.5; classification 1.0; quality 1.0; compatibility 1.0
- gradient clip 1.0; patience 8; minimum epochs 5; min delta 1e-6

## Checkpoint selection

Checkpoint selection uses clean validation Macro-F1 only, maximized. Ties are broken by lower clean validation classification loss. Robustness, mismatch, intervention, utility, calibration, and official-test metrics are prohibited for checkpoint selection.

## Scientific integrity

Hyperparameter search, post-result threshold tuning, seed dropping, checkpoint reselection after formal evaluation, validation-derived gate/calibration tuning, and official-test access are prohibited.
