# AEGIS v0.31 Step6 鈥?Formal-Training Evidence / Provenance Freeze

Step6 freezes the artifacts and provenance of the nine completed v0.31 formal-training runs. No retraining, formal evaluation, hypothesis analysis, checkpoint reselection, or official-test access occurs in this step.

## Seed identity correction

The first Step6 provenance attempt assumed experiment.json or summary.json exposed a top-level seed. They do not. This is a provenance-schema issue only. The authoritative seed identity is the prospectively frozen Step4 roster together with the deterministic Step5 run-directory mapping. Any seed metadata present inside run artifacts is checked for agreement, but its absence is not an error.

## Frozen execution set

- Architectures: M4qcesi-c, M4qcesi-u, M4qcesi
- Seeds: 42, 43, 44
- Formal runs completed: 9/9
- Official test samples accessed: 0
- V31-H1..H5: NOT_COMPUTED

## Frozen evidence

Each run records SHA256 identities for experiment.json, metrics.jsonl, summary.json, best_validation_predictions.json, best_model.pt, final_model.pt, plus any additional run-local artifacts present at freeze time. Each run directory and the overall Step5 root also receive deterministic tree hashes.

Validation metrics are provenance and checkpoint-selection metadata only. They are not formal robustness evidence and cannot be used to revise the prospective hypotheses.

## Scientific boundary

Additional v0.31 formal training before Step7 is prohibited. Formal evaluation and formal hypothesis decisions remain prohibited until their prospectively frozen stages.
