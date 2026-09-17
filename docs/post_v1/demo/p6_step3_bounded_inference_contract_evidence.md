# AEGIS Post-v1 P6 Step3 鈥?Bounded Inference Contract Evidence

## Decision

**NOT_AUTHORIZED_PENDING_REVIEW_OF_FROZEN_CONTRACT_EVIDENCE**

This is a read-only static evidence audit. It does not import source modules, load a checkpoint, execute a model, train, evaluate, or access the official test set.

## Candidate source surfaces

| Path | main | argparse | predict/infer | checkpoint load | input signal | output signal | training signal | evaluation signal | test signal |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `aegis/ablation/__init__.py` | False | False | False | False | False | False | False | False | False |
| `aegis/ablation/aggregate.py` | False | False | False | False | False | False | False | False | False |
| `aegis/ablation/batch.py` | False | False | False | False | False | False | False | False | False |
| `aegis/ablation/comparison.py` | False | False | False | False | False | False | False | False | False |
| `aegis/ablation/multiseed.py` | False | False | False | False | False | False | False | False | False |
| `aegis/ablation/registry.py` | False | False | False | False | False | False | False | False | False |
| `aegis/ablation/result.py` | False | False | False | False | False | False | False | False | False |
| `aegis/ablation/runner.py` | False | False | False | False | False | False | False | False | False |
| `aegis/ablation/spec.py` | False | False | False | False | False | False | False | False | False |
| `aegis/ablation/statistics.py` | False | False | False | False | False | False | False | False | False |
| `aegis/acquisition/__init__.py` | False | False | False | False | False | False | False | False | False |
| `aegis/acquisition/base.py` | False | False | False | False | False | False | False | False | False |
| `aegis/acquisition/loader.py` | False | False | False | False | True | False | False | False | False |
| `aegis/acquisition/record.py` | False | False | False | False | True | False | False | False | False |
| `aegis/acquisition/validators.py` | False | False | False | False | True | False | False | False | False |
| `aegis/evaluation/__init__.py` | False | False | False | False | False | False | False | True | False |
| `aegis/evaluation/calibration.py` | False | False | False | False | False | True | False | True | False |
| `aegis/evaluation/classification.py` | False | False | False | False | False | True | False | True | False |
| `aegis/evaluation/errors.py` | False | False | False | False | False | True | False | True | False |
| `aegis/evaluation/evaluator.py` | False | False | False | False | False | True | False | True | False |
| `aegis/evaluation/hierarchical.py` | False | False | False | False | False | True | False | True | False |
| `aegis/evaluation/labels.py` | False | False | False | False | False | False | False | True | False |
| `aegis/evaluation/report.py` | False | False | False | False | False | False | False | True | False |
| `aegis/evaluation/stratified.py` | False | False | False | False | False | True | False | True | False |
| `aegis/multilingual_evaluation/__init__.py` | False | False | False | False | False | False | False | True | False |
| `aegis/multilingual_evaluation/comparison.py` | False | False | False | False | False | False | False | True | False |
| `aegis/multilingual_evaluation/evaluator.py` | False | False | False | False | False | True | False | True | False |
| `aegis/multilingual_evaluation/language.py` | False | False | False | False | False | False | False | True | True |
| `aegis/multilingual_evaluation/report.py` | False | False | False | False | False | False | False | True | False |
| `aegis/multilingual_evaluation/result.py` | False | False | False | False | False | False | False | True | True |
| `aegis/multilingual_evaluation/transfer.py` | False | False | False | False | False | False | False | True | True |
| `scripts/build_fakeddit_cache.py` | True | True | False | False | False | False | False | False | True |
| `scripts/run_fakeddit_ablation.py` | True | True | False | True | False | True | True | True | True |
| `scripts/run_fakeddit_alignment_ablation.py` | True | True | False | False | False | True | False | True | True |
| `scripts/run_fakeddit_unified_evaluation_v028.py` | True | True | True | True | True | True | False | True | True |
| `scripts/run_fakeddit_unified_evaluation_v029.py` | True | True | True | True | True | True | False | True | True |
| `scripts/run_fakeddit_unified_evaluation_v030.py` | True | True | True | True | True | True | False | True | True |
| `scripts/run_fakeddit_unified_evaluation_v031.py` | True | True | True | True | True | True | False | True | True |
| `scripts/run_fakeddit_unified_evaluation_v033.py` | True | True | True | True | True | True | False | True | True |
| `scripts/test_ablation.py` | False | False | False | False | False | False | False | True | False |
| `scripts/test_evaluation.py` | False | False | False | False | False | True | False | True | False |
| `scripts/test_multilingual_evaluation.py` | False | False | False | False | False | True | False | True | False |
| `scripts/test_robustness_evaluation.py` | False | False | False | False | False | True | False | True | False |
| `tests/test_ablation.py` | True | False | False | False | False | False | False | True | False |
| `tests/test_acquisition.py` | True | False | False | False | True | False | False | False | True |
| `tests/test_evaluation.py` | True | False | False | False | False | True | False | True | False |
| `tests/test_fakeddit_unified_evaluation_v028.py` | False | False | False | True | False | True | True | True | True |
| `tests/test_multilingual_evaluation.py` | True | False | False | False | False | True | False | True | False |
| `tests/test_robustness_evaluation.py` | True | False | False | False | False | True | False | True | False |
| `tests/test_v030_unified_evaluator_adaptation.py` | False | False | False | True | False | False | False | True | True |
| `tests/test_v031_unified_evaluator.py` | False | False | False | True | False | True | False | True | False |
| `tests/test_v033_unified_evaluator.py` | False | False | False | True | False | True | False | True | False |

Static token presence is observational only. It is not proof that a file is a supported v1.0 end-user inference interface.

## Tracked model-like artifacts

- None tracked.

## Required contract before D4 authorization

D4 requires all of the following to be established together: an explicit supported v1.0 inference entry point; exact compatible frozen checkpoint/artifact identity; stable input contract; defined non-evaluative output semantics; official-test isolation; no training or weight update; no new benchmark/evaluation side effects; and a claim-safe research-demonstration presentation contract.

## Execution boundary

Source import: **NONE**. Model load: **NONE**. Model execution: **NONE**. Training: **NONE**. New evaluation: **NONE**. Official test samples: **0**.

The next decision must use this frozen evidence. If the evidence does not establish the complete contract, D4 must become NOT_INCLUDED and P6 should close as a documentary demonstration package.
