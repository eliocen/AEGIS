# AEGIS v0.27 - M4q Multi-Seed Training Results

## Experiment status

**Step:** 8  
**Variant:** M4q - M1b gated-interaction backbone with explicitly supervised intrinsic modality-quality heads  
**Dataset:** Fakeddit  
**Evaluation:** Held-out validation/development protocol  
**Seeds:** 42, 43, 44  
**Official test split:** SEALED / NOT ACCESSED

## Research purpose

Step 8 evaluates whether adding explicit corruption-aware intrinsic
modality-quality supervision to the M1b backbone preserves clean
classification performance.

This step does **not** establish that the learned quality scores have
meaningful degradation semantics. That question is reserved for the
Step 9 corruption diagnostics and Step 10 hypothesis analysis.

## Per-seed results

| Seed | Best epoch | Epochs completed | Best Macro-F1 | Classification loss | Stop reason |
|---:|---:|---:|---:|---:|---|
| 42 | 12 | 20 | 0.872979 | 0.324977 | early_stopping |
| 43 | 8 | 16 | 0.877741 | 0.311323 | early_stopping |
| 44 | 7 | 15 | 0.866970 | 0.313682 | early_stopping |

## Aggregate classification result

Best validation Macro-F1 values:

0.872978533, 0.877741301, 0.866970068

Mean:

**0.872563301**

Sample standard deviation:

**0.005397608**

Accordingly, the M4q three-seed validation result is:

**0.8726 +/- 0.0054**

## H6-CLS assessment

Frozen M1b reference:

**0.8736 +/- 0.0029**

M4q mean:

**0.8726**

Difference:

**Delta Macro-F1(M4q - M1b) = -0.001037**

Preregistered descriptive criterion:

**Delta Macro-F1 >= -0.01**

Result:

**H6-CLS: SATISFIED on the frozen validation protocol.**

This result supports the limited conclusion that explicit intrinsic
quality supervision preserves clean validation classification
performance within the preregistered margin.

It is not a formal statistical non-inferiority test.

## Training integrity

All three runs:

- used the frozen 5,000-example training cache;
- used the frozen 1,000-example held-out validation cache;
- used the M4q quality-supervised architecture;
- used quality-loss weight lambda_q = 1.0;
- retained the M1b gated-interaction backbone;
- selected checkpoints using validation Macro-F1 with validation
  classification loss as the tie-breaker;
- completed with parameter updates to both modality-quality
  estimators and the intended shared/classification components; and
- recorded the official Fakeddit test split as sealed and not
  accessed.

## Scientific interpretation

Step 8 establishes classification retention only.

The following claims are **not yet established**:

1. that text-quality scores decrease appropriately with increasing
   text corruption;
2. that vision-quality scores decrease appropriately with increasing
   vision corruption;
3. that quality degradation is modality-selective;
4. that zeroed modalities receive appropriately low quality scores
   while intact modalities retain high scores;
5. that cross-modal compatibility is learned;
6. that the system performs factual verification; or
7. that the quality scores generalize to real-world/raw-input
   corruption.

These questions remain subject to the preregistered Step 9 and Step
10 diagnostics.

## Next phase

Step 9 will evaluate the frozen best M4q checkpoints from seeds
42, 43, and 44 under deterministic controlled validation corruption.

The primary subsequent hypotheses are H1-Q, H2-Q, and H3-Q.

M4qc and quality-informed fusion remain outside the scope of this
record.
