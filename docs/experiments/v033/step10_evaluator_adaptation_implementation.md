# AEGIS v0.33 Step10 鈥?Evaluator Adaptation Implementation

The frozen v0.31 unified evaluator was adapted only for the prospectively
frozen v0.33 formal roster and M4qusli utility-supervision diagnostics.

## Frozen implementation boundary

- Parent Step9 commit: `b208021818e5984e9b93893b4436f2a27db4f5b9`
- Parent evaluator SHA256: `46685C69198071E0D9114DA264647BF3DD77910C616000B7DD771531EC2DB919`
- v0.33 evaluator SHA256: `49678E9B0E4E0CB9F22253DC00B00453D7FFD3CC40C64B5A42EC66D5897B2623`
- v0.33 evaluator regression SHA256: `96E8C50A448A6A66F0A627C4ABD39F938FBD723CFB7A4261DBCBD0923E2FA7EC`

The inherited 19-condition corruption matrix, class-preserving mismatch
construction, binary metrics, mismatch-transition metrics, and historical
comparator semantics remain unchanged.

## M4qusli adaptation

The evaluator adds M4qusli checkpoint/model routing and records utility
probability, utility gate, intervention gate, active-intervention indicator,
p_ref_true, p_candidate_true, delta_u, and u_target.

Counterfactual target quantities are recomputed from the stable reference and
candidate fused embeddings through the same frozen classifier and exact
utility-target transform used in training.

The v0.33 active-intervention contract is preserved at utility probability
>= 0.60; v0.31's 0.50 gate threshold is not reused for M4qusli.

## Engineering smoke

A non-formal seed-42 smoke over all four architectures and two smoke
conditions completed with 8 quality summaries and 8 mismatch summaries.

The evaluator persists four root evidence JSONs: `summary.json`,
`manifest.json`, `quality_condition_summaries.json`, and
`mismatch_summaries.json`. The class-preserving mapping itself is used in
memory; its frozen SHA256 `975967D77622BCDAD7E28597CB6F9F967C73BFE16D4A7455C1AB56CC7FBC6464`
is persisted and verified identically in both summary and manifest.

The post-smoke recovery corrected two verifier assumptions only:
1. sample cardinality is in `metrics.sample_count`, while diagnostic summaries
   expose `mean`, `std_population`, `min`, and `max`;
2. no standalone root `mismatch_mapping.json` is part of the evaluator output
   contract.

The evaluator and smoke outputs were not changed or rerun.

Engineering-smoke tree SHA256:
`EFD7DD57B0B24671E99748378239ED7556D472424468A2E9C0A5B67FD189613A`

No training, formal evaluation, checkpoint reselection, threshold tuning,
hypothesis decision, or official-test access occurred. V33-H1 through
V33-H5 remain NOT_COMPUTED.

Next stage: Step11 formal unified evaluation execution.
