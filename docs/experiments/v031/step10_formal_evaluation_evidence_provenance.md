# AEGIS v0.31 Step10 鈥?Formal-Evaluation Evidence/Provenance Freeze

## Frozen source boundary

- Step8 source commit: `1468262092d1da6cb56c118b31359e923428c9ac`
- Evaluator SHA256: `46685C69198071E0D9114DA264647BF3DD77910C616000B7DD771531EC2DB919`
- Step7 operationalization SHA256: `6E072218C150E7BA2457298C7A945F7D8CEA3C234F518F147BE9C4672693C30F`
- Step8 implementation record SHA256: `611944E209DC6191FDB8E74F4272F05AFFA23AF688CC7D1DF5A5FCE5046C7602`

## Frozen formal evaluation evidence

- Formal output root: `experiments/fakeddit/v031_step9_formal_evaluation`
- Formal output tree SHA256: `17110ED5139AF09034D4EDECBF083339DEFC2AF2C7E03D6CCB1F9CA1D3399DFF`
- Formal artifact count: `888`
- Architectures: 7
- Seeds: 42, 43, 44
- Frozen checkpoints: 21
- Quality conditions per checkpoint: 19
- Quality summaries: 399
- Mismatch summaries: 42
- Mapping SHA256: `975967D77622BCDAD7E28597CB6F9F967C73BFE16D4A7455C1AB56CC7FBC6464`

The JSON provenance record contains the complete per-file formal-artifact manifest
and the exact 21-checkpoint manifest.

## Scientific boundary

Step10 performs evidence freezing only. It does not rerun evaluation, train any
model, reselect checkpoints, tune thresholds, compute V31-H1 through V31-H5, or
access the official test split.

V31-H1, V31-H2, V31-H3, V31-H4, and V31-H5 remain **NOT_COMPUTED**.
Official-test samples accessed remain **0**.

## Next stage

Step11 may compute the prospectively frozen v0.31 hypothesis criteria from this
immutable Step10 evidence. No post-exposure rebinning, checkpoint reselection,
threshold tuning, or official-test access is permitted.
