# AEGIS T1 Step2D12 鈥?SD1.4 Audit Evidence and Prospective Anomaly Disposition

The Step2D11 full-population audit is bound to manifest SHA-256 `D06AF407F08B0839FB7BDA28FF324293F2F0A418DB434DDA620BC5100BE31D2D` and summary SHA-256 `9E3F50136FB8FBE558F5C0D778A7320F83FA5D193082BF7F23ACE188931AE750`. It observed **336,000 files**, **120 exact duplicate hash groups**, **0 cross-class duplicate groups**, **17 cross-split duplicate groups**, and **3 decode failures**.

All 17 cross-split collisions occur within the authentic/nature class between native train and validation. The three decode failures occur in `train/ai`. The audit also establishes a perfect file-format/class association in this subset: AI files are PNG while nature files are JPEG. Median file sizes also differ strongly by class (train AI 460,668 B vs train nature 107,535 B; validation AI 458,176 B vs validation nature 127,222 B). These are dataset-level shortcut risks and are not evidence of semantic authenticity discrimination.

Prospectively, the raw dataset remains immutable. Decode-failing files shall be excluded only through the derived usable-population manifest. For each exact train-validation collision, the validation occurrence shall be excluded from the usable validation population while the training occurrence is preserved. No raw file is deleted, repaired, moved, relabeled, or overwritten. Other same-split exact duplicates remain unchanged pending any separately governed decision.

Before model training, AEGIS-Vision must freeze an input-standardization and shortcut-mitigation design addressing format and related low-level acquisition artifacts. Preprocessing execution, model execution, training, formal evaluation, and official-test access remain NOT_AUTHORIZED.
