# AEGIS v0.30 Step6 鈥?Formal Training Evidence & Provenance Freeze

## Status

- Formal training: **COMPLETE (9/9)**
- Formal evaluation: **NO**
- Formal hypothesis decisions: **NO**
- Official Fakeddit test samples accessed: **0**
- Training/retraining in Step6: **NO**
- Checkpoint reselection in Step6: **NO**

## Frozen source boundary

- Step4 operationalization commit: `7483adf96f51dfc6a243f1cf855ecf9f5098ca7e`
- Step5A routing-correction commit: `3230843532421f274b2ba0b70f0f7e09d8c7e600`
- Corrected runner SHA256: `2AB9072852A578E1CFC4A87DBD5553E40A5501EE132833440119526E48973D1A`

## Formal training evidence

| Run | Architecture | Seed | Root tree SHA256 |
|---:|---|---:|---|
| 1 | M4qesri-w | 42 | `3F801714FBB569BF6E3CD627A3A5C3FC46249DD856CEE572ED17944A4178960A` |
| 2 | M4qesri-w | 43 | `87DEA62F1C53C4B17C4FA76F6B9F3492BAEB3E22401CA0B9A23871403CB18A83` |
| 3 | M4qesri-w | 44 | `7AA128E507FFA15A57E46ABC889C02623494D9D3D8F8D0A4A6ACB34920E7F0CE` |
| 4 | M4qesri-t | 42 | `25C5D2D875C7E9A470637C93755AD488D1357322B4E99CC55121789081378638` |
| 5 | M4qesri-t | 43 | `1BE9D8EEEE1A512893CC1DDD86E4738FB99EFF10FC32E99546FCBBD05FDAF088` |
| 6 | M4qesri-t | 44 | `B41C7211475AD7DCB1793AE77026A3A4CAAFCB9B08C137D969423B2E445D3B73` |
| 7 | M4qesri | 42 | `29B9265D5DF99C555657B242AE90457B0F546213405B6BA7DBA76F53684F446D` |
| 8 | M4qesri | 43 | `7326E272A12C0E0435AC52D20EC79BF67EA59E9161B89F32EF4B92A0E3BCDA43` |
| 9 | M4qesri | 44 | `84FFFE620FDF4C7F3684450C397DA6680D3EE67256C398A58F6EB64A287A97B8` |

Each run is bound in the JSON provenance manifest to the exact frozen Step4 command, command SHA256, training log, seven required Step5 evidence artifacts, every file in the experiment root, byte size, and SHA256.

## Preserved technical failure

The original pre-correction run-1 failure is retained as provenance and is explicitly classified as a technical training-routing defect that produced no formal training result. Its directory tree and failure log are hashed in the JSON manifest.

## Scientific boundary

The Step6 freeze does not perform corruption/mismatch evaluation, does not load checkpoints for evaluation, does not compute V30-H1鈥揤30-H4, does not access the official test split, and does not alter any scientific configuration.

## Next stage

**v0.30 Step7 鈥?Formal Evaluation Operationalization**
