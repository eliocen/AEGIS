# AEGIS v0.29 Step3 Formal Training Results

Status: **FORMAL STEP3 TRAINING COMPLETE**

Evidence class: `CONTROLLED_VALIDATION_DEVELOPMENT_EVIDENCE`

Nine prospectively locked formal training runs completed across M4qgr, M4qtc, and M4qgrt with seeds 42, 43, and 44.

The official Fakeddit test split remained sealed and zero official-test samples were accessed.

## Frozen selected checkpoints

| Run | Architecture | Seed | Best epoch | Clean Macro-F1 | Clean cls loss | Checkpoint SHA256 |
|---:|---|---:|---:|---:|---:|---|
| 1 | M4qgr | 42 | 12 | 0.865935 | 0.370874 | `43D66CA55327CDD5D590905BDF6950AE49C3D48A54871A119D2DEE2AF024FCFE` |
| 2 | M4qgr | 43 | 9 | 0.861876 | 0.340912 | `34FE692A7FF966EDD33D4D14D30713F9E2C6D62CDB9C472B9EE39EB53606FCB6` |
| 3 | M4qgr | 44 | 5 | 0.864940 | 0.346842 | `613A14DCDFF526651905B2A2B4419D80A068B8CEFF7DC2408C65D2891507D9E3` |
| 4 | M4qtc | 42 | 12 | 0.854861 | 0.379009 | `251BD3871727C881EA9159EEDE13835DF7ECF0AEEF513802193F9586FCF094B0` |
| 5 | M4qtc | 43 | 9 | 0.865895 | 0.357214 | `C35BA7A87467461DB5C973442F155431278703534F73424D0F96001D140E1D74` |
| 6 | M4qtc | 44 | 13 | 0.862900 | 0.391258 | `8079A5938456A9575B7F388CC1CAA2AB44C73F3A9E8879A0AC4CCB461928F34D` |
| 7 | M4qgrt | 42 | 8 | 0.858807 | 0.357467 | `543EEA54A3A34303D2581969347B9C13B2E4DFD6CDEEEED27856A646081AA48A` |
| 8 | M4qgrt | 43 | 6 | 0.859838 | 0.353500 | `20B655E2AB26F940C6583A8B005400F55DEC84C2B172A5AAFFF201BF75986EB1` |
| 9 | M4qgrt | 44 | 5 | 0.863824 | 0.355661 | `EABBD1FC5BECC959967DDA7AB662BEB51B8C452F2B20880C9BEE537F6366752D` |

## Scientific boundary

- No formal v0.29 evaluation has yet been performed.
- No formal v0.29 hypothesis decision has yet been computed.
- No checkpoint reselection is permitted from later diagnostic results.
- No parameter search or architecture modification occurred during formal training.
- Official test access remains 0.

Next gate: prospective v0.29 formal evaluation operationalization derived from the frozen Step1 protocol.
