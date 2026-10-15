# T1 Step 8M - B1 seed44 Windows Update interruption and restart clearance

- Status: local evidence prepared; remote freeze pending.
- Frozen parent Step8K: 8fcac6792e664299ab82c8f334a8472e86d7e136
- Variant/seed: B1 / 44.
- Attempt: interrupted, not formally completed; no final_summary.json.
- Archived at: `artifacts/post_v1/t1_aegis_vision/failed_attempts/B1/seed_44/windows_update_restart_after_epoch14`.
- Verified best checkpoint: epoch 13; validation Macro-F1: 0.9905698351655304 (selection-exposed).
- History: 14 epochs, 70,844 recorded optimizer updates.
- Restart evidence: Windows Event 1074 at 2026-10-14 23:30:04 (MoUsoCoreWorker.exe) and 2026-10-15 09:05:40 (TrustedInstaller.exe).
- Restart timing does not prove the precise moment of training termination.
- Canonical run root is empty. No resume is authorized.
- Next: evidence review, allowlisted commit, remote verification, then fresh epoch-1 restart.
- Formal evaluation and official test access remain unauthorized.

## Archived artifact SHA256

- `best.json`: `7A0320E3D77189404D30F016150514075B8A2B72EABA35DD7D8CB3A56FA21DC3`
- `best.pt`: `726DB36598EC09098EDB480DA3C5CCCE65DC4A6803FB3F64EC415106C863CA5E`
- `history.json`: `456661845605EB140A081B5F8ED5029445FBF8BDE66EDA51B5A4717248611E5B`
- `provenance.json`: `ABB705C9B81F6091D7C7E040FCAE4C0BFD40B1019259219FBCE5B9BF05F26F95`
