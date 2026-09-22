# AEGIS T1 Step8E 鈥?B0/Seed42 Failed-Attempt Restart Clearance

The prior B0/seed42 execution failed before the first optimizer update. Its sole artifact, provenance.json, SHA256 $ProvSha, has been copied byte-identically to:

$Archive

Only after hash verification was the original copy removed from the formal run root. The formal B0/seed42 run root is therefore empty and eligible for a clean restart under corrected runner blob $RunnerBlob and authorization SHA256 $AuthSha.

The failed attempt remains classified as PRE_OPTIMIZER_EXECUTION_FAILURE, optimizer_steps=0, completed_formal_run=false. Formal evaluation and official-test access remain unauthorized.
