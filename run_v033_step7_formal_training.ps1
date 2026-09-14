param([string]$RepoRoot=(Get-Location).Path)

$ErrorActionPreference="Stop"
Set-StrictMode -Version Latest

function Fail([string]$m){throw $m}
function Check([string]$m){if($LASTEXITCODE -ne 0){throw $m}}
function Sha([string]$p){(Get-FileHash -Algorithm SHA256 -LiteralPath $p).Hash.ToUpperInvariant()}
function Banner([string]$t){Write-Host "";Write-Host("="*78);Write-Host $t;Write-Host("="*78)}

Set-Location $RepoRoot

$Branch="feature/v0.33-prospective-utility-supervision-selector-learning-redesign"
$Architecture="quality_compatibility_utility_supervised_intervention"
$Expected=@{
  "aegis/reliability/reliability_controller.py"="6807597395E0FAD7D295645A416CB5506825845DEA9F6D342F3642035D6E137F"
  "aegis/alignment/model.py"="6EDC8D06233DF4BFB7D868165E1BDA7607D083308370426AAE9E92232252793C"
  "scripts/run_fakeddit_ablation.py"="1D6100D5AC0E8858F0B54846FC5167A97E20C0EEF4D70B5762EF7F3BF41B63CF"
  "tests/test_v033_utility_supervision_selector.py"="96C5AC55143D40DE9EB2C8B7194E294B36A1E2B12ECF017D52B5D4335CAC4C9E"
  "tests/test_v033_training_path.py"="C5B485D0C36CE4BE17D4149AA4CD5AF133C8164DF7A3F8D6C4C5A0ABBF9E6691"
}

Banner "AEGIS V0.33 STEP7 FORMAL M4QUSLI TRAINING"
Write-Host "AUTHORIZED BY V0.33 STEP6 OPERATIONALIZATION"
Write-Host "PRIMARY ONLY: M4qusli x seeds 42/43/44"
Write-Host "OFFICIAL TEST SPLIT MUST REMAIN SEALED"

if((git branch --show-current).Trim() -ne $Branch){Fail "Wrong branch."}
git fetch origin $Branch | Out-Null
Check "Fetch failed."
if((git rev-parse HEAD).Trim() -ne (git rev-parse "origin/$Branch").Trim()){
    Fail "Local and remote branch HEAD differ."
}
if(@(git status --porcelain --untracked-files=no).Count -ne 0){
    Fail "Tracked working tree must be clean before formal training."
}
foreach($p in $Expected.Keys){
    if(-not (Test-Path -LiteralPath $p -PathType Leaf)){Fail "Missing frozen file: $p"}
    if((Sha $p) -ne $Expected[$p]){Fail "Frozen source/test SHA mismatch: $p"}
}

python -m pytest -q `
  .\tests\test_v033_utility_supervision_selector.py `
  .\tests\test_v033_training_path.py
Check "v0.33 pre-training hard-gate suite failed."

$TrainCache="data/processed/fakeddit/frozen_embeddings/train_n5000_seed42"
$ValidationCache="data/processed/fakeddit/frozen_embeddings/validation_n1000_seed42"
if(-not (Test-Path -LiteralPath $TrainCache -PathType Container)){Fail "Training cache missing."}
if(-not (Test-Path -LiteralPath $ValidationCache -PathType Container)){Fail "Validation cache missing."}

$Seeds=@(42,43,44)
foreach($Seed in $Seeds){
    Banner "FORMAL M4QUSLI SEED $Seed"
    $RunRoot="experiments/fakeddit/v033/formal_training/m4qusli_seed$Seed"
    $Log="run_v033_step7_formal_m4qusli_seed$Seed.txt"

    if(Test-Path -LiteralPath $RunRoot){
        Fail "Formal run directory already exists: $RunRoot"
    }
    if(Test-Path -LiteralPath $Log){
        Fail "Formal run log already exists: $Log"
    }

    & python -m scripts.run_fakeddit_ablation `
      --mode multimodal `
      --fusion-architecture $Architecture `
      --train-cache $TrainCache `
      --validation-cache $ValidationCache `
      --experiment-root $RunRoot `
      --epochs 50 `
      --batch-size 32 `
      --seed $Seed `
      --learning-rate 0.001 `
      --weight-decay 0.0001 `
      --shared-dim 128 `
      --hidden-dim 128 `
      --dropout 0.1 `
      --temperature 0.07 `
      --alignment-weight 0.5 `
      --classification-weight 1.0 `
      --quality-weight 1.0 `
      --compatibility-weight 1.0 `
      --transition-objective-weight 0.0 `
      --gradient-clip 1.0 `
      --patience 8 `
      --min-epochs 5 `
      --min-delta 1e-6 `
      --log-every 1 2>&1 | Tee-Object -FilePath $Log

    $Exit=$LASTEXITCODE
    if($Exit -ne 0){Fail "Formal M4qusli seed $Seed failed with exit code $Exit."}

    $Summary=Join-Path $RunRoot "summary.json"
    $Metrics=Join-Path $RunRoot "metrics.jsonl"
    $Experiment=Join-Path $RunRoot "experiment.json"
    $Best=Join-Path $RunRoot "best_model.pt"
    $Final=Join-Path $RunRoot "final_model.pt"
    foreach($p in @($Summary,$Metrics,$Experiment,$Best,$Final)){
        if(-not (Test-Path -LiteralPath $p -PathType Leaf)){Fail "Missing formal artifact: $p"}
    }

    $S=Get-Content -Raw -LiteralPath $Summary | ConvertFrom-Json
    if($S.fusion_architecture -ne $Architecture){Fail "Summary architecture mismatch seed $Seed."}
    if($S.test_split_status -ne "sealed_not_accessed"){Fail "Official test status violation seed $Seed."}

    $Rows=@(Get-Content -LiteralPath $Metrics | Where-Object {$_.Trim()} | ForEach-Object {$_ | ConvertFrom-Json})
    if($Rows.Count -lt 1){Fail "No metrics rows for seed $Seed."}
    foreach($R in $Rows){
        foreach($k in @(
            "effective_utility_loss_weight",
            "selector_gradient_norm",
            "selector_parameter_l2_movement",
            "utility_probability_mean",
            "active_intervention_rate",
            "delta_u_mean",
            "delta_u_std",
            "u_target_mean",
            "transition_loss"
        )){
            if($null -eq $R.train.$k){Fail "Missing train diagnostic '$k' seed $Seed epoch $($R.epoch)."}
        }
        foreach($k in @(
            "utility_probability_mean",
            "utility_gate_mean",
            "intervention_gate_mean",
            "active_intervention_rate",
            "macro_f1",
            "classification_loss"
        )){
            if($null -eq $R.validation.$k){Fail "Missing validation diagnostic '$k' seed $Seed epoch $($R.epoch)."}
        }
        if([double]$R.train.effective_utility_loss_weight -ne 0.50){
            Fail "Effective utility weight changed seed $Seed epoch $($R.epoch)."
        }
        if([math]::Abs([double]$R.train.transition_loss) -gt 1e-12){
            Fail "Transition loss nonzero seed $Seed epoch $($R.epoch)."
        }
    }

    Write-Host "PASS: seed $Seed formal run + machine-readable diagnostics"
}

Banner "AEGIS V0.33 STEP7 FORMAL TRAINING COMPLETE"
Write-Host "Runs completed: 3/3"
Write-Host "Architecture: M4qusli"
Write-Host "Seeds: 42,43,44"
Write-Host "Utility loss weight: 0.50"
Write-Host "Transition objective: 0.0"
Write-Host "Official test samples: 0"
Write-Host "Formal evaluation: NOT YET AUTHORIZED"
Write-Host "Next: freeze Step7 formal-training evidence/provenance"
Write-Host ("="*78)
Write-Host "V033_STEP7_FORMAL_TRAINING_COMPLETE"
