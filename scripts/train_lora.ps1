$ErrorActionPreference = "Stop"
$env:PYTHONPATH = (Resolve-Path ".\src").Path
$env:PYTHONDONTWRITEBYTECODE = "1"

python -m tmc_llm.train_lora `
  --config .\configs\train_lora.yaml

$adapterDir = ".\models\adapters\tmc-lm-tinyllama-lora-v1.0"
if (-not (Test-Path $adapterDir)) {
    throw "Training finished but adapter directory was not found: $adapterDir. Check configs/train_lora.yaml and rerun."
}

Write-Host ""
Write-Host "======================================================"
Write-Host " TRAINING COMPLETE - ADAPTER SAVED TO $adapterDir"
Write-Host " Next step: run scripts/merge_lora.ps1 to merge with base model."
Write-Host "======================================================"
Get-ChildItem $adapterDir | Select-Object Name, Length | Format-Table -AutoSize
