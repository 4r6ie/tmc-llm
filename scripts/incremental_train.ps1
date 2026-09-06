param(
    [string]$NewSource = "",
    [string]$NewSourceDir = ""
)
$ErrorActionPreference = "Stop"
$env:PYTHONPATH = (Resolve-Path ".\src").Path
$env:PYTHONDONTWRITEBYTECODE = "1"
$argsList = @("--config", ".\configs\train_lora.yaml")
if ($NewSource) { $argsList += @("--new-source", $NewSource) }
if ($NewSourceDir) { $argsList += @("--new-source-dir", $NewSourceDir) }
python -m tmc_llm.incremental_train @argsList
