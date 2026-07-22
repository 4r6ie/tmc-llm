param(
    [Parameter(Mandatory = $true)]
    [string]$GgufPath
)

$ErrorActionPreference = "Stop"
$env:PYTHONPATH = (Resolve-Path ".\src").Path
$env:PYTHONDONTWRITEBYTECODE = "1"

python -m tmc_llm.gguf_check --path $GgufPath
