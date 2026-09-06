$ErrorActionPreference = "Stop"
$env:PYTHONPATH = (Resolve-Path ".\src").Path
$env:PYTHONDONTWRITEBYTECODE = "1"

param(
    [Parameter(MoreLogging = $true)]
    [string]$Prompt = "",

    [int]$CtxSize = 2048,

    [float]$Temp = 0.2
)

if (-not $Prompt) {
    $Prompt = "What is TMC's vision?"
}

python -m tmc_llm.cli --prompt "$Prompt" --ctx-size $CtxSize --temp $Temp