param(
    [string]$LlamaCppDir = ".\external\llama.cpp",
    [string]$MergedModelDir = ".\models\merged\tmc-lm-tinyllama-v1.0",
    [string]$OutputDir = ".\models\gguf",
    [string]$Quantization = "Q4_K_M"
)

$ErrorActionPreference = "Stop"
$env:PYTHONPATH = (Resolve-Path ".\src").Path
$env:PYTHONDONTWRITEBYTECODE = "1"

$convertScript = Join-Path $LlamaCppDir "convert_hf_to_gguf.py"
if (-not (Test-Path $convertScript)) {
    throw "Missing llama.cpp converter: $convertScript. Clone llama.cpp into external/llama.cpp first."
}

New-Item -ItemType Directory -Force -Path $OutputDir | Out-Null

$f16Path = Join-Path $OutputDir "tmc-lm-tinyllama-f16.gguf"
$q4Path = Join-Path $OutputDir "tmc-lm-tinyllama-q4_k_m.gguf"

python $convertScript `
  $MergedModelDir `
  --outfile $f16Path `
  --outtype f16

$quantizeCandidates = @(
    (Join-Path $LlamaCppDir "build\bin\Release\llama-quantize.exe"),
    (Join-Path $LlamaCppDir "build\bin\llama-quantize.exe"),
    (Join-Path $LlamaCppDir "llama-quantize.exe")
)

$quantizeExe = $quantizeCandidates | Where-Object { Test-Path $_ } | Select-Object -First 1
if (-not $quantizeExe) {
    Write-Warning "F16 GGUF created, but llama-quantize.exe was not found. Build llama.cpp, then rerun this script."
    python -m tmc_llm.gguf_check --path $f16Path
    exit 0
}

& $quantizeExe $f16Path $q4Path $Quantization
python -m tmc_llm.gguf_check --path $q4Path
