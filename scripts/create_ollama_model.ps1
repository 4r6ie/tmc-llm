param(
    [string]$ModelName = "tmc-lm",
    [string]$GgufPath = ".\models\gguf\tmc-lm-tinyllama-q4_k_m.gguf"
)

$ErrorActionPreference = "Stop"
$env:PYTHONPATH = (Resolve-Path ".\src").Path
$env:PYTHONDONTWRITEBYTECODE = "1"

$ollama = "$env:LOCALAPPDATA\Programs\Ollama\ollama.exe"
if (-not (Test-Path $ollama)) {
    $ollama = "ollama"
}

python -m tmc_llm.gguf_check --path $GgufPath

$modelfile = ".\configs\Modelfile"
$resolvedGguf = (Resolve-Path $GgufPath).Path.Replace("\", "/")
$content = Get-Content -Raw $modelfile
$content = $content -replace "FROM ./models/gguf/tmc-lm-tinyllama-q4_k_m.gguf", "FROM $resolvedGguf"

$tempModelfile = ".\models\gguf\Modelfile.ollama"
Set-Content -Path $tempModelfile -Value $content -Encoding UTF8

& $ollama create $ModelName -f $tempModelfile
& $ollama show $ModelName
