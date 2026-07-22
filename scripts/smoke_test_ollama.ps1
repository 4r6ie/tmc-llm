param(
    [string]$ModelName = "tmc-lm"
)

$ErrorActionPreference = "Stop"

$ollama = "$env:LOCALAPPDATA\Programs\Ollama\ollama.exe"
if (-not (Test-Path $ollama)) {
    $ollama = "ollama"
}

$questions = @(
    "What is the vision of TMC?",
    "What is the mission of TMC?",
    "What academic programs are offered by Trinidad Municipal College?",
    "When did Trinidad Junior College become registered with the SEC?",
    "What is the weather today in Bohol?"
)

foreach ($question in $questions) {
    Write-Host ""
    Write-Host "QUESTION: $question"
    Write-Host "ANSWER:"
    & $ollama run $ModelName $question
}

