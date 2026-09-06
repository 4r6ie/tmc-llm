param(
    [int]$SampleSize = 50,
    [string]$Output = ""
)
$ErrorActionPreference = "Stop"
$env:PYTHONPATH = (Resolve-Path ".\src").Path
$env:PYTHONDONTWRITEBYTECODE = "1"
$argsList = @("--test-data", ".\data\processed\test.jsonl", "--sample-size", "$SampleSize")
if ($Output) { $argsList += @("--output", $Output) }
python -m tmc_llm.evaluate @argsList
