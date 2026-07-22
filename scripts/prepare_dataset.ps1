$ErrorActionPreference = "Stop"
$env:PYTHONPATH = (Resolve-Path ".\src").Path
$env:PYTHONDONTWRITEBYTECODE = "1"

python -m tmc_llm.dataset_builder `
  --source-dir .\data\raw\tmc_sources `
  --output-dir .\data\processed
