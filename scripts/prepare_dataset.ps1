$ErrorActionPreference = "Stop"
$env:PYTHONPATH = (Resolve-Path ".\src").Path
$env:PYTHONDONTWRITEBYTECODE = "1"

python -m tmc_llm.dataset_builder `
  --source .\data\raw\tmc_sources\train.txt `
  --source-dir .\data\raw\tmc_sources `
  --output-dir .\data\processed
