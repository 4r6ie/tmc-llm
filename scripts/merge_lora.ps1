$ErrorActionPreference = "Stop"
$env:PYTHONPATH = (Resolve-Path ".\src").Path
$env:PYTHONDONTWRITEBYTECODE = "1"

python -m tmc_llm.merge_lora `
  --base-model TinyLlama/TinyLlama-1.1B-Chat-v1.0 `
  --adapter-dir .\models\adapters\tmc-lm-tinyllama-lora-v1.0 `
  --output-dir .\models\merged\tmc-lm-tinyllama-v1.0
