$ErrorActionPreference = "Stop"
$env:PYTHONPATH = (Resolve-Path ".\src").Path
$env:PYTHONDONTWRITEBYTECODE = "1"
Write-Host "WARNING: This custom tokenizer is EXPERIMENTAL and is NOT used by" -ForegroundColor Yellow
Write-Host "train_lora.py or evaluate.py (they use the base model's 32K-vocab tokenizer)." -ForegroundColor Yellow
python -m tmc_llm.train_tokenizer `
  --corpus .\data\processed\corpus.txt `
  --output-dir .\models\tokenizer `
  --vocab-size 32000 `
  --model-type bpe
