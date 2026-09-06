$ErrorActionPreference = "Stop"
$env:PYTHONPATH = (Resolve-Path ".\src").Path
$env:PYTHONDONTWRITEBYTECODE = "1"
python -m tmc_llm.train_tokenizer `
  --corpus .\data\processed\corpus.txt `
  --output-dir .\models\tokenizer `
  --vocab-size 32000 `
  --model-type bpe
