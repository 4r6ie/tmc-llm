# TMC-LM

Local TinyLlama-based training scaffold for Trinidad Municipal College (TMC) knowledge.

This project uses:

- `documents.md` as the project/development reference
- `data/raw/tmc_sources/train.txt` as the first source institutional knowledge file
- TinyLlama 1.1B as the base model
- LoRA fine-tuning for lightweight local training
- GGUF conversion for Ollama usage

## Folder Structure

```text
tmc-llm/
  configs/
    train_lora.yaml
    Modelfile
  data/
    raw/
      tmc_sources/
        train.txt
    processed/
  models/
    adapters/
    merged/
    gguf/
  scripts/
    prepare_dataset.ps1
    train_lora.ps1
    merge_lora.ps1
    convert_to_gguf.ps1
    check_gguf.ps1
    create_ollama_model.ps1
    smoke_test_ollama.ps1
  src/
    tmc_llm/
  tests/
  documents.md
```

## Recommended Environment

Use Python 3.11 or 3.12 for ML dependencies. Your default Python is currently 3.14, which may not work with PyTorch packages.

Create a virtual environment after installing Python 3.11/3.12:
if not py 3.12
just install
```
py install 3.12
```
Then
```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python -m pip install -e .
```

If your Python launcher does not expose `py -3.12`, install Python 3.12 first, then rerun the commands above.

## Source Documents

Do not manually edit `data/processed`. That folder is generated.

Put official source files here:

```text
data/raw/tmc_sources/
```

Supported files:

```text
.txt
.md
.pdf
.docx
.xlsx
.csv
.json
```

The existing `train.txt` now lives in that same source folder. If you add another manual, for example:

```text
data/raw/tmc_sources/student_handbook_2026.pdf
data/raw/tmc_sources/faculty_manual.docx
data/raw/tmc_sources/programs.xlsx
```

then rerun:

```powershell
.\scripts\prepare_dataset.ps1
```

The script will regenerate `data/processed` and write `sources_manifest.json` so you can check which files were included.

You do not need to edit Python code when adding new source information. The dataset builder dynamically creates training examples from:

- document file names
- section headings
- `LABEL: value` lines such as `VISION: ...`
- extracted PDF/DOCX/XLSX/CSV/JSON/TXT text chunks

It does not create a fixed menu of user questions. The model is trained on official source knowledge, then the finished chat model can accept the user's own free-form questions.

Add files to `data/raw/tmc_sources`, rerun the prepare script, then train again.

## 1. Prepare Dataset

```powershell
.\scripts\prepare_dataset.ps1
```

Output:

```text
data/processed/dataset.jsonl
data/processed/train.jsonl
data/processed/validation.jsonl
data/processed/test.jsonl
data/processed/corpus.txt
data/processed/metadata.json
data/processed/sources_manifest.json
```

Run tests:

```powershell
python -m pytest
```

## 2. Fine-Tune TinyLlama with LoRA

```powershell
.\scripts\train_lora.ps1
```

Default base model:

```text
TinyLlama/TinyLlama-1.1B-Chat-v1.0
```

Output:

```text
models/adapters/tmc-lm-tinyllama-lora/
```

Note: On your Ryzen 3 2200G CPU, training can be slow. Start with the default small config first.

## 3. Merge LoRA Adapter

```powershell
.\scripts\merge_lora.ps1
```

Output:

```text
models/merged/tmc-lm-tinyllama/
```

## 4. Convert Merged Model to GGUF

This script expects `llama.cpp` under `external/llama.cpp`.

One-time setup:

```powershell
git clone https://github.com/ggml-org/llama.cpp .\external\llama.cpp
```

Build `llama.cpp` before quantizing. If it is not built yet, the script will still create the F16 GGUF and tell you that `llama-quantize.exe` is missing.

```powershell
.\scripts\convert_to_gguf.ps1
```

Output:

```text
models/gguf/tmc-lm-tinyllama-f16.gguf
models/gguf/tmc-lm-tinyllama-q4_k_m.gguf
```

## 5. Check GGUF

```powershell
.\scripts\check_gguf.ps1 -GgufPath .\models\gguf\tmc-lm-tinyllama-q4_k_m.gguf
```

This checks that the file exists and has a valid GGUF header.

## 6. Create Ollama Model

```powershell
.\scripts\create_ollama_model.ps1 -GgufPath .\models\gguf\tmc-lm-tinyllama-q4_k_m.gguf
```

Run:

```powershell
& "$env:LOCALAPPDATA\Programs\Ollama\ollama.exe" run tmc-lm
```

## 7. Smoke Test with TMC Questions

```powershell
.\scripts\smoke_test_ollama.ps1
```

Expected model behavior:

- Answers only from official TMC knowledge
- Says it does not know when the answer is outside the trained source
- Correctly answers vision, mission, programs, and brief history questions
