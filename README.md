# TMC-LM

Local TinyLlama-based institutional chatbot for Trinidad Municipal College (TMC).

## Overview

TMC-LM fine-tunes **TinyLlama-1.1B-Chat** on official TMC documents (vision, mission, programs, history, policies) so students, faculty, and staff get accurate, offline-capable answers instead of generic or hallucinated responses from public LLMs.

**Who it's for:** TMC students/faculty/staff, IT admins deploying on low-resource local machines (Ryzen 3 / 8GB VRAM / CPU-only friendly).
**What it does:** Builds a private Q&A model from `data/raw/tmc_sources/` and serves it via CLI, Docker, FastAPI, or web chat. Answers only from trained official knowledge and says it does not know when out of scope.

## Features

- LoRA fine-tuning of TinyLlama-1.1B-Chat-v1.0 for lightweight local training
- Multi-format dataset builder (`.txt`, `.md`, `.pdf`, `.docx`, `.xlsx`, `.csv`, `.json`)
- Merged + GGUF quantized output (`F16` / `Q4_K_M`) for CPU inference
- Run via Docker (llama.cpp), local Python (`llama-cpp-python`), CLI, FastAPI, or `web_chat.html`
- Grounded responses with unknown-query refusal behavior

## Architecture

```text
data/raw/tmc_sources/ (train.txt, pdfs, docs)
  -> prepare_dataset.ps1 -> data/processed/ (jsonl, corpus.txt)
  -> train_lora.ps1 (LoRA adapter) -> merge_lora.ps1 (merged HF model)
  -> GGUF convert + quantize (llama.cpp) -> models/gguf/*.gguf
  -> Inference: Docker CLI / tmc_llm.cli / tmc_llm.api + web_chat.html
```

This project uses:

- `documents.md` as the project/development reference
- `data/raw/tmc_sources/train.txt` as the first source institutional knowledge file
- **TinyLlama/TinyLlama-1.1B-Chat-v1.0** as the base model (auto-downloaded from Hugging Face)
- LoRA fine-tuning for lightweight local training
- GGUF conversion via llama.cpp (Docker or local) for local inference
- FastAPI for HTTP API endpoint
- Web-based chat interface

## Contents

- [Prerequisites](#prerequisites)
- [Download Base Model](#download-base-model)
- [Quick Start (Windows PowerShell)](#quick-start-windows-powershell)
- [Docker Setup (Full Pipeline)](#docker-setup-full-pipeline)
- [Local Inference](#local-inference-no-docker-required)
- [Model Details](#model-details)
- [Folder Structure](#folder-structure)
- [Adding New Source Documents](#adding-new-source-documents)
- [Configuration](#configuration)
- [Troubleshooting](#troubleshooting)
- [License](#license)

## Prerequisites

### Required Tools

| Tool | Version | Purpose |
|------|---------|---------|
| **Python** | 3.11 or 3.12 | ML dependencies (PyTorch, Transformers, FastAPI) |
| **Git** | Latest | Clone repository |
| **Docker** | Latest | Run llama.cpp in Linux container (no Windows build needed) |
| **uvicorn** | Latest | FastAPI ASGI server for API endpoint |

### Hardware Recommendations

- **GPU**: NVIDIA GPU with 8GB+ VRAM (CUDA 11.8+)
- **CPU**: Ryzen 3 2200G or better (training will be slow on CPU-only)
- **RAM**: 16GB+ system memory
- **Disk**: 10GB+ free space for models and datasets

## Download Base Model

The base model **TinyLlama/TinyLlama-1.1B-Chat-v1.0** (~2.2 GB) is **automatically downloaded** on first run of `.\scripts\train_lora.ps1` to `~/.cache/huggingface/hub/`. No manual action needed (see Quick Start step 5).

<details>
<summary>Manual / offline download (optional)</summary>

```powershell
huggingface-cli download TinyLlama/TinyLlama-1.1B-Chat-v1.0 --local-dir ./models/base/tinyllama-1.1b-chat
```

Then in `configs/train_lora.yaml`:
```yaml
base_model: ./models/base/tinyllama-1.1b-chat
```

Verify:
```powershell
ls ./models/base/tinyllama-1.1b-chat/
# expected: config.json, model.safetensors, tokenizer.json, tokenizer.model, tokenizer_config.json
```

</details>

---

## Quick Start (Windows PowerShell)

### 1. Install Python 3.12

```powershell
# Using winget
winget install Python.Python.3.12

# Or download from python.org
```

### 2. Install Docker Desktop

Download and install from: https://www.docker.com/products/docker-desktop/

Enable WSL 2 backend and Ubuntu integration in Docker Desktop settings.

### 3. Create Virtual Environment & Install Dependencies

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip

# CPU-only machine? Install the much smaller CPU-only PyTorch build first:
# python -m pip install torch --index-url https://download.pytorch.org/whl/cpu

# Runtime + training dependencies (canonical list lives in pyproject.toml;
# the "local" extra adds llama-cpp-python for GGUF inference)
python -m pip install -r requirements.txt

# Dev tooling (pytest, ruff, mypy) — needed to run the test suite
python -m pip install -e ".[dev]"
```

### 4. Prepare Dataset

```powershell
.\scripts\prepare_dataset.ps1
```

Output:
```
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

### 5. Fine-Tune TinyLlama with LoRA

The base model **TinyLlama/TinyLlama-1.1B-Chat-v1.0** is automatically downloaded from Hugging Face on first run (~2.2 GB).

```powershell
.\scripts\train_lora.ps1
```

Output:
```
models/adapters/tmc-lm-tinyllama-lora-v1.0/
```

> **Note**: On CPU-only (e.g., Ryzen 3 2200G), training can be slow. The default config uses small batch sizes.

### 6. Merge LoRA Adapter

```powershell
.\scripts\merge_lora.ps1
```

Output:
```
models/merged/tmc-lm-tinyllama-v1.0/
```

### 7. Convert to GGUF using Docker (llama.cpp)

No need to build llama.cpp on Windows. Use the pre-built Docker image:

```powershell
# Convert merged model to F16 GGUF (--convert is the :full image's tools.sh entrypoint)
docker run --rm -v ${PWD}:/app ghcr.io/ggml-org/llama.cpp:full --convert /app/models/merged/tmc-lm-tinyllama-v1.0 --outfile /app/models/gguf/tmc-lm-tinyllama-f16.gguf

# Quantize to Q4_K_M (smaller, faster)
docker run --rm -v ${PWD}:/app ghcr.io/ggml-org/llama.cpp:full --quantize /app/models/gguf/tmc-lm-tinyllama-f16.gguf /app/models/gguf/tmc-lm-tinyllama-q4_k_m.gguf Q4_K_M
```

Output:
```
models/gguf/tmc-lm-tinyllama-f16.gguf
models/gguf/tmc-lm-tinyllama-q4_k_m.gguf
```

### 8. Verify GGUF

```powershell
python -m tmc_llm.gguf_check --path .\models\gguf\tmc-lm-tinyllama-q4_k_m.gguf
```

### 9. Run Inference with llama.cpp (Docker)

Interactive chat (conversation mode automatically applies TinyLlama's embedded
`<|user|>`/`<|assistant|>` chat template, which matches how the model was trained):

```powershell
docker run --rm -it -v ${PWD}/models:/models ghcr.io/ggml-org/llama.cpp:light -m /models/gguf/tmc-lm-tinyllama-q4_k_m.gguf -c 2048 --temp 0.2 --repeat-penalty 1.12 -cnv -p "What is TMC's vision?"
```

> **Important**: Do **not** pass `--chat-template llama-2-chat` or an
> `[INST] <<SYS>>...<</SYS>>...[/INST]` prompt. That is the Llama-2 format, not
> TinyLlama's. Using it bypasses the template the model was trained with and
> degrades answer quality. Let llama.cpp apply the template stored in the GGUF
> file (the default behavior in conversation mode).

Expected behavior:
- Answers only from official TMC knowledge
- Says it does not know when the answer is outside the trained source
- Correctly answers vision, mission, programs, and brief history questions

---

## Docker Setup (Full Pipeline)

The repository ships a ready-to-use `Dockerfile` (Python 3.12 + CUDA base,
installs dependencies from `requirements.txt` — including the `local` extra —
and clones + builds llama.cpp with CUDA for GGUF conversion). Review it at the
project root; there is no need to create it manually.

### Build Docker Image

```bash
docker build -t tmc-llm .
```

### Run Full Pipeline in Docker

```bash
# With GPU support
docker run --gpus all -it --rm \
  -v ${PWD}/data:/app/data \
  -v ${PWD}/models:/app/models \
  -v ${PWD}/configs:/app/configs \
  tmc-llm \
  bash -c "
    python -m tmc_llm.dataset_builder --source-dir /app/data/raw/tmc_sources --output-dir /app/data/processed &&
    python -m tmc_llm.train_lora --config /app/configs/train_lora.yaml &&
    python -m tmc_llm.merge_lora --base-model TinyLlama/TinyLlama-1.1B-Chat-v1.0 --adapter-dir /app/models/adapters/tmc-lm-tinyllama-lora-v1.0 --output-dir /app/models/merged/tmc-lm-tinyllama-v1.0 &&
    python /app/external/llama.cpp/convert_hf_to_gguf.py /app/models/merged/tmc-lm-tinyllama-v1.0 --outfile /app/models/gguf/tmc-lm-tinyllama-f16.gguf --outtype f16 &&
    /app/external/llama.cpp/build/bin/llama-quantize /app/models/gguf/tmc-lm-tinyllama-f16.gguf /app/models/gguf/tmc-lm-tinyllama-q4_k_m.gguf Q4_K_M
  "
```

### Run Inference in Docker

```bash
docker run --gpus all -it --rm \
  -v ${PWD}/models:/app/models \
  tmc-llm \
  /app/external/llama.cpp/build/bin/llama-cli -m /app/models/gguf/tmc-lm-tinyllama-q4_k_m.gguf \
  -c 2048 --temp 0.2 --repeat-penalty 1.12 -cnv -p "What is TMC's vision?"
```

---

## Local Inference (No Docker Required)

You can run inference locally using a pre-converted GGUF model without Docker.
This requires the optional `local` extra (llama-cpp-python), which is already
installed if you set up via `requirements.txt`:

```powershell
python -m pip install -e ".[local]"
```

1. **Convert to GGUF** first (see the Convert to GGUF section), then:

```powershell
# List available GGUF models
Get-ChildItem .\models\gguf\

# Run inference with a PowerShell script
.\scripts\inference.ps1 --prompt "What is TMC's vision?"
```

Alternatively, use the Python CLI directly:

```powershell
python -m tmc_llm.cli --prompt "What is TMC's vision?" --ctx-size 2048 --temp 0.2
```

Or use the HTTP API:

```powershell
# Start the API
python -m tmc_llm.api

# Then query it
curl -X POST http://localhost:8000/query \
  -H "Content-Type: application/json" \
  -d '{"prompt": "What is TMC\'s vision?"}'
```

If no GGUF model is found, the script will print the Docker command you can run instead.

2. **Inference with Docker** (see the Docker Setup section above)

---

## API Reference (FastAPI)

```powershell
# Start server
python -m tmc_llm.api

# Query endpoint
curl -X POST http://localhost:8000/query \
  -H "Content-Type: application/json" \
  -d '{"prompt": "What is TMC\'s vision?"}'
```

| Endpoint | Method | Params | Response |
|----------|--------|--------|----------|
| `/query` | POST | `prompt` (str, required) | `{"answer": str, "source": str|none, "confidence": float}` |
| `/health` | GET | none | `{"status": "ok"}` |

- Port: `8000`
- Uses GGUF model from `models/gguf/` (or falls back to Docker command)
- Unknown answers return `{"answer": "I do not know", "source": null}`

## Web Chat Interface

Open `web_chat.html` in a browser for a UI that sends queries to `/query`.

---

## Model Details

| Property | Value |
|----------|-------|
| **Model ID** | `TinyLlama/TinyLlama-1.1B-Chat-v1.0` |
| **Parameters** | 1.1 Billion |
| **Architecture** | Llama 2 |
| **Context Length** | 2048 tokens |
| **License** | Apache 2.0 |
| **Hugging Face** | https://huggingface.co/TinyLlama/TinyLlama-1.1B-Chat-v1.0 |

See [Download Base Model](#download-base-model) for automatic vs. manual download.

## Evaluation

### How to run the test suite

```powershell
python -m pytest
```

### Sample Q&A test prompts (5 examples)

| # | Prompt | Expected answer pattern |
|---|--------|------------------------|
| 1 | `What is TMC's vision?` | Starts with "TMC aspires to be a premier..." |
| 2 | `What is TMC's mission?` | Starts with "To provide quality education..." |
| 3 | `List TMC's programs` | Accountancy, Business, IT (or similar) |
| 4 | `When was TMC founded?` | "Founded in 1952" or similar |
| 5 | `What is the capital of the Philippines?` | `I do not know` (out of scope) |

### Quality criteria

- ✅ Answers only from officially trained TMC knowledge
- ✅ Correctly says `I do not know` when query is outside trained sources
- ✅ Refuses to answer non-TMC questions (hallucination prevention)
- ✅ Response latency acceptable for ctx-size 2048 / temp 0.2

### Limitations

- 1.1B parameter model has limited reasoning capacity beyond retrieved facts
- Context length 2048 tokens — long documents may be truncated
- CPU-only training is slow (default small batch sizes)
- Quantized GGUF models may lose some nuance vs. full-precision

---

---

```text
tmc-llm/
  configs/
    train_lora.yaml       # LoRA training configuration
  data/
    raw/
      tmc_sources/        # Put source documents here (.txt, .md, .pdf, .docx, .xlsx, .csv, .json)
        train.txt
    processed/            # Generated dataset (do not edit manually)
  models/
    adapters/             # LoRA adapters output
    merged/               # Merged model output
    gguf/                 # GGUF quantized models
  external/
    llama.cpp/            # Cloned llama.cpp (inside Docker only)
  scripts/
    prepare_dataset.ps1   # Build dataset from source documents
    train_lora.ps1        # Fine-tune with LoRA
    merge_lora.ps1        # Merge LoRA into base model
    check_gguf.ps1        # Validate GGUF file
    inference.ps1         # Run local GGUF inference (no Docker)
  src/
    tmc_llm/              # Python package
  tests/
  documents.md            # Project reference
  requirements.txt
  Dockerfile              # Docker setup for training + conversion
  web_chat.html         # Web-based chat interface
  api.py                # FastAPI HTTP endpoint
```

---

## Data Format

The dataset builder reads files from `data/raw/tmc_sources/` and extracts:

- **File name** (without extension) → document title
- **Section headings** (`# `, `## `, etc.) → chunk titles  
- **`LABEL: value` lines** (e.g., `VISION: ...`, `MISSION: ...`)
- **Plain text chunks** from PDF/DOCX/XLSX/CSV/JSON/TXT

### Minimal `train.txt` example

```text
VISION: TMC aspires to be a premier ... 
MISSION: To provide quality education ... 
PROGRAMS: Accountancy, Business, IT 
HISTORY: Founded in 1952 ...

Q: What is TMC's vision?
A: TMC aspires to be a premier ...
```

### Supported source files

| Extension | Example |
|-----------|---------|
| `.txt` | `announcements.txt` |
| `.md` | `student_handbook.md` |
| `.pdf` | `faculty_manual.pdf` |
| `.docx` | `policy.docx` |
| `.xlsx` | `programs.xlsx` |
| `.csv` | `grades.csv` |
| `.json` | `events.json` |

Add files then re-run:
```powershell
.\scripts\prepare_dataset.ps1
```
---

1. Add files to `data/raw/tmc_sources/`:
   ```text
   data/raw/tmc_sources/student_handbook_2026.pdf
   data/raw/tmc_sources/faculty_manual.docx
   data/raw/tmc_sources/programs.xlsx
   data/raw/tmc_sources/announcements.txt
   ```

2. Re-run dataset preparation:
   ```powershell
   .\scripts\prepare_dataset.ps1
   ```

3. Re-train:
   ```powershell
   .\scripts\train_lora.ps1
   ```

The dataset builder dynamically creates training examples from:
- Document file names
- Section headings
- `LABEL: value` lines (e.g., `VISION: ...`)
- Extracted text chunks from PDF/DOCX/XLSX/CSV/JSON/TXT

---

## Configuration & Environment

### `configs/train_lora.yaml`

Key settings (see [Configuration](#configuration) for full table):

```yaml
base_model: TinyLlama/TinyLlama-1.1B-Chat-v1.0
max_seq_length: 768
num_train_epochs: 5
max_steps: -1          # -1 = run full epochs
per_device_train_batch_size: 1
gradient_accumulation_steps: 16
learning_rate: 0.0002
lr_scheduler_type: cosine
weight_decay: 0.01
fp16: true             # auto-disabled on CPU-only
bf16: false

lora:
  r: 16
  alpha: 32
  dropout: 0.05
  target_modules: [q_proj, k_proj, v_proj, o_proj, gate_proj, up_proj, down_proj]
```

### `.env.example` (copy to `.env`)

```env
# Hugging Face token (needed if rate-limited or private repos)
HF_HUB_TOKEN=your_token_here

# Inference defaults
INFERENCE_TEMP=0.2
INFERENCE_CTX_SIZE=2048
INFERENCE_REPEAT_PENALTY=1.12

# API
API_PORT=8000
```

### VRAM / Disk summary

| Component | Min | Recommended |
|-----------|-----|-------------|
| VRAM | 4GB (FP16) | 8GB+ (FP16/Q4_K_M) |
| System RAM | 8GB | 16GB+ |
| Disk | 10GB | 20GB+ (models + datasets) |

### `configs/train_lora.yaml`

Key settings:
```yaml
base_model: TinyLlama/TinyLlama-1.1B-Chat-v1.0
max_seq_length: 768
num_train_epochs: 5
max_steps: -1          # -1 = run full epochs (do not cap at a fixed step count)
per_device_train_batch_size: 1
gradient_accumulation_steps: 16
learning_rate: 0.0002
lr_scheduler_type: cosine
weight_decay: 0.01
fp16: true             # auto-disabled on CPU-only machines
bf16: false            # T4 (Colab) does not support bf16

lora:
  r: 16
  alpha: 32
  dropout: 0.05
  target_modules: [q_proj, k_proj, v_proj, o_proj, gate_proj, up_proj, down_proj]
```

---

## Prerequisites (Expanded)

### Required Tools

| Tool | Version | Purpose |
|------|---------|---------|
| **Python** | 3.11 or 3.12 | ML dependencies (PyTorch, Transformers, FastAPI) |
| **Git** | Latest | Clone repository |
| **Docker Desktop** | Latest | Run llama.cpp in Linux container (no Windows build needed) |
| **uvicorn** | Latest | FastAPI ASGI server for API endpoint |
| **winget** (optional) | Latest | Install Python 3.12 |

### Execution Policy (Windows PowerShell)

```powershell
# Allow scripts to run (required for .ps1 files)
Set-ExecutionPolicy -Scope Process -ExecutionPolicy RemoteSigned
# Or per-file: .\scripts\train_lora.ps1 -ExecutionPolicy Bypass
```

### Hardware Recommendations

- **GPU**: NVIDIA GPU with 8GB+ VRAM (CUDA 11.8+). Training on 4GB VRAM is possible with `fp16: false` + reduced batch size.
- **CPU**: Ryzen 3 2200G or better (training will be slow on CPU-only; expect 10-30x slower than GPU).
- **RAM**: 16GB+ system memory.
- **Disk**: 10GB+ free space for models and datasets.

### Docker WSL2 Setup

1. Install Docker Desktop from https://www.docker.com/products/docker-desktop/
2. Launch Docker Desktop and sign in
3. Go to **Settings** > **General** > **Use the WSL 2 based engine**
4. Go to **Resources** > **WSIL Integration** and enable Ubuntu integration
5. For GPU: Go to **Resources** > **Advanced** and enable "Enable NVIDIA GPU support"

### Hugging Face Authentication

If you hit rate limits or need private repos:

```powershell
huggingface-cli login
# Or set token in environment:
$env:HF_HUB_TOKEN = "hf_..."
```

Set `HF_HUB_DISABLE_TELEMETRY=1` to suppress telemetry (already in scripts).

### Troubleshooting

| Issue | Fix |
|-------|-----|
| **Python Version Issues** | Use Python 3.11 or 3.12. PyTorch wheels for 3.13+ may not exist. |
| **CUDA/GPU Not Detected** | Verify: `python -c "import torch; print(torch.cuda.is_available(), torch.version.cuda)"`. Ensure Docker WSL2 GPU support is enabled. |
| **Docker Issues** | Ensure Docker Desktop is running with WSL 2 backend. Enable "Use the WSL 2 based engine". |
| **Out of Memory (OOM)** | Reduce `per_device_train_batch_size` to 1. Increase `gradient_accumulation_steps`. Use `max_seq_length: 512` instead of 768. |
| **Model Download Fails** | Check internet connection. Set `HF_HUB_DISABLE_TELEMETRY=1`. Use `huggingface-cli login` if rate limited. |
| **Scripts Won't Run** | Run `Set-ExecutionPolicy -Scope Process -ExecutionPolicy RemoteSigned` in PowerShell. |

## License

### Code

This project is licensed under **Apache 2.0** - see the [LICENSE](LICENSE) file in the repository root.

### Base Model

The TinyLlama base model `TinyLlama/TinyLlama-1.1B-Chat-v1.0` is licensed under **Apache 2.0** at https://huggingface.co/TinyLlama/TinyLlama-1.1B-Chat-v1.0. That license governs use of the base model weights; this project adds LoRA adapters, dataset tools, and conversion scripts which are separate under Apache 2.0.

### Third-Party

- `llama.cpp` (GGUF conversion) - https://github.com/ggml-org/llama.cpp (MIT license)
- `fastapi` - MIT license
- `transformers` - Apache 2.0 license

---

## Contributing

1. Fork the repository
2. Create a feature branch: `git checkout -b feature/foo`
3. Commit your changes: `git commit -m "Add foo"`
4. Push to branch: `git push origin feature/foo`
5. Open a Pull Request

Please read [CONTRIBUTING.md](CONTRIBUTING.md) for detailed guidelines.

---

## Limitations & Disclaimer

- **Model scope**: TMC-LM only answers questions based on officially provided TMC source documents. It does not general knowledge beyond what it was trained on.
- **Hallucination risk**: Even with grounded training, 1.1B parameter models can produce incorrect or nonsensical answers. Always verify critical information.
- **Out-of-scope responses**: The model is designed to say "I do not know" when questions are outside the trained TMC knowledge base. However, this behavior is not guaranteed 100% of the time.
- **Privacy**: No user data is stored or transmitted unless you use the FastAPI endpoint (`/query`). Docker runs are ephemeral.
- **Academic/research use only**: This scaffold is intended for educational and institutional AI experimentation. Do not deploy as a production customer-facing system without additional safety guards.
- **Quantization trade-offs**: GGUF quantization (Q4_K_M, etc.) reduces file size and speeds up inference but may slightly change answer quality compared to full-precision FP16.

### Roadmap (planned)

- [ ] Add RAG pipeline instead of pure LoRA fine-tuning
- [ ] Support multi-modal inputs (images, tables from Excel)
- [ ] Web UI with user authentication
- [ ] Evaluation dataset generation from `documents.md`
- [ ] Windows installer (no PowerShell required)