# TMC-LM

Local TinyLlama-based training scaffold for Trinidad Municipal College (TMC) knowledge.

This project uses:

- `documents.md` as the project/development reference
- `data/raw/tmc_sources/train.txt` as the first source institutional knowledge file
- **TinyLlama/TinyLlama-1.1B-Chat-v1.0** as the base model (auto-downloaded from Hugging Face)
- LoRA fine-tuning for lightweight local training
- GGUF conversion via llama.cpp (Docker or local) for local inference
- FastAPI for HTTP API endpoint
- Web-based chat interface

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

## Download Base Model (Required First Step)

The base model **TinyLlama/TinyLlama-1.1B-Chat-v1.0** (~2.2 GB) must be downloaded from Hugging Face before training.

### Option 1: Automatic (Recommended)

The model is **automatically downloaded** on first run of `train_lora.ps1`. No manual action needed.

```powershell
# This will download the model automatically
.\scripts\train_lora.ps1
```

Download location: `~/.cache/huggingface/hub/models--TinyLlama--TinyLlama-1.1B-Chat-v1.0/`

### Option 2: Manual Pre-Download (For Offline/Air-gapped Environments)

If you need to pre-download or work offline:

```powershell
# Using Python
python -c "
from transformers import AutoModelForCausalLM, AutoTokenizer
model_id = 'TinyLlama/TinyLlama-1.1B-Chat-v1.0'
tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForCausalLM.from_pretrained(model_id)
tokenizer.save_pretrained('./models/base/tinyllama-1.1b-chat')
model.save_pretrained('./models/base/tinyllama-1.1b-chat')
print('Model saved to ./models/base/tinyllama-1.1b-chat')
"
```

Then update `configs/train_lora.yaml`:
```yaml
base_model: ./models/base/tinyllama-1.1b-chat
```

### Option 3: Using huggingface-cli

```powershell
# Install huggingface_hub
pip install huggingface_hub

# Download model
huggingface-cli download TinyLlama/TinyLlama-1.1B-Chat-v1.0 --local-dir ./models/base/tinyllama-1.1b-chat
```

Then update `configs/train_lora.yaml`:
```yaml
base_model: ./models/base/tinyllama-1.1b-chat
```

### Verify Download

```powershell
# Check if model exists in cache
ls ~/.cache/huggingface/hub/models--TinyLlama--TinyLlama-1.1B-Chat-v1.0/

# Or verify local path
ls ./models/base/tinyllama-1.1b-chat/
```

Expected files:
```
config.json
generation_config.json
model.safetensors
tokenizer.json
tokenizer.model
tokenizer_config.json
special_tokens_map.json
```

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
python -m pip install -r requirements.txt
python -m pip install -e .
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
# Convert merged model to F16 GGUF
docker run --rm -v ${PWD}:/app ghcr.io/ggml-org/llama.cpp:full \
  python /app/llama.cpp/convert_hf_to_gguf.py /app/models/merged/tmc-lm-tinyllama-v1.0 \
  --outfile /app/models/gguf/tmc-lm-tinyllama-f16.gguf --outtype f16

# Quantize to Q4_K_M (smaller, faster)
docker run --rm -v ${PWD}:/app ghcr.io/ggml-org/llama.cpp:full \
  /app/llama.cpp/build/bin/llama-quantize /app/models/gguf/tmc-lm-tinyllama-f16.gguf \
  /app/models/gguf/tmc-lm-tinyllama-q4_k_m.gguf Q4_K_M
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

Interactive chat:

```powershell
docker run --rm -it -v ${PWD}:/app ghcr.io/ggml-org/llama.cpp:full \
  /app/llama.cpp/build/bin/llama-cli -m /app/models/gguf/tmc-lm-tinyllama-q4_k_m.gguf \
  -c 2048 --temp 0.2 --repeat-penalty 1.12 -p "[INST] <<SYS>>You are TMC-LM, an offline assistant for Trinidad Municipal College. Answer using only the provided official TMC knowledge. If the source does not contain the answer, say that the available TMC source does not contain it.<</SYS>>What is TMC's vision?[/INST]"
```

Or use the chat template:

```powershell
docker run --rm -it -v ${PWD}:/app ghcr.io/ggml-org/llama.cpp:full \
  /app/llama.cpp/build/bin/llama-cli -m /app/models/gguf/tmc-lm-tinyllama-q4_k_m.gguf \
  -c 2048 --temp 0.2 --repeat-penalty 1.12 --chat-template llama-2-chat -i
```

Expected behavior:
- Answers only from official TMC knowledge
- Says it does not know when the answer is outside the trained source
- Correctly answers vision, mission, programs, and brief history questions

---

## Docker Setup (Full Pipeline)

### Dockerfile for Training + Conversion

### Local Inference (No Docker Required)
You can run inference locally using a pre-converted GGUF model without Docker:

1. **Convert to GGUF** first (see the Convert to GGUF section), then:

```powershell
# List available GGUF models
Get-ChildItem .\models\gguf\

# Run inference with a prompt
.\scripts\inference.ps1 --prompt "What is TMC's vision?"
```

Alternatively, use the Python CLI directly:

```powershell
python -m tmc_llm.cli --prompt "What is TMC's vision?" --ctx-size 2048 --temp 0.2
```

If no GGUF model is found, the script will print the Docker command you can run instead.
```

2. **Inference with Docker** (unchanged from below)
```

### Dockerfile for Training + Conversion

Create a `Dockerfile` in the project root:

```dockerfile
# Base image with Python 3.12 and CUDA support
FROM nvidia/cuda:12.1.0-devel-ubuntu22.04

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    HF_HUB_DISABLE_TELEMETRY=1

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3.12 python3.12-venv python3.12-dev \
    git cmake build-essential \
    && rm -rf /var/lib/apt/lists/*

# Set Python 3.12 as default
RUN update-alternatives --install /usr/bin/python python /usr/bin/python3.12 1 \
    && update-alternatives --install /usr/bin/pip pip /usr/bin/pip3.12 1

WORKDIR /app

# Copy requirements first for better caching
COPY requirements.txt .
RUN pip install --upgrade pip && pip install -r requirements.txt

# Copy project
COPY . .

# Install package in development mode
RUN pip install -e .

# Clone and build llama.cpp for GGUF conversion
RUN git clone https://github.com/ggml-org/llama.cpp /app/external/llama.cpp \
    && cd /app/external/llama.cpp \
    && cmake -B build -DCMAKE_BUILD_TYPE=Release -DGGML_CUDA=ON \
    && cmake --build build --config Release -j$(nproc)

# Default command
CMD ["bash"]
```

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
  -c 2048 --temp 0.2 --repeat-penalty 1.12 --chat-template llama-2-chat -i
```

---

## Local Inference (No Docker Required)

You can run inference locally using a pre-converted GGUF model without Docker:

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

2. **Inference with Docker** (unchanged from above)
```

## Model Details

### Base Model

| Property | Value |
|----------|-------|
| **Model ID** | `TinyLlama/TinyLlama-1.1B-Chat-v1.0` |
| **Parameters** | 1.1 Billion |
| **Architecture** | Llama 2 |
| **Context Length** | 2048 tokens |
| **License** | Apache 2.0 |
| **Hugging Face** | https://huggingface.co/TinyLlama/TinyLlama-1.1B-Chat-v1.0 |

### Automatic Download

The model is **automatically downloaded** from Hugging Face when you run `train_lora.ps1` (first run only). The transformers library handles caching in `~/.cache/huggingface/hub/`.

**No manual download needed** - just run the training script.

### Manual Download (if needed)

If you want to pre-download or use offline:

```python
from transformers import AutoModelForCausalLM, AutoTokenizer

model_id = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForCausalLM.from_pretrained(model_id)

# Save locally
tokenizer.save_pretrained("./models/base/tinyllama-1.1b-chat")
model.save_pretrained("./models/base/tinyllama-1.1b-chat")
```

Then update `configs/train_lora.yaml`:
```yaml
base_model: ./models/base/tinyllama-1.1b-chat
```

---

## Folder Structure

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

## Adding New Source Documents

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

## Configuration

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

## Troubleshooting

### Python Version Issues
- Use Python 3.11 or 3.12. Python 3.13+ may not have compatible PyTorch wheels.
- Check: `py -3.12 --version`

### CUDA/GPU Not Detected
```powershell
# Verify CUDA
python -c "import torch; print(torch.cuda.is_available(), torch.version.cuda)"
```

### Docker Issues
- Ensure Docker Desktop is running with WSL 2 backend
- Enable "Use the WSL 2 based engine" in Docker Desktop settings
- For GPU: Enable "Enable NVIDIA GPU support" in Docker Desktop > Resources > Advanced

### Out of Memory (OOM)
- Reduce `per_device_train_batch_size` to 1
- Increase `gradient_accumulation_steps`
- Use `max_seq_length: 512` instead of 768

### Model Download Fails
- Check internet connection
- Set `HF_HUB_DISABLE_TELEMETRY=1` (already in scripts)
- Use `huggingface-cli login` if rate limited

---

## License

Apache 2.0 - See base model license at https://huggingface.co/TinyLlama/TinyLlama-1.1B-Chat-v1.0