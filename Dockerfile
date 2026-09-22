# Base image with Python 3.12 and CUDA support
FROM nvidia/cuda:12.1.0-devel-ubuntu22.04

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    HF_HUB_DISABLE_TELEMETRY=1

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3.12 python3.12-venv python3.12-dev \
    git cmake build-essential tesseract-ocr \
    && rm -rf /var/lib/apt/lists/*

# Set Python 3.12 as default
RUN update-alternatives --install /usr/bin/python python /usr/bin/python3.12 1 \
    && update-alternatives --install /usr/bin/pip pip /usr/bin/pip3.12 1

WORKDIR /app

# Copy package metadata and source first so the editable install inside
# requirements.txt (-e .[local]) can resolve the project itself
COPY pyproject.toml ./
COPY src ./src
COPY requirements.txt .
RUN pip install --upgrade pip && pip install -r requirements.txt

# Copy the rest of the project
COPY . .

# Clone and build llama.cpp for GGUF conversion
RUN git clone https://github.com/ggml-org/llama.cpp /app/external/llama.cpp \
    && cd /app/external/llama.cpp \
    && cmake -B build -DCMAKE_BUILD_TYPE=Release -DGGML_CUDA=ON \
    && cmake --build build --config Release -j$(nproc)

# Default command
CMD ["bash"]