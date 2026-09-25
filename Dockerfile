# Base image with Python 3.12 and CUDA support
FROM nvidia/cuda:12.1.0-devel-ubuntu22.04

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    HF_HUB_DISABLE_TELEMETRY=1

# Install system dependencies.
# Ubuntu 22.04 (jammy) ships Python 3.10 by default and has no python3.12
# package, so we pull 3.12 from the deadsnakes PPA.
RUN apt-get update && apt-get install -y --no-install-recommends \
    software-properties-common \
    && add-apt-repository ppa:deadsnakes/ppa \
    && apt-get update && apt-get install -y --no-install-recommends \
    python3.12 python3.12-venv python3.12-dev \
    git cmake build-essential tesseract-ocr \
    && rm -rf /var/lib/apt/lists/*

# Set Python 3.12 as default and bootstrap pip (the .deb does not ship pip3.12)
RUN update-alternatives --install /usr/bin/python python /usr/bin/python3.12 1 \
    && python3.12 -m ensurepip --upgrade

WORKDIR /app

# Copy package metadata and source first so the editable install inside
# requirements.txt (-e .[local]) can resolve the project itself
COPY pyproject.toml requirements.txt ./
COPY src ./src
RUN python3.12 -m pip install --upgrade pip && python3.12 -m pip install -r requirements.txt

# Copy the rest of the project
COPY . .

# Clone and build llama.cpp for GGUF conversion
RUN git clone https://github.com/ggml-org/llama.cpp /app/external/llama.cpp \
    && cd /app/external/llama.cpp \
    && cmake -B build -DCMAKE_BUILD_TYPE=Release -DGGML_CUDA=ON \
    && cmake --build build --config Release -j$(nproc)

# Default command
CMD ["bash"]