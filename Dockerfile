# Single-stage Dockerfile (reference-style) for Qwen3-TTS.
#
# Why single-stage:
# - Avoids copying a Python venv between images, which can break CUDA dynamic linking.
# - Matches the known-working VibeVoice pattern: one coherent CUDA devel environment.

FROM nvidia/cuda:12.8.1-devel-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive

# Limit parallel builds to reduce memory pressure during CUDA/C++ extension compilation
ARG MAX_JOBS=1
ENV MAX_JOBS=${MAX_JOBS}
ENV CMAKE_BUILD_PARALLEL_LEVEL=${MAX_JOBS}

# Generate comprehensive list of architectures including Blackwell (sm_120)
ENV TORCH_CUDA_ARCH_LIST="12.0;8.9;8.6;8.0"

# System dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    software-properties-common \
    gnupg \
    dirmngr \
    lsb-release \
    && add-apt-repository ppa:deadsnakes/ppa -y \
    && apt-get update && apt-get install -y --no-install-recommends \
    python3.12 \
    python3.12-dev \
    python3.12-venv \
    git \
    build-essential \
    cmake \
    ninja-build \
    curl \
    ca-certificates \
    sox \
    libsox-fmt-all \
    libsndfile1 \
    ffmpeg \
    pkg-config \
    libgomp1 \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    && rm -rf /var/lib/apt/lists/*

RUN update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.12 1 && \
    update-alternatives --install /usr/bin/python python /usr/bin/python3.12 1

RUN python3.12 -m ensurepip --upgrade && \
    python3.12 -m pip install --upgrade pip setuptools wheel

# Virtual environment (keeps runtime isolated and predictable)
RUN python3.12 -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Torch constraints to prevent dependency-driven downgrades (e.g., Whisper).
COPY constraints-cu128-py312.txt /tmp/constraints-cu128-py312.txt

# Install the PyTorch stack from the CUDA 12.8 wheel index.
# This ensures we don't end up with a CUDA 12.1 (cu121) build.
RUN pip install \
    --index-url https://download.pytorch.org/whl/cu128 \
    -c /tmp/constraints-cu128-py312.txt \
    torch==2.10.0+cu128 torchvision torchaudio

# Install prebuilt FlashAttention wheel (compatible with PyTorch 2.10 + cu128).
# Using the provided prebuilt wheel to accelerate attention kernels.
RUN pip install --no-cache-dir \
    https://github.com/mjun0812/flash-attention-prebuild-wheels/releases/download/v0.7.12/flash_attn-2.7.4+cu128torch2.10-cp312-cp312-linux_x86_64.whl

# Python deps for this project
COPY requirements.txt /tmp/requirements.txt
RUN pip install \
    --extra-index-url https://download.pytorch.org/whl/cu128 \
    -c /tmp/constraints-cu128-py312.txt \
    -r /tmp/requirements.txt

# Ensure torch stays pinned (some deps like whisper can try to pull different torch)
RUN pip install \
    --index-url https://download.pytorch.org/whl/cu128 \
    --force-reinstall --no-deps \
    -c /tmp/constraints-cu128-py312.txt \
    torch==2.10.0+cu128 torchvision torchaudio

# Install Qwen3-TTS (editable) without letting it resolve deps (we control deps above)
RUN git clone https://github.com/QwenLM/Qwen3-TTS.git /opt/Qwen3-TTS && \
    cd /opt/Qwen3-TTS && \
    pip install -e . --no-deps

# App code
WORKDIR /app
COPY api/ /app/api/
COPY utils/ /app/utils/
COPY config/ /app/config/
COPY inference/ /app/inference/
COPY ui/ /app/ui/

# Bake models into the image at build time.
# This avoids slow bind-mount I/O from Windows during runtime.
# NOTE: We intentionally only copy `models/Qwen/` to avoid Docker Desktop issues
# with deep HuggingFace cache directories under `models/`.

# Create non-root user and set permissions
RUN useradd -m -u 1000 -s /bin/bash appuser && \
    mkdir -p /app/logs /workspace/outputs /models /home/appuser/.cache && \
    chown -R appuser:appuser /app /workspace /models /home/appuser/.cache /opt/Qwen3-TTS

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    CUDA_HOME=/usr/local/cuda \
    ORT_DISABLE_DEVICE_DISCOVERY=1 \
    PYTHONPATH="/app"

USER appuser

EXPOSE 8000 7860

CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"]

