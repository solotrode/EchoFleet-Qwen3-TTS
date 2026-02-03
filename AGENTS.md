# AGENTS.md - Qwen3-TTS Docker Project Coding Standards

## Purpose

This document establishes comprehensive coding standards, conventions, and best practices for AI agents (including AI-assisted development tools, copilots, and automated code generators) working on the Qwen3-TTS Docker containerization project. All generated code MUST conform to these standards.

## Table of Contents

1. [General Principles](#general-principles)
2. [Python Coding Standards](#python-coding-standards)
3. [Project Structure](#project-structure)
4. [Docker Best Practices](#docker-best-practices)
5. [API Development Standards](#api-development-standards)
6. [Model Handling Standards](#model-handling-standards)
7. [Error Handling](#error-handling)
8. [Testing Requirements](#testing-requirements)
9. [Documentation Standards](#documentation-standards)
10. [Security Guidelines](#security-guidelines)
11. [Performance Optimization](#performance-optimization)
12. [Git Workflow](#git-workflow)

---

## General Principles

### Code Philosophy
1. **Clarity over Cleverness**: Write code that is immediately understandable. Avoid clever one-liners that sacrifice readability.
2. **Explicit over Implicit**: Be explicit in your intentions. Don't rely on implicit behaviors.
3. **Fail Fast**: Detect and report errors as early as possible. Don't let errors propagate.
4. **Single Responsibility**: Each function/class should have one clear purpose.
5. **DRY (Don't Repeat Yourself)**: Abstract common patterns, but not prematurely.
6. **YAGNI (You Aren't Gonna Need It)**: Don't add functionality until it's needed.

### AI Agent Responsibilities
- **ALWAYS** run code formatters before committing
- **ALWAYS** validate code against these standards
- **ALWAYS** include type hints in Python code
- **ALWAYS** write docstrings for public functions and classes
- **ALWAYS** include unit tests for new functionality
- **NEVER** commit code with TODO comments without creating corresponding issues
- **NEVER** hardcode values that should be configurable
- **NEVER** commit debugging print statements

---

## Python Coding Standards

### Version and Compatibility
- **Python Version**: 3.12+
- **Type Checking**: Use mypy for static type checking
- **Compatibility**: Code should work on Linux (Ubuntu 22.04)

### Code Style

#### PEP 8 Compliance
- Follow PEP 8 with the following modifications:
  - **Line Length**: 100 characters (not 79)
  - **Docstring Style**: Google style docstrings

#### Formatting Tools
```bash
# Required formatters (run in this order)
black --line-length 100 .
isort --profile black .
autoflake --remove-all-unused-imports --recursive --in-place .
```

#### Import Organization
```python
# Standard library imports
import os
import sys
from typing import Dict, List, Optional

# Third-party imports
import torch
import numpy as np
from fastapi import FastAPI

# Local imports
from models.loader import ModelManager
from utils.logging import get_logger
```

**Rules**:
- Group imports: standard library, third-party, local
- Alphabetize within groups
- Use absolute imports for project modules
- Avoid wildcard imports (`from module import *`)

### Type Hints

**REQUIRED** for all function signatures:

```python
# Good
def load_model(
    model_name: str,
    device_map: str = "auto",
    dtype: torch.dtype = torch.bfloat16
) -> Qwen3TTSModel:
    """Load TTS model with specified configuration."""
    pass

# Bad
def load_model(model_name, device_map="auto", dtype=torch.bfloat16):
    pass
```

**Use typing module for complex types**:
```python
from typing import Dict, List, Optional, Union, Tuple, Any

# Good
def process_batch(
    texts: List[str],
    speakers: Optional[List[str]] = None
) -> Dict[str, torch.Tensor]:
    pass

# Use Union for multiple types
def load_audio(source: Union[str, bytes, np.ndarray]) -> torch.Tensor:
    pass

# Use Optional for None-able types
def get_model(model_type: str) -> Optional[Qwen3TTSModel]:
    pass
```

### Naming Conventions

#### Variables and Functions
```python
# Snake case for variables and functions
model_name = "qwen3-tts-base"
inference_engine = InferenceEngine()

def generate_audio(text: str) -> torch.Tensor:
    pass

def load_model_from_cache(cache_dir: str) -> Model:
    pass
```

#### Classes
```python
# PascalCase for classes
class VoiceCloneEngine:
    pass

class AudioProcessor:
    pass

class ModelManager:
    pass
```

#### Constants
```python
# UPPER_SNAKE_CASE for constants
MAX_TEXT_LENGTH = 5000
DEFAULT_SAMPLE_RATE = 24000
SUPPORTED_LANGUAGES = ["en", "zh", "ja", "ko"]
MODEL_CACHE_DIR = "/models"
```

#### Private Members
```python
class ModelManager:
    def __init__(self):
        self._model = None  # Private attribute (single underscore)
        self.__internal_cache = {}  # Name mangling (double underscore, rare use)
    
    def _load_model_internal(self):  # Private method
        pass
```

### Docstrings

**Use Google-style docstrings** for all public functions and classes:

```python
def generate_voice_clone(
    text: str,
    ref_audio: Union[str, bytes],
    ref_text: str,
    language: Optional[str] = None,
    x_vector_only_mode: bool = False
) -> Tuple[torch.Tensor, int]:
    """Generate cloned voice audio from reference sample.
    
    This function performs voice cloning by extracting speaker embeddings
    from the reference audio and generating new speech with those characteristics.
    
    Args:
        text: Text to synthesize in the cloned voice.
        ref_audio: Reference audio file path, URL, or bytes.
        ref_text: Transcript of the reference audio.
        language: Target language code (e.g., 'en', 'zh'). Auto-detected if None.
        x_vector_only_mode: If True, use only x-vector embeddings without reference text.
    
    Returns:
        A tuple containing:
            - Generated audio tensor of shape (samples,)
            - Sample rate as integer (typically 24000)
    
    Raises:
        ValueError: If ref_audio cannot be loaded or text is empty.
        ModelNotLoadedError: If the voice clone model is not loaded.
        CUDAOutOfMemoryError: If insufficient GPU memory for generation.
    
    Example:
        >>> audio, sr = generate_voice_clone(
        ...     "Hello world",
        ...     ref_audio="speaker.wav",
        ...     ref_text="This is a reference",
        ...     language="en"
        ... )
        >>> audio.shape
        torch.Size([240000])  # 10 seconds at 24kHz
    
    Note:
        Reference audio should be at least 3 seconds long for best results.
        Longer reference audio (up to 30 seconds) may improve quality.
    """
    pass
```

**Class docstrings**:
```python
class ModelManager:
    """Manages loading and lifecycle of Qwen3-TTS models.
    
    This class handles lazy loading, caching, and multi-GPU distribution
    of the three Qwen3-TTS model variants. It ensures efficient memory
    usage and provides a unified interface for model access.
    
    Attributes:
        cache_dir: Directory where model weights are cached.
        device_map: Strategy for distributing model across GPUs.
        dtype: Data type for model weights (typically torch.bfloat16).
        models: Dictionary mapping model types to loaded models.
    
    Example:
        >>> manager = ModelManager(cache_dir="/models")
        >>> model = manager.get_model("base")
        >>> audio = model.generate_voice_clone(...)
    """
    
    def __init__(self, cache_dir: str, device_map: str = "auto"):
        """Initialize ModelManager with configuration.
        
        Args:
            cache_dir: Directory for caching model weights.
            device_map: GPU distribution strategy ('auto', 'balanced', etc.).
        """
        pass
```

### Error Handling

#### Custom Exceptions
```python
# Define project-specific exceptions
class Qwen3TTSError(Exception):
    """Base exception for Qwen3-TTS errors."""
    pass

class ModelNotFoundError(Qwen3TTSError):
    """Raised when requested model is not found."""
    pass

class AudioProcessingError(Qwen3TTSError):
    """Raised when audio processing fails."""
    pass

class CUDAOutOfMemoryError(Qwen3TTSError):
    """Raised when GPU memory is exhausted."""
    pass
```

#### Exception Handling Patterns
```python
# Good: Specific exceptions, proper logging, cleanup
def load_audio(file_path: str) -> torch.Tensor:
    """Load audio file and convert to tensor."""
    logger = get_logger(__name__)
    
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Audio file not found: {file_path}")
    
    try:
        audio, sr = sf.read(file_path)
        audio_tensor = torch.from_numpy(audio)
        
        if sr != TARGET_SAMPLE_RATE:
            audio_tensor = resample_audio(audio_tensor, sr, TARGET_SAMPLE_RATE)
        
        return audio_tensor
        
    except sf.LibsndfileError as e:
        logger.error(f"Failed to read audio file {file_path}: {e}")
        raise AudioProcessingError(f"Cannot read audio file: {e}") from e
    
    except Exception as e:
        logger.exception(f"Unexpected error loading audio {file_path}")
        raise AudioProcessingError(f"Audio loading failed: {e}") from e

# Bad: Bare except, no logging, swallowing exceptions
def load_audio(file_path):
    try:
        audio, sr = sf.read(file_path)
        return torch.from_numpy(audio)
    except:
        return None
```

### Logging

#### Logger Setup
```python
import logging
from utils.logging import get_logger

# Get logger for current module
logger = get_logger(__name__)
```

#### Logging Patterns
```python
# Log levels and usage
logger.debug("Detailed model parameters: %s", model_config)  # Development info
logger.info("Model loaded successfully: %s", model_name)  # General info
logger.warning("GPU memory usage high: %.2f GB", memory_used)  # Potential issues
logger.error("Failed to load model: %s", error_msg)  # Errors
logger.exception("Critical failure in inference")  # Exceptions with traceback

# Good: Structured logging with context
logger.info(
    "Audio generated",
    extra={
        "model": "voice-clone",
        "text_length": len(text),
        "duration_ms": duration,
        "gpu_id": device_id
    }
)

# Use lazy formatting (% style, not f-strings)
logger.info("Processing request %s with model %s", request_id, model_name)

# Never log sensitive data
logger.info("User request", extra={"user_id": hash(user_id)})  # Hashed
```

### Configuration Management

```python
from pydantic import BaseSettings, Field

class Settings(BaseSettings):
    """Application configuration from environment variables."""
    
    # Model configuration
    model_cache_dir: str = Field(default="/models", env="MODEL_CACHE_DIR")
    enable_base_model: bool = Field(default=True, env="ENABLE_BASE_MODEL")
    default_dtype: str = Field(default="bfloat16", env="DEFAULT_DTYPE")
    
    # GPU configuration
    cuda_visible_devices: str = Field(default="0,1,2,3", env="CUDA_VISIBLE_DEVICES")
    gpu_memory_fraction: float = Field(default=0.9, env="GPU_MEMORY_FRACTION")
    
    # API configuration
    api_host: str = Field(default="0.0.0.0", env="API_HOST")
    api_port: int = Field(default=8000, env="API_PORT")
    
    class Config:
        env_file = ".env"
        case_sensitive = False

# Usage
settings = Settings()
```

---

## Project Structure

### Directory Layout

```
qwen3-tts-docker/
├── Dockerfile                  # Main Dockerfile
├── docker-compose.yml          # Docker Compose configuration
├── .dockerignore              # Docker ignore patterns
├── requirements.txt           # Python dependencies
├── setup.py                   # Package setup
├── README.md                  # Project overview
├── AGENTS.md                  # This file
├── .env.example              # Environment variable template
│
├── config/                    # Configuration files
│   ├── __init__.py
│   └── settings.py           # Pydantic settings
│
├── models/                    # Model management
│   ├── __init__.py
│   ├── download.py           # Model download utilities
│   ├── loader.py             # Model loading and management
│   └── registry.py           # Model registry/catalog
│
├── api/                       # FastAPI application
│   ├── __init__.py
│   ├── main.py               # FastAPI app initialization
│   ├── routes.py             # API route definitions
│   ├── schemas.py            # Pydantic request/response schemas
│   ├── dependencies.py       # FastAPI dependencies
│   └── middleware.py         # Custom middleware
│
├── inference/                 # Inference engines
│   ├── __init__.py
│   ├── base.py               # Base inference class
│   ├── voice_clone.py        # Voice cloning implementation
│   ├── custom_voice.py       # Custom voice implementation
│   ├── voice_design.py       # Voice design implementation
│   └── batch_processor.py    # Batch processing
│
├── utils/                     # Utility modules
│   ├── __init__.py
│   ├── audio_utils.py        # Audio processing utilities
│   ├── audio_output.py       # Audio output formatting
│   ├── gpu_utils.py          # GPU management utilities
│   ├── logging.py            # Logging configuration
│   └── errors.py             # Custom exceptions
│
├── tests/                     # Test suite
│   ├── __init__.py
│   ├── conftest.py           # Pytest configuration
│   ├── unit/                 # Unit tests
│   │   ├── test_models.py
│   │   ├── test_audio.py
│   │   └── test_inference.py
│   ├── integration/          # Integration tests
│   │   ├── test_api.py
│   │   └── test_workflows.py
│   └── performance/          # Performance tests
│       └── test_benchmarks.py
│
├── scripts/                   # Utility scripts
│   ├── download_models.sh    # Download all models
│   ├── test_gpu.py           # GPU testing script
│   └── benchmark.py          # Performance benchmarking
│
├── docs/                      # Documentation
│   ├── API.md                # API reference
│   ├── DEPLOYMENT.md         # Deployment guide
│   └── ARCHITECTURE.md       # Architecture documentation
│
└── examples/                  # Usage examples
    ├── python_client.py
    ├── curl_examples.sh
    └── gradio_demo.py
```

### Module Organization Rules

1. **One class per file** (for major classes)
2. **Group related functions** in utility modules
3. **Keep files under 500 lines** (split if larger)
4. **Use `__init__.py`** for package exports

```python
# models/__init__.py - Export public API
from .loader import ModelManager
from .download import download_all_models

__all__ = ["ModelManager", "download_all_models"]
```

---

## Docker Best Practices

### Dockerfile Standards

```dockerfile
# ============================================================================
# Multi-stage Dockerfile for Qwen3-TTS API Service
# Base Image: nvidia/cuda:13.1.0-devel-ubuntu22.04
#
# CUDA REQUIREMENT: Minimum CUDA 13.0
# - Builder: nvidia/cuda:13.1.0-devel-ubuntu22.04
# - Runtime: nvidia/cuda:13.1.0-runtime-ubuntu22.04
# - PyTorch: Compiled for CUDA 13.0 (cu130)
# ============================================================================

# ============================================================================
# Stage 1: Builder - Compile all dependencies
# ============================================================================
FROM nvidia/cuda:13.1.0-devel-ubuntu22.04 AS builder

# Prevent interactive prompts
ENV DEBIAN_FRONTEND=noninteractive

# Install system dependencies (Python 3.12 from deadsnakes PPA)
RUN apt-get update && apt-get install -y --no-install-recommends \
    software-properties-common \
    && add-apt-repository ppa:deadsnakes/ppa -y \
    && apt-get update && apt-get install -y --no-install-recommends \
    python3.12 \
    python3.12-dev \
    python3.12-venv \
    python3-pip \
    git \
    build-essential \
    cmake \
    ninja-build \
    wget \
    ca-certificates \
    sox \
    libsox-fmt-all \
    libsndfile1 \
    ffmpeg \
    && rm -rf /var/lib/apt/lists/*

# Set Python 3.12 as default
RUN update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.12 1 && \
    update-alternatives --install /usr/bin/python python /usr/bin/python3.12 1

# Bootstrap pip using ensurepip
RUN python3.12 -m ensurepip --upgrade && \
    python3.12 -m pip install --no-cache-dir --upgrade pip setuptools wheel

# Create virtual environment
RUN python3 -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Limit parallel builds to reduce memory pressure during CUDA/C++ extension compilation
ARG MAX_JOBS=1
ENV MAX_JOBS=${MAX_JOBS}
ENV CMAKE_BUILD_PARALLEL_LEVEL=${MAX_JOBS}

# Generate conservative list of architectures for building CUDA extensions
ENV TORCH_CUDA_ARCH_LIST="8.9;8.6;8.0"

# Install PyTorch with CUDA support
RUN pip install --no-cache-dir \
    torch \
    torchvision \
    --index-url https://download.pytorch.org/whl/cu130

# Install build dependencies
RUN pip install --no-cache-dir \
    packaging \
    wheel \
    setuptools \
    ninja \
    einops

# Install flash-attn from pre-built wheel (avoids 10+ min CUDA compilation)
RUN pip install --no-cache-dir flash-attn --no-build-isolation || \
    echo "Flash Attention installation failed - will use SDPA fallback"

# Copy requirements and install
COPY requirements.txt /tmp/requirements.txt
RUN pip install --no-cache-dir -r /tmp/requirements.txt

# ============================================================================
# Stage 2: Runtime - Minimal production image
# ============================================================================
FROM nvidia/cuda:13.1.0-runtime-ubuntu22.04 AS runtime

# Prevent interactive prompts
ENV DEBIAN_FRONTEND=noninteractive

# Install minimal runtime dependencies (Python 3.12 from deadsnakes PPA)
RUN apt-get update && apt-get install -y --no-install-recommends \
    software-properties-common \
    && add-apt-repository ppa:deadsnakes/ppa -y \
    && apt-get update && apt-get install -y --no-install-recommends \
    python3.12 \
    python3.12-venv \
    curl \
    libgomp1 \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    sox \
    libsox-fmt-all \
    libsndfile1 \
    ffmpeg \
    && rm -rf /var/lib/apt/lists/*

# Create non-root user
RUN useradd -m -u 1000 -s /bin/bash appuser

# Set Python 3.12 as default
RUN update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.12 1 && \
    update-alternatives --install /usr/bin/python python /usr/bin/python3.12 1

# Copy virtual environment from builder
COPY --from=builder /opt/venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Set environment variables
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    CUDA_HOME=/usr/local/cuda \
    LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH

# Set working directory
WORKDIR /app

# Copy application code
COPY --chown=appuser:appuser . .

# Create necessary directories
RUN mkdir -p /app/models /app/logs && \
    chown -R appuser:appuser /app

# Switch to non-root user
USER appuser

# Expose API port
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD curl -f http://localhost:8000/v1/health || exit 1

# Default command
CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"]
```

### Docker Compose Standards

```yaml
version: '3.8'

services:
  qwen3-tts:
    build:
      context: .
      dockerfile: Dockerfile
      args:
        - CUDA_VERSION=13.1.0
        - MAX_JOBS=1
    
    image: qwen3-tts:latest
    
    container_name: qwen3-tts-server
    
    # GPU configuration
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              count: all
              capabilities: [gpu]
    
    # Environment variables
    env_file:
      - .env
    environment:
      - MODEL_CACHE_DIR=/models
      - LOG_LEVEL=INFO
    
    # Volume mounts
    volumes:
      - ./models:/models
      - ./logs:/logs
      - ./config:/app/config
    
    # Port mapping
    ports:
      - "8000:8000"
    
    # Health check
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:8000/v1/health"]
      interval: 30s
      timeout: 10s
      retries: 3
      start_period: 60s
    
    # Restart policy
    restart: unless-stopped
    
    # Logging
    logging:
      driver: "json-file"
      options:
        max-size: "10m"
        max-file: "3"
    
    # Network
    networks:
      - qwen3-network

networks:
  qwen3-network:
    driver: bridge
```

### .dockerignore

```
# Python
__pycache__/
*.py[cod]
*$py.class
*.so
.Python
env/
venv/
.venv/
pip-log.txt
.pytest_cache/
.mypy_cache/

# IDE
.vscode/
.idea/
*.swp
*.swo
*~

# Git
.git/
.gitignore
.gitattributes

# Documentation
*.md
docs/

# Tests
tests/
*.test

# CI/CD
.github/
.gitlab-ci.yml

# Local development
.env
.env.local
*.log

# Models (download in container)
models/
*.safetensors
*.bin
*.pt

# Temporary
tmp/
temp/
```

---

## API Development Standards

### FastAPI Application Structure

```python
# api/main.py
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.gzip import GZipMiddleware
from fastapi.responses import JSONResponse

from api.routes import router
from api.middleware import LoggingMiddleware, ErrorHandlerMiddleware
from config.settings import settings
from utils.logging import get_logger

logger = get_logger(__name__)

def create_app() -> FastAPI:
    """Create and configure FastAPI application."""
    
    app = FastAPI(
        title="Qwen3-TTS API",
        description="Multi-model text-to-speech API with voice cloning",
        version="1.0.0",
        docs_url="/docs",
        redoc_url="/redoc",
        openapi_url="/openapi.json"
    )
    
    # CORS middleware
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],  # Configure appropriately for production
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    
    # Compression middleware
    app.add_middleware(GZipMiddleware, minimum_size=1000)
    
    # Custom middleware
    app.add_middleware(LoggingMiddleware)
    app.add_middleware(ErrorHandlerMiddleware)
    
    # Include routers
    app.include_router(router, prefix="/v1")
    
    @app.on_event("startup")
    async def startup_event():
        """Initialize resources on startup."""
        logger.info("Starting Qwen3-TTS API server")
        # Initialize model manager, etc.
    
    @app.on_event("shutdown")
    async def shutdown_event():
        """Cleanup resources on shutdown."""
        logger.info("Shutting down Qwen3-TTS API server")
        # Cleanup models, close connections, etc.
    
    return app

app = create_app()
```

### Request/Response Schemas

```python
# api/schemas.py
from typing import Optional, List, Literal
from pydantic import BaseModel, Field, validator

class VoiceCloneRequest(BaseModel):
    """Request schema for voice cloning endpoint."""
    
    text: str = Field(
        ...,
        description="Text to synthesize",
        min_length=1,
        max_length=5000,
        example="Hello, this is a test of voice cloning."
    )
    
    ref_audio: str = Field(
        ...,
        description="Reference audio (file path, URL, or base64)",
        example="https://example.com/audio.wav"
    )
    
    ref_text: str = Field(
        ...,
        description="Transcript of reference audio",
        example="This is the reference speech."
    )
    
    language: Optional[str] = Field(
        default=None,
        description="Target language code (auto-detected if not specified)",
        example="en"
    )
    
    x_vector_only_mode: bool = Field(
        default=False,
        description="Use only x-vector embeddings (faster, lower quality)"
    )
    
    output_format: Literal["wav", "mp3", "base64"] = Field(
        default="wav",
        description="Output audio format"
    )
    
    @validator("text")
    def validate_text(cls, v):
        """Validate text is not empty."""
        if not v.strip():
            raise ValueError("Text cannot be empty")
        return v
    
    @validator("language")
    def validate_language(cls, v):
        """Validate language code."""
        if v and v not in SUPPORTED_LANGUAGES:
            raise ValueError(f"Unsupported language: {v}")
        return v
    
    class Config:
        schema_extra = {
            "example": {
                "text": "Welcome to our voice cloning system.",
                "ref_audio": "https://example.com/speaker.wav",
                "ref_text": "This is my voice sample.",
                "language": "en",
                "output_format": "wav"
            }
        }

class AudioResponse(BaseModel):
    """Response schema for audio generation."""
    
    audio_data: str = Field(
        ...,
        description="Audio data (base64 encoded or URL)"
    )
    
    sample_rate: int = Field(
        ...,
        description="Audio sample rate in Hz",
        example=24000
    )
    
    duration_seconds: float = Field(
        ...,
        description="Audio duration in seconds",
        example=5.2
    )
    
    format: str = Field(
        ...,
        description="Audio format",
        example="wav"
    )

class ErrorResponse(BaseModel):
    """Error response schema."""
    
    error: str = Field(..., description="Error type")
    message: str = Field(..., description="Error message")
    detail: Optional[str] = Field(None, description="Detailed error information")
```

### Route Definitions

```python
# api/routes.py
from fastapi import APIRouter, HTTPException, Depends, BackgroundTasks
from fastapi.responses import StreamingResponse

from api.schemas import VoiceCloneRequest, AudioResponse, ErrorResponse
from api.dependencies import get_model_manager
from inference.voice_clone import VoiceCloneEngine
from utils.logging import get_logger

logger = get_logger(__name__)
router = APIRouter()

@router.post(
    "/tts/voice-clone",
    response_model=AudioResponse,
    responses={
        400: {"model": ErrorResponse},
        500: {"model": ErrorResponse}
    },
    tags=["Voice Cloning"],
    summary="Generate cloned voice audio"
)
async def voice_clone(
    request: VoiceCloneRequest,
    model_manager = Depends(get_model_manager),
    background_tasks: BackgroundTasks = BackgroundTasks()
):
    """
    Generate speech in a cloned voice from reference audio.
    
    This endpoint performs voice cloning by extracting speaker characteristics
    from a 3+ second reference audio sample and generating new speech.
    """
    try:
        logger.info(
            "Voice clone request received",
            extra={
                "text_length": len(request.text),
                "language": request.language,
                "output_format": request.output_format
            }
        )
        
        # Get voice clone engine
        engine = VoiceCloneEngine(model_manager.get_model("base"))
        
        # Generate audio
        audio, sr = await engine.generate(
            text=request.text,
            ref_audio=request.ref_audio,
            ref_text=request.ref_text,
            language=request.language,
            x_vector_only_mode=request.x_vector_only_mode
        )
        
        # Process output
        audio_data = await process_audio_output(
            audio=audio,
            sample_rate=sr,
            format=request.output_format
        )
        
        duration = len(audio) / sr
        
        logger.info(
            "Voice clone completed",
            extra={"duration_seconds": duration}
        )
        
        return AudioResponse(
            audio_data=audio_data,
            sample_rate=sr,
            duration_seconds=duration,
            format=request.output_format
        )
    
    except ValueError as e:
        logger.warning(f"Invalid request: {e}")
        raise HTTPException(status_code=400, detail=str(e))
    
    except Exception as e:
        logger.exception("Voice clone failed")
        raise HTTPException(status_code=500, detail="Internal server error")

@router.get(
    "/health",
    tags=["System"],
    summary="Health check endpoint"
)
async def health_check():
    """Check API health and model availability."""
    return {
        "status": "healthy",
        "models_loaded": True,
        "gpu_available": torch.cuda.is_available()
    }
```

### Dependency Injection

```python
# api/dependencies.py
from functools import lru_cache
from fastapi import Depends

from models.loader import ModelManager
from config.settings import Settings

@lru_cache()
def get_settings() -> Settings:
    """Get cached application settings."""
    return Settings()

_model_manager = None

def get_model_manager(
    settings: Settings = Depends(get_settings)
) -> ModelManager:
    """Get or create model manager singleton."""
    global _model_manager
    
    if _model_manager is None:
        _model_manager = ModelManager(
            cache_dir=settings.model_cache_dir,
            device_map="auto",
            dtype=settings.default_dtype
        )
    
    return _model_manager
```

---

## Model Handling Standards

### Model Loading

```python
# models/loader.py
import torch
from typing import Dict, Optional
from qwen_tts import Qwen3TTSModel

from utils.logging import get_logger
from utils.gpu_utils import get_device_map, get_optimal_dtype
from utils.errors import ModelNotFoundError, CUDAOutOfMemoryError

logger = get_logger(__name__)

class ModelManager:
    """Manages Qwen3-TTS model lifecycle."""
    
    MODEL_REGISTRY = {
        "base": "Qwen/Qwen3-TTS-12Hz-1.7B-Base",
        "custom-voice": "Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice",
        "voice-design": "Qwen/Qwen3-TTS-12Hz-1.7B-VoiceDesign"
    }
    
    def __init__(
        self,
        cache_dir: str,
        device_map: str = "auto",
        dtype: str = "bfloat16"
    ):
        """Initialize model manager.
        
        Args:
            cache_dir: Directory for model cache.
            device_map: GPU distribution strategy.
            dtype: Model data type ('bfloat16' or 'float16').
        """
        self.cache_dir = cache_dir
        self.device_map = device_map
        self.dtype = get_optimal_dtype(dtype)
        self._models: Dict[str, Qwen3TTSModel] = {}
        
        logger.info(
            "ModelManager initialized",
            extra={
                "cache_dir": cache_dir,
                "device_map": device_map,
                "dtype": dtype,
                "cuda_available": torch.cuda.is_available(),
                "gpu_count": torch.cuda.device_count()
            }
        )
    
    def get_model(self, model_type: str) -> Qwen3TTSModel:
        """Get or load model by type.
        
        Args:
            model_type: Model type ('base', 'custom-voice', 'voice-design').
        
        Returns:
            Loaded model instance.
        
        Raises:
            ModelNotFoundError: If model type is invalid.
            CUDAOutOfMemoryError: If insufficient GPU memory.
        """
        if model_type not in self.MODEL_REGISTRY:
            raise ModelNotFoundError(f"Unknown model type: {model_type}")
        
        # Return cached model if already loaded
        if model_type in self._models:
            logger.debug(f"Returning cached model: {model_type}")
            return self._models[model_type]
        
        # Load model
        logger.info(f"Loading model: {model_type}")
        try:
            model = self._load_model(model_type)
            self._models[model_type] = model
            logger.info(f"Model loaded successfully: {model_type}")
            return model
        
        except torch.cuda.OutOfMemoryError as e:
            logger.error(f"CUDA OOM loading {model_type}: {e}")
            raise CUDAOutOfMemoryError(
                f"Insufficient GPU memory to load {model_type}"
            ) from e
    
    def _load_model(self, model_type: str) -> Qwen3TTSModel:
        """Internal method to load model from HuggingFace Hub."""
        model_name = self.MODEL_REGISTRY[model_type]
        
        device_map = self._get_device_map()
        
        model = Qwen3TTSModel.from_pretrained(
            model_name,
            cache_dir=self.cache_dir,
            torch_dtype=self.dtype,
            device_map=device_map,
            attn_implementation="flash_attention_2",
            low_cpu_mem_usage=True
        )
        
        return model
    
    def _get_device_map(self) -> dict:
        """Get device map for multi-GPU distribution."""
        if self.device_map == "auto":
            return "auto"  # Let accelerate handle it
        
        # Custom device mapping logic here if needed
        return self.device_map
    
    def unload_model(self, model_type: str) -> None:
        """Unload model to free GPU memory."""
        if model_type in self._models:
            del self._models[model_type]
            torch.cuda.empty_cache()
            logger.info(f"Model unloaded: {model_type}")
```

---

## Testing Requirements

### Unit Test Standards

```python
# tests/unit/test_models.py
import pytest
import torch
from unittest.mock import Mock, patch

from models.loader import ModelManager
from utils.errors import ModelNotFoundError

class TestModelManager:
    """Unit tests for ModelManager class."""
    
    @pytest.fixture
    def model_manager(self, tmp_path):
        """Create ModelManager instance for testing."""
        return ModelManager(
            cache_dir=str(tmp_path),
            device_map="cpu",  # Use CPU for testing
            dtype="float32"
        )
    
    def test_init_creates_model_manager(self, model_manager):
        """Test ModelManager initialization."""
        assert model_manager.cache_dir is not None
        assert model_manager.device_map == "cpu"
        assert isinstance(model_manager._models, dict)
    
    def test_get_model_raises_error_for_unknown_type(self, model_manager):
        """Test get_model raises error for invalid model type."""
        with pytest.raises(ModelNotFoundError) as exc_info:
            model_manager.get_model("invalid-model")
        
        assert "Unknown model type" in str(exc_info.value)
    
    @patch("models.loader.Qwen3TTSModel.from_pretrained")
    def test_get_model_loads_and_caches(self, mock_from_pretrained, model_manager):
        """Test model loading and caching."""
        mock_model = Mock()
        mock_from_pretrained.return_value = mock_model
        
        # First call should load
        model1 = model_manager.get_model("base")
        assert model1 is mock_model
        assert mock_from_pretrained.call_count == 1
        
        # Second call should return cached
        model2 = model_manager.get_model("base")
        assert model2 is model1
        assert mock_from_pretrained.call_count == 1  # Not called again
    
    def test_unload_model_removes_from_cache(self, model_manager):
        """Test model unloading."""
        # Setup: Load a model (mocked)
        model_manager._models["base"] = Mock()
        
        # Unload
        model_manager.unload_model("base")
        
        # Verify
        assert "base" not in model_manager._models
```

### Integration Test Standards

```python
# tests/integration/test_api.py
import pytest
from fastapi.testclient import TestClient

from api.main import app

@pytest.fixture
def client():
    """Create test client."""
    return TestClient(app)

class TestVoiceCloneEndpoint:
    """Integration tests for voice clone endpoint."""
    
    def test_voice_clone_success(self, client):
        """Test successful voice cloning request."""
        payload = {
            "text": "Hello world",
            "ref_audio": "path/to/audio.wav",
            "ref_text": "Reference text",
            "language": "en"
        }
        
        response = client.post("/v1/tts/voice-clone", json=payload)
        
        assert response.status_code == 200
        data = response.json()
        assert "audio_data" in data
        assert data["sample_rate"] == 24000
    
    def test_voice_clone_invalid_language(self, client):
        """Test error handling for invalid language."""
        payload = {
            "text": "Hello",
            "ref_audio": "audio.wav",
            "ref_text": "Ref",
            "language": "invalid"
        }
        
        response = client.post("/v1/tts/voice-clone", json=payload)
        
        assert response.status_code == 422  # Validation error
    
    def test_voice_clone_empty_text(self, client):
        """Test error handling for empty text."""
        payload = {
            "text": "",
            "ref_audio": "audio.wav",
            "ref_text": "Ref"
        }
        
        response = client.post("/v1/tts/voice-clone", json=payload)
        
        assert response.status_code == 422
```

### Test Organization Rules

1. **One test class per class/module** being tested
2. **Descriptive test names** that explain what is being tested
3. **AAA pattern**: Arrange, Act, Assert
4. **Use fixtures** for setup/teardown
5. **Mock external dependencies** (API calls, file I/O, GPU operations)
6. **Test edge cases** and error conditions

### Coverage Requirements

- **Minimum coverage**: 80% overall
- **Critical paths**: 95%+ coverage (model loading, inference, API endpoints)
- **Run coverage** before committing:

```bash
pytest --cov=. --cov-report=html --cov-report=term
```

---

## Performance Optimization

### GPU Memory Management

#### Aggressive VRAM Cleanup Pattern (VERIFIED WORKING)

When switching between large models or completing multi-phase GPU workflows, use this **production-verified** pattern:

```python
import torch
import gc

def aggressive_vram_cleanup(job_id: str = ""):
    """Aggressively clear VRAM between model phases.
    
    CRITICAL for multi-model workflows (e.g., TTS → Whisper).
    Prevents Docker crashes from accumulated VRAM leaks.
    
    Verified working for back-to-back multi-candidate jobs.
    """
    if not torch.cuda.is_available():
        return
    
    logger.info(f"Performing aggressive VRAM cleanup", extra={"job_id": job_id})
    
    # Synchronize all CUDA devices first
    for device_id in range(torch.cuda.device_count()):
        with torch.cuda.device(device_id):
            torch.cuda.synchronize()
            torch.cuda.empty_cache()
    
    # Multiple GC passes to catch circular references
    for _ in range(3):
        gc.collect()
    
    # Final cache clear after GC freed references
    for device_id in range(torch.cuda.device_count()):
        with torch.cuda.device(device_id):
            torch.cuda.empty_cache()
    
    logger.info(f"VRAM cleanup completed", extra={"job_id": job_id})

# Usage: After unloading models
model = None  # Delete model reference
aggressive_vram_cleanup(job_id="abc123")

# Legacy context manager (use aggressive_vram_cleanup instead for critical paths)
from contextlib import contextmanager

@contextmanager
def torch_gc():
    """Context manager to ensure GPU memory cleanup."""
    try:
        yield
    finally:
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()

# Usage
with torch_gc():
    audio = model.generate(text)
```

#### Model Unloading Best Practice

**ALWAYS unload models before loading a different large model:**

```python
# BAD: Loading Whisper while TTS models still in memory
tts_result = tts_model.generate(text)
whisper_model = load_whisper()  # OOM risk!

# GOOD: Explicit unload sequence
tts_result = tts_model.generate(text)
tts_model.unload()  # Free TTS memory
aggressive_vram_cleanup()  # Ensure cleanup
whisper_model = load_whisper()  # Safe to load
```

### Batch Processing

```python
def process_batch(
    texts: List[str],
    batch_size: int = 4
) -> List[torch.Tensor]:
    """Process texts in batches for efficiency.
    
    Args:
        texts: List of texts to process.
        batch_size: Number of texts per batch.
    
    Returns:
        List of generated audio tensors.
    """
    results = []
    
    for i in range(0, len(texts), batch_size):
        batch = texts[i:i + batch_size]
        
        with torch.no_grad(), torch_gc():
            batch_results = model.generate_batch(batch)
        
        results.extend(batch_results)
    
    return results
```

### Caching Strategies

```python
from functools import lru_cache

@lru_cache(maxsize=128)
def load_speaker_embedding(speaker_id: str) -> torch.Tensor:
    """Load and cache speaker embeddings."""
    return _load_embedding_from_disk(speaker_id)
```

---

## Security Guidelines

### Input Validation

```python
from pydantic import validator, Field

class SecureRequest(BaseModel):
    """Secure request with validation."""
    
    text: str = Field(..., max_length=5000)
    
    @validator("text")
    def sanitize_text(cls, v):
        """Remove potentially harmful characters."""
        # Remove control characters
        v = "".join(char for char in v if ord(char) >= 32 or char == '\n')
        return v.strip()
```

### File Path Security

```python
import os
from pathlib import Path

def safe_path_join(base_dir: str, user_path: str) -> Path:
    """Safely join paths preventing directory traversal.
    
    Raises:
        ValueError: If path attempts to escape base directory.
    """
    base = Path(base_dir).resolve()
    target = (base / user_path).resolve()
    
    # Ensure target is within base directory
    if not target.is_relative_to(base):
        raise ValueError("Path traversal detected")
    
    return target
```

### API Rate Limiting

```python
from fastapi import FastAPI, Request
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded

limiter = Limiter(key_func=get_remote_address)
app = FastAPI()
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

@app.post("/v1/tts/voice-clone")
@limiter.limit("10/minute")  # 10 requests per minute
async def voice_clone(request: Request):
    pass
```

---

## Git Workflow

### Commit Message Format

```
<type>(<scope>): <subject>

<body>

<footer>
```

**Types**:
- `feat`: New feature
- `fix`: Bug fix
- `docs`: Documentation changes
- `style`: Code style changes (formatting, no logic change)
- `refactor`: Code refactoring
- `perf`: Performance improvements
- `test`: Adding or updating tests
- `chore`: Maintenance tasks

**Examples**:
```
feat(api): add voice design endpoint

Implement voice design endpoint with natural language
voice description parsing and generation.

Closes #123

---

fix(models): resolve CUDA OOM in multi-GPU setup

Adjust device_map to balance memory across GPUs more evenly.
Reduces memory usage per GPU by ~20%.

---

docs(readme): update installation instructions

Add CUDA 12.8 requirement and multi-GPU setup guide.
```

### Branch Naming

- `feature/voice-design-endpoint`
- `bugfix/cuda-memory-leak`
- `refactor/model-loading`
- `docs/api-documentation`

### Pre-commit Checklist

Before committing, AI agents MUST:

```bash
# 1. Format code
black --line-length 100 .
isort --profile black .

# 2. Type check
mypy .

# 3. Lint
flake8 .
pylint src/

# 4. Run tests
pytest tests/

# 5. Check coverage
pytest --cov=. --cov-fail-under=80

# 6. Security check
bandit -r . -ll

# 7. Check for secrets
detect-secrets scan
```

---

## AI Agent Checklist

Before submitting any code, verify:

### Code Quality
- [ ] Code follows PEP 8 and project style guide
- [ ] All functions have type hints
- [ ] All public functions have docstrings (Google style)
- [ ] No hardcoded values (use config)
- [ ] No print() statements (use logger)
- [ ] No commented-out code
- [ ] No TODO comments without GitHub issues

### Functionality
- [ ] Code accomplishes intended purpose
- [ ] Edge cases handled
- [ ] Error handling implemented
- [ ] Logging added at appropriate levels
- [ ] Input validation present

### Testing
- [ ] Unit tests written and passing
- [ ] Integration tests added for new endpoints
- [ ] Test coverage ≥ 80%
- [ ] All existing tests still passing

### Documentation
- [ ] Docstrings complete and accurate
- [ ] API documentation updated (if applicable)
- [ ] README updated (if applicable)
- [ ] Example code provided (if applicable)

### Performance
- [ ] No obvious performance bottlenecks
- [ ] GPU memory managed properly
- [ ] Batch processing used where appropriate
- [ ] Caching implemented for repeated operations

### Security
- [ ] Input validation implemented
- [ ] No SQL injection vulnerabilities
- [ ] No path traversal vulnerabilities
- [ ] Secrets not hardcoded
- [ ] Rate limiting considered

### Git
- [ ] Meaningful commit message
- [ ] Branch named appropriately
- [ ] No merge conflicts
- [ ] Changes focused and atomic

---

## Common Patterns and Anti-Patterns

### ✅ Good Patterns

```python
# Good: Resource management with context managers
with torch.no_grad():
    with torch_gc():
        result = model.generate(text)

# Good: Clear variable names
audio_sample_rate = 24000
reference_audio_tensor = load_audio(ref_path)

# Good: Early returns for validation
def process_request(text: str) -> Result:
    if not text:
        raise ValueError("Text cannot be empty")
    
    if len(text) > MAX_LENGTH:
        raise ValueError(f"Text exceeds maximum length: {MAX_LENGTH}")
    
    return generate_audio(text)

# Good: Dependency injection
def create_engine(model: Model, config: Config) -> Engine:
    return Engine(model, config)
```

### ❌ Anti-Patterns

```python
# Bad: Bare except
try:
    result = risky_operation()
except:
    pass

# Bad: Mutable default arguments
def process_batch(items, config={}):  # BUG: shared dict
    config['processed'] = True

# Bad: Global state
MODEL = None  # Avoid global mutable state

def load_model():
    global MODEL
    MODEL = load()

# Bad: String concatenation for paths
path = base_dir + "/" + file_name  # Use Path or os.path.join

# Bad: Ignoring errors
result = model.generate(text)  # What if this fails?
save_to_disk(result)  # This will crash if result is None

# Bad: Cryptic names
def p(t, r, l):  # What do these mean?
    return m.g(t, r, l)
```

---

## Final Notes for AI Agents

### Priority Order
1. **Correctness**: Code must work
2. **Safety**: Handle errors, validate inputs
3. **Clarity**: Code must be readable
4. **Performance**: Optimize where it matters
5. **Style**: Follow conventions

### When in Doubt
- **Ask for clarification** rather than guessing
- **Search documentation** before implementing
- **Follow existing patterns** in the codebase
- **Write tests first** for complex logic (TDD)
- **Refactor incrementally** rather than all at once

### Remember
- This is a **production system** handling GPU resources
- **GPU memory is limited** - always clean up
- **Users depend on this API** - handle errors gracefully
- **Security matters** - validate and sanitize inputs
- **Performance matters** - profile before optimizing

**The goal is reliable, maintainable, high-performance code that serves users well.**