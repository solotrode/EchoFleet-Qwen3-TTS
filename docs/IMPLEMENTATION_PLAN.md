# Qwen3-TTS Docker Implementation Plan (REVISED)

## Project Overview

Build a production-ready Docker container for Qwen3-TTS models supporting all three 1.7B model variants:
- **Qwen3-TTS-12Hz-1.7B-Base**: Voice cloning with reference audio
- **Qwen3-TTS-12Hz-1.7B-CustomVoice**: Predefined voices with style control
- **Qwen3-TTS-12Hz-1.7B-VoiceDesign**: Natural language voice design

**Target Environment**: Multi-GPU system with NVIDIA 5000 series GPUs requiring CUDA 13.0+

## System Architecture

### Container Components
```
qwen3-tts-container/
├── models/               # Model weights storage (volume mount)
│   ├── tokenizer/       # Qwen3-TTS-Tokenizer-12Hz (REQUIRED)
│   ├── base/
│   ├── custom-voice/
│   └── voice-design/
├── api/                  # FastAPI application
├── inference/            # Inference engine
├── utils/                # Utility modules
└── tests/                # Test suite
```

### Technology Stack
- **Base Image**: nvidia/cuda:13.1.0-devel-ubuntu22.04 (builder), nvidia/cuda:13.1.0-runtime-ubuntu22.04 (runtime)
- **Python**: 3.12 (from deadsnakes PPA)
- **Deep Learning**: PyTorch with CUDA 13.0 (cu130)
- **Web Framework**: FastAPI with Uvicorn
- **Model Management**: Hugging Face Hub
- **Attention**: Flash Attention 2 from pre-built wheel (with SDPA fallback)
- **Multi-GPU**: torch.nn.DataParallel or accelerate

## Implementation Phases

### Phase 1: Docker Environment Setup (Priority: Critical)

#### Task 1.1: Dockerfile Creation with Multi-Stage Build
**Objective**: Create optimized multi-stage Dockerfile with CUDA 13.1.0 support

**Requirements**:
- Use multi-stage build for layer caching and smaller final image
- Base on nvidia/cuda:13.1.0-devel-ubuntu22.04 for build stage
- Base on nvidia/cuda:13.1.0-runtime-ubuntu22.04 for runtime stage
- Install Python 3.12 from deadsnakes PPA (using add-apt-repository)
- Configure NVIDIA runtime
- Install build tools: ninja-build, build-essential, cmake
- Set up environment variables for GPU detection
- Install system dependencies (sox, ffmpeg, libsndfile, libsox-fmt-all)
- Install audio processing tools (pydub dependencies)
- Set MAX_JOBS environment variable to control parallel builds

**Multi-Stage Structure**:
```dockerfile
# Stage 1: Builder - Compile all dependencies
FROM nvidia/cuda:13.1.0-devel-ubuntu22.04 AS builder
# - Install Python 3.12 from deadsnakes PPA
# - Create virtual environment at /opt/venv
# - Install PyTorch with cu130
# - Install flash-attn from pre-built wheel
# - Install all Python dependencies
# - Compile CUDA extensions

# Stage 2: Runtime - Minimal production image
FROM nvidia/cuda:13.1.0-runtime-ubuntu22.04 AS runtime
# - Install minimal runtime dependencies
# - Copy virtual environment from builder
# - Copy application code
# - Set up non-root user
```

**Deliverable**: `Dockerfile` with optimized layer caching and ~1-2 hour initial build time

**Validation**:
- Container builds successfully in under 2 hours
- CUDA 13.1 accessible via nvidia-smi
- Python 3.12 installed and functional
- All GPUs detected (verify with `torch.cuda.device_count()`)
- Build cache works on subsequent builds (under 5 minutes)
- Virtual environment correctly copied from builder stage

#### Task 1.2: CUDA and PyTorch Installation
**Objective**: Install PyTorch with CUDA 13.0 support

**Requirements**:
- Install PyTorch with cu130 variant (latest compatible version)
- Verify CUDA toolkit compatibility (13.0+)
- Configure multi-GPU support
- Add environment variables for optimal CUDA performance
- Set TORCH_CUDA_ARCH_LIST for targeted GPU architectures

**Dependencies**:
```bash
# Install PyTorch with CUDA 13.0 support
pip install --no-cache-dir torch torchvision --index-url https://download.pytorch.org/whl/cu130
```

**Environment Configuration**:
```dockerfile
ENV CUDA_HOME=/usr/local/cuda
ENV LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH
ENV CUDA_LAUNCH_BLOCKING=0
ENV TORCH_CUDA_ARCH_LIST="8.9;8.6;8.0"  # 5000 series + 4000 series + A100 support
ENV MAX_JOBS=1  # Limit parallel builds to avoid memory pressure
ENV CMAKE_BUILD_PARALLEL_LEVEL=1
```

**Deliverable**: Working PyTorch installation with all GPUs accessible

**Validation**:
```python
import torch
assert torch.cuda.is_available()
assert torch.cuda.device_count() > 0
assert torch.version.cuda.startswith("13.")  # CUDA 13.x
print(f"PyTorch version: {torch.__version__}")
print(f"CUDA version: {torch.version.cuda}")
print(f"GPUs available: {torch.cuda.device_count()}")
for i in range(torch.cuda.device_count()):
    print(f"GPU {i}: {torch.cuda.get_device_name(i)}")
```

#### Task 1.3: Flash Attention 2 Installation with Fallback Strategy
**Objective**: Install Flash Attention 2 from pre-built wheel with graceful fallback to PyTorch SDPA

**CRITICAL NOTES**:
- Use pre-built wheel to avoid 1-2 hour compilation time
- Pre-built wheels are compatible with CUDA 13.0
- Flash Attention may still fail on some systems - fallback is required
- Runtime compatibility issues may occur with libcudart.so versions

**Installation Approach** (Using Pre-built Wheel):
```dockerfile
# Install flash-attn from pre-built wheel (avoids 10+ min CUDA compilation)
RUN pip install --no-cache-dir packaging wheel setuptools ninja einops && \
    pip install --no-cache-dir flash-attn --no-build-isolation || \
    echo "Flash Attention installation failed - will use SDPA fallback at runtime"
```

**Note**: This approach avoids the lengthy compilation process by using pre-built wheels. The `--no-build-isolation` flag is still used to ensure dependencies are available.

**Code-Level Fallback** (REQUIRED):
```python
# utils/attention_utils.py
try:
    from flash_attn import flash_attn_func
    FLASH_ATTENTION_AVAILABLE = True
    ATTENTION_IMPLEMENTATION = "flash_attention_2"
except ImportError:
    FLASH_ATTENTION_AVAILABLE = False
    ATTENTION_IMPLEMENTATION = "sdpa"  # PyTorch Scaled Dot Product Attention
    logger.warning("Flash Attention not available, using SDPA fallback")

def get_attention_implementation():
    """Get available attention implementation."""
    return ATTENTION_IMPLEMENTATION
```

**Model Loading with Fallback**:
```python
model = Qwen3TTSModel.from_pretrained(
    model_name,
    attn_implementation=get_attention_implementation(),
    # ... other params
)
```

**Deliverable**: 
- `utils/attention_utils.py` with detection and fallback
- Dockerfile with pre-built wheel installation
- Documentation on fallback behavior
- Note: Build time reduced from 1-2 hours to under 5 minutes

**Validation**:
- Check which attention implementation is active at runtime
- Verify performance (Flash Attention ~2x faster than SDPA when available)
- Ensure inference works with both implementations
- Test fallback behavior when flash-attn import fails

#### Task 1.4: Core Dependencies Installation with Audio Support
**Objective**: Install all required Python packages including comprehensive audio support

**Requirements** (from pyproject.toml + audio + CUDA build tools):
```
# Build dependencies (install first)
packaging
wheel
setuptools
ninja

# Core TTS
transformers==4.57.3
accelerate==1.12.0
librosa
torchaudio
soundfile
einops
qwen-tts  # Official package

# Audio Processing (COMPLETE SUPPORT)
pydub  # MP3, OGG, M4A support
audioread  # Multiple format reading
resampy  # High-quality resampling
sox  # Audio effects
ffmpeg-python  # Video/audio extraction

# API
fastapi
uvicorn[standard]
pydantic
python-multipart  # File uploads

# Utilities
numpy
scipy
aiofiles  # Async file operations
tenacity  # Retry logic
langdetect  # Language detection

# Monitoring
prometheus-client  # Metrics
psutil  # System monitoring

# Development
pytest
pytest-asyncio
pytest-cov
httpx
black
isort
mypy
flake8
```

**Deliverable**: `requirements.txt` with pinned versions + `requirements-dev.txt` for development

**Validation**: All imports successful, no version conflicts

### Phase 2: Model Management (Priority: Critical)

#### Task 2.1: Model Download System with Retry Logic
**Objective**: Create robust automated model download and caching system

**CRITICAL**: Download FOUR models (not three):
1. **Qwen/Qwen3-TTS-Tokenizer-12Hz** - REQUIRED by all models (~1-2 GB)
2. **Qwen/Qwen3-TTS-12Hz-1.7B-Base** - Voice cloning (~4.5 GB)
3. **Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice** - Custom voices (~4.5 GB)
4. **Qwen/Qwen3-TTS-12Hz-1.7B-VoiceDesign** - Voice design (~4.5 GB)

**Total Download Size**: 16-20 GB

**Requirements**:
- Download all four models on first run
- Implement retry logic with exponential backoff (3 attempts)
- Support both Hugging Face and ModelScope sources
- Enable partial download resumption
- Show download progress
- Verify checksums after download
- Handle network interruptions gracefully
- Cache to persistent volume

**Implementation Details**:
```python
# models/download.py
from tenacity import retry, stop_after_attempt, wait_exponential
from huggingface_hub import snapshot_download
import logging

logger = logging.getLogger(__name__)

MODEL_REGISTRY = {
    "tokenizer": "Qwen/Qwen3-TTS-Tokenizer-12Hz",  # ESSENTIAL
    "base": "Qwen/Qwen3-TTS-12Hz-1.7B-Base",
    "custom-voice": "Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice",
    "voice-design": "Qwen/Qwen3-TTS-12Hz-1.7B-VoiceDesign"
}

@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=2, min=4, max=60),
    reraise=True
)
def download_model_with_retry(model_name: str, cache_dir: str):
    """Download model with automatic retry on failure."""
    logger.info(f"Downloading {model_name}...")
    
    try:
        # Try HuggingFace first
        path = snapshot_download(
            repo_id=model_name,
            cache_dir=cache_dir,
            resume_download=True,
            local_files_only=False
        )
        logger.info(f"Downloaded {model_name} to {path}")
        return path
    
    except Exception as e:
        logger.warning(f"HuggingFace download failed: {e}")
        # TODO: Fallback to ModelScope for China users
        raise

def download_all_models(cache_dir: str):
    """Download all required models including tokenizer."""
    for model_type, model_name in MODEL_REGISTRY.items():
        if not check_model_exists(cache_dir, model_type):
            download_model_with_retry(model_name, cache_dir)
        else:
            logger.info(f"Model {model_type} already cached")
```

**Deliverable**: `models/download.py` with robust download manager

**Validation**:
- All four models download correctly (test on clean cache)
- Retry works after simulated network failure
- Models cached in persistent volume
- Re-runs skip existing models (< 1 minute startup)
- Progress displayed during download

#### Task 2.2: Model Loading System with Memory Management
**Objective**: Implement efficient model loading with multi-GPU support and memory awareness

**CRITICAL MEMORY CALCULATIONS**:
- Each 1.7B model: ~4.54 GB in bfloat16
- Tokenizer: ~1-2 GB
- Runtime overhead: +30-50% (activations, KV cache)
- **Practical per-model requirement**: 6-7 GB
- **All models + tokenizer**: 22-25 GB minimum

**GPU Configuration Strategies**:
```
Single 24GB GPU:  Load 1 model at a time + tokenizer (with switching)
Single 32GB GPU:  Load 2 models + tokenizer comfortably
Dual 24GB GPUs:   Load 2 models distributed + tokenizer
Dual 32GB GPUs:   Load all 3 models + tokenizer
Quad 24GB GPUs:   Load all models with headroom for batching
```

**Requirements**:
- Load tokenizer ONCE (shared by all models)
- Load model with device_map for multi-GPU distribution
- Support torch.bfloat16 precision (REQUIRED for these models)
- Enable Flash Attention 2 if available
- Lazy loading (load on first request)
- LRU cache for model switching (keep last 2 models if memory permits)
- Automatic model unloading when switching if memory constrained
- Memory monitoring and alerts
- Graceful degradation to CPU offloading if needed

**Implementation Details**:
```python
# models/loader.py
import torch
from typing import Dict, Optional
from qwen_tts import Qwen3TTSModel
from collections import OrderedDict

from utils.logging import get_logger
from utils.gpu_utils import get_device_map, get_available_gpu_memory
from utils.errors import ModelNotFoundError, CUDAOutOfMemoryError

logger = get_logger(__name__)

class ModelManager:
    """Manages Qwen3-TTS model lifecycle with memory awareness."""
    
    MODEL_REGISTRY = {
        "tokenizer": "Qwen/Qwen3-TTS-Tokenizer-12Hz",  # ALWAYS LOADED
        "base": "Qwen/Qwen3-TTS-12Hz-1.7B-Base",
        "custom-voice": "Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice",
        "voice-design": "Qwen/Qwen3-TTS-12Hz-1.7B-VoiceDesign"
    }
    
    # Memory requirements per model (GB)
    MODEL_MEMORY_REQUIREMENTS = {
        "tokenizer": 2,
        "base": 7,
        "custom-voice": 7,
        "voice-design": 7
    }
    
    def __init__(
        self,
        cache_dir: str,
        device_map: str = "auto",
        dtype: str = "bfloat16",
        max_models_in_memory: int = 2
    ):
        """Initialize model manager with memory constraints.
        
        Args:
            cache_dir: Directory for model cache.
            device_map: GPU distribution strategy.
            dtype: Model data type (bfloat16 required).
            max_models_in_memory: Max TTS models to keep loaded (not including tokenizer).
        """
        self.cache_dir = cache_dir
        self.device_map = device_map
        self.dtype = self._get_dtype(dtype)
        self.max_models_in_memory = max_models_in_memory
        
        self._tokenizer = None  # Loaded once, kept always
        self._models: OrderedDict = OrderedDict()  # LRU cache
        
        # Calculate available memory
        self.available_memory_gb = self._get_total_available_memory()
        
        logger.info(
            "ModelManager initialized",
            extra={
                "cache_dir": cache_dir,
                "device_map": device_map,
                "dtype": dtype,
                "cuda_available": torch.cuda.is_available(),
                "gpu_count": torch.cuda.device_count(),
                "available_memory_gb": self.available_memory_gb,
                "max_models_in_memory": max_models_in_memory
            }
        )
    
    def _get_dtype(self, dtype_str: str) -> torch.dtype:
        """Convert string dtype to torch dtype."""
        dtype_map = {
            "bfloat16": torch.bfloat16,
            "float16": torch.float16,
            "float32": torch.float32
        }
        return dtype_map.get(dtype_str, torch.bfloat16)
    
    def _get_total_available_memory(self) -> float:
        """Get total available GPU memory in GB."""
        if not torch.cuda.is_available():
            return 0.0
        
        total_memory = sum(
            torch.cuda.get_device_properties(i).total_memory 
            for i in range(torch.cuda.device_count())
        ) / (1024 ** 3)  # Convert to GB
        
        return total_memory
    
    def get_tokenizer(self):
        """Get shared tokenizer (loaded once)."""
        if self._tokenizer is None:
            logger.info("Loading shared tokenizer...")
            self._tokenizer = self._load_tokenizer()
            logger.info("Tokenizer loaded successfully")
        return self._tokenizer
    
    def _load_tokenizer(self):
        """Load tokenizer from cache."""
        from qwen_tts import Qwen3TTSTokenizer
        
        tokenizer_name = self.MODEL_REGISTRY["tokenizer"]
        return Qwen3TTSTokenizer.from_pretrained(
            tokenizer_name,
            cache_dir=self.cache_dir
        )
    
    def get_model(self, model_type: str):
        """Get or load model by type with LRU caching.
        
        Args:
            model_type: Model type ('base', 'custom-voice', 'voice-design').
        
        Returns:
            Loaded model instance.
        
        Raises:
            ModelNotFoundError: If model type is invalid.
            CUDAOutOfMemoryError: If insufficient GPU memory.
        """
        if model_type not in ["base", "custom-voice", "voice-design"]:
            raise ModelNotFoundError(f"Unknown model type: {model_type}")
        
        # Return cached model if already loaded (move to end for LRU)
        if model_type in self._models:
            logger.debug(f"Returning cached model: {model_type}")
            self._models.move_to_end(model_type)
            return self._models[model_type]
        
        # Check if we need to unload a model first
        if len(self._models) >= self.max_models_in_memory:
            self._evict_least_recently_used()
        
        # Load model
        logger.info(f"Loading model: {model_type}")
        try:
            model = self._load_model(model_type)
            self._models[model_type] = model
            
            # Log memory usage
            self._log_memory_usage()
            
            logger.info(f"Model loaded successfully: {model_type}")
            return model
        
        except torch.cuda.OutOfMemoryError as e:
            logger.error(f"CUDA OOM loading {model_type}: {e}")
            # Try to free memory
            self._emergency_memory_cleanup()
            raise CUDAOutOfMemoryError(
                f"Insufficient GPU memory to load {model_type}. "
                f"Available: {self.available_memory_gb:.1f}GB, "
                f"Required: {self.MODEL_MEMORY_REQUIREMENTS[model_type]}GB"
            ) from e
    
    def _load_model(self, model_type: str):
        """Internal method to load model from cache."""
        from qwen_tts import Qwen3TTSModel
        from utils.attention_utils import get_attention_implementation
        
        model_name = self.MODEL_REGISTRY[model_type]
        
        device_map = self._get_device_map()
        
        model = Qwen3TTSModel.from_pretrained(
            model_name,
            cache_dir=self.cache_dir,
            torch_dtype=self.dtype,
            device_map=device_map,
            attn_implementation=get_attention_implementation(),
            low_cpu_mem_usage=True
        )
        
        return model
    
    def _get_device_map(self) -> dict:
        """Get device map for multi-GPU distribution."""
        if self.device_map == "auto":
            return "auto"  # Let accelerate handle it
        
        # Custom device mapping logic here if needed
        return self.device_map
    
    def _evict_least_recently_used(self):
        """Unload least recently used model to free memory."""
        if not self._models:
            return
        
        # Get first item (least recently used)
        lru_model_type = next(iter(self._models))
        logger.info(f"Evicting LRU model: {lru_model_type}")
        self.unload_model(lru_model_type)
    
    def unload_model(self, model_type: str) -> None:
        """Unload model to free GPU memory."""
        if model_type in self._models:
            del self._models[model_type]
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
            logger.info(f"Model unloaded: {model_type}")
            self._log_memory_usage()
    
    def _emergency_memory_cleanup(self):
        """Emergency cleanup when OOM occurs."""
        logger.warning("Performing emergency memory cleanup")
        # Unload all models except tokenizer
        for model_type in list(self._models.keys()):
            self.unload_model(model_type)
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
    
    def _log_memory_usage(self):
        """Log current GPU memory usage."""
        if not torch.cuda.is_available():
            return
        
        for i in range(torch.cuda.device_count()):
            allocated = torch.cuda.memory_allocated(i) / (1024 ** 3)
            reserved = torch.cuda.memory_reserved(i) / (1024 ** 3)
            total = torch.cuda.get_device_properties(i).total_memory / (1024 ** 3)
            
            logger.info(
                f"GPU {i} memory",
                extra={
                    "gpu_id": i,
                    "allocated_gb": f"{allocated:.2f}",
                    "reserved_gb": f"{reserved:.2f}",
                    "total_gb": f"{total:.2f}",
                    "utilization_pct": f"{(allocated/total)*100:.1f}"
                }
            )
    
    def is_loaded(self, model_type: str) -> bool:
        """Check if model is currently loaded."""
        return model_type in self._models
    
    def get_loaded_models(self) -> list:
        """Get list of currently loaded models."""
        return list(self._models.keys())
```

**Deliverable**: `models/loader.py` with smart memory-aware model management

**Validation**:
- Tokenizer loads once and stays loaded
- Models load across all GPUs appropriately
- Memory usage tracked and logged
- Flash Attention 2 enabled (or SDPA fallback)
- LRU eviction works when hitting memory limits
- Switching between models works smoothly
- Emergency cleanup prevents crashes

#### Task 2.3: Multi-GPU Configuration and Optimization
**Objective**: Optimize multi-GPU usage for inference

**Requirements**:
- Detect all available GPUs
- Distribute model layers across GPUs using accelerate
- Balance memory usage across GPUs
- Support tensor parallelism
- Environment variable configuration
- Prevent GPU memory fragmentation

**Configuration**:
```python
# utils/gpu_utils.py
import torch
import os

def setup_multi_gpu():
    """Configure multi-GPU environment."""
    if not torch.cuda.is_available():
        return None
    
    gpu_count = torch.cuda.device_count()
    
    # Set CUDA devices if not already set
    if "CUDA_VISIBLE_DEVICES" not in os.environ:
        os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(str(i) for i in range(gpu_count))
    
    # Configure distributed backend if multiple GPUs
    if gpu_count > 1:
        os.environ.setdefault("TORCH_DISTRIBUTED_BACKEND", "nccl")
    
    return gpu_count

def get_device_map(gpu_count: int = None):
    """Get optimal device map for model distribution."""
    if gpu_count is None:
        gpu_count = torch.cuda.device_count()
    
    if gpu_count == 0:
        return "cpu"
    elif gpu_count == 1:
        return "cuda:0"
    else:
        return "auto"  # Let accelerate distribute automatically

def get_available_gpu_memory(device_id: int = 0) -> float:
    """Get available GPU memory in GB."""
    if not torch.cuda.is_available():
        return 0.0
    
    torch.cuda.set_device(device_id)
    free, total = torch.cuda.mem_get_info(device_id)
    return free / (1024 ** 3)
```

**Deliverable**: `utils/gpu_utils.py` with GPU management utilities

**Validation**:
- All GPUs utilized during inference
- Balanced memory distribution across GPUs
- No GPU idle during generation
- Memory monitoring per GPU

#### Task 2.4: Model Warm-up on Startup
**Objective**: Pre-warm models to avoid slow first request

**Requirements**:
- Run dummy inference on each model during startup
- Initialize CUDA kernels and JIT compilation
- Load commonly used speakers for custom-voice
- Benchmark and log warm-up performance
- Make warm-up optional via environment variable

**Implementation**:
```python
# api/main.py - startup event
@app.on_event("startup")
async def startup_event():
    """Initialize resources on startup."""
    logger.info("Starting Qwen3-TTS API server")
    
    # Initialize model manager
    global model_manager
    model_manager = ModelManager(
        cache_dir=settings.model_cache_dir,
        device_map=settings.device_map,
        max_models_in_memory=settings.max_models_in_memory
    )
    
    # Load tokenizer (always needed)
    model_manager.get_tokenizer()
    
    # Optional: Warm up models
    if settings.warmup_on_startup:
        await warmup_models()

async def warmup_models():
    """Warm up models with dummy inference to initialize CUDA."""
    logger.info("Warming up models (this may take 1-2 minutes)...")
    
    dummy_text = "Hello, this is a test."
    dummy_audio_duration = 3  # seconds
    
    # Create dummy reference audio (silence)
    dummy_ref_audio = torch.zeros(24000 * dummy_audio_duration)
    
    models_to_warmup = []
    if settings.enable_base_model:
        models_to_warmup.append("base")
    if settings.enable_custom_voice:
        models_to_warmup.append("custom-voice")
    if settings.enable_voice_design:
        models_to_warmup.append("voice-design")
    
    for model_type in models_to_warmup:
        try:
            start_time = time.time()
            logger.info(f"Warming up {model_type}...")
            
            model = model_manager.get_model(model_type)
            
            # Run dummy inference
            with torch.no_grad():
                if model_type == "base":
                    _ = model.generate(
                        text=dummy_text,
                        ref_audio=dummy_ref_audio,
                        ref_text=dummy_text
                    )
                elif model_type == "custom-voice":
                    _ = model.generate(
                        text=dummy_text,
                        speaker="default"
                    )
                elif model_type == "voice-design":
                    _ = model.generate(
                        text=dummy_text,
                        voice_description="A friendly voice"
                    )
            
            elapsed = time.time() - start_time
            logger.info(f"{model_type} warmed up in {elapsed:.2f}s")
            
        except Exception as e:
            logger.warning(f"Failed to warm up {model_type}: {e}")
    
    logger.info("Model warm-up completed")
```

**Deliverable**: Startup warm-up implementation

**Validation**:
- First request after startup is fast (< 2s)
- Warm-up completes in 1-2 minutes
- All CUDA kernels initialized
- Can be disabled via config

### Phase 3: API Development (Priority: High)

#### Task 3.1: FastAPI Application Structure
**Objective**: Create RESTful API for TTS services

**Endpoints**:

```
POST /v1/tts/voice-clone
- Voice cloning with reference audio
- Body: {text, ref_audio, ref_text, language, x_vector_only_mode}

POST /v1/tts/custom-voice
- Custom voice with predefined speakers
- Body: {text, language, speaker, instruct}

POST /v1/tts/voice-design
- Voice design from description
- Body: {text, voice_description}

POST /v1/tts/voice-clone/stream
- Streaming voice clone (dual-track generation)
- Returns chunked audio for low-latency playback

GET /v1/speakers
- List available speakers for custom-voice

GET /v1/languages
- List supported languages (10 languages)

GET /v1/models
- List loaded models and memory usage

GET /v1/health
- Basic health check endpoint

GET /v1/health/detailed
- Detailed health with GPU stats and model status

GET /v1/metrics
- Prometheus metrics
```

**Deliverable**: `api/main.py` with complete API

**Validation**:
- All endpoints functional
- Proper error handling
- OpenAPI documentation generated
- Request validation with Pydantic

#### Task 3.2: Request Processing Pipeline with Concurrency Control
**Objective**: Build efficient request processing with concurrency limits

**CRITICAL REQUIREMENTS**:
- Implement request queuing with semaphore
- Limit concurrent inference operations (4-8 max depending on GPU)
- Prevent GPU OOM from concurrent requests
- Graceful request rejection when overloaded
- Request timeout handling (5 minute default)
- Client disconnect detection and cleanup

**Requirements**:
- Async request handling
- Input validation and sanitization
- Audio format conversion (support WAV, MP3, OGG, FLAC, M4A, URLs, base64)
- Batch processing support
- Request queuing for load management
- Response streaming for real-time generation
- Concurrency semaphore to prevent OOM
- Timeout handling for long-running requests
- Cleanup on client disconnect

**Implementation Details**:
```python
# api/pipeline.py
import asyncio
from asyncio import Semaphore, TimeoutError
from typing import Union, Optional
import torch

from utils.logging import get_logger
from utils.errors import ValidationError, TimeoutError as RequestTimeoutError
from config.settings import settings

logger = get_logger(__name__)

# Global semaphore for concurrency control
MAX_CONCURRENT_REQUESTS = settings.max_concurrent_requests  # Default: 4
inference_semaphore = Semaphore(MAX_CONCURRENT_REQUESTS)

class RequestProcessor:
    """Process TTS requests with concurrency control."""
    
    def __init__(self, model_manager):
        self.model_manager = model_manager
        self.active_requests = 0
    
    async def process_voice_clone(
        self,
        text: str,
        ref_audio: Union[str, bytes],
        ref_text: str,
        language: Optional[str] = None,
        timeout: int = 300
    ):
        """Process voice clone request with timeout and concurrency control."""
        
        # Validate inputs
        self._validate_text(text)
        
        # Acquire semaphore (wait if too many concurrent requests)
        async with inference_semaphore:
            self.active_requests += 1
            logger.info(
                f
                try:
            # Apply timeout
            result = await asyncio.wait_for(
                self._generate_voice_clone(text, ref_audio, ref_text, language),
                timeout=timeout
            )
            return result
        
        except TimeoutError:
            logger.error(f"Request timeout after {timeout}s")
            raise RequestTimeoutError(f"Request exceeded timeout of {timeout}s")
        
        finally:
            self.active_requests -= 1

async def _generate_voice_clone(self, text, ref_audio, ref_text, language):
    """Internal generation method."""
    # Preprocess audio
    processed_audio = await self._load_and_process_audio(ref_audio)
    
    # Get model
    model = self.model_manager.get_model("base")
    tokenizer = self.model_manager.get_tokenizer()
    
    # Generate audio
    with torch.no_grad():
        audio_output = model.generate(
            text=text,
            ref_audio=processed_audio,
            ref_text=ref_text,
            language=language,
            tokenizer=tokenizer
        )
    
    return audio_output

async def _load_and_process_audio(self, ref_audio: Union[str, bytes]):
    """Load and process reference audio from various sources."""
    from utils.audio_utils import (
        load_audio_from_path,
        load_audio_from_url,
        load_audio_from_bytes,
        resample_audio,
        convert_to_mono
    )
    
    # Determine source type and load
    if isinstance(ref_audio, bytes):
        audio_data, sr = load_audio_from_bytes(ref_audio)
    elif ref_audio.startswith(('http://', 'https://')):
        audio_data, sr = await load_audio_from_url(ref_audio)
    else:
        audio_data, sr = load_audio_from_path(ref_audio)
    
    # Validate duration (3-30 seconds recommended)
    duration = len(audio_data) / sr
    if duration < 3:
        logger.warning(f"Reference audio is short ({duration:.1f}s), quality may be reduced")
    elif duration > 30:
        logger.warning(f"Reference audio is long ({duration:.1f}s), truncating to 30s")
        audio_data = audio_data[:30 * sr]
    
    # Resample to 24kHz if needed
    if sr != 24000:
        audio_data = resample_audio(audio_data, sr, 24000)
    
    # Convert to mono if stereo
    audio_data = convert_to_mono(audio_data)
    
    return torch.from_numpy(audio_data)

def _validate_text(self, text: str):
    """Validate input text."""
    if not text or not text.strip():
        raise ValidationError("Text cannot be empty")
    
    if len(text) > settings.max_text_length:
        raise ValidationError(
            f"Text exceeds maximum length of {settings.max_text_length} characters"
        )
**Deliverable**: `api/pipeline.py` with processing pipeline and concurrency control

**Validation**:
- Handles concurrent requests without OOM
- Validates all inputs
- Processes various audio formats
- Respects timeout limits
- Cleans up on errors

#### Task 3.3: Comprehensive Error Handling and Logging
**Objective**: Comprehensive error handling and monitoring

**Requirements**:
- Structured logging (JSON format)
- Error codes and messages
- Request tracing with correlation IDs
- Performance metrics tracking
- GPU memory monitoring
- Graceful degradation
- Detailed error responses

**Logging Levels**:
- DEBUG: Detailed execution flow, model parameters
- INFO: Request/response logs, successful operations
- WARNING: Resource warnings, degraded performance
- ERROR: Failures with context
- CRITICAL: System failures requiring intervention

**Implementation**:
```python
# utils/logging.py
import logging
import json
import sys
from datetime import datetime

class JSONFormatter(logging.Formatter):
    """Format logs as JSON for structured logging."""
    
    def format(self, record):
        log_data = {
            "timestamp": datetime.utcnow().isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
            "module": record.module,
            "function": record.funcName,
            "line": record.lineno
        }
        
        # Add extra fields if present
        if hasattr(record, "extra"):
            log_data.update(record.extra)
        
        # Add exception info if present
        if record.exc_info:
            log_data["exception"] = self.formatException(record.exc_info)
        
        return json.dumps(log_data)

def setup_logging(log_level: str = "INFO"):
    """Configure application logging."""
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(JSONFormatter())
    
    root_logger = logging.getLogger()
    root_logger.setLevel(getattr(logging, log_level.upper()))
    root_logger.addHandler(handler)

def get_logger(name: str):
    """Get logger for module."""
    return logging.getLogger(name)
```
```python
# utils/errors.py
class Qwen3TTSError(Exception):
    """Base exception for all Qwen3-TTS errors."""
    pass

class ModelNotFoundError(Qwen3TTSError):
    """Model not found or invalid model type."""
    pass

class CUDAOutOfMemoryError(Qwen3TTSError):
    """GPU memory exhausted."""
    pass

class AudioProcessingError(Qwen3TTSError):
    """Audio processing failed."""
    pass

class ValidationError(Qwen3TTSError):
    """Input validation failed."""
    pass

class TimeoutError(Qwen3TTSError):
    """Request timeout exceeded."""
    pass

class ConcurrencyLimitError(Qwen3TTSError):
    """Too many concurrent requests."""
    pass
```

**Deliverable**: `utils/logging.py` and `utils/errors.py`

**Validation**:
- All errors logged properly with context
- Stack traces captured for debugging
- Structured logs parseable by log aggregators
- Metrics exported correctly

#### Task 3.4: Detailed Health Checks
**Objective**: Comprehensive health monitoring

**Requirements**:
- Basic health endpoint (fast, < 100ms)
- Detailed health with full diagnostics
- GPU status and memory per device
- Model loading status
- Disk space monitoring
- Recent error rate
- Average response time

**Implementation**:
```python
# api/routes.py - health endpoints
@router.get("/health", tags=["System"])
async def health_check():
    """Fast health check for load balancers."""
    return {
        "status": "healthy",
        "timestamp": datetime.utcnow().isoformat()
    }

@router.get("/health/detailed", tags=["System"])
async def detailed_health_check(
    model_manager = Depends(get_model_manager)
):
    """Detailed health check with full diagnostics."""
    import shutil
    
    # GPU information
    gpu_info = []
    if torch.cuda.is_available():
        for i in range(torch.cuda.device_count()):
            props = torch.cuda.get_device_properties(i)
            allocated = torch.cuda.memory_allocated(i) / (1024 ** 3)
            reserved = torch.cuda.memory_reserved(i) / (1024 ** 3)
            total = props.total_memory / (1024 ** 3)
            
            gpu_info.append({
                "gpu_id": i,
                "name": props.name,
                "allocated_gb": round(allocated, 2),
                "reserved_gb": round(reserved, 2),
                "total_gb": round(total, 2),
                "utilization_pct": round((allocated / total) * 100, 1)
            })
    
    # Model status
    model_status = {
        "tokenizer": model_manager._tokenizer is not None,
        "base": model_manager.is_loaded("base"),
        "custom_voice": model_manager.is_loaded("custom-voice"),
        "voice_design": model_manager.is_loaded("voice-design"),
        "loaded_models": model_manager.get_loaded_models()
    }
    
    # Disk space
    disk_usage = shutil.disk_usage(settings.model_cache_dir)
    disk_info = {
        "cache_dir": settings.model_cache_dir,
        "total_gb": round(disk_usage.total / (1024 ** 3), 2),
        "used_gb": round(disk_usage.used / (1024 ** 3), 2),
        "free_gb": round(disk_usage.free / (1024 ** 3), 2),
        "percent_used": round((disk_usage.used / disk_usage.total) * 100, 1)
    }
    
    return {
        "status": "healthy",
        "timestamp": datetime.utcnow().isoformat(),
        "gpu": {
            "available": torch.cuda.is_available(),
            "count": torch.cuda.device_count(),
            "devices": gpu_info
        },
        "models": model_status,
        "disk": disk_info,
        "config": {
            "max_concurrent_requests": settings.max_concurrent_requests,
            "max_text_length": settings.max_text_length,
            "flash_attention": FLASH_ATTENTION_AVAILABLE
        }
    }
```

**Deliverable**: Comprehensive health check endpoints

**Validation**:
- Basic health responds in < 100ms
- Detailed health provides full system state
- All metrics accurate
- Useful for monitoring and debugging

### Phase 4: Inference Engine (Priority: Critical)

#### Task 4.1: Voice Clone Implementation
**Objective**: Implement voice cloning inference

**Requirements**:
- Accept audio in multiple formats (wav, mp3, ogg, flac, m4a, URL, base64)
- Process reference audio and transcript
- Generate cloned voice
- Support x_vector_only_mode
- Handle long-form text (chunking if needed)
- Support 10 languages with auto-detection

**Implementation Details**:
```python
# inference/voice_clone.py
class VoiceCloneEngine:
    """Voice cloning inference engine."""
    
    def __init__(self, model, tokenizer):
        self.model = model
        self.tokenizer = tokenizer
        self.logger = get_logger(__name__)
    
    async def generate(
        self,
        text: str,
        ref_audio: torch.Tensor,
        ref_text: str,
        language: Optional[str] = None,
        x_vector_only_mode: bool = False,
        **kwargs
    ):
        """Generate cloned voice audio.
        
        Args:
            text: Text to synthesize.
            ref_audio: Reference audio tensor (24kHz, mono).
            ref_text: Transcript of reference audio.
            language: Target language (auto-detected if None).
            x_vector_only_mode: Use only x-vector embeddings.
        
        Returns:
            Tuple of (audio_tensor, sample_rate)
        """
        # Detect language if not specified
        if language is None:
            language = self._detect_language(text)
            self.logger.info(f"Auto-detected language: {language}")
        
        # Chunk long text if needed
        if len(text) > 500:  # Adjust based on model limits
            return await self._generate_long_form(
                text, ref_audio, ref_text, language, x_vector_only_mode, **kwargs
            )
        
        # Generate audio
        with torch.no_grad():
            audio = self.model.generate_voice_clone(
                text=text,
                ref_audio=ref_audio,
                ref_text=ref_text,
                language=language,
                x_vector_only_mode=x_vector_only_mode,
                tokenizer=self.tokenizer,
                **kwargs
            )
        
        return audio, 24000  # Return audio and sample rate
    
    def _detect_language(self, text: str) -> str:
        """Auto-detect language from text."""
        from utils.language_utils import detect_language
        return detect_language(text)
    
    async def _generate_long_form(self, text, ref_audio, ref_text, language, x_vector_only_mode, **kwargs):
        """Generate long-form text by chunking."""
        chunks = self._chunk_text(text, max_length=500)
        audio_chunks = []
        
        for i, chunk in enumerate(chunks):
            self.logger.info(f"Generating chunk {i+1}/{len(chunks)}")
            audio, sr = await self.generate(
                chunk, ref_audio, ref_text, language, x_vector_only_mode, **kwargs
            )
            audio_chunks.append(audio)
        
        # Concatenate chunks
        full_audio = torch.cat(audio_chunks, dim=0)
        return full_audio, 24000
    
    def _chunk_text(self, text: str, max_length: int = 500) -> list:
        """Chunk text at sentence boundaries."""
        # Split on sentence boundaries
        import re
        sentences = re.split(r'([.!?]+)', text)
        
        chunks = []
        current_chunk = ""
        
        for sentence in sentences:
            if len(current_chunk) + len(sentence) <= max_length:
                current_chunk += sentence
            else:
                if current_chunk:
                    chunks.append(current_chunk.strip())
                current_chunk = sentence
        
        if current_chunk:
            chunks.append(current_chunk.strip())
        
        return chunks
```

**Deliverable**: `inference/voice_clone.py`

**Validation**:
- Generates high-quality cloned voices
- Handles various audio inputs
- Works with long texts via chunking
- Maintains voice consistency
- Auto-detects language correctly

#### Task 4.2: Custom Voice Implementation
**Objective**: Implement custom voice inference with predefined speakers

**Requirements**:
- Load available speakers from model
- Support all 10 languages
- Apply instruction-based control (emotion, tone, prosody)
- Handle speaker validation

**Implementation**: Similar pattern to voice_clone.py

**Deliverable**: `inference/custom_voice.py`

#### Task 4.3: Voice Design Implementation
**Objective**: Implement voice design inference from natural language descriptions

**Requirements**:
- Parse natural language voice descriptions
- Generate novel voices from descriptions
- Maintain consistency across generations
- Validate description format

**Deliverable**: `inference/voice_design.py`

#### Task 4.4: Batch Processing with Dynamic Sizing
**Objective**: Implement efficient batch inference with dynamic batch sizing

**Requirements**:
- Process multiple texts in single inference
- Dynamically calculate optimal batch size based on:
  - Available GPU memory
  - Text lengths
  - Model type
- Optimize GPU utilization
- Maintain individual request tracking
- Return results in order

**Implementation Details**:
```python
# inference/batch_processor.py
class BatchProcessor:
    """Dynamic batch processing for TTS inference."""
    
    def __init__(self, model_manager):
        self.model_manager = model_manager
        self.logger = get_logger(__name__)
    
    def calculate_optimal_batch_size(
        self,
        text_lengths: list,
        model_type: str
    ) -> int:
        """Calculate optimal batch size based on inputs and resources."""
        # Get available GPU memory
        available_memory_gb = get_available_gpu_memory()
        
        # Estimate memory per sample (rough heuristic)
        avg_text_length = sum(text_lengths) / len(text_lengths)
        memory_per_sample_gb = (avg_text_length / 1000) * 0.5  # Rough estimate
        
        # Calculate max batch size
        max_batch_size = int(available_memory_gb / memory_per_sample_gb)
        
        # Apply limits
        max_batch_size = max(1, min(max_batch_size, 8))  # Between 1-8
        
        self.logger.info(
            f"Calculated batch size: {max_batch_size}",
            extra={
                "available_memory_gb": available_memory_gb,
                "avg_text_length": avg_text_length,
                "num_samples": len(text_lengths)
            }
        )
        
        return max_batch_size
```

**Deliverable**: `inference/batch_processor.py`

**Validation**:
- Improves throughput vs sequential processing
- Correct result mapping
- No memory leaks
- Dynamic sizing prevents OOM

### Phase 5: Audio Processing (Priority: Medium)

#### Task 5.1: Comprehensive Audio Input Handling
**Objective**: Support all major audio input formats

**Requirements**:
- Local file paths (WAV, MP3, OGG, FLAC, M4A)
- URLs (HTTP/HTTPS with async download)
- Base64 encoded audio
- Raw audio arrays with sample rate
- Format conversion to WAV 24kHz mono
- Audio validation (duration, size, format)

**Implementation Details**:
```python
# utils/audio_utils.py
import soundfile as sf
import numpy as np
from pydub import AudioSegment
import resampy
import aiohttp
import base64
import io
from typing import Tuple

async def load_audio_from_url(url: str) -> Tuple[np.ndarray, int]:
    """Load audio from URL asynchronously."""
    async with aiohttp.ClientSession() as session:
        async with session.get(url, timeout=30) as response:
            if response.status != 200:
                raise AudioProcessingError(f"Failed to download audio: {response.status}")
            
            audio_bytes = await response.read()
            return load_audio_from_bytes(audio_bytes)

def load_audio_from_bytes(audio_bytes: bytes) -> Tuple[np.ndarray, int]:
    """Load audio from bytes."""
    try:
        # Try soundfile first (WAV, FLAC, OGG)
        audio_data, sr = sf.read(io.BytesIO(audio_bytes))
        return audio_data, sr
    except:
        # Fallback to pydub for MP3, M4A, etc.
        audio_segment = AudioSegment.from_file(io.BytesIO(audio_bytes))
        audio_data = np.array(audio_segment.get_array_of_samples(), dtype=np.float32)
        audio_data = audio_data / (2**15)  # Normalize to [-1, 1]
        sr = audio_segment.frame_rate
        
        # Handle stereo
        if audio_segment.channels == 2:
            audio_data = audio_data.reshape((-1, 2))
        
        return audio_data, sr

def load_audio_from_path(file_path: str) -> Tuple[np.ndarray, int]:
    """Load audio from file path."""
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Audio file not found: {file_path}")
    
    # Get file extension
    ext = os.path.splitext(file_path)[1].lower()
    
    if ext in ['.wav', '.flac', '.ogg']:
        audio_data, sr = sf.read(file_path)
    else:
        # Use pydub for other formats
        audio_segment = AudioSegment.from_file(file_path)
        audio_data = np.array(audio_segment.get_array_of_samples(), dtype=np.float32)
        audio_data = audio_data / (2**15)
        sr = audio_segment.frame_rate
        
        if audio_segment.channels == 2:
            audio_data = audio_data.reshape((-1, 2))
    
    return audio_data, sr

def load_audio_from_base64(base64_str: str) -> Tuple[np.ndarray, int]:
    """Load audio from base64 encoded string."""
    audio_bytes = base64.b64decode(base64_str)
    return load_audio_from_bytes(audio_bytes)

def resample_audio(audio: np.ndarray, orig_sr: int, target_sr: int) -> np.ndarray:
    """Resample audio to target sample rate."""
    if orig_sr == target_sr:
        return audio
    
    # Use resampy for high-quality resampling
    return resampy.resample(audio, orig_sr, target_sr)

def convert_to_mono(audio: np.ndarray) -> np.ndarray:
    """Convert stereo audio to mono."""
    if audio.ndim == 1:
        return audio
    elif audio.ndim == 2:
        return np.mean(audio, axis=1)
    else:
        raise ValueError(f"Unexpected audio shape: {audio.shape}")

def normalize_audio(audio: np.ndarray, target_level: float = -20.0) -> np.ndarray:
    """Normalize audio to target RMS level in dB."""
    rms = np.sqrt(np.mean(audio**2))
    if rms == 0:
        return audio
    
    target_rms = 10 ** (target_level / 20)
    return audio * (target_rms / rms)

def validate_audio(
    audio: np.ndarray,
    sr: int,
    min_duration: float = 3.0,
    max_duration: float = 30.0,
    max_size_mb: float = 10.0
) -> None:
    """Validate audio meets requirements."""
    duration = len(audio) / sr
    
    if duration < min_duration:
        raise ValidationError(
            f"Audio too short: {duration:.1f}s (minimum: {min_duration}s)"
        )
    
    if duration > max_duration:
        logger.warning(f"Audio too long: {duration:.1f}s, will truncate to {max_duration}s")
    
    # Check size
    size_mb = audio.nbytes / (1024 ** 2)
    if size_mb > max_size_mb:
        raise ValidationError(
            f"Audio file too large: {size_mb:.1f}MB (maximum: {max_size_mb}MB)"
        )
```

**Deliverable**: `utils/audio_utils.py` with comprehensive audio handling

**Validation**:
- All input types work correctly
- Correct sample rate conversion
- Maintains audio quality
- Proper error handling for invalid files

#### Task 5.2: Audio Output Processing
**Objective**: Provide audio in requested formats

**Requirements**:
- WAV output (primary, lossless)
- MP3 output (compressed, smaller size)
- Base64 encoding option for API responses
- Streaming output support for real-time playback
- Sample rate conversion if requested

**Implementation Details**:
```python
# utils/audio_output.py
import soundfile as sf
from pydub import AudioSegment
import base64
import io

def encode_audio_wav(audio: np.ndarray, sr: int) -> bytes:
    """Encode audio as WAV bytes."""
    buffer = io.BytesIO()
    sf.write(buffer, audio, sr, format='WAV')
    buffer.seek(0)
    return buffer.read()

def encode_audio_mp3(audio: np.ndarray, sr: int, bitrate: str = "192k") -> bytes:
    """Encode audio as MP3 bytes."""
    # Convert to int16
    audio_int16 = (audio * 32767).astype(np.int16)
    
    # Create AudioSegment
    audio_segment = AudioSegment(
        audio_int16.tobytes(),
        frame_rate=sr,
        sample_width=2,
        channels=1
    )
    
    # Export as MP3
    buffer = io.BytesIO()
    audio_segment.export(buffer, format="mp3", bitrate=bitrate)
    buffer.seek(0)
    return buffer.read()

def encode_audio_base64(audio_bytes: bytes) -> str:
    """Encode audio bytes as base64 string."""
    return base64.b64encode(audio_bytes).decode('utf-8')

async def stream_audio_chunks(audio: np.ndarray, sr: int, chunk_duration: float = 0.5):
    """Stream audio in chunks for real-time playback."""
    chunk_samples = int(sr * chunk_duration)
    
    for i in range(0, len(audio), chunk_samples):
        chunk = audio[i:i + chunk_samples]
        chunk_bytes = encode_audio_wav(chunk, sr)
        yield chunk_bytes
```

**Deliverable**: `utils/audio_output.py`

**Validation**:
- All output formats functional
- Proper MIME types set
- Streaming works correctly
- No quality degradation

#### Task 5.3: Language Detection and Mixed-Language Handling
**Objective**: Auto-detect language and handle mixed-language text

**Requirements**:
- Support all 10 languages: zh, en, ja, ko, de, fr, ru, pt, es, it
- Auto-detect primary language
- Handle mixed-language text (segment by language)
- Per-language text normalization
- Fallback for unsupported languages

**Implementation**:
```python
# utils/language_utils.py
from langdetect import detect, detect_langs
from typing import Optional, List, Tuple

SUPPORTED_LANGUAGES = ["zh", "en", "ja", "ko", "de", "fr", "ru", "pt", "es", "it"]

def detect_language(text: str, fallback: str = "en") -> str:
    """Auto-detect language with fallback."""
    try:
        lang = detect(text)
        
        # Map language codes
        lang_map = {
            "zh-cn": "zh",
            "zh-tw": "zh"
        }
        lang = lang_map.get(lang, lang)
        
        if lang in SUPPORTED_LANGUAGES:
            return lang
        else:
            logger.warning(f"Detected unsupported language: {lang}, using fallback: {fallback}")
            return fallback
    
    except Exception as e:
        logger.warning(f"Language detection failed: {e}, using fallback: {fallback}")
        return fallback

def detect_mixed_languages(text: str, threshold: float = 0.3) -> List[Tuple[str, float]]:
    """Detect multiple languages in text."""
    try:
        langs = detect_langs(text)
        return [(lang.lang, lang.prob) for lang in langs if lang.prob > threshold]
    except:
        return [("en", 1.0)]

def segment_by_language(text: str) -> List[Tuple[str, str]]:
    """Segment text by language (simplified version)."""
    # For MVP, just return single segment with detected language
    lang = detect_language(text)
    return [(text, lang)]
```

**Deliverable**: `utils/language_utils.py`

**Validation**:
- Accurate language detection (>90% accuracy)
- Supports all 10 languages
- Fallback works correctly

### Phase 6: Configuration Management (Priority: Medium)

#### Task 6.1: Environment Configuration
**Objective**: Flexible configuration via environment variables

**Configuration Variables**:
```bash
# Model Configuration
MODEL_CACHE_DIR=/models
ENABLE_BASE_MODEL=true
ENABLE_CUSTOM_VOICE=true
ENABLE_VOICE_DESIGN=true
DEFAULT_DTYPE=bfloat16
USE_FLASH_ATTENTION=true
MAX_MODELS_IN_MEMORY=2

# GPU Configuration
CUDA_VISIBLE_DEVICES=0,1,2,3
GPU_MEMORY_FRACTION=0.9
DEVICE_MAP=auto

# API Configuration
API_HOST=0.0.0.0
API_PORT=8000
MAX_CONCURRENT_REQUESTS=4
REQUEST_TIMEOUT=300
MAX_TEXT_LENGTH=5000
MAX_REF_AUDIO_DURATION=30
MAX_REF_AUDIO_SIZE_MB=10

# Inference Configuration
DEFAULT_LANGUAGE=Auto
DEFAULT_SAMPLE_RATE=24000
ENABLE_STREAMING=true
WARMUP_ON_STARTUP=true

# Logging
LOG_LEVEL=INFO
LOG_FORMAT=json

# Download Configuration
ENABLE_MODEL_DOWNLOAD=true
USE_MODELSCOPE_MIRROR=false
```

**Deliverable**: `config/settings.py` with pydantic BaseSettings

**Validation**:
- All configs loaded correctly
- Defaults work
- Overrides functional
- Validation prevents invalid configs

#### Task 6.2: Docker Compose Setup
**Objective**: Easy deployment with compose.yaml

**Requirements**:
- Volume mounts for models and logs
- GPU passthrough configuration
- Port mapping
- Environment file support
- Health checks
- Restart policy
- Resource limits

**Deliverable**: `compose.yaml`

**Example**:
```yaml
version: '3.8'

services:
  qwen3-tts:
    build:
      context: .
      dockerfile: Dockerfile
    
    image: qwen3-tts:latest
    container_name: qwen3-tts-server
    
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              count: all
              capabilities: [gpu]
        limits:
          memory: 32G
    
    env_file:
      - .env
    
    environment:
      - MODEL_CACHE_DIR=/models
      - LOG_LEVEL=INFO
      - WARMUP_ON_STARTUP=true
    
    volumes:
      - ./models:/models
      - ./logs:/logs
    
    ports:
      - "8000:8000"
    
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:8000/v1/health"]
      interval: 30s
      timeout: 10s
      retries: 3
      start_period: 120s
    
    restart: unless-stopped
    
    logging:
      driver: "json-file"
      options:
        max-size: "10m"
        max-file: "3"
```

**Validation**:
- Container starts successfully
- GPUs accessible
- Volumes mounted correctly
- Health checks pass

### Phase 7: Testing and Validation (Priority: High)

#### Task 7.1: Unit Tests
**Objective**: Test individual components

**Test Coverage**:
- Model loading and caching
- Audio processing (all formats)
- Language detection
- Input validation
- Error handling
- GPU utilities
- Batch processing

**Deliverable**: `tests/unit/` directory with pytest tests

**Target**: >80% code coverage

#### Task 7.2: Integration Tests
**Objective**: Test end-to-end workflows

**Test Scenarios**:
- Complete voice clone workflow
- Custom voice generation
- Voice design generation
- Multi-GPU distribution
- Concurrent requests (4-8 simultaneous)
- Error recovery
- Model switching
- Timeout handling

**Deliverable**: `tests/integration/` directory

**Validation**: All workflows pass

#### Task 7.3: Performance Tests
**Objective**: Validate performance targets

**Metrics**:
- Inference latency (target: < 2s for 10s audio)
- First request latency after startup (target: < 5s with warmup)
- Throughput (requests/second)
- GPU utilization (target: > 80%)
- Memory usage stability
- Concurrent request handling

**Deliverable**: `tests/performance/` directory with benchmarks

**Validation**: Meet performance targets

### Phase 8: Documentation (Priority: Medium)
Medium)

#### Task 8.1: API Documentation
**Objective**: Complete API reference

**Contents**:
- Endpoint descriptions
- Request/response schemas
- Example requests with curl and Python
- Error codes reference
- Rate limits
- Supported languages and formats

**Deliverable**: `docs/API.md` and OpenAPI spec

#### Task 8.2: Deployment Guide
**Objective**: Step-by-step deployment instructions

**Contents**:
- Prerequisites (CUDA 13.0+, Docker, GPU drivers)
- Docker installation
- GPU driver setup for 5000 series
- Container build and run (expect 1-2 hours first build)
- Configuration options
- Memory requirements per GPU configuration
- Troubleshooting guide
- Performance tuning

**Deliverable**: `docs/DEPLOYMENT.md`

#### Task 8.3: Usage Examples
**Objective**: Code examples for all features

**Examples**:
- Python client
- JavaScript/Node.js client
- curl examples
- Gradio UI integration
- Batch processing
- Streaming audio

**Deliverable**: `examples/` directory

## Project Timeline

### Week 1: Foundation (Critical Path)
- **Day 1-2**: Multi-stage Dockerfile creation (Task 1.1)
- **Day 2-3**: CUDA 13.0 + PyTorch installation (Task 1.2)
- **Day 3**: Flash Attention from pre-built wheel (Task 1.3) - FAST INSTALL
- **Day 4-5**: Core dependencies (Task 1.4)
- **Day 5-7**: Model download system with retry (Task 2.1) - FOUR MODELS

**Checkpoint**: Container builds, all 4 models download, GPU accessible

### Week 2: Core Implementation
- **Day 1-2**: Model loading with memory management (Task 2.2)
- **Day 2-3**: Multi-GPU configuration (Task 2.3)
- **Day 3**: Model warm-up (Task 2.4)
- **Day 4-5**: FastAPI structure and concurrency (Tasks 3.1, 3.2)
- **Day 6-7**: Voice clone engine (Task 4.1)

**Checkpoint**: API functional, voice cloning works, concurrency controlled

### Week 3: Features and Optimization
- **Day 1-2**: Custom voice and voice design (Tasks 4.2, 4.3)
- **Day 3**: Batch processing (Task 4.4)
- **Day 4-5**: Comprehensive audio processing (Task 5.1, 5.2)
- **Day 6**: Language detection (Task 5.3)
- **Day 7**: Error handling and health checks (Tasks 3.3, 3.4)

**Checkpoint**: All three models working, all features implemented

### Week 4: Testing and Polish
- **Day 1-2**: Unit tests (Task 7.1)
- **Day 3-4**: Integration tests (Task 7.2)
- **Day 4-5**: Performance testing and optimization (Task 7.3)
- **Day 6**: Configuration and compose.yaml (Tasks 6.1, 6.2)
- **Day 7**: Documentation (Tasks 8.1, 8.2, 8.3)

**Checkpoint**: Production ready, tested, documented

## Success Criteria

### Functional Requirements
✅ All FOUR models operational (3x TTS + 1x Tokenizer)
✅ Multi-GPU support with all GPUs utilized
✅ All API endpoints functional
✅ Support for 10 languages with auto-detection
✅ Multiple audio input/output formats
✅ Batch processing capability
✅ Concurrency control preventing OOM

### Performance Requirements
✅ < 2 seconds latency for 10 second audio generation
✅ < 5 seconds first request (with warm-up)
✅ > 80% GPU utilization during inference
✅ Support 4-8 concurrent requests (GPU-dependent)
✅ < 1% error rate under normal load
✅ Graceful degradation under overload

### Quality Requirements
✅ High-quality voice cloning (3s reference audio)
✅ Accurate voice design from descriptions
✅ Stable long-form audio generation
✅ No memory leaks or resource exhaustion
✅ Flash Attention 2 working OR SDPA fallback active

### Operational Requirements
✅ One-command deployment (`docker compose up`)
✅ Automatic model downloading (all 4 models)
✅ Comprehensive health monitoring
✅ Structured JSON logging
✅ Graceful error handling
✅ Clear documentation

## Critical Risks and Mitigation

### Technical Risks

1. **Flash Attention Compilation Failure** (HIGH RISK)
   - **Mitigation**: SDPA fallback built into code
   - **Testing**: Validate both paths work
   - **Documentation**: Warn about 1-2 hour build time

2. **GPU Memory Constraints** (MEDIUM RISK)
   - **Mitigation**: LRU model caching, memory monitoring
   - **Testing**: Test on various GPU configurations
   - **Documentation**: Clear memory requirements per setup

3. **Model Download Failures** (MEDIUM RISK)
   - **Mitigation**: Retry logic, dual-source support
   - **Testing**: Test with interrupted downloads
   - **Fallback**: Manual download instructions

4. **Concurrent Request OOM** (HIGH RISK)
   - **Mitigation**: Semaphore limiting, request queuing
   - **Testing**: Load testing with 10+ concurrent requests
   - **Monitoring**: GPU memory alerts

5. **First Build Time** (LOW RISK, HIGH FRUSTRATION)
   - **Mitigation**: Multi-stage build with caching
   - **Documentation**: Clear warning about 2-3 hour first build
   - **Option**: Provide pre-built image

## Deployment Checklist

Before production deployment:
- [ ] All tests passing (unit, integration, performance)
- [ ] All 4 models downloaded and verified
- [ ] Performance benchmarks met
- [ ] Documentation complete and reviewed
- [ ] Security review completed
- [ ] Logging and monitoring configured
- [ ] Health checks operational
- [ ] Resource limits configured
- [ ] Flash Attention status confirmed (working or fallback active)
- [ ] GPU memory monitoring active
- [ ] Concurrency limits tested
- [ ] Error handling validated

## Maintenance Plan

### Regular Tasks
- Monitor GPU memory usage daily
- Review error logs weekly
- Update dependencies monthly
- Performance optimization quarterly
- Model updates when Qwen releases new versions
- Security patches as needed

### Monitoring Metrics
- Request latency (p50, p95, p99)
- Error rate by endpoint
- GPU utilization per device
- Memory usage trends
- Disk usage for model cache
- Concurrent connection count
- Model switch frequency

### Alerts
- GPU memory > 90%
- Error rate > 5%
- Request latency > 10s
- Disk space < 10GB
- Health check failures

