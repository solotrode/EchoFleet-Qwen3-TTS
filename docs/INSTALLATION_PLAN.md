# Qwen3-TTS Installation Plan

## Executive Summary

This project currently **DOES NOT** follow the official Qwen3-TTS installation instructions from HuggingFace. Critical gaps have been identified that prevent proper functionality.

## Official Installation Requirements (from HuggingFace)

### 1. Core Package Installation
**Official Method:**
```bash
conda create -n qwen3-tts python=3.12 -y
conda activate qwen3-tts
pip install -U qwen-tts
```

**Current Project Status:** ❌ MISSING
- The `qwen-tts` package is listed in requirements.txt but never properly installed in a clean environment
- No conda environment is used (Docker uses system Python)
- The package installation order may cause dependency conflicts

### 2. Flash Attention Installation
**Official Method:**
```bash
pip install -U flash-attn --no-build-isolation

# OR for low-memory machines:
MAX_JOBS=4 pip install -U flash-attn --no-build-isolation
```

**Current Project Status:** ⚠️ PARTIALLY IMPLEMENTED
- Project tries to use pre-built wheel from `/flash-attn/` directory
- Dockerfile has strict requirements that may fail if wheel is incompatible
- No fallback mechanism to build from source or download from PyPI

### 3. Model Initialization Code
**Official Method:**
```python
import torch
from qwen_tts import Qwen3TTSModel

model = Qwen3TTSModel.from_pretrained(
    "Qwen/Qwen3-TTS-12Hz-1.7B-Base",
    device_map="cuda:0",
    dtype=torch.bfloat16,
    attn_implementation="flash_attention_2",
)
```

**Current Project Status:** ❌ MISSING
- No actual model loading code exists in the codebase
- `api/main.py` exists but has no model initialization
- No inference implementation for any of the three model types

## Critical Issues Identified

### Issue 1: Missing `qwen-tts` Package Integration
**Problem:** The official `qwen-tts` package provides:
- `Qwen3TTSModel` class for model loading
- `Qwen3TTSTokenizer` for audio encoding/decoding
- `generate_voice_clone()`, `generate_custom_voice()`, `generate_voice_design()` methods
- Built-in audio processing utilities

**Current State:** Package is in requirements but never imported or used

**Impact:** ⚠️ CRITICAL - Cannot load or run any models without this

### Issue 2: Incorrect Installation Sequence
**Official Order:**
1. Create fresh Python 3.12 environment
2. Install `qwen-tts` (which pulls in dependencies)
3. Optionally install `flash-attn` afterward

**Current Order:**
1. Install PyTorch manually
2. Install flash-attn from local wheel
3. Install requirements.txt (including qwen-tts)

**Impact:** ⚠️ HIGH - May cause version conflicts, especially with transformers

### Issue 3: Missing Model Implementation
**Required Code (not present):**
- Model loader class using `Qwen3TTSModel.from_pretrained()`
- Voice clone inference using `model.generate_voice_clone()`
- Custom voice inference using `model.generate_custom_voice()`
- Voice design inference using `model.generate_voice_design()`
- Audio output handling with `soundfile`

**Current State:** No implementation exists

### Issue 4: Flash Attention Handling
**Official Recommendation:**
- Optional but recommended
- Install from PyPI with `--no-build-isolation`
- Can fallback to standard attention if not available

**Current Implementation:**
- Requires local wheel file
- Fails build if wheel is missing or incompatible
- No graceful degradation

**Impact:** ⚠️ MEDIUM - Build may fail unnecessarily

## Recommended Fixes

### Fix 1: Align Dockerfile with Official Installation
**Changes Required:**

```dockerfile
# Stage 1: Builder
# 1. Install qwen-tts FIRST (let it manage dependencies)
RUN pip install --no-cache-dir qwen-tts

# 2. OPTIONALLY install flash-attn (with fallback)
RUN pip install --no-cache-dir flash-attn --no-build-isolation || \
    echo "⚠️  Flash Attention installation failed - will use standard attention fallback"

# 3. Install additional requirements (avoiding duplicates)
COPY requirements.txt /tmp/requirements.txt
RUN pip install --no-cache-dir -r /tmp/requirements.txt
```

**Benefits:**
- Matches official installation process
- Lets `qwen-tts` manage its own dependencies
- Graceful flash-attn fallback

### Fix 2: Implement Model Loading (models/loader.py)
**Required Implementation:**

```python
from qwen_tts import Qwen3TTSModel
import torch
from typing import Optional, Dict

class ModelManager:
    """Manages Qwen3-TTS model lifecycle following official API."""
    
    MODEL_REGISTRY = {
        "base": "Qwen/Qwen3-TTS-12Hz-1.7B-Base",
        "custom-voice": "Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice",
        "voice-design": "Qwen/Qwen3-TTS-12Hz-1.7B-VoiceDesign"
    }
    
    def __init__(
        self,
        cache_dir: str,
        device_map: str = "cuda:0",
        dtype: torch.dtype = torch.bfloat16
    ):
        self.cache_dir = cache_dir
        self.device_map = device_map
        self.dtype = dtype
        self._models: Dict[str, Qwen3TTSModel] = {}
    
    def get_model(self, model_type: str) -> Qwen3TTSModel:
        """Load model using official API."""
        if model_type in self._models:
            return self._models[model_type]
        
        model_name = self.MODEL_REGISTRY[model_type]
        
        # Use official loading API
        model = Qwen3TTSModel.from_pretrained(
            model_name,
            cache_dir=self.cache_dir,
            device_map=self.device_map,
            dtype=self.dtype,
            attn_implementation="flash_attention_2",
        )
        
        self._models[model_type] = model
        return model
```

### Fix 3: Implement Inference Engines
**Required Files:**

1. **inference/voice_clone.py** - Use `model.generate_voice_clone()`
2. **inference/custom_voice.py** - Use `model.generate_custom_voice()`
3. **inference/voice_design.py** - Use `model.generate_voice_design()`

**Example (voice_clone.py):**
```python
import torch
import soundfile as sf
from typing import Union, Tuple
from qwen_tts import Qwen3TTSModel

class VoiceCloneEngine:
    def __init__(self, model: Qwen3TTSModel):
        self.model = model
    
    async def generate(
        self,
        text: str,
        ref_audio: Union[str, bytes],
        ref_text: str,
        language: Optional[str] = None,
        x_vector_only_mode: bool = False
    ) -> Tuple[torch.Tensor, int]:
        """Generate cloned voice using official API."""
        
        # Use official generate_voice_clone method
        wavs, sr = self.model.generate_voice_clone(
            text=text,
            ref_audio=ref_audio,
            ref_text=ref_text,
            language=language,
            x_vector_only_mode=x_vector_only_mode
        )
        
        return wavs[0], sr
```

### Fix 4: Update requirements.txt
**Remove duplicates/conflicts:**

```txt
# Let qwen-tts manage these dependencies:
# - transformers (qwen-tts specifies compatible version)
# - accelerate (qwen-tts specifies compatible version)  
# - torch (install separately with CUDA)
# - torchaudio (install separately with CUDA)

# Keep only additional dependencies:
fastapi>=0.104.0
uvicorn[standard]>=0.24.0
pydantic>=2.0.0
python-multipart>=0.0.6
aiofiles>=23.0.0
prometheus-client>=0.18.0
psutil>=5.9.0
```

## Implementation Checklist

### Phase 1: Fix Docker Installation (Critical)
- [ ] Reorder Dockerfile to install `qwen-tts` first
- [ ] Make flash-attn optional with fallback
- [ ] Remove redundant package versions from requirements.txt
- [ ] Test build completes successfully

### Phase 2: Implement Core Models (Critical)
- [ ] Create `models/loader.py` with `ModelManager` using official API
- [ ] Create `inference/voice_clone.py` with official `generate_voice_clone()`
- [ ] Create `inference/custom_voice.py` with official `generate_custom_voice()`
- [ ] Create `inference/voice_design.py` with official `generate_voice_design()`
- [ ] Create `utils/audio_output.py` for format conversion (wav, mp3, base64)

### Phase 3: Implement API Endpoints (High Priority)
- [ ] Update `api/main.py` to initialize ModelManager on startup
- [ ] Implement `/v1/tts/voice-clone` endpoint
- [ ] Implement `/v1/tts/custom-voice` endpoint
- [ ] Implement `/v1/tts/voice-design` endpoint
- [ ] Implement `/v1/health` endpoint
- [ ] Add proper error handling and logging

### Phase 4: Testing (High Priority)
- [ ] Create test script using official examples
- [ ] Test voice clone with reference audio
- [ ] Test custom voice with supported speakers
- [ ] Test voice design with natural language descriptions
- [ ] Performance benchmarking on target hardware

### Phase 5: Documentation (Medium Priority)
- [ ] Update README with correct installation steps
- [ ] Document API endpoints with examples
- [ ] Add troubleshooting guide
- [ ] Create quickstart guide matching official docs

## Missing Components Summary

### Files That Need to Be Created:
1. `models/loader.py` - Model management using official API
2. `inference/base.py` - Base inference class
3. `inference/voice_clone.py` - Voice cloning implementation
4. `inference/custom_voice.py` - Custom voice implementation
5. `inference/voice_design.py` - Voice design implementation
6. `utils/audio_output.py` - Audio format conversion utilities
7. `api/routes.py` - FastAPI route definitions
8. `api/schemas.py` - Pydantic request/response models
9. `api/dependencies.py` - Dependency injection
10. `examples/test_voice_clone.py` - Working example script

### Current Files That Need Major Updates:
1. `Dockerfile` - Reorder installation, add qwen-tts first
2. `requirements.txt` - Remove conflicts, minimize dependencies
3. `api/main.py` - Add model initialization and startup
4. `compose.yaml` - Verify GPU configuration matches requirements

## References

- **Official Docs:** https://huggingface.co/Qwen/Qwen3-TTS-12Hz-1.7B-Base
- **GitHub Repo:** https://github.com/QwenLM/Qwen3-TTS
- **PyPI Package:** https://pypi.org/project/qwen-tts/
- **Example Scripts:** https://github.com/QwenLM/Qwen3-TTS/tree/main/examples

## Risk Assessment

### HIGH RISK (Must Fix):
- ❌ No working model loading code
- ❌ No inference implementation
- ❌ Incorrect package installation order

### MEDIUM RISK (Should Fix):
- ⚠️  Flash attention handling too strict
- ⚠️  No graceful degradation
- ⚠️  Missing audio processing utilities

### LOW RISK (Nice to Have):
- ℹ️  Could optimize for multi-GPU
- ℹ️  Could add streaming support
- ℹ️  Could add batch processing optimizations

## Next Steps

1. **IMMEDIATE:** Review this plan with team
2. **PRIORITY 1:** Fix Dockerfile installation order (1-2 hours)
3. **PRIORITY 2:** Implement model loader using official API (2-4 hours)
4. **PRIORITY 3:** Implement inference engines (4-6 hours)
5. **PRIORITY 4:** Implement API endpoints (4-6 hours)
6. **PRIORITY 5:** End-to-end testing (2-4 hours)

**Total Estimated Effort:** 13-22 hours of development work

---

**Document Version:** 1.0  
**Last Updated:** January 24, 2026  
**Status:** DRAFT - REQUIRES APPROVAL
