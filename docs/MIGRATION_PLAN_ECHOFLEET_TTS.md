# EchoFleet-TTS Migration Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use **superpowers:subagent-driven-development** (recommended for this plan — see rationale at bottom) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace the in-process Qwen3-TTS and Fish SGLang backends with short-lived vllm-omni subprocesses, add Voxtral support, and rename the project from EchoFleet-Qwen3-TTS to EchoFleet-TTS.

**Architecture:** Each GPU worker thread spawns a `vllm serve` subprocess per candidate (with `CUDA_VISIBLE_DEVICES=N`), polls `/v1/audio/voices` for readiness, POSTs to `/v1/audio/speech`, terminates the process to fully release VRAM, then scores the result with a persistent per-GPU Whisper instance. Whisper is unloaded when the GPU's work queue drains. The dispatcher uses a new `GpuPool` class to route candidates only to GPUs with sufficient VRAM for the requested model family.

**Tech Stack:** Python 3.12, FastAPI, Redis, vllm-omni (`vllm serve` CLI), httpx, soundfile, Whisper via HuggingFace Transformers, Docker Compose, pytest.

---

## Codebase Orientation

> Read this before touching any code.

### What exists today

| File | Role |
|------|------|
| `api/main.py` | FastAPI app + `job_worker_loop` (runs in worker subprocess). Contains inference singletons (`_SYNC_TTS_SERVER`, `_SYNC_FISH_AUDIO_SERVICE`, `_FISH_SGLANG_CONTROLLER`) and `_generate_candidate()`. |
| `worker/run.py` | Spawns `job_worker_loop` processes + `dispatch_loop` thread. Uses `settings.tts_gpu_list()`. |
| `config/settings.py` | Env-based config. Has Fish/SGLang/Qwen-specific fields that are being removed. |
| `api/schemas.py` | Pydantic request models. `CustomVoiceRequest` uses `speaker`/`instruct` (old names). `S2ProRequest` is being removed. |
| `utils/errors.py` | Exception hierarchy rooted at `Qwen3TTSError` (renaming to `TTSError`). |
| `utils/gpu_pool.py` | `GpuLeasePool` — thread-safe scheduler. **Keep as-is.** Not the same as the new `inference/gpu_pool.py`. |
| `inference/accuracy_scorer.py` | WER/CER scoring. **Keep as-is.** |
| `inference/whisper_service.py` | `WhisperTranscriberHF`. **Keep as-is.** |

### What is being deleted

| File | Reason |
|------|--------|
| `inference/qwen_tts_service.py` | In-process model → vllm subprocess |
| `inference/fish_audio_service.py` | SGLang proxy → vllm subprocess |
| `inference/fish_sglang_controller.py` | Docker container lifecycle → not needed |
| `Dockerfile.sglang` | SGLang container gone |
| `Dockerfile.sglang-base` | SGLang base image gone |
| `tests/unit/test_fish_audio_service.py` | Tests deleted module |
| `tests/unit/test_speech_tokenizer_device.py` | Tests deleted module |
| `tests/unit/test_generation_limits.py` | Tests deprecated module |

### What is being created

| File | Responsibility |
|------|---------------|
| `inference/gpu_pool.py` | `GpuPool`: parse `GPU_POOL` env var; `eligible_gpus(model)`; `assign_candidates(n, model)` round-robin |
| `inference/vllm_process.py` | `VllmProcess`: spawn/poll/generate/terminate a vllm-omni subprocess for one GPU |
| `tests/unit/test_inference_gpu_pool.py` | Tests for `GpuPool` |
| `tests/unit/test_vllm_process.py` | Tests for `VllmProcess` (mocked subprocess + httpx) |
| `tests/unit/test_settings.py` | Tests for new `Settings` fields |
| `tests/unit/test_schemas.py` | Tests for updated API schemas |
| `tests/unit/test_errors.py` | Tests for `TTSError` rename |

### What is being modified

| File | Key changes |
|------|-------------|
| `utils/errors.py` | `Qwen3TTSError` → `TTSError`; update all subclasses |
| `config/settings.py` | Remove Fish/Qwen-specific fields; add `gpu_pool`, `vllm_*`, per-model VRAM + model ID fields |
| `api/schemas.py` | Remove `S2ProRequest`; `speaker`→`voice`, `instruct`→`instructions`; add `TTSRequest` |
| `api/main.py` | Add `_model_id_for_job`, `_generate_with_vllm`; replace `job_worker_loop`; update `_score_candidates`; remove inference singletons + S2Pro endpoint |
| `worker/run.py` | Use `GpuPool.assign_candidates()` in dispatcher; derive GPU list from `settings.gpu_pool_parsed()` |
| `Dockerfile` | Remove qwen_tts git install + docker.io; add `pip install vllm-omni` |
| `compose.yaml` | Remove fish-sglang services; rename containers/networks/volumes; swap env vars |
| `AGENTS.md` | Update project name, structure, build commands |

---

## Task 1: Delete Legacy Inference Files

**Files:**
- Delete: `inference/qwen_tts_service.py`
- Delete: `inference/fish_audio_service.py`
- Delete: `inference/fish_sglang_controller.py`
- Delete: `Dockerfile.sglang`
- Delete: `Dockerfile.sglang-base`
- Delete: `tests/unit/test_fish_audio_service.py`
- Delete: `tests/unit/test_speech_tokenizer_device.py`
- Delete: `tests/unit/test_generation_limits.py`

- [ ] **Step 1: Delete the inference source files**

```bash
rm inference/qwen_tts_service.py inference/fish_audio_service.py inference/fish_sglang_controller.py
```

- [ ] **Step 2: Delete the SGLang Dockerfiles**

```bash
rm Dockerfile.sglang Dockerfile.sglang-base
```

- [ ] **Step 3: Delete obsolete tests**

```bash
rm tests/unit/test_fish_audio_service.py tests/unit/test_speech_tokenizer_device.py tests/unit/test_generation_limits.py
```

- [ ] **Step 4: Verify surviving unit tests still pass**

Run: `pytest tests/unit/test_gpu_pool.py tests/unit/test_accuracy_scorer.py tests/unit/test_text_chunker.py -v`

Expected: All PASS

- [ ] **Step 5: Commit**

```bash
git add -A
git commit -m "chore: delete legacy Fish SGLang and in-process Qwen3-TTS inference files"
```

---

## Task 2: Rename `Qwen3TTSError` → `TTSError`

**Files:**
- Modify: `utils/errors.py`
- Create: `tests/unit/test_errors.py`

- [ ] **Step 1: Write the failing test**

Create `tests/unit/test_errors.py`:

```python
import utils.errors as m


class TestTTSError:
    def test_tts_error_is_base_exception(self):
        err = m.TTSError("test")
        assert isinstance(err, Exception)

    def test_subclasses_inherit_from_tts_error(self):
        assert issubclass(m.ModelNotFoundError, m.TTSError)
        assert issubclass(m.ModelLoadError, m.TTSError)
        assert issubclass(m.AudioProcessingError, m.TTSError)

    def test_qwen3_tts_error_name_does_not_exist(self):
        assert not hasattr(m, "Qwen3TTSError")
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/unit/test_errors.py -v`

Expected: FAIL — `ImportError: cannot import name 'TTSError'`

- [ ] **Step 3: Rewrite `utils/errors.py`**

```python
"""Custom exceptions for EchoFleet-TTS service."""


class TTSError(Exception):
    """Base exception for all EchoFleet-TTS errors."""
    pass


class ModelNotFoundError(TTSError):
    """Raised when requested model is not found or not loaded."""
    pass


class ModelLoadError(TTSError):
    """Raised when model loading fails."""
    pass


class AudioProcessingError(TTSError):
    """Raised when audio processing fails."""
    pass


class CUDAOutOfMemoryError(TTSError):
    """Raised when GPU memory is exhausted."""
    pass


class InferenceError(TTSError):
    """Raised when TTS inference fails."""
    pass


class TranscriptionError(TTSError):
    """Raised when Whisper transcription fails."""
    pass


class ValidationError(TTSError):
    """Raised when request validation fails."""
    pass


class JobNotFoundError(TTSError):
    """Raised when job ID is not found in Redis."""
    pass


class GPUNotAvailableError(TTSError):
    """Raised when no GPUs are available for processing."""
    pass


__all__ = [
    "TTSError",
    "ModelNotFoundError",
    "ModelLoadError",
    "AudioProcessingError",
    "CUDAOutOfMemoryError",
    "InferenceError",
    "TranscriptionError",
    "ValidationError",
    "JobNotFoundError",
    "GPUNotAvailableError",
]
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/unit/test_errors.py -v`

Expected: All PASS

- [ ] **Step 5: Commit**

```bash
git add utils/errors.py tests/unit/test_errors.py
git commit -m "refactor: rename Qwen3TTSError to TTSError"
```

---

## Task 3: Rewrite `config/settings.py`

**Files:**
- Modify: `config/settings.py`
- Create: `tests/unit/test_settings.py`

**What's removed:** `tts_gpus`, `tts_preferred_gpu`, `tts_gpu_capacity`, `enable_base_model`, `enable_custom_voice`, `enable_voice_design`, `default_dtype`, `device_map`, `tts_attn_impl`, deprecated token fields, `tts_unload_idle_seconds`, all Fish/SGLang/S2Pro fields, `gpu_memory_threshold_mb`.

**What's added:** `gpu_pool`, `qwen_min_vram_gb`, `fish_min_vram_gb`, `voxtral_min_vram_gb`, five model ID fields, `vllm_port_base`, `vllm_startup_timeout_seconds`, `vllm_gpu_memory_utilization`, `gpu_pool_parsed()`.

**What's kept:** Redis, directories, `tts_worker_concurrency`, `tts_sample_rate`, `tts_chunk_max_chars`, API ports, Whisper config, job limits, retention, logging.

- [ ] **Step 1: Write the failing tests**

Create `tests/unit/test_settings.py`:

```python
import os
from unittest.mock import patch
import pytest


def _make(overrides=None):
    # Import inside function so lru_cache doesn't interfere between tests
    from importlib import reload
    import config.settings as mod
    reload(mod)
    env = {k: v for k, v in os.environ.items()}
    for key in [
        "GPU_POOL", "QWEN_MIN_VRAM_GB", "FISH_MIN_VRAM_GB", "VOXTRAL_MIN_VRAM_GB",
        "QWEN_MODEL_ID", "QWEN_VOICE_DESIGN_MODEL_ID", "QWEN_BASE_MODEL_ID",
        "FISH_MODEL_ID", "VOXTRAL_MODEL_ID",
        "VLLM_PORT_BASE", "VLLM_STARTUP_TIMEOUT_SECONDS", "VLLM_GPU_MEMORY_UTILIZATION",
    ]:
        env.pop(key, None)
    if overrides:
        env.update(overrides)
    with patch.dict(os.environ, env, clear=True):
        return mod.Settings()


class TestNewFields:
    def test_gpu_pool_default(self):
        assert _make().gpu_pool == "0:16,1:16,2:8,3:8"

    def test_gpu_pool_parsed_returns_tuples(self):
        s = _make({"GPU_POOL": "0:16,1:16,2:8,3:8"})
        assert s.gpu_pool_parsed() == [(0, 16), (1, 16), (2, 8), (3, 8)]

    def test_gpu_pool_parsed_single_gpu(self):
        s = _make({"GPU_POOL": "0:24"})
        assert s.gpu_pool_parsed() == [(0, 24)]

    def test_vllm_port_base_default(self):
        assert _make().vllm_port_base == 8100

    def test_vllm_startup_timeout_default(self):
        assert _make().vllm_startup_timeout_seconds == 180

    def test_vllm_gpu_memory_utilization_default(self):
        assert _make().vllm_gpu_memory_utilization == pytest.approx(0.90)

    def test_model_id_defaults(self):
        s = _make()
        assert s.qwen_model_id == "Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice"
        assert s.qwen_voice_design_model_id == "Qwen/Qwen3-TTS-12Hz-1.7B-VoiceDesign"
        assert s.qwen_base_model_id == "Qwen/Qwen3-TTS-12Hz-1.7B-Base"
        assert s.fish_model_id == "fishaudio/s2-pro"
        assert s.voxtral_model_id == "mistralai/Voxtral-4B-TTS-2603"

    def test_min_vram_defaults(self):
        s = _make()
        assert s.qwen_min_vram_gb == 8
        assert s.fish_min_vram_gb == 12
        assert s.voxtral_min_vram_gb == 12


class TestRemovedFields:
    def test_no_tts_gpus(self):
        assert not hasattr(_make(), "tts_gpus")

    def test_no_fish_idle_unload_seconds(self):
        assert not hasattr(_make(), "fish_idle_unload_seconds")

    def test_no_fish_sglang_container_name(self):
        assert not hasattr(_make(), "fish_sglang_container_name")

    def test_no_enable_base_model(self):
        assert not hasattr(_make(), "enable_base_model")


class TestPreservedFields:
    def test_tts_worker_concurrency_preserved(self):
        s = _make({"TTS_WORKER_CONCURRENCY": "4"})
        assert s.tts_worker_concurrency == 4

    def test_whisper_model_id_shorthand(self):
        s = _make({"WHISPER_MODEL": "large-v3-turbo"})
        assert s.whisper_model_id() == "openai/whisper-large-v3-turbo"

    def test_whisper_model_id_passthrough(self):
        s = _make({"WHISPER_MODEL": "openai/whisper-large-v3-turbo"})
        assert s.whisper_model_id() == "openai/whisper-large-v3-turbo"
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/unit/test_settings.py -v 2>&1 | head -30`

Expected: Multiple FAIL — missing attributes on current `Settings`

- [ ] **Step 3: Rewrite `config/settings.py`**

```python
"""Centralized configuration using environment variables."""

from __future__ import annotations

import os
from functools import lru_cache
from typing import List


class Settings:
    """Settings for EchoFleet-TTS multi-GPU service."""

    def __init__(self) -> None:
        # Redis
        self.redis_host: str = os.getenv("REDIS_HOST", "redis")
        self.redis_port: int = int(os.getenv("REDIS_PORT", "6379"))

        # Directories
        self.model_cache_dir: str = os.getenv("MODEL_CACHE_DIR", "/models")
        self.output_dir: str = os.getenv("OUTPUT_DIR", "/workspace/outputs")

        # GPU pool: "0:16,1:16,2:8,3:8"  (gpu_id:vram_gb, comma-separated)
        self.gpu_pool: str = os.getenv("GPU_POOL", "0:16,1:16,2:8,3:8")

        # Per-model minimum VRAM (GB) — controls which GPUs are eligible
        self.qwen_min_vram_gb: int = int(os.getenv("QWEN_MIN_VRAM_GB", "8"))
        self.fish_min_vram_gb: int = int(os.getenv("FISH_MIN_VRAM_GB", "12"))
        self.voxtral_min_vram_gb: int = int(os.getenv("VOXTRAL_MIN_VRAM_GB", "12"))

        # Model IDs (HuggingFace repo IDs or local paths)
        self.qwen_model_id: str = os.getenv(
            "QWEN_MODEL_ID", "Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice"
        )
        self.qwen_voice_design_model_id: str = os.getenv(
            "QWEN_VOICE_DESIGN_MODEL_ID", "Qwen/Qwen3-TTS-12Hz-1.7B-VoiceDesign"
        )
        self.qwen_base_model_id: str = os.getenv(
            "QWEN_BASE_MODEL_ID", "Qwen/Qwen3-TTS-12Hz-1.7B-Base"
        )
        self.fish_model_id: str = os.getenv("FISH_MODEL_ID", "fishaudio/s2-pro")
        self.voxtral_model_id: str = os.getenv(
            "VOXTRAL_MODEL_ID", "mistralai/Voxtral-4B-TTS-2603"
        )

        # vllm-omni subprocess settings
        # Port assignment: cuda:N uses vllm_port_base + N
        # e.g. base=8100 → cuda:0→8100, cuda:1→8101, cuda:2→8102, cuda:3→8103
        self.vllm_port_base: int = int(os.getenv("VLLM_PORT_BASE", "8100"))
        self.vllm_startup_timeout_seconds: int = int(
            os.getenv("VLLM_STARTUP_TIMEOUT_SECONDS", "180")
        )
        self.vllm_gpu_memory_utilization: float = float(
            os.getenv("VLLM_GPU_MEMORY_UTILIZATION", "0.90")
        )

        # Number of per-GPU worker processes
        self.tts_worker_concurrency: int = int(os.getenv("TTS_WORKER_CONCURRENCY", "4"))

        # TTS generation
        self.tts_sample_rate: int = int(os.getenv("TTS_SAMPLE_RATE", "24000"))
        self.tts_chunk_max_chars: int = int(os.getenv("TTS_CHUNK_MAX_CHARS", "1000"))

        # API
        self.api_host: str = os.getenv("API_HOST", "0.0.0.0")
        self.api_port: int = int(os.getenv("API_PORT", "8000"))
        self.gradio_port: int = int(os.getenv("GRADIO_PORT", "7860"))

        # Whisper (HF Transformers)
        self.whisper_model: str = os.getenv("WHISPER_MODEL", "openai/whisper-large-v3-turbo")
        self.whisper_device_pool: str = os.getenv("WHISPER_DEVICE_POOL", "0,1,2,3")
        self.whisper_model_path: str = os.getenv(
            "WHISPER_MODEL_PATH", "/stt_models/whisper-large-v3-turbo"
        )

        # Job limits
        self.max_text_length: int = int(os.getenv("MAX_TEXT_LENGTH", "5000"))
        self.max_candidates: int = int(os.getenv("MAX_CANDIDATES", "10"))
        self.default_candidates: int = int(os.getenv("DEFAULT_CANDIDATES", "1"))
        self.job_timeout: int = int(os.getenv("JOB_TIMEOUT", "3600"))
        self.tts_candidate_stale_seconds: int = int(
            os.getenv("TTS_CANDIDATE_STALE_SECONDS", "900")
        )
        self.tts_candidate_rescue_attempts: int = int(
            os.getenv("TTS_CANDIDATE_RESCUE_ATTEMPTS", "2")
        )

        # Retention
        self.retention_hours: int = int(os.getenv("RETENTION_HOURS", "24"))
        self.cleanup_schedule: str = os.getenv("CLEANUP_SCHEDULE", "0:00")
        self.log_retention_days: int = int(os.getenv("LOG_RETENTION_DAYS", "10"))
        self.job_artifact_retention_days: int = int(
            os.getenv("JOB_ARTIFACT_RETENTION_DAYS", "10")
        )

        # Logging
        self.log_level: str = os.getenv("LOG_LEVEL", "INFO")
        self.structured_logging: bool = (
            os.getenv("STRUCTURED_LOGGING", "true").lower() == "true"
        )
        self.log_to_file: bool = os.getenv("LOG_TO_FILE", "false").lower() == "true"
        self.log_file_path: str = os.getenv("LOG_FILE_PATH", "")

    def gpu_pool_parsed(self) -> list[tuple[int, int]]:
        """Parse gpu_pool string into a list of (gpu_id, vram_gb) tuples.

        Example:
            >>> Settings().gpu_pool_parsed()
            [(0, 16), (1, 16), (2, 8), (3, 8)]
        """
        result = []
        for part in self.gpu_pool.split(","):
            gpu_id_str, vram_str = part.strip().split(":")
            result.append((int(gpu_id_str), int(vram_str)))
        return result

    def whisper_device_list(self) -> List[str]:
        """Return Whisper device pool as CUDA device strings, e.g. ['cuda:0', 'cuda:1']."""
        try:
            parts = [p.strip() for p in self.whisper_device_pool.split(",") if p.strip()]
            return [f"cuda:{int(p)}" for p in parts]
        except Exception:
            return ["cuda:0"]

    def whisper_model_id(self) -> str:
        """Return a HuggingFace repo ID for Whisper. Accepts shorthand like 'large-v3-turbo'."""
        raw = (self.whisper_model or "").strip()
        if "/" in raw:
            return raw
        shorthand_map = {
            "large-v3-turbo": "openai/whisper-large-v3-turbo",
            "large-v3": "openai/whisper-large-v3",
            "large": "openai/whisper-large",
            "medium": "openai/whisper-medium",
            "small": "openai/whisper-small",
            "base": "openai/whisper-base",
            "tiny": "openai/whisper-tiny",
        }
        return shorthand_map.get(raw, f"openai/whisper-{raw}")

    def whisper_local_path(self) -> str:
        """Return local filesystem path for a pre-downloaded Whisper model."""
        return self.whisper_model_path or "/stt_models/whisper-large-v3-turbo"


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Return a cached Settings instance."""
    return Settings()


__all__ = ["Settings", "get_settings"]
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/unit/test_settings.py -v`

Expected: All PASS

- [ ] **Step 5: Commit**

```bash
git add config/settings.py tests/unit/test_settings.py
git commit -m "feat: rewrite settings for vllm-omni GPU pool and model configuration"
```

---

## Task 4: Create `inference/gpu_pool.py`

**Files:**
- Create: `inference/gpu_pool.py`
- Create: `tests/unit/test_inference_gpu_pool.py`

> Note: `utils/gpu_pool.py` (`GpuLeasePool`) is a different class — a thread-safe scheduler used for the old in-process model loading. It is **not** the same as the new `inference/gpu_pool.py` (`GpuPool`). Do not confuse them.

- [ ] **Step 1: Write the failing tests**

Create `tests/unit/test_inference_gpu_pool.py`:

```python
import pytest
from inference.gpu_pool import GpuPool


@pytest.fixture
def pool():
    return GpuPool("0:16,1:16,2:8,3:8", {"qwen": 8, "fish": 12, "voxtral": 12})


class TestEligibility:
    def test_qwen_eligible_on_all_four_gpus(self, pool):
        assert pool.eligible_gpus("qwen") == [0, 1, 2, 3]

    def test_fish_eligible_only_on_16gb_gpus(self, pool):
        assert pool.eligible_gpus("fish") == [0, 1]

    def test_voxtral_eligible_only_on_16gb_gpus(self, pool):
        assert pool.eligible_gpus("voxtral") == [0, 1]

    def test_unknown_model_treated_as_zero_minimum(self, pool):
        # Unknown model has no min_vram entry → 0 → all GPUs qualify
        assert pool.eligible_gpus("unknown") == [0, 1, 2, 3]

    def test_no_gpu_meets_requirement_returns_empty(self):
        p = GpuPool("0:4,1:4", {"qwen": 8})
        assert p.eligible_gpus("qwen") == []


class TestAssignment:
    def test_round_robin_fish_four_candidates_two_gpus(self, pool):
        assignments = pool.assign_candidates(4, "fish")
        assert [g for g, _ in assignments] == [0, 1, 0, 1]

    def test_all_parallel_qwen_four_candidates(self, pool):
        assignments = pool.assign_candidates(4, "qwen")
        assert [g for g, _ in assignments] == [0, 1, 2, 3]

    def test_candidate_ids_are_sequential(self, pool):
        assignments = pool.assign_candidates(3, "fish")
        assert [c for _, c in assignments] == [0, 1, 2]

    def test_single_candidate_goes_to_first_eligible_gpu(self, pool):
        assert pool.assign_candidates(1, "fish") == [(0, 0)]

    def test_no_eligible_gpus_raises_value_error(self):
        p = GpuPool("0:4,1:4", {"qwen": 8, "fish": 12, "voxtral": 12})
        with pytest.raises(ValueError, match="No eligible GPUs"):
            p.assign_candidates(1, "qwen")


class TestParsing:
    def test_invalid_format_missing_colon_raises(self):
        with pytest.raises(ValueError):
            GpuPool("0-16,1-16", {"qwen": 8})

    def test_single_entry(self):
        p = GpuPool("2:16", {"qwen": 8})
        assert p.eligible_gpus("qwen") == [2]
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/unit/test_inference_gpu_pool.py -v 2>&1 | head -10`

Expected: FAIL — `ModuleNotFoundError: No module named 'inference.gpu_pool'`

- [ ] **Step 3: Implement `inference/gpu_pool.py`**

```python
"""Model-aware GPU eligibility and candidate assignment.

Torch-free: safe to import in test environments without CUDA.
"""

from __future__ import annotations


class GpuPool:
    """Assigns generation candidates to eligible GPUs via round-robin.

    Args:
        gpu_pool_str: Comma-separated ``gpu_id:vram_gb`` pairs,
            e.g. ``"0:16,1:16,2:8,3:8"``.
        model_min_vram: Mapping from model family name to minimum required
            VRAM in GB, e.g. ``{"qwen": 8, "fish": 12, "voxtral": 12}``.
    """

    def __init__(self, gpu_pool_str: str, model_min_vram: dict[str, int]) -> None:
        self._gpus: list[tuple[int, int]] = self._parse(gpu_pool_str)
        self._model_min_vram = model_min_vram

    def _parse(self, pool_str: str) -> list[tuple[int, int]]:
        try:
            result = []
            for part in pool_str.split(","):
                gpu_id_str, vram_str = part.strip().split(":")
                result.append((int(gpu_id_str), int(vram_str)))
            return result
        except ValueError as exc:
            raise ValueError(
                f"Invalid GPU_POOL format {pool_str!r}: expected 'id:vram_gb,...'"
            ) from exc

    def eligible_gpus(self, model: str) -> list[int]:
        """Return GPU ids whose VRAM meets or exceeds the minimum for ``model``."""
        min_vram = self._model_min_vram.get(model, 0)
        return [gpu_id for gpu_id, vram in self._gpus if vram >= min_vram]

    def assign_candidates(
        self, num_candidates: int, model: str
    ) -> list[tuple[int, int]]:
        """Return ``[(gpu_id, candidate_id), ...]`` round-robin across eligible GPUs.

        Raises:
            ValueError: If no GPUs are eligible for ``model``.
        """
        eligible = self.eligible_gpus(model)
        if not eligible:
            raise ValueError(
                f"No eligible GPUs for model {model!r} — check GPU_POOL "
                f"and {model.upper()}_MIN_VRAM_GB settings"
            )
        return [(eligible[i % len(eligible)], i) for i in range(num_candidates)]
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/unit/test_inference_gpu_pool.py -v`

Expected: All PASS

- [ ] **Step 5: Commit**

```bash
git add inference/gpu_pool.py tests/unit/test_inference_gpu_pool.py
git commit -m "feat: add GpuPool for model-aware GPU eligibility and candidate assignment"
```

---

## Task 5: Create `inference/vllm_process.py`

**Files:**
- Create: `inference/vllm_process.py`
- Create: `tests/unit/test_vllm_process.py`

**What this class does:**
1. `start()` — runs `vllm serve <model_id> --port N --gpu-memory-utilization X` with `CUDA_VISIBLE_DEVICES=gpu_id` in the env, then polls `GET http://localhost:<port>/v1/audio/voices` until HTTP 200.
2. `generate(**kwargs)` — `POST http://localhost:<port>/v1/audio/speech` with the kwargs as JSON (None values stripped). Returns raw bytes (WAV).
3. `stop()` — `terminate()` + `wait()` the subprocess. VRAM fully released after this.
4. Context manager (`with VllmProcess(...) as proc`) — calls `start()`/`stop()` automatically.

- [ ] **Step 1: Write the failing tests**

Create `tests/unit/test_vllm_process.py`:

```python
import subprocess
from unittest.mock import MagicMock, patch

import httpx
import pytest

from inference.vllm_process import VllmProcess


def _proc(**overrides) -> VllmProcess:
    kwargs = dict(model_id="test/model", gpu_id=0, port=8100, startup_timeout=5)
    kwargs.update(overrides)
    return VllmProcess(**kwargs)


class TestCommand:
    def test_starts_with_vllm_serve(self):
        cmd = _proc()._build_cmd()
        assert cmd[:2] == ["vllm", "serve"]

    def test_includes_model_id(self):
        assert "test/model" in _proc()._build_cmd()

    def test_includes_port(self):
        assert "8103" in _proc(port=8103)._build_cmd()

    def test_includes_gpu_memory_utilization(self):
        assert "0.85" in _proc(gpu_memory_utilization=0.85)._build_cmd()


class TestStart:
    def test_sets_cuda_visible_devices(self):
        p = _proc(gpu_id=2)
        with patch("subprocess.Popen", return_value=MagicMock()) as popen_call, \
             patch.object(p, "_wait_for_ready"):
            p.start()
            env = popen_call.call_args[1]["env"]
            assert env["CUDA_VISIBLE_DEVICES"] == "2"

    def test_stops_if_wait_for_ready_raises(self):
        p = _proc()
        with patch("subprocess.Popen", return_value=MagicMock()), \
             patch.object(p, "_wait_for_ready", side_effect=TimeoutError("timeout")), \
             patch.object(p, "stop") as mock_stop:
            with pytest.raises(TimeoutError):
                p.start()
            mock_stop.assert_called_once()


class TestReadiness:
    def test_returns_on_http_200(self):
        p = _proc()
        p._process = MagicMock()
        with patch("httpx.get", return_value=MagicMock(status_code=200)):
            p._wait_for_ready()  # must not raise

    def test_raises_timeout_on_persistent_connect_error(self):
        p = _proc(startup_timeout=0)
        p._process = MagicMock()
        with patch("httpx.get", side_effect=httpx.ConnectError("refused")):
            with pytest.raises(TimeoutError):
                p._wait_for_ready()

    def test_polls_correct_url(self):
        p = _proc(port=8102)
        p._process = MagicMock()
        with patch("httpx.get", return_value=MagicMock(status_code=200)) as get_call:
            p._wait_for_ready()
            url = get_call.call_args[0][0]
            assert "8102" in url
            assert "/v1/audio/voices" in url


class TestGenerate:
    def test_posts_to_speech_endpoint(self):
        p = _proc(port=8101)
        p._process = MagicMock()
        mock_resp = MagicMock(content=b"wav")
        mock_resp.raise_for_status = MagicMock()
        with patch("httpx.post", return_value=mock_resp) as post_call:
            result = p.generate(input="hello")
            url = post_call.call_args[0][0]
            assert "8101" in url
            assert "/v1/audio/speech" in url
            assert result == b"wav"

    def test_strips_none_values_from_payload(self):
        p = _proc()
        p._process = MagicMock()
        mock_resp = MagicMock(content=b"wav")
        mock_resp.raise_for_status = MagicMock()
        with patch("httpx.post", return_value=mock_resp) as post_call:
            p.generate(input="hello", voice=None, ref_audio=None)
            payload = post_call.call_args[1]["json"]
            assert "voice" not in payload
            assert payload["input"] == "hello"

    def test_raises_on_http_error(self):
        p = _proc()
        p._process = MagicMock()
        mock_resp = MagicMock()
        mock_resp.raise_for_status.side_effect = httpx.HTTPStatusError(
            "500", request=MagicMock(), response=MagicMock()
        )
        with patch("httpx.post", return_value=mock_resp):
            with pytest.raises(httpx.HTTPStatusError):
                p.generate(input="hello")


class TestStop:
    def test_terminates_and_waits(self):
        p = _proc()
        mock_proc = MagicMock()
        p._process = mock_proc
        p.stop()
        mock_proc.terminate.assert_called_once()
        mock_proc.wait.assert_called_once()
        assert p._process is None

    def test_kills_if_terminate_times_out(self):
        p = _proc()
        mock_proc = MagicMock()
        mock_proc.wait.side_effect = [subprocess.TimeoutExpired("vllm", 30), None]
        p._process = mock_proc
        p.stop()
        mock_proc.kill.assert_called_once()

    def test_idempotent_when_not_started(self):
        _proc().stop()  # must not raise


class TestContextManager:
    def test_calls_start_and_stop(self):
        p = _proc()
        with patch.object(p, "start") as s, patch.object(p, "stop") as e:
            with p:
                s.assert_called_once()
            e.assert_called_once()

    def test_stop_called_even_on_exception(self):
        p = _proc()
        with patch.object(p, "start"), patch.object(p, "stop") as mock_stop:
            with pytest.raises(RuntimeError):
                with p:
                    raise RuntimeError("oops")
            mock_stop.assert_called_once()
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/unit/test_vllm_process.py -v 2>&1 | head -10`

Expected: FAIL — `ModuleNotFoundError: No module named 'inference.vllm_process'`

- [ ] **Step 3: Implement `inference/vllm_process.py`**

```python
"""Manages a single vllm-omni subprocess for one GPU.

Typical usage:
    with VllmProcess(model_id="Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice",
                     gpu_id=0, port=8100) as proc:
        wav_bytes = proc.generate(
            input="Hello world",
            voice="vivian",
            task_type="CustomVoice",
            language="English",
        )
"""

from __future__ import annotations

import os
import subprocess
import time
from typing import Any

import httpx


class VllmProcess:
    """Manages a single vllm-omni subprocess for one GPU.

    Lifecycle:
        start() → _wait_for_ready() → generate() → stop()

    Always use as a context manager to guarantee stop() is called:
        with VllmProcess(...) as proc: ...

    Args:
        model_id: HuggingFace model ID or local path.
        gpu_id: CUDA device index; sets CUDA_VISIBLE_DEVICES in the subprocess.
        port: HTTP port for the vllm-omni server.
        gpu_memory_utilization: Fraction of GPU memory vllm may use (0.0–1.0).
        startup_timeout: Seconds to wait for the /v1/audio/voices readiness endpoint.
    """

    def __init__(
        self,
        model_id: str,
        gpu_id: int,
        port: int,
        gpu_memory_utilization: float = 0.90,
        startup_timeout: int = 180,
    ) -> None:
        self.model_id = model_id
        self.gpu_id = gpu_id
        self.port = port
        self.gpu_memory_utilization = gpu_memory_utilization
        self.startup_timeout = startup_timeout
        self._process: subprocess.Popen | None = None

    def _build_cmd(self) -> list[str]:
        return [
            "vllm",
            "serve",
            self.model_id,
            "--port",
            str(self.port),
            "--gpu-memory-utilization",
            str(self.gpu_memory_utilization),
        ]

    def start(self) -> None:
        """Spawn the subprocess and wait for the readiness endpoint."""
        env = {**os.environ, "CUDA_VISIBLE_DEVICES": str(self.gpu_id)}
        self._process = subprocess.Popen(
            self._build_cmd(),
            env=env,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        try:
            self._wait_for_ready()
        except TimeoutError:
            self.stop()
            raise

    def _wait_for_ready(self) -> None:
        url = f"http://localhost:{self.port}/v1/audio/voices"
        deadline = time.monotonic() + self.startup_timeout
        while time.monotonic() < deadline:
            try:
                resp = httpx.get(url, timeout=2.0)
                if resp.status_code == 200:
                    return
            except (httpx.ConnectError, httpx.TimeoutException):
                pass
            time.sleep(1.0)
        raise TimeoutError(
            f"vllm-omni on port {self.port} not ready after {self.startup_timeout}s"
        )

    def generate(self, **kwargs: Any) -> bytes:
        """POST /v1/audio/speech and return raw WAV bytes.

        Pass vllm-omni fields as keyword arguments.
        None values are stripped so vllm-omni uses its own defaults.
        """
        url = f"http://localhost:{self.port}/v1/audio/speech"
        payload = {k: v for k, v in kwargs.items() if v is not None}
        resp = httpx.post(url, json=payload, timeout=300.0)
        resp.raise_for_status()
        return resp.content

    def stop(self) -> None:
        """Terminate the subprocess and wait for exit. VRAM fully released after return."""
        if self._process is not None:
            self._process.terminate()
            try:
                self._process.wait(timeout=30)
            except subprocess.TimeoutExpired:
                self._process.kill()
                self._process.wait()
            self._process = None

    def __enter__(self) -> "VllmProcess":
        self.start()
        return self

    def __exit__(self, *_: Any) -> None:
        self.stop()
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/unit/test_vllm_process.py -v`

Expected: All PASS

- [ ] **Step 5: Commit**

```bash
git add inference/vllm_process.py tests/unit/test_vllm_process.py
git commit -m "feat: add VllmProcess for per-GPU vllm-omni subprocess management"
```

---

## Task 6: Update `api/schemas.py`

**Files:**
- Modify: `api/schemas.py`
- Create: `tests/unit/test_schemas.py`

**Field renames:**
- `CustomVoiceRequest.speaker` → `voice`
- `CustomVoiceRequest.instruct` → `instructions`
- `VoiceDesignRequest.instruct` → `instructions`

**Additions:** `VoiceCloneRequest.task_type = "Base"`, new `TTSRequest` unified model.

**Removals:** `S2ProRequest` class deleted entirely.

- [ ] **Step 1: Write the failing tests**

Create `tests/unit/test_schemas.py`:

```python
import pytest
from pydantic import ValidationError


class TestCustomVoiceRequest:
    def test_voice_field_accepted(self):
        from api.schemas import CustomVoiceRequest
        req = CustomVoiceRequest(text="hello", voice="vivian")
        assert req.voice == "vivian"

    def test_speaker_field_rejected(self):
        from api.schemas import CustomVoiceRequest
        with pytest.raises((ValidationError, TypeError)):
            CustomVoiceRequest(text="hello", speaker="vivian")

    def test_instructions_optional(self):
        from api.schemas import CustomVoiceRequest
        req = CustomVoiceRequest(text="hello", voice="ryan", instructions="Speak slowly")
        assert req.instructions == "Speak slowly"

    def test_instruct_field_rejected(self):
        from api.schemas import CustomVoiceRequest
        with pytest.raises((ValidationError, TypeError)):
            CustomVoiceRequest(text="hello", voice="ryan", instruct="Speak slowly")


class TestVoiceDesignRequest:
    def test_instructions_field_accepted(self):
        from api.schemas import VoiceDesignRequest
        req = VoiceDesignRequest(text="hello", instructions="A warm voice")
        assert req.instructions == "A warm voice"

    def test_instruct_field_rejected(self):
        from api.schemas import VoiceDesignRequest
        with pytest.raises((ValidationError, TypeError)):
            VoiceDesignRequest(text="hello", instruct="A warm voice")

    def test_instructions_required(self):
        from api.schemas import VoiceDesignRequest
        with pytest.raises(ValidationError):
            VoiceDesignRequest(text="hello")


class TestVoiceCloneRequest:
    def test_task_type_defaults_to_base(self):
        from api.schemas import VoiceCloneRequest
        req = VoiceCloneRequest(text="hello", ref_audio="data:audio/wav;base64,abc")
        assert req.task_type == "Base"

    def test_ref_audio_required(self):
        from api.schemas import VoiceCloneRequest
        with pytest.raises(ValidationError):
            VoiceCloneRequest(text="hello")


class TestTTSRequest:
    def test_model_family_defaults_to_qwen(self):
        from api.schemas import TTSRequest
        assert TTSRequest(text="hello").model_family == "qwen"

    def test_accepts_fish_and_voxtral(self):
        from api.schemas import TTSRequest
        assert TTSRequest(text="hello", model_family="fish").model_family == "fish"
        assert TTSRequest(text="hello", model_family="voxtral").model_family == "voxtral"

    def test_rejects_unknown_model_family(self):
        from api.schemas import TTSRequest
        with pytest.raises(ValidationError):
            TTSRequest(text="hello", model_family="unknown")

    def test_num_candidates_defaults_to_one(self):
        from api.schemas import TTSRequest
        assert TTSRequest(text="hello").num_candidates == 1

    def test_optional_fields_default_to_none(self):
        from api.schemas import TTSRequest
        req = TTSRequest(text="hello")
        for field in ("voice", "ref_audio", "instructions", "task_type", "model_id"):
            assert getattr(req, field) is None


class TestS2ProRequestRemoved:
    def test_does_not_exist(self):
        import api.schemas as schemas
        assert not hasattr(schemas, "S2ProRequest")
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/unit/test_schemas.py -v 2>&1 | head -30`

Expected: Multiple FAIL — `CustomVoiceRequest` has `speaker`, not `voice`; `S2ProRequest` still exists; etc.

- [ ] **Step 3: Rewrite `api/schemas.py`**

```python
"""Pydantic schemas for the FastAPI service."""

from __future__ import annotations

from typing import List, Literal, Optional

from pydantic import BaseModel, Field, validator


class ErrorResponse(BaseModel):
    error: str = Field(..., description="Error type")
    message: str = Field(..., description="Human-readable message")
    detail: Optional[str] = Field(default=None)


class AudioResponse(BaseModel):
    audio_base64: str = Field(..., description="Base64-encoded WAV")
    sample_rate: int = Field(..., description="Sample rate in Hz")
    duration_seconds: float = Field(..., description="Audio duration in seconds")
    format: Literal["wav"] = Field(default="wav")
    model: str = Field(..., description="Model family used (qwen/fish/voxtral)")
    job_id: Optional[str] = Field(default=None)
    download_url: Optional[str] = Field(default=None)


class VoiceCloneRequest(BaseModel):
    """Qwen3-TTS Base model: voice cloning from a reference audio file."""

    text: str = Field(..., min_length=1, max_length=5000)
    language: Optional[str] = Field(default=None, description="e.g. English, Chinese, Auto")
    ref_audio: str = Field(..., description="Reference audio: path, URL, or base64 data URI")
    ref_text: Optional[str] = Field(default=None, description="Transcript of ref_audio")
    task_type: Literal["Base"] = Field(default="Base")
    num_candidates: int = Field(default=1, ge=1, le=12)
    return_all_candidates: bool = Field(default=False)

    @validator("language", pre=True)
    def _map_short_language_codes(cls, v: Optional[str]) -> Optional[str]:
        if not v:
            return v
        mapping = {
            "en": "english", "eng": "english",
            "zh": "chinese", "cn": "chinese",
            "ja": "japanese", "jp": "japanese",
            "ko": "korean", "kr": "korean",
        }
        return mapping.get(str(v).strip().lower(), v)


class CustomVoiceRequest(BaseModel):
    """Qwen3-TTS CustomVoice: predefined voice with optional style instruction."""

    text: str = Field(..., min_length=1, max_length=5000)
    language: Optional[str] = Field(default=None)
    voice: str = Field(..., description="CustomVoice name, e.g. Vivian, Ryan")
    instructions: Optional[str] = Field(default=None, description="Optional style instruction")


class VoiceDesignRequest(BaseModel):
    """Qwen3-TTS VoiceDesign: voice described in natural language."""

    text: str = Field(..., min_length=1, max_length=5000)
    language: Optional[str] = Field(default=None)
    instructions: str = Field(..., min_length=1, description="Natural-language voice description")


class TTSRequest(BaseModel):
    """Unified TTS request stored in the Redis job queue.

    All fields except model_family/model_id/num_candidates/return_all_candidates
    map directly to vllm-omni's /v1/audio/speech parameters.
    """

    text: str = Field(..., min_length=1, max_length=5000)
    model_family: Literal["qwen", "fish", "voxtral"] = Field(
        default="qwen",
        description="Determines GPU eligibility (VRAM requirement)",
    )
    model_id: Optional[str] = Field(default=None, description="Override default model for family")
    task_type: Optional[Literal["CustomVoice", "VoiceDesign", "Base"]] = Field(default=None)
    voice: Optional[str] = Field(default=None)
    language: Optional[str] = Field(default=None)
    instructions: Optional[str] = Field(default=None)
    ref_audio: Optional[str] = Field(default=None)
    ref_text: Optional[str] = Field(default=None)
    num_candidates: int = Field(default=1, ge=1, le=12)
    return_all_candidates: bool = Field(default=False)


class CandidateScore(BaseModel):
    candidate_id: int
    accuracy_score: float = Field(..., description="1 - WER (0–1, higher is better)")
    word_error_rate: float
    char_error_rate: float
    duration_seconds: float
    reference_words: int
    transcribed_words: int
    transcribed_text: str
    tts_gpu: int
    stt_device: str


class EnhancedAudioResponse(AudioResponse):
    best_candidate_score: Optional[CandidateScore] = Field(default=None)
    all_candidates: Optional[List[CandidateScore]] = Field(default=None)
    num_candidates_generated: Optional[int] = Field(default=None)


class JobSubmitResponse(BaseModel):
    job_id: str
    status_url: str
    audio_url: str
    status: Literal["queued"] = Field(default="queued")
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/unit/test_schemas.py -v`

Expected: All PASS

- [ ] **Step 5: Commit**

```bash
git add api/schemas.py tests/unit/test_schemas.py
git commit -m "feat: update schemas (remove S2ProRequest, rename fields, add TTSRequest)"
```

---

## Task 7: Add `_generate_with_vllm` and replace `job_worker_loop` in `api/main.py`

**Files:**
- Modify: `api/main.py`

**Context:** `api/main.py` is ~2400 lines. This task adds two new functions and replaces the body of `job_worker_loop`. The old inference singletons are left untouched here — they are removed in Task 9 after verifying this works.

**Important:** `soundfile` is already available in the environment (installed as part of the audio stack). `base64` and `os` are already imported in `api/main.py`.

- [ ] **Step 1: Add `_model_id_for_job` helper**

In `api/main.py`, find the line `def _generate_candidate(` (around line 304). Insert the following two functions **immediately before** it:

```python
def _model_id_for_job(job: Dict[str, Any]) -> str:
    """Return the vllm-omni model ID for a job payload.

    Prefers an explicit ``job["model_id"]``; falls back to the configured
    default for the job's ``model_family``.
    """
    if job.get("model_id"):
        return str(job["model_id"])
    family = str(job.get("model_family", "qwen")).lower()
    return {
        "qwen": settings.qwen_model_id,
        "fish": settings.fish_model_id,
        "voxtral": settings.voxtral_model_id,
    }.get(family, settings.qwen_model_id)


def _generate_with_vllm(
    job: Dict[str, Any],
    job_id: str,
    candidate_id: int,
    device: str,
) -> Optional[Dict[str, Any]]:
    """Generate audio for one candidate via vllm-omni subprocess, persist WAV.

    Spawns a VllmProcess for the job's model, generates audio, terminates the
    subprocess, writes the WAV to disk, and returns a candidate result dict.

    The returned dict has the same shape as the old ``_generate_candidate``
    result so all downstream Redis and scoring code is unchanged:
        candidate_id, wav_path, audio_base64, sample_rate, duration_seconds,
        tts_gpu, timings.

    Returns None on any failure (error is logged).
    """
    import io

    import soundfile as sf

    from inference.vllm_process import VllmProcess

    gpu_id = int(device.split(":")[-1]) if ":" in device else 0
    port = settings.vllm_port_base + gpu_id
    model_id = _model_id_for_job(job)

    gen_start = time.time()
    try:
        with VllmProcess(
            model_id=model_id,
            gpu_id=gpu_id,
            port=port,
            gpu_memory_utilization=settings.vllm_gpu_memory_utilization,
            startup_timeout=settings.vllm_startup_timeout_seconds,
        ) as proc:
            audio_bytes = proc.generate(
                input=job.get("text"),
                voice=job.get("voice"),
                task_type=job.get("task_type"),
                language=job.get("language"),
                instructions=job.get("instructions"),
                ref_audio=job.get("ref_audio"),
                ref_text=job.get("ref_text"),
            )
    except Exception:
        logger.exception(
            "vllm generation failed",
            extra={"job_id": job_id, "candidate_id": candidate_id, "device": device},
        )
        return None
    gen_end = time.time()

    candidate_dir = os.path.join(settings.output_dir, job_id, "candidates")
    os.makedirs(candidate_dir, exist_ok=True)
    wav_path = os.path.join(candidate_dir, f"candidate_{candidate_id}.wav")
    with open(wav_path, "wb") as f:
        f.write(audio_bytes)
        try:
            f.flush()
            os.fsync(f.fileno())
        except Exception:
            logger.warning("Failed to fsync candidate file", extra={"wav_path": wav_path})

    with sf.SoundFile(io.BytesIO(audio_bytes)) as sf_file:
        sample_rate = sf_file.samplerate
        duration = len(sf_file) / sample_rate

    audio_b64 = base64.b64encode(audio_bytes).decode("utf-8")
    logger.info(
        "candidate_saved",
        extra={"job_id": job_id, "candidate_id": candidate_id, "wav_path": wav_path},
    )
    return {
        "candidate_id": candidate_id,
        "wav_path": wav_path,
        "audio_base64": audio_b64,
        "sample_rate": sample_rate,
        "duration_seconds": duration,
        "tts_gpu": gpu_id,
        "timings": {
            "generation_start": gen_start,
            "generation_end": gen_end,
            "generation_duration": round(gen_end - gen_start, 3),
        },
    }
```

- [ ] **Step 2: Update `_score_candidates` to accept an optional pre-loaded transcriber**

Find `def _score_candidates(job_id: str, candidates:` (around line 447). Replace just the function signature and the transcriber-initialisation block (leave the rest of the function body untouched):

Old (replace only these lines):
```python
def _score_candidates(job_id: str, candidates: list[Dict[str, Any]], job: Dict[str, Any]) -> None:
    stt_devices = settings.whisper_device_list()
    transcribers: Dict[str, WhisperTranscriberHF] = {}
    for dev in stt_devices:
        try:
            transcribers[dev] = WhisperTranscriberHF(settings.whisper_model_id(), device=dev)
        except Exception:
            logger.exception("Failed to init transcriber on %s", dev)
```

New:
```python
def _score_candidates(
    job_id: str,
    candidates: list[Dict[str, Any]],
    job: Dict[str, Any],
    transcriber: Optional["WhisperTranscriberHF"] = None,
) -> None:
    stt_devices = settings.whisper_device_list()
    transcribers: Dict[str, Any] = {}
    if transcriber is not None:
        # Re-use the per-worker persistent Whisper instance for all devices.
        for dev in stt_devices:
            transcribers[dev] = transcriber
    else:
        for dev in stt_devices:
            try:
                transcribers[dev] = WhisperTranscriberHF(settings.whisper_model_id(), device=dev)
            except Exception:
                logger.exception("Failed to init transcriber on %s", dev)
```

- [ ] **Step 3: Replace `job_worker_loop` body**

Find `def job_worker_loop(worker_id: int, device: str, queue_name: str) -> None:` (around line 802).

Replace the **entire function** (signature through the end of the `while True` block) with:

```python
def job_worker_loop(worker_id: int, device: str, queue_name: str) -> None:
    """Background worker loop — one process per GPU.

    Three task types arrive on the per-GPU Redis queue:

    - ``candidate``: spawn vllm-omni subprocess, generate one audio sample,
      terminate subprocess (VRAM freed), record result to Redis.
    - ``score``: transcribe candidates with Whisper (loaded on first use,
      kept loaded until queue drains), rank, finalise job in Redis.
    - ``job``: single-shot generation (no multi-candidate flow).

    Whisper is unloaded when blpop times out (queue drained), so VRAM is
    free between jobs.

    Args:
        worker_id: Integer ID for logging.
        device: CUDA device string, e.g. ``"cuda:0"``.
        queue_name: Redis list key this worker pops from.
    """
    gpu_id = int(device.split(":")[-1]) if ":" in device else 0
    whisper: Optional["WhisperTranscriberHF"] = None

    logger.info(
        "Starting job worker loop",
        extra={"worker_id": worker_id, "device": device, "queue": queue_name},
    )

    while True:
        try:
            item = redis_client.blpop([queue_name, "tts:commands"], timeout=5)

            if not item:
                # Queue drained — unload Whisper to release VRAM.
                if whisper is not None:
                    try:
                        whisper.unload()
                        logger.info("Unloaded Whisper on idle", extra={"device": device})
                    except Exception:
                        logger.warning("Failed to unload Whisper", exc_info=True)
                    whisper = None
                continue

            queue_received, payload_json = item

            # ── Command queue ──────────────────────────────────────────────
            if queue_received == "tts:commands":
                try:
                    cmd = json.loads(payload_json)
                    cmd_id = cmd.get("cmd_id")
                    if cmd_id:
                        # No persistent models to unload in the new architecture.
                        redis_client.hset(
                            f"tts:cmd:{cmd_id}",
                            mapping={
                                "status": "done",
                                "result": "[]",
                                "completed_at": str(time.time()),
                            },
                        )
                        redis_client.expire(f"tts:cmd:{cmd_id}", 60)
                except Exception:
                    logger.exception("Failed to process command")
                continue

            # ── Job queue ──────────────────────────────────────────────────
            job = json.loads(payload_json)
            job_id = job.get("job_id")
            if not job_id:
                continue

            task_type = job.get("task", "job")

            # ── Candidate: generate one audio sample ───────────────────────
            if task_type == "candidate":
                candidate_id = int(job.get("candidate_id", 0))
                num_candidates = int(job.get("num_candidates", 1))
                assigned_gpu = str(job.get("assigned_gpu") or device)
                rescue_count = int(job.get("rescue_count", 0) or 0)

                _set_candidate_state(
                    job_id,
                    candidate_id,
                    status="started",
                    assigned_gpu=assigned_gpu,
                    updated_at=time.time(),
                    rescue_count=rescue_count,
                )

                result = _generate_with_vllm(job, job_id, candidate_id, device)
                terminal_state = "completed" if result else "failed"
                was_first = bool(
                    redis_client.hsetnx(
                        _job_candidate_terminal_key(job_id),
                        str(candidate_id),
                        terminal_state,
                    )
                )
                if was_first:
                    if result:
                        redis_client.rpush(
                            f"tts:job:{job_id}:candidates", json.dumps(result)
                        )
                        redis_client.hincrby(f"tts:job:{job_id}", "completed_candidates", 1)
                    else:
                        redis_client.hincrby(f"tts:job:{job_id}", "failed_candidates", 1)

                    settled = redis_client.hincrby(
                        f"tts:job:{job_id}", "settled_candidates", 1
                    )
                    _set_candidate_state(
                        job_id,
                        candidate_id,
                        status=terminal_state,
                        assigned_gpu=assigned_gpu,
                        updated_at=time.time(),
                        rescue_count=rescue_count,
                    )
                    if settled >= num_candidates:
                        _maybe_enqueue_score_task(job_id, num_candidates)
                continue

            # ── Score: transcribe + rank + finalise ────────────────────────
            if task_type == "score":
                job_payload = _load_job_payload(job_id)
                if not job_payload:
                    redis_client.hset(
                        f"tts:job:{job_id}",
                        mapping={"status": "failed", "error": "missing_job_payload"},
                    )
                    continue

                cand_items = redis_client.lrange(f"tts:job:{job_id}:candidates", 0, -1)
                candidates = []
                for raw in cand_items:
                    try:
                        candidates.append(json.loads(raw))
                    except Exception:
                        logger.warning(
                            "Failed to decode candidate", extra={"job_id": job_id}
                        )

                if not candidates:
                    redis_client.hset(
                        f"tts:job:{job_id}",
                        mapping={"status": "failed", "error": "no_candidates"},
                    )
                    continue

                # Load Whisper on first use; keep it for subsequent score tasks.
                if whisper is None and WhisperTranscriberHF is not None:
                    try:
                        whisper = WhisperTranscriberHF(
                            settings.whisper_model_id(), device=device
                        )
                        logger.info("Loaded Whisper for scoring", extra={"device": device})
                    except Exception:
                        logger.exception("Failed to load Whisper on %s", device)

                try:
                    _score_candidates(
                        job_id=job_id,
                        candidates=candidates,
                        job=job_payload,
                        transcriber=whisper,
                    )
                except Exception as exc:
                    logger.exception("Scoring failed", extra={"job_id": job_id})
                    redis_client.hset(
                        f"tts:job:{job_id}",
                        mapping={"status": "failed", "error": str(exc)[:200]},
                    )
                continue

            # ── Single-shot job: one candidate, no scoring ─────────────────
            key = f"tts:job:{job_id}"
            redis_client.hset(
                key, mapping={"status": "running", "started_at": str(time.time())}
            )
            result = _generate_with_vllm(job, job_id, candidate_id=0, device=device)
            if result:
                redis_client.hset(
                    key,
                    mapping={
                        "status": "done",
                        "audio_base64": result["audio_base64"],
                        "sample_rate": str(result["sample_rate"]),
                        "duration_seconds": str(result["duration_seconds"]),
                        "wav_path": result["wav_path"],
                        "download_url": f"/v1/audio/{job_id}.wav",
                        "completed_at": str(time.time()),
                        "timings": json.dumps(result["timings"]),
                    },
                )
            else:
                redis_client.hset(
                    key,
                    mapping={"status": "failed", "error": "generation_failed"},
                )

        except Exception:
            logger.exception("Worker loop error; continuing")
            time.sleep(1)
```

- [ ] **Step 4: Verify `api/main.py` imports cleanly**

Run: `python -c "import api.main; print('OK')" 2>&1 | head -10`

Expected: `OK`

- [ ] **Step 5: Run all unit tests**

Run: `pytest tests/unit/ -v --ignore=tests/unit/test_whisper_transcriber.py 2>&1 | tail -20`

Expected: All PASS

- [ ] **Step 6: Commit**

```bash
git add api/main.py
git commit -m "feat: add _generate_with_vllm helper and replace job_worker_loop with vllm-omni-based implementation"
```

---

## Task 8: Update `worker/run.py` Dispatcher

**Files:**
- Modify: `worker/run.py`

**What changes:**
1. `main()`: derive GPU device list from `settings.gpu_pool_parsed()` instead of `settings.tts_gpu_list()`.
2. `dispatch_loop()`: use `GpuPool.assign_candidates()` for model-aware routing; fall back gracefully if no eligible GPUs.

- [ ] **Step 1: Add the `GpuPool` import**

At the top of `worker/run.py`, after the existing imports, add:

```python
from inference.gpu_pool import GpuPool
```

- [ ] **Step 2: Replace GPU device derivation in `main()`**

Find in `main()`:
```python
    gpu_ids = settings.tts_gpu_list()
    if not gpu_ids:
        logger.warning("No TTS_GPUS configured; all workers will run on CPU.")
        gpu_devices = ["cpu"]
    else:
        gpu_devices = [f"cuda:{gid}" for gid in gpu_ids]
```

Replace with:
```python
    try:
        gpu_devices = [f"cuda:{gid}" for gid, _ in settings.gpu_pool_parsed()]
    except Exception:
        logger.warning("Failed to parse GPU_POOL; falling back to CPU.", exc_info=True)
        gpu_devices = []
    if not gpu_devices:
        logger.warning("GPU_POOL produced empty list; running on CPU.")
        gpu_devices = ["cpu"]
```

- [ ] **Step 3: Replace candidate dispatch logic in `dispatch_loop()`**

Find (the full block from the `# Single-candidate` comment through the last `redis_client.rpush`):
```python
            # Single-candidate or non-base model: send directly to a GPU queue.
            if model_type != "base" or num_candidates <= 1:
                target = gpu_devices[0] if gpu_devices else "cpu"
                job["task"] = "job"
                redis_client.rpush(f"tts:jobs:{target}", json.dumps(job))
                continue

            # Multi-candidate voice clone: distribute candidates across GPUs.
            for i in range(num_candidates):
                target = gpu_devices[i % len(gpu_devices)] if gpu_devices else "cpu"
                cand_job = dict(job)
                cand_job["task"] = "candidate"
                cand_job["candidate_id"] = i
                cand_job["assigned_gpu"] = target
                cand_job["rescue_count"] = 0
                _set_candidate_state(
                    job_id,
                    i,
                    status="queued",
                    assigned_gpu=target,
                    updated_at=time.time(),
                    rescue_count=0,
                )
                redis_client.rpush(f"tts:jobs:{target}", json.dumps(cand_job))
            _rescue_stale_candidate_jobs(gpu_devices)
```

Replace with:
```python
            # model_family drives GPU eligibility; fall back to model_type for compat.
            model_family = str(job.get("model_family") or model_type).lower()

            gpu_pool = GpuPool(
                settings.gpu_pool,
                {
                    "qwen": settings.qwen_min_vram_gb,
                    "fish": settings.fish_min_vram_gb,
                    "voxtral": settings.voxtral_min_vram_gb,
                },
            )

            # Single-candidate: straight to a GPU queue, no tracking overhead.
            if num_candidates <= 1:
                try:
                    eligible = gpu_pool.eligible_gpus(model_family)
                    target = f"cuda:{eligible[0]}" if eligible else (gpu_devices[0] if gpu_devices else "cpu")
                except Exception:
                    target = gpu_devices[0] if gpu_devices else "cpu"
                job["task"] = "job"
                redis_client.rpush(f"tts:jobs:{target}", json.dumps(job))
                continue

            # Multi-candidate: round-robin across eligible GPUs.
            try:
                assignments = gpu_pool.assign_candidates(num_candidates, model_family)
            except ValueError:
                logger.error(
                    "No eligible GPUs for %s; sending all candidates to first GPU",
                    model_family,
                    extra={"job_id": job_id},
                )
                fallback_id = int(gpu_devices[0].split(":")[-1]) if gpu_devices and ":" in gpu_devices[0] else 0
                assignments = [(fallback_id, i) for i in range(num_candidates)]

            for gpu_id, i in assignments:
                target = f"cuda:{gpu_id}"
                cand_job = dict(job)
                cand_job["task"] = "candidate"
                cand_job["candidate_id"] = i
                cand_job["assigned_gpu"] = target
                cand_job["rescue_count"] = 0
                _set_candidate_state(
                    job_id,
                    i,
                    status="queued",
                    assigned_gpu=target,
                    updated_at=time.time(),
                    rescue_count=0,
                )
                redis_client.rpush(f"tts:jobs:{target}", json.dumps(cand_job))
            _rescue_stale_candidate_jobs(gpu_devices)
```

- [ ] **Step 4: Verify `worker/run.py` imports cleanly**

Run: `python -c "import worker.run; print('OK')" 2>&1 | head -10`

Expected: `OK`

- [ ] **Step 5: Run unit tests**

Run: `pytest tests/unit/ -v --ignore=tests/unit/test_whisper_transcriber.py 2>&1 | tail -15`

Expected: All PASS

- [ ] **Step 6: Commit**

```bash
git add worker/run.py
git commit -m "feat: update worker dispatcher to use GpuPool for model-aware GPU assignment"
```

---

## Task 9: Remove Inference Singletons from `api/main.py`

**Files:**
- Modify: `api/main.py`

These singletons are no longer called after Tasks 7 and 8. Remove them to eliminate dead code and broken imports.

- [ ] **Step 1: Remove the `QwenTTSServer` import block**

Find and delete this entire block (around line 38):
```python
try:
    from inference.qwen_tts_service import QwenTTSServer, wav_to_wav_bytes
except Exception:
    QwenTTSServer = None

    def wav_to_wav_bytes(wav, sr):
        raise RuntimeError("inference.qwen_tts_service not available in this environment")
```

- [ ] **Step 2: Remove `_SYNC_TTS_SERVER` and `get_sync_tts_server()`**

Delete these lines (around line 82):
```python
_SYNC_TTS_SERVER: Optional[QwenTTSServer] = None
_SYNC_TTS_LOCK = threading.Lock()
```

Delete the entire `get_sync_tts_server()` function.

- [ ] **Step 3: Remove Fish Audio singletons and helpers**

Delete these lines (around line 207):
```python
_SYNC_FISH_AUDIO_SERVICE = None
_SYNC_FISH_AUDIO_LOCK = threading.Lock()
_FISH_IDLE_WATCHER_THREAD: Optional[threading.Thread] = None
_FISH_IDLE_WATCHER_STOP = threading.Event()
```

Delete the entire `get_sync_fish_audio_service()` function.

Delete the entire `get_fish_sglang_controller()` function.

Delete `_FISH_SGLANG_CONTROLLER = None` and `_FISH_SGLANG_CONTROLLER_LOCK = threading.Lock()`.

Delete the entire `_run_fish_idle_watcher()` function.

- [ ] **Step 4: Remove `_generate_candidate()`**

Delete the entire `_generate_candidate()` function (around line 304). It has been replaced by `_generate_with_vllm()`.

- [ ] **Step 5: Remove the S2Pro endpoint and its imports**

Search for `s2-pro` and `S2ProRequest` in `api/main.py`. Delete:
- The `S2ProRequest` import at the top of the file
- The `POST /v1/tts/s2-pro/sync` endpoint handler function

- [ ] **Step 6: Remove Fish references from startup/shutdown event handlers**

Search for `_FISH_IDLE_WATCHER_THREAD` and `_SYNC_FISH_AUDIO_SERVICE` in the `@app.on_event("startup")` and `@app.on_event("shutdown")` handlers. Delete those lines.

- [ ] **Step 7: Update `_select_score_device()`**

Find:
```python
def _select_score_device() -> str:
    """Select a GPU device for scoring tasks."""
    gpu_ids = settings.tts_gpu_list()
    preferred = settings.tts_preferred_gpu
    if preferred in gpu_ids:
        return f"cuda:{preferred}"
    if gpu_ids:
        return f"cuda:{gpu_ids[0]}"
    return "cpu"
```

Replace with:
```python
def _select_score_device() -> str:
    """Select the first GPU in the pool for scoring tasks."""
    try:
        entries = settings.gpu_pool_parsed()
        if entries:
            return f"cuda:{entries[0][0]}"
    except Exception:
        pass
    return "cpu"
```

- [ ] **Step 8: Verify clean import**

Run: `python -c "import api.main; print('OK')" 2>&1 | head -10`

Expected: `OK`

- [ ] **Step 9: Run all unit tests**

Run: `pytest tests/unit/ -v --ignore=tests/unit/test_whisper_transcriber.py 2>&1 | tail -20`

Expected: All PASS

- [ ] **Step 10: Commit**

```bash
git add api/main.py
git commit -m "refactor: remove QwenTTSServer, FishAudioService, SGLangController singletons and S2Pro endpoint from api/main.py"
```

---

## Task 10: Update `Dockerfile`

**Files:**
- Modify: `Dockerfile`

- [ ] **Step 1: Read the current Dockerfile to find exact line ranges**

Run: `cat -n Dockerfile`

Note the line numbers for:
- The `git clone https://github.com/QwenLM/Qwen3-TTS.git` block
- The apt package line that includes `docker.io`
- Any image name label

- [ ] **Step 2: Remove the Qwen3-TTS git install**

Find and delete the `RUN` block that contains:
```dockerfile
RUN git clone https://github.com/QwenLM/Qwen3-TTS.git \
    && pip install -e ./Qwen3-TTS --no-deps
```

(The exact phrasing may differ slightly — delete the entire RUN command that clones and pip-installs Qwen3-TTS.)

- [ ] **Step 3: Remove `docker.io` from the apt-get install block**

Find the line inside the `apt-get install` block:
```dockerfile
    docker.io \
```
Delete that line (and adjust the trailing backslash on the preceding line if needed).

- [ ] **Step 4: Add vllm-omni install as a separate layer**

After the main `pip install -r requirements.txt` RUN block, add:

```dockerfile
RUN pip install --no-cache-dir vllm-omni
```

Placing it as a separate layer allows Docker to cache it independently of requirements changes.

- [ ] **Step 5: Verify Dockerfile syntax**

Run: `docker buildx build --check . 2>&1 | head -20`

Expected: No syntax errors (actual build is not required here)

- [ ] **Step 6: Commit**

```bash
git add Dockerfile
git commit -m "feat: update Dockerfile — remove Qwen3-TTS git install and docker.io, add vllm-omni"
```

---

## Task 11: Update `compose.yaml`

**Files:**
- Modify: `compose.yaml`

- [ ] **Step 1: Remove `fish-sglang` and `fish-sglang-base` service blocks**

Delete the entire `fish-sglang:` block and the `fish-sglang-base:` block (if present) from the `services:` section.

- [ ] **Step 2: Rename service keys and container/image names**

Make these replacements throughout `compose.yaml`:

| Old | New |
|-----|-----|
| `echofleet-qwen3-tts:` (service key) | `echofleet-tts:` |
| `container_name: echofleet-qwen3-tts` | `container_name: echofleet-tts` |
| `image: echofleet-qwen3-tts` | `image: echofleet-tts` |
| `echofleet-qwen3-worker:` (service key) | `echofleet-worker:` |
| `container_name: echofleet-qwen3-worker` | `container_name: echofleet-worker` |
| `echofleet-qwen3-ui:` (service key) | `echofleet-ui:` |
| `container_name: echofleet-qwen3-ui` | `container_name: echofleet-ui` |
| `echofleet-qwen3-network` | `echofleet-network` |
| `echofleet-qwen3-redis-data` | `echofleet-redis-data` |

- [ ] **Step 3: Remove volumes no longer needed**

From the `echofleet-tts` service's `volumes:` section, delete:
- `/var/run/docker.sock:/var/run/docker.sock`
- The `fish-tts` volume mount line

From the top-level `volumes:` section, delete the `fish-tts:` entry.

- [ ] **Step 4: Remove obsolete environment variables**

From the `echofleet-tts` and `echofleet-worker` `environment:` sections, delete these keys:
```
SGLANG_BASE_URL
FISH_REF_AUDIO_DIR
FISH_IDLE_UNLOAD_SECONDS
FISH_STARTUP_TIMEOUT_SECONDS
FISH_STOP_TIMEOUT_SECONDS
FISH_SGLANG_CONTAINER_NAME
HOST_DOCKER_GID
TTS_GPUS
TTS_PREFERRED_GPU
TTS_GPU_CAPACITY
ENABLE_BASE_MODEL
ENABLE_CUSTOM_VOICE
ENABLE_VOICE_DESIGN
```

- [ ] **Step 5: Add new environment variables**

Add the following to the `environment:` section of both `echofleet-tts` and `echofleet-worker`:

```yaml
      GPU_POOL: "0:16,1:16,2:8,3:8"
      QWEN_MIN_VRAM_GB: "8"
      FISH_MIN_VRAM_GB: "12"
      VOXTRAL_MIN_VRAM_GB: "12"
      QWEN_MODEL_ID: "Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice"
      QWEN_VOICE_DESIGN_MODEL_ID: "Qwen/Qwen3-TTS-12Hz-1.7B-VoiceDesign"
      QWEN_BASE_MODEL_ID: "Qwen/Qwen3-TTS-12Hz-1.7B-Base"
      FISH_MODEL_ID: "fishaudio/s2-pro"
      VOXTRAL_MODEL_ID: "mistralai/Voxtral-4B-TTS-2603"
      VLLM_PORT_BASE: "8100"
      VLLM_STARTUP_TIMEOUT_SECONDS: "180"
      VLLM_GPU_MEMORY_UTILIZATION: "0.90"
      TTS_WORKER_CONCURRENCY: "4"
```

- [ ] **Step 6: Remove `fish-sglang` from `depends_on`**

The `echofleet-tts` service previously had `fish-sglang` in its `depends_on:` block. Delete that entry.

- [ ] **Step 7: Verify compose syntax**

Run: `docker compose config 2>&1 | head -30`

Expected: Valid YAML output showing services: `redis`, `echofleet-tts`, `echofleet-worker`, `echofleet-ui`

- [ ] **Step 8: Commit**

```bash
git add compose.yaml
git commit -m "feat: update compose.yaml — remove fish-sglang, rename services, add vllm-omni env vars"
```

---

## Task 12: Global Rename + Update `AGENTS.md`

**Files:**
- Modify: `AGENTS.md`
- Modify: any remaining files referencing `echofleet-qwen3` or old field names

- [ ] **Step 1: Find remaining old-name references**

Run: `grep -rn "echofleet-qwen3\|EchoFleet-Qwen3\|tts_gpu_list\|tts_preferred_gpu\|fish_idle" --include="*.py" --include="*.md" --include="*.yaml" --include="*.txt" . | grep -v ".worktrees" | grep -v "__pycache__"`

Note all files returned.

- [ ] **Step 2: Fix any remaining `tts_gpu_list()` call sites**

If any `.py` file other than `config/settings.py` still calls `settings.tts_gpu_list()`, replace with `settings.gpu_pool_parsed()` and adjust the usage (the old method returned `[int, ...]`; the new one returns `[(int, int), ...]` — extract just the GPU IDs with `[gid for gid, _ in settings.gpu_pool_parsed()]`).

- [ ] **Step 3: Rewrite `AGENTS.md`**

```markdown
# AGENTS.md - EchoFleet-TTS

Quick reference for AI agents working on this codebase.

## Project Overview

EchoFleet-TTS is a multi-GPU TTS service supporting:
- **Qwen3-TTS** — CustomVoice, VoiceDesign, Base/voice-clone task types
- **Fish Audio S2 Pro**
- **Voxtral** (mistralai/Voxtral-4B-TTS-2603)

All models are served via short-lived **vllm-omni** subprocesses.
Each GPU worker spawns `vllm serve <model>` per candidate, generates audio,
terminates the process (VRAM freed), then scores with Whisper.

## Build & Run

```bash
# Install dependencies
pip install -r requirements.txt

# Docker build
docker build -t echofleet-tts .

# Docker Compose (full stack)
docker compose up

# API server only (dev)
python -m uvicorn api.main:app --host 0.0.0.0 --port 8000 --reload

# Worker only (dev)
python -m worker.run
```

## Lint & Format

```bash
black --line-length 100 .
isort --profile black .
mypy .
flake8 .
```

## Testing

```bash
# All unit tests (no GPU required)
pytest tests/unit/ -v

# Single file
pytest tests/unit/test_inference_gpu_pool.py -v
```

## Key Source Layout

```
api/
  main.py           FastAPI app; job submission, status, scoring,
                    _generate_with_vllm, job_worker_loop
  schemas.py        Pydantic models: TTSRequest, VoiceCloneRequest, etc.
config/
  settings.py       Env config (GPU_POOL, vllm_*, model IDs, Whisper, etc.)
inference/
  gpu_pool.py       GpuPool: eligible_gpus(model), assign_candidates(n, model)
  vllm_process.py   VllmProcess: spawn/generate/stop vllm-omni subprocess
  whisper_service.py  WhisperTranscriberHF
  accuracy_scorer.py  AccuracyScorer: WER/CER for candidate ranking
utils/
  gpu_pool.py       GpuLeasePool (thread-safe scheduler — legacy, unused by workers)
  audio_utils.py    Audio loading and format conversion
  text_chunker.py   Sentence-aware text splitting
  errors.py         TTSError hierarchy
worker/
  run.py            dispatch_loop + spawns job_worker_loop processes
ui/
  gradio_app.py     Gradio web UI (port 7860)
tests/
  unit/             Torch-free unit tests
  integration/      End-to-end tests (require running services)
```

## GPU Pool

`GPU_POOL=0:16,1:16,2:8,3:8` — comma-separated `gpu_id:vram_gb` pairs.

Per-model minimums (GB): `QWEN_MIN_VRAM_GB=8`, `FISH_MIN_VRAM_GB=12`, `VOXTRAL_MIN_VRAM_GB=12`.

## vllm-omni Port Assignment

Port = `VLLM_PORT_BASE` + `gpu_id` (default base: 8100).
cuda:0→8100, cuda:1→8101, cuda:2→8102, cuda:3→8103.
```

- [ ] **Step 4: Run the full unit test suite**

Run: `pytest tests/unit/ -v --ignore=tests/unit/test_whisper_transcriber.py --ignore=tests/unit/test_dockerfile_torch_stack.py 2>&1 | tail -30`

Expected: All PASS

- [ ] **Step 5: Final commit**

```bash
git add -A
git commit -m "chore: rename project to EchoFleet-TTS and update AGENTS.md"
```

---

## Execution Skill Recommendation

**Use `superpowers:subagent-driven-development`.**

Reasons:
1. **12 sequential tasks** — each one depends on the previous (settings before GpuPool, GpuPool before worker dispatch, etc.). Fresh context per task prevents stale type signatures and method names from bleeding across tasks.
2. **Two-stage review between tasks** — the subagent skill reviews output before proceeding, catching issues like a wrong method signature in Task 4 before Task 8 depends on it.
3. **Tasks 4 and 5 are independent** — `inference/gpu_pool.py` and `inference/vllm_process.py` have no shared state. The subagent skill can be asked to run them in parallel.
4. **Large target files** — `api/main.py` is 2400 lines. A fresh subagent for Tasks 7 and 9 avoids context accumulation errors when making surgical edits to a large file.

`superpowers:executing-plans` is also viable if you prefer a single inline session with explicit checkpoints — but the subagent approach is lower risk for a migration of this size.
