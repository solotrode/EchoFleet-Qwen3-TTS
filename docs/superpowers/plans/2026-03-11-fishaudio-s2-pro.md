# Fish Audio S2 Pro Integration Implementation Plan

> **For agentic workers:** REQUIRED: Use superpowers:subagent-driven-development (if subagents available) or superpowers:executing-plans to implement this plan. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add Fish Audio S2 Pro as a dedicated plain TTS endpoint without changing the existing Qwen3 voice-clone or voice-design contracts.

**Scope decision:** S2 Pro does **not** support the same feature set as EchoFleet's existing `voice-clone` and `voice-design` flows. Do not add `s2-pro` to those request schemas or route those endpoints to Fish Audio. Do not route `custom-voice` to S2 Pro either; that endpoint is speaker-preset based and does not map cleanly to Fish Audio.

**Architecture:** Add a new S2 Pro request schema, create a Fish Audio service wrapper, expose a dedicated synchronous endpoint, and keep the Redis worker / multi-candidate Qwen3 paths unchanged.

**Tech Stack:** Python, FastAPI, pydantic, existing Qwen3 infrastructure, Fish Audio library (exact package/API must be verified before implementation)

---

## File Structure

| File | Action | Purpose |
|------|--------|---------|
| `api/schemas.py` | Modify | Add S2 Pro request schema and update response docs |
| `config/settings.py` | Modify | Add S2 Pro config settings |
| `inference/fish_audio_service.py` | Create | Fish Audio S2 Pro service wrapper |
| `api/main.py` | Modify | Add S2 Pro singleton accessor and sync endpoint |
| `requirements.txt` | Modify | Add Fish Audio dependency after verifying package name |
| `tests/unit/test_schemas.py` | Create | Schema tests for S2 Pro request |
| `tests/unit/test_fish_audio_service.py` | Create | Service-layer tests |
| `tests/integration/test_s2_pro_endpoint.py` | Create | Endpoint tests with mocked service |

---

## Constraints and Non-Goals

- [ ] Do **not** add `model="s2-pro"` to `VoiceCloneRequest`.
- [ ] Do **not** add `model="s2-pro"` to `VoiceDesignRequest`.
- [ ] Do **not** route `/v1/tts/voice-clone`, `/v1/tts/voice-clone/sync`, `/v1/tts/voice-clone/submit`, `/v1/tts/voice-design`, or `/v1/tts/voice-design/sync` to Fish Audio.
- [ ] Do **not** route `/v1/tts/custom-voice` to Fish Audio.
- [ ] Do **not** modify the Redis worker job format in this plan.
- [ ] Keep S2 Pro sync-only for the first implementation pass.

Why: the current Qwen3 endpoints have feature-specific contracts and worker behavior that assume voice cloning, speaker presets, or natural-language voice design. S2 Pro does not match those capabilities, so forcing it into those surfaces would create invalid API behavior.

---

## Chunk 1: Schema and Configuration

### Task 1: Add a dedicated S2 Pro request schema

**Files:**
- Modify: `api/schemas.py`
- Create: `tests/unit/test_schemas.py`

- [ ] **Step 1: Write failing tests for the new schema**

```python
from api.schemas import S2ProRequest


def test_s2_pro_request_accepts_text_and_language():
    req = S2ProRequest(text="Hello world", language="english")
    assert req.text == "Hello world"
    assert req.language == "english"


def test_s2_pro_request_defaults_language_to_none():
    req = S2ProRequest(text="Hello world")
    assert req.language is None
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/unit/test_schemas.py -v`
Expected: FAIL because `S2ProRequest` does not exist yet

- [ ] **Step 3: Add the new schema**

In `api/schemas.py`, add:

```python
class S2ProRequest(BaseModel):
    """Request payload for Fish Audio S2 Pro plain TTS."""

    text: str = Field(..., min_length=1, max_length=5000)
    language: Optional[str] = Field(default=None, description="e.g. English, Chinese, or 'auto' for auto-detection (case-insensitive)")
```

Also update `AudioResponse.model` description to explicitly allow `s2-pro` (change from `str` to a `Literal` type with values `"base"`, `"custom-voice"`, `"voice-design"`, `"s2-pro"` or use `Union` if arbitrary strings are needed).

- [ ] **Step 4: Confirm existing feature schemas stay unchanged**

Do not add a `model` field to:
- `VoiceCloneRequest`
- `CustomVoiceRequest`
- `VoiceDesignRequest`

- [ ] **Step 5: Run tests to verify they pass**

Run: `pytest tests/unit/test_schemas.py -v`
Expected: PASS

---

### Task 2: Add S2 Pro configuration settings

**Files:**
- Modify: `config/settings.py`

- [ ] **Step 1: Add S2 Pro settings**

Add to `Settings.__init__` in `config/settings.py`:

```python
# Fish Audio S2 Pro configuration
self.s2_pro_model_dir: str = os.getenv("S2_PRO_MODEL_DIR", "/models/s2-pro")
self.s2_pro_repo_id: str = os.getenv("S2_PRO_REPO_ID", "fishaudio/s2-pro")
self.s2_pro_device: str = os.getenv("S2_PRO_DEVICE", f"cuda:{str(self.tts_preferred_gpu)}")
```

- [ ] **Step 2: Keep the config style consistent**

Match the existing `Settings` style:
- add plain attributes inside `__init__`
- avoid introducing Pydantic settings machinery
- preserve existing defaults and ordering as much as practical

- [ ] **Step 3: Add or update tests only if config tests already exist**

There are currently no config unit tests in this repo. Do not create a large new config test suite just for these fields.

---

## Chunk 2: Fish Audio Service Layer

### Task 3: Add a focused Fish Audio service wrapper

**Files:**
- Create: `inference/fish_audio_service.py`
- Create: `tests/unit/test_fish_audio_service.py`

- [ ] **Step 1: Write failing unit tests for initialization behavior**

```python
from config.settings import Settings
from inference.fish_audio_service import FishAudioService


def test_fish_audio_service_exposes_model_dir(tmp_path, monkeypatch):
    monkeypatch.setenv("S2_PRO_MODEL_DIR", str(tmp_path))
    settings = Settings()
    service = FishAudioService(settings)
    assert service.model_dir == str(tmp_path)
```

- [ ] **Step 2: Write the minimal implementation**

Create `inference/fish_audio_service.py`:

```python
"""Fish Audio S2 Pro TTS service."""

from __future__ import annotations

from pathlib import Path
from threading import Lock
from typing import Optional, Tuple

import numpy as np

from config.settings import Settings
from utils.logging import get_logger

logger = get_logger(__name__)


class FishAudioService:
    """Lazy-loading Fish Audio S2 Pro wrapper for plain TTS."""

    def __init__(self, settings: Settings) -> None:
        self._settings = settings
        self._lock = Lock()
        self._model = None
        self._model_dir = settings.s2_pro_model_dir
        self._device = settings.s2_pro_device

    @property
    def model_dir(self) -> str:
        return self._model_dir

    def generate(self, text: str, language: Optional[str] = None) -> Tuple[np.ndarray, int]:
        """Generate audio from text using Fish Audio S2 Pro.
        
        Args:
            text: Input text to synthesize.
            language: Optional language hint (e.g., "English", "Chinese", "auto").
        
        Returns:
            Tuple of (audio_array, sample_rate).
        
        Raises:
            NotImplementedError: When real Fish Audio inference is not yet integrated.
        """
        if self._model is None:
            self._load_model()
        raise NotImplementedError(
            "FishAudioService.generate() not yet implemented - awaiting Fish Audio package verification"
        )

    def cleanup_vram(self) -> None:
        """Aggressively clean up GPU memory after generation.
        
        Uses the same pattern as QwenTTSServer to ensure VRAM is released
        when switching between models.
        """
        import gc
        import torch

        if torch.cuda.is_available():
            for device_id in range(torch.cuda.device_count()):
                with torch.cuda.device(device_id):
                    torch.cuda.synchronize()
                    torch.cuda.empty_cache()
            for _ in range(3):
                gc.collect()
        self._model = None
        logger.info("Fish Audio S2 Pro VRAM cleanup completed")

    def _load_model(self) -> None:
        with self._lock:
            if self._model is not None:
                return

            model_path = Path(self._model_dir)
            if not model_path.exists():
                raise FileNotFoundError(
                    f"S2 Pro model not found at {model_path}. Download from {self._settings.s2_pro_repo_id}"
                )

            # TODO: replace with the verified Fish Audio loading API.
            self._model = object()
            logger.info("Fish Audio S2 Pro model loaded", extra={"path": str(model_path), "device": self._device})
```

- [ ] **Step 3: Keep the service scope narrow**

Do not add:
- voice cloning arguments
- reference audio handling
- voice design instructions
- speaker preset abstractions

Those capabilities are intentionally out of scope for S2 Pro in this plan.

- [ ] **Step 4: Run unit tests**

Run: `pytest tests/unit/test_fish_audio_service.py -v`
Expected: PASS for init-level tests; generation can remain stubbed

---

### Task 4: Verify and add the Fish Audio dependency

**Files:**
- Modify: `requirements.txt`

- [ ] **Step 1: Verify the exact install package and import path**

Do not guess. Confirm from Fish Audio documentation or package metadata whether the install target is something like:
- `fish-speech`
- `fish_audio`
- another package name entirely

- [ ] **Step 2: Add the dependency only after verification**

When verified, add it to `requirements.txt` with a version floor or exact pin that matches the repo's dependency policy.

- [ ] **Step 3: Record any unresolved packaging issue in the plan or follow-up note**

If the package cannot be installed cleanly on this stack yet, stop before implementation of real inference and document the blocker clearly.

---

## Chunk 3: API Integration

### Task 5: Add a dedicated Fish Audio singleton accessor

**Files:**
- Modify: `api/main.py`

- [ ] **Step 1: Add a sync-only service singleton**

Follow the existing `get_sync_tts_server()` pattern. Add:

```python
from config.settings import get_settings  # Ensure settings is imported at module level

_SYNC_FISH_AUDIO_SERVICE: Optional[FishAudioService] = None
_SYNC_FISH_AUDIO_LOCK = threading.Lock()


def get_sync_fish_audio_service() -> FishAudioService:
    global _SYNC_FISH_AUDIO_SERVICE
    if _SYNC_FISH_AUDIO_SERVICE is None:
        with _SYNC_FISH_AUDIO_LOCK:
            if _SYNC_FISH_AUDIO_SERVICE is None:
                from inference.fish_audio_service import FishAudioService

                _settings = get_settings()  # Use get_settings() helper to access settings
                _SYNC_FISH_AUDIO_SERVICE = FishAudioService(_settings)
                logger.info("Initialized sync Fish Audio service")
    return _SYNC_FISH_AUDIO_SERVICE
```

- [ ] **Step 2: Do not add a generic engine registry in this pass**

The current API is not organized around a shared engine interface across all routes, and introducing one here would create extra refactor risk without solving a real problem. Keep the integration local and explicit.

---

### Task 6: Add a dedicated S2 Pro synchronous endpoint

**Files:**
- Modify: `api/main.py`

- [ ] **Step 1: Add a new endpoint path**

Add a dedicated endpoint instead of overloading existing feature-specific routes:

```python
@app.post(
    "/v1/tts/s2-pro/sync",
    response_model=AudioResponse,
    responses={400: {"model": ErrorResponse}, 500: {"model": ErrorResponse}, 503: {"model": ErrorResponse}},
    tags=["TTS"],
    summary="Fish Audio S2 Pro plain TTS (sync)",
)
async def tts_s2_pro_sync(request: S2ProRequest) -> AudioResponse:
    ...
```

- [ ] **Step 2: Implement the endpoint using the Fish Audio service**

The endpoint should:
- call `get_sync_fish_audio_service()`
- call `service.generate(text=request.text, language=request.language)`
- convert the result with the existing WAV/base64 helpers
- persist the WAV using the same disk-write pattern used by other sync endpoints
- write a minimal Redis job record so the existing download endpoint can serve the file
- call `service.cleanup_vram()` after generation to release GPU memory before returning
- return `AudioResponse(..., model="s2-pro", ...)`

- [ ] **Step 3: Use correct error mapping**

Map errors as follows:
- `FileNotFoundError` from missing model files -> HTTP 503
- input validation issues -> HTTP 400
- generation failures -> HTTP 500

- [ ] **Step 4: Leave existing endpoints untouched**

Do not change behavior of:
- `tts_voice_clone`
- `tts_voice_clone_sync`
- `tts_voice_clone_submit`
- `tts_custom_voice`
- `tts_voice_design_sync`
- `tts_voice_design`

This is the core scope correction for this plan.

---

## Chunk 4: Testing and Validation

### Task 7: Add realistic schema and service unit tests

**Files:**
- Create: `tests/unit/test_schemas.py`
- Create: `tests/unit/test_fish_audio_service.py`

- [ ] **Step 1: Add schema tests**

Cover:
- `S2ProRequest` accepts `text`
- `S2ProRequest` accepts optional `language`
- empty `text` is rejected

- [ ] **Step 2: Add service tests**

Cover:
- constructor stores config-derived model path
- `_load_model()` raises `FileNotFoundError` when model dir is missing
- generation remains intentionally unimplemented until the real package API is verified

- [ ] **Step 3: Keep tests cheap and local**

Do not require:
- downloaded S2 Pro model files
- CUDA
- real Fish Audio package execution

Use monkeypatching and temporary directories instead.

---

### Task 8: Add endpoint tests that match the repo's actual test harness

**Files:**
- Create: `tests/integration/test_s2_pro_endpoint.py`

- [ ] **Step 1: Build tests with `fastapi.testclient.TestClient` directly**

Do not assume a `loaded_client` fixture exists. The current `tests/conftest.py` only sets up `sys.path`.

- [ ] **Step 2: Mock the Fish Audio service and Redis writes**

Use monkeypatching to:
- replace `get_sync_fish_audio_service()` with a fake service that returns a small deterministic waveform
- replace `redis_client` with a lightweight fake object that implements the methods used by the sync endpoint
- optionally redirect `settings.output_dir` to a temp directory

- [ ] **Step 3: Add integration tests for the real supported behavior**

Example cases:

```python
def test_s2_pro_sync_returns_audio(monkeypatch, tmp_path):
    ...
    response = client.post(
        "/v1/tts/s2-pro/sync",
        json={"text": "Hello world", "language": "english"},
    )
    assert response.status_code == 200
    assert response.json()["model"] == "s2-pro"


def test_s2_pro_sync_returns_503_when_model_missing(monkeypatch):
    ...
    response = client.post("/v1/tts/s2-pro/sync", json={"text": "Hello world"})
    assert response.status_code == 503
```

- [ ] **Step 4: Do not add tests for unsupported routes**

Do not add tests asserting that `voice-clone` or `voice-design` accept `model="s2-pro"`. That would encode the wrong product behavior.

---

### Task 9: Final verification

- [ ] **Step 1: Run focused tests first**

Run:
- `pytest tests/unit/test_schemas.py -v`
- `pytest tests/unit/test_fish_audio_service.py -v`
- `pytest tests/integration/test_s2_pro_endpoint.py -v`

Expected: PASS without needing downloaded S2 Pro weights

- [ ] **Step 2: Run the full test suite**

Run: `pytest --cov=. --cov-fail-under=80 -v`
Expected: no regressions in existing Qwen3 tests (minimum 80% coverage)

- [ ] **Step 3: Run formatting and static checks**

Run:
- `black --line-length 100 .`
- `isort --profile black .`
- `mypy .`

Expected: no new errors introduced by the S2 Pro changes

- [ ] **Step 4: Manual smoke test when model files are available**

Once the real dependency and model files are installed, manually verify:
- `/v1/tts/s2-pro/sync` returns playable audio
- generated WAV is persisted to the output directory
- `/v1/audio/{job_id}.wav` serves the saved file

---

## Summary

| Task | Description | Est. Time |
|------|-------------|-----------|
| 1 | Add S2 Pro request schema | 20 min |
| 2 | Add S2 Pro config | 10 min |
| 3 | Create Fish Audio service stub | 30 min |
| 4 | Verify dependency | 20 min |
| 5 | Add singleton accessor | 15 min |
| 6 | Add dedicated sync endpoint | 30 min |
| 7-8 | Add unit/integration tests | 45 min |
| 9 | Final verification | 20 min |

**Total estimated: ~3 hours**

---

## Future Work (Out of Scope)

- Real Fish Audio inference integration after package/API verification
- Async S2 Pro job execution via Redis workers
- Generic text-to-speech endpoint unification across providers
- Provider capability matrix in API docs
- S2 Pro-specific controls such as inline tags, if supported by the final integration