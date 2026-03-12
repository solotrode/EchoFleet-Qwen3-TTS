# Design: Add Fish Audio S2 Pro as Dedicated Plain TTS Endpoint

**Date:** 2026-03-11  
**Status:** Draft  
**Author:** opencode

## Summary

Add Fish Audio S2 Pro as a separate plain-TTS capability with its own synchronous endpoint. Keep the existing Qwen3 `voice-clone`, `custom-voice`, and `voice-design` APIs unchanged.

## Goals

1. Add S2 Pro without regressing the current Qwen3 API surface.
2. Expose S2 Pro through a dedicated endpoint that matches its actual capabilities.
3. Keep the first implementation small, local, and sync-only.
4. Reuse existing response, WAV persistence, and download patterns where practical.

## Non-Goals

- Add `model="s2-pro"` to existing request schemas.
- Route `voice-clone`, `custom-voice`, or `voice-design` requests to S2 Pro.
- Refactor the entire TTS service into a provider-agnostic engine registry.
- Add S2 Pro support to Redis workers in the first pass.
- Support streaming in this design.

## Background

### Current State
- EchoFleet currently exposes feature-specific Qwen3 endpoints rather than a generic provider-selection API.
- `voice-clone` assumes reference-audio cloning semantics.
- `custom-voice` assumes Qwen3 speaker presets.
- `voice-design` assumes natural-language voice-design prompts.
- Sync endpoints already share a response pattern: generate WAV, persist it, return `AudioResponse` or `EnhancedAudioResponse`, and store a minimal Redis record for download.

### Fish Audio S2 Pro
- Dual-autoregressive TTS model.
- Supports plain TTS and inline expressive controls, subject to the final library/API used.
- Does not map cleanly onto EchoFleet's existing voice-clone, custom-voice, or voice-design contracts.
- License terms still need to be surfaced clearly for commercial use.

## Key Design Decision

S2 Pro will be integrated as its own plain-TTS endpoint rather than as a model option on existing Qwen3 routes.

Reasoning:
- The current endpoints are capability-specific, not generic provider-selection surfaces.
- Reusing `voice-clone` for a model that does not support that feature would create an invalid contract.
- Reusing `custom-voice` would imply support for named speaker presets that S2 Pro does not provide.
- Reusing `voice-design` would imply support for the existing natural-language voice-design flow, which is not established for S2 Pro.
- A dedicated route keeps the change isolated and avoids touching the current Redis worker path.

## Proposed API Surface

### New Request Schema

**File:** `api/schemas.py`

Add a dedicated request model:

```python
class S2ProRequest(BaseModel):
    """Request payload for Fish Audio S2 Pro plain TTS."""

    text: str = Field(..., min_length=1, max_length=5000)
    language: Optional[str] = Field(default=None, description="e.g. English, Chinese, Auto")
```

Keep these existing schemas unchanged:
- `VoiceCloneRequest`
- `CustomVoiceRequest`
- `VoiceDesignRequest`

Update `AudioResponse.model` documentation so `s2-pro` is an allowed returned model value.

### New Endpoint

**File:** `api/main.py`

Add a dedicated synchronous endpoint:

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

This endpoint should:
- accept plain text and optional language
- call the Fish Audio service wrapper
- convert the generated waveform to WAV/base64 using the existing helpers
- persist the WAV to the existing output directory pattern
- create the same minimal Redis record used by other sync endpoints so the download route keeps working
- return `AudioResponse(..., model="s2-pro", ...)`

### Explicitly Unchanged Endpoints

Do not route S2 Pro through:
- `/v1/tts/voice-clone`
- `/v1/tts/voice-clone/sync`
- `/v1/tts/voice-clone/submit`
- `/v1/tts/custom-voice`
- `/v1/tts/voice-design`
- `/v1/tts/voice-design/sync`

## Configuration

**File:** `config/settings.py`

Add:
- `s2_pro_model_dir`: local model directory, default `/models/s2-pro`
- `s2_pro_repo_id`: model source identifier, default `fishaudio/s2-pro`
- `s2_pro_device`: explicit runtime device, defaulting to the preferred TTS GPU

Example:

```python
self.s2_pro_model_dir: str = os.getenv("S2_PRO_MODEL_DIR", "/models/s2-pro")
self.s2_pro_repo_id: str = os.getenv("S2_PRO_REPO_ID", "fishaudio/s2-pro")
self.s2_pro_device: str = os.getenv("S2_PRO_DEVICE", f"cuda:{self.tts_preferred_gpu}")
```

## Service Layer

**New file:** `inference/fish_audio_service.py`

Create a narrow service wrapper:

```python
class FishAudioService:
    """Lazy-loading Fish Audio S2 Pro wrapper for plain TTS."""

    def __init__(self, settings: Settings) -> None:
        self._settings = settings
        self._lock = Lock()
        self._model = None
        self._model_dir = settings.s2_pro_model_dir
        self._device = settings.s2_pro_device

    def generate(self, text: str, language: Optional[str] = None) -> tuple[np.ndarray, int]:
        if self._model is None:
            self._load_model()
        return self._model.generate(...)
```

The wrapper should not expose:
- reference-audio cloning arguments
- Qwen-style speaker preset arguments
- voice-design prompt arguments

That functionality is intentionally not part of the S2 Pro contract in this design.

## Service Lifetime and Loading

**File:** `api/main.py`

Mirror the existing sync Qwen3 singleton pattern with a dedicated Fish Audio accessor:

```python
_SYNC_FISH_AUDIO_SERVICE: Optional[FishAudioService] = None
_SYNC_FISH_AUDIO_LOCK = threading.Lock()


def get_sync_fish_audio_service() -> FishAudioService:
    ...
```

Why this approach:
- Keeps model loading lazy.
- Avoids import-time GPU initialization.
- Stays consistent with the current sync Qwen3 integration style.

## Dependencies

**File:** `requirements.txt`

Add the Fish Audio dependency only after verifying the exact install package and import path. Do not guess between names like `fish-speech` and `fish_audio` without confirming the actual supported package for this stack.

## Data Flow

```text
POST /v1/tts/s2-pro/sync
    |
    v
S2ProRequest(text, language)
    |
    v
get_sync_fish_audio_service()
    |
    v
FishAudioService.generate(text, language)
    |
    v
waveform + sample_rate
    |
    v
WAV bytes + persisted output + Redis job record
    |
    v
AudioResponse(model="s2-pro", ...)
```

## Error Handling

- Missing or unavailable S2 Pro model files: 503 Service Unavailable.
- Input validation errors: 400 Bad Request.
- Inference failures: 500 Internal Server Error.

There is no `unknown model` error path in this design because S2 Pro is not selected via a shared `model` field.

## Testing Strategy

1. **Unit tests**
   - Validate `S2ProRequest` schema behavior.
   - Validate `FishAudioService` initialization and missing-model behavior.

2. **Integration tests**
   - Test `/v1/tts/s2-pro/sync` with a mocked Fish Audio service.
   - Test 503 handling when the model is unavailable.
   - Test persistence/download wiring through the existing sync endpoint pattern.

3. **Manual testing**
   - Install the verified Fish Audio dependency.
   - Place model files in the configured directory.
   - Confirm generated WAV playback and response correctness.

The tests should use `fastapi.testclient.TestClient` directly rather than assuming a preexisting fixture that does not exist in the repo.

## Security Considerations

- Fish Audio licensing still needs an explicit operator-facing note.
- Model files should remain in a controlled local path.
- Keep request validation identical in strictness to existing text-based TTS endpoints.

## Performance Considerations

- Lazy-load the S2 Pro model on first use.
- Reuse a singleton service instance for sync calls.
- Defer worker-based concurrency until the provider API and runtime footprint are validated.

## Open Questions

1. What is the exact supported Python package and import path for S2 Pro on this runtime?
2. What is the final inference API shape for plain text generation?
3. Which inline expressive controls should be documented once the provider API is confirmed?
4. Does S2 Pro need provider-specific GPU memory cleanup behavior beyond the current sync-service pattern?

## Implementation Order

1. Add the dedicated S2 Pro request schema.
2. Add config fields.
3. Create a narrow Fish Audio service stub.
4. Verify and add the real dependency.
5. Add the sync singleton accessor.
6. Add `/v1/tts/s2-pro/sync`.
7. Add unit and integration tests.

## Timeline Estimate

- Schema/config: 30 minutes
- Fish Audio service stub: 1 hour
- Endpoint wiring: 1 hour
- Testing: 1 to 2 hours
- Package/API verification: variable

**Total:** ~3 to 5 hours, excluding any provider-package or model-download blockers

## Alternatives Considered

### Option A: Reuse `voice-clone` with `model="s2-pro"`
Rejected because it would advertise a capability S2 Pro does not provide in this integration.

### Option B: Reuse `custom-voice` with `model="s2-pro"`
Rejected because `custom-voice` is a speaker-preset contract specific to Qwen3.

### Option C: Reuse `voice-design` with `model="s2-pro"`
Rejected because it would imply support for the existing voice-design behavior without a matching provider contract.

### Option D: Add a dedicated endpoint
Selected because it is the smallest correct change and keeps provider capabilities explicit.