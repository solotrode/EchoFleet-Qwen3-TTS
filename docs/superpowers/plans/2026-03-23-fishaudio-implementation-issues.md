# Fish Audio S2 Pro Implementation Issues Resolution Plan

> **For agentic workers:** REQUIRED: Use superpowers:subagent-driven-development (if subagents available) or superpowers:executing-plans to implement this plan. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Resolve all 11 issues documented in `docs/2026-03-23 FishAudioImplementationReview.md` covering build process, performance, and code quality problems in the Fish Audio S2 Pro integration.

**Architecture:** High and Medium priority issues first (blocking build failures and performance), then Low priority fixes. Issues grouped by file for efficient batch implementation.

**Tech Stack:** Docker, Docker Compose, Python (FastAPI), SGLang

---

## Chunk 1: High Priority Issues (Blocking/Performance)

### Task 1: Pin SGLANG_OMNI_REF to specific commit

**Files:**
- Modify: `Dockerfile.sglang:11`

- [ ] **Step 1: Check for latest stable tag in sglang-omni repo**

Run: `git ls-remote --tags https://github.com/sgl-project/sglang-omni.git | tail -5`
Expected: List of tags (e.g., v0.x.x)

- [ ] **Step 2: Update Dockerfile.sglang to pin to latest tag**

Replace:
```dockerfile
ARG SGLANG_OMNI_REF=main
```

With:
```dockerfile
ARG SGLANG_OMNI_REF=v0.1.0  # Or latest stable tag
```

- [ ] **Step 3: Commit**

```bash
git add Dockerfile.sglang
git commit -m "fix: pin SGLANG_OMNI_REF to specific tag"
```

---

### Task 2: Add fish-sglang healthcheck and depends_on

**Files:**
- Modify: `compose.yaml:155-183` (fish-sglang service)
- Modify: `compose.yaml:36-38` (echofleet-qwen3-tts depends_on)

- [ ] **Step 1: Add healthcheck to fish-sglang service**

In compose.yaml, add after line 167 (before environment):
```yaml
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:8000/v1/models"]
      interval: 30s
      timeout: 10s
      retries: 5
      start_period: 300s
```

- [ ] **Step 2: Add depends_on for fish-sglang in echofleet-qwen3-tts**

Replace lines 36-38:
```yaml
    depends_on:
      redis:
        condition: service_healthy
```

With:
```yaml
    depends_on:
      redis:
        condition: service_healthy
      fish-sglang:
        condition: service_healthy
```

- [ ] **Step 3: Commit**

```bash
git add compose.yaml
git commit -m "fix: add fish-sglang healthcheck and depends_on"
```

---

### Task 3: Fix blocking svc.generate() in async handler

**Files:**
- Modify: `api/main.py:1842`

- [ ] **Step 1: Write the failing test (if test file exists, create if not)**

Find test location:
```bash
grep -r "s2-pro" tests/ --include="*.py" | head -5
```

If no existing test, create `tests/integration/test_fish_audio.py`:
```python
import pytest
import asyncio
from unittest.mock import patch, MagicMock, AsyncMock

@pytest.mark.asyncio
async def test_s2_pro_sync_endpoint_uses_executor():
    """Verify svc.generate() is called via run_in_executor to avoid blocking event loop."""
    from api.main import app
    from httpx import AsyncClient
    
    mock_wav = MagicMock()
    mock_wav.__len__ = lambda self: 48000
    
    with patch("api.main.get_sync_fish_audio_service") as mock_svc_factory:
        mock_svc = MagicMock()
        mock_svc.generate = MagicMock(return_value=(mock_wav, 24000))
        mock_svc_factory.return_value = mock_svc
        
        # Patch run_in_executor to verify it's called (not direct svc.generate)
        with patch("asyncio.get_event_loop") as mock_loop:
            mock_loop_instance = MagicMock()
            mock_loop_instance.run_in_executor = MagicMock(return_value=(mock_wav, 24000))
            mock_loop.return_value = mock_loop_instance
            
            async with AsyncClient(app=app, base_url="http://test") as client:
                response = await client.post(
                    "/v1/tts/s2-pro/sync",
                    json={"text": "Hello world"}
                )
            
            # Verify run_in_executor was called (not direct svc.generate)
            mock_loop_instance.run_in_executor.assert_called_once()
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/integration/test_fish_audio.py -v`
Expected: FAIL with "AssertionError: assert run_in_executor call not made" because current code calls svc.generate() directly

- [ ] **Step 3: Update api/main.py to wrap in run_in_executor**

Replace line 1842:
```python
        wav, sr = svc.generate(request.text, request.language)
```

With:
```python
        loop = asyncio.get_event_loop()
        wav, sr = await loop.run_in_executor(None, svc.generate, request.text, request.language)
```

Add import at top of file (if not present):
```python
import asyncio
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/integration/test_fish_audio.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add api/main.py tests/integration/test_fish_audio.py
git commit -m "fix: wrap svc.generate in run_in_executor to avoid blocking event loop"
```

---

## Chunk 2: Medium Priority Issues

### Task 4: Test moving Stage 2 decoder to GPU

**Files:**
- Modify: `compose.yaml:180`

- [ ] **Step 1: Test removing CPU pinning**

In compose.yaml line 180, change:
```yaml
      --stages.2.executor.args.device cpu
```

To:
```yaml
      --stages.2.executor.args.device cuda
```

- [ ] **Step 2: Build and test**

Run: `docker compose build fish-sglang && docker compose up -d fish-sglang`
Wait for model to load, then test endpoint:
```bash
curl -X POST http://localhost:18001/v1/audio/speech \
  -H "Content-Type: application/json" \
  -d '{"input": "Test audio generation"}'
```

- [ ] **Step 3: Verify GPU usage**

Run: `nvidia-smi` to check if stage 2 uses GPU memory

- [ ] **Step 4: If successful, commit; if fails, revert**

If works:
```bash
git add compose.yaml
git commit -m "perf: move stage 2 decoder to GPU"
```

If fails (CUDA OOM or errors), revert only the stage 2 device change:
```bash
# Revert only the stage 2 device change, preserve other compose.yaml changes
sed -i 's/--stages.2.executor.args.device cuda/--stages.2.executor.args.device cpu/' compose.yaml
git add compose.yaml
git commit -m "perf: stage 2 decoder remains on CPU (GPU incompatible)"
```

---

### Task 5: Change FISH_FORCE_SDPA_KVCACHE default to false

**Files:**
- Modify: `Dockerfile.sglang:126-133`

- [ ] **Step 1: Update default in Dockerfile.sglang**

Replace lines 126-133:
```python
FISH_FORCE_SDPA_KVCACHE = os.getenv("FISH_FORCE_SDPA_KVCACHE", "true").lower() in (
    "true",
    "1",
    "yes",
)
```

With:
```python
FISH_FORCE_SDPA_KVCACHE = os.getenv("FISH_FORCE_SDPA_KVCACHE", "false").lower() in (
    "true",
    "1",
    "yes",
)
```

- [ ] **Step 2: Add comment explaining when to enable**

Add inline comment:
```python
# Default to false - flash-attn-kvcache is faster on supported GPUs
# Set to true only on GPUs where flash-attn kernel is unavailable
FISH_FORCE_SDPA_KVCACHE = os.getenv("FISH_FORCE_SDPA_KVCACHE", "false").lower() in (
```

- [ ] **Step 3: Commit**

```bash
git add Dockerfile.sglang
git commit -m "perf: default FISH_FORCE_SDPA_KVCACHE to false"
```

---

### Task 6: Consolidate patch RUN layers

**Files:**
- Modify: `Dockerfile.sglang:80-230`

- [ ] **Step 1: Review all patch blocks**

Identify patches at lines:
- 80-90 (attention backend)
- 92-113 (runtime import)
- 115-218 (SDPA KV-cache fallback)
- 220-230 (disable_cuda_graph)

- [ ] **Step 2: Consolidate into 1-2 RUN blocks**

Replace the 4 separate RUN blocks with a single consolidated patch RUN block:
```dockerfile
# Apply all sglang-omni patches in a single layer
RUN python - <<'PY'
from pathlib import Path

# Patch 1: Attention backend
factory = Path('/opt/venv/lib/python3.12/site-packages/sglang_omni/models/fishaudio_s2_pro/factory.py')
text = factory.read_text(encoding='utf-8')
old = '    if server_args.attention_backend is None:\n        server_args.attention_backend = "fa3"\n'
new = '    if server_args.attention_backend is None:\n        server_args.attention_backend = "triton"\n'
if old not in text:
    raise SystemExit('Expected attention backend snippet not found in factory.py')
factory.write_text(text.replace(old, new, 1), encoding='utf-8')

# Patch 2: Runtime import
text = factory.read_text(encoding='utf-8')
old = '''from .runtime.s2pro_sglang_ar import (
    S2ProSGLangIterationController,
    S2ProSGLangModelRunner,
    S2ProSGLangResourceManager,
)
'''
new = '''from .runtime.s2pro_sglang_ar import (
    S2ProSGLangIterationController,
    S2ProSGLangModelRunner,
    S2ProSGLangOutputProcessor,
    S2ProSGLangResourceManager,
)
'''
if old not in text:
    raise SystemExit('Expected runtime import block not found in factory.py')
factory.write_text(text.replace(old, new, 1), encoding='utf-8')

# Patch 3: SDPA KV-cache fallback (full implementation from Dockerfile.sglang lines 115-218)
modeling = Path('/opt/venv/lib/python3.12/site-packages/sglang_omni/models/fishaudio_s2_pro/fish_speech/models/text2semantic/modeling.py')
text = modeling.read_text(encoding='utf-8')

old_flag = '''FISH_BATCH_INVARIANT = os.getenv("FISH_BATCH_INVARIANT", "false").lower() in (
    "true",
    "1",
    "yes",
)
'''
new_flag = old_flag + '''
FISH_FORCE_SDPA_KVCACHE = os.getenv("FISH_FORCE_SDPA_KVCACHE", "true").lower() in (
    "true",
    "1",
    "yes",
)
'''
if old_flag not in text:
    raise SystemExit('Expected FISH_BATCH_INVARIANT snippet not found in modeling.py')
text = text.replace(old_flag, new_flag, 1)

old_block = '''        # Use flash_attn_with_kvcache - it handles KV cache update internally
        # q: (batch_size, seqlen, nheads, headdim)
        # k_cache/v_cache: (batch_size, seqlen_cache, nheads_k, headdim)
        # k/v: (batch_size, seqlen_new, nheads_k, headdim)
        k_cache, v_cache = self.kv_cache.get(bsz)
        y = flash_attn_kvcache_op(
            q=q,
            k_cache=k_cache,
            v_cache=v_cache,
            k=k,
            v=v,
            cache_seqlens=cache_seqlens,
            causal=True,
            num_splits=1 if FISH_BATCH_INVARIANT else 0,
        )

        y = y.contiguous().view(bsz, seqlen, q_size)
'''

new_block = '''        # Prefer a portable SDPA fallback for KV-cache decode on GPUs where the
        # flash-attn kernel is unavailable.
        k_cache, v_cache = self.kv_cache.get(bsz)
        y = None

        if not FISH_FORCE_SDPA_KVCACHE:
            try:
                y = flash_attn_kvcache_op(
                    q=q,
                    k_cache=k_cache,
                    v_cache=v_cache,
                    k=k,
                    v=v,
                    cache_seqlens=cache_seqlens,
                    causal=True,
                    num_splits=1 if FISH_BATCH_INVARIANT else 0,
                )
            except RuntimeError as exc:
                if "no kernel image is available for execution on the device" not in str(exc):
                    raise
                log.warning("Falling back to SDPA KV-cache decode: %s", exc)

        if y is None:
            start_positions = cache_seqlens.to(torch.int64)
            for batch_idx in range(bsz):
                start = int(start_positions[batch_idx].item())
                end = start + seqlen
                k_cache[batch_idx, start:end] = k[batch_idx]
                v_cache[batch_idx, start:end] = v[batch_idx]

            total_lengths = start_positions + seqlen
            max_total = int(total_lengths.max().item())

            q = q.transpose(1, 2)
            full_k = k_cache[:bsz, :max_total].transpose(1, 2)
            full_v = v_cache[:bsz, :max_total].transpose(1, 2)
            full_k = full_k.repeat_interleave(self.n_head // self.n_local_heads, dim=1)
            full_v = full_v.repeat_interleave(self.n_head // self.n_local_heads, dim=1)

            key_positions = torch.arange(max_total, device=q.device)
            query_positions = start_positions[:, None] + torch.arange(seqlen, device=q.device)[None, :]
            valid_mask = key_positions.view(1, 1, max_total) <= query_positions.unsqueeze(-1)
            valid_mask = valid_mask & (key_positions.view(1, 1, max_total) < total_lengths.view(-1, 1, 1))
            attn_mask = valid_mask.unsqueeze(1)

            y = self._scaled_dot_product_attention(
                q,
                full_k,
                full_v,
                attn_mask=attn_mask,
                is_causal=False,
            )
            y = y.transpose(1, 2)

        y = y.contiguous().view(bsz, seqlen, q_size)
'''

if old_block not in text:
    raise SystemExit('Expected KV-cache block not found in modeling.py')
text = text.replace(old_block, new_block, 1)
modeling.write_text(text, encoding='utf-8')

# Patch 4: Disable CUDA graphs
stages = Path('/opt/venv/lib/python3.12/site-packages/sglang_omni/models/fishaudio_s2_pro/pipeline/stages.py')
text = stages.read_text(encoding='utf-8')
old = '        disable_cuda_graph=False,\n'
new = '        disable_cuda_graph=True,\n'
if old not in text:
    raise SystemExit('Expected disable_cuda_graph snippet not found in stages.py')
stages.write_text(text.replace(old, new, 1), encoding='utf-8')
PY
```

- [ ] **Step 3: Build to verify**

Run: `docker build -f Dockerfile.sglang -t fish-sglang-test .`
Expected: Build completes without patch errors

- [ ] **Step 4: Commit**

```bash
git add Dockerfile.sglang
git commit -m "build: consolidate patch RUN layers into single block"
```

---

## Chunk 3: Low Priority Issues

### Task 7: Use requests.Session for HTTP connection reuse

**Files:**
- Modify: `inference/fish_audio_service.py:73`

- [ ] **Step 1: Update FishAudioService to use Session**

Replace lines 22-27:
```python
    def __init__(self) -> None:
        self._settings = get_settings()
        self._lock = threading.Lock()
        self._model_dir = getattr(self._settings, "s2_pro_model_dir", "/models/s2-pro")
        self._sglang_url = getattr(self._settings, "sglang_base_url", "http://fish-sglang:8000")
        self._ref_audio_dir = getattr(self._settings, "fish_ref_audio_dir", "/ref_audio")
```

With:
```python
    def __init__(self) -> None:
        self._settings = get_settings()
        self._lock = threading.Lock()
        self._model_dir = getattr(self._settings, "s2_pro_model_dir", "/models/s2-pro")
        self._sglang_url = getattr(self._settings, "sglang_base_url", "http://fish-sglang:8000")
        self._ref_audio_dir = getattr(self._settings, "fish_ref_audio_dir", "/ref_audio")
        self._session = requests.Session()

    def __del__(self) -> None:
        if hasattr(self, "_session"):
            self._session.close()
```

- [ ] **Step 2: Update generate() to use session**

Replace line 73:
```python
        response = requests.post(endpoint, json=payload, timeout=300)
```

With:
```python
        response = self._session.post(endpoint, json=payload, timeout=300)
```

- [ ] **Step 3: Write test for session reuse**

```python
import pytest
import requests
from inference.fish_audio_service import FishAudioService

def test_fish_audio_service_uses_session():
    svc = FishAudioService()
    assert hasattr(svc, "_session")
    assert isinstance(svc._session, requests.Session)
```

- [ ] **Step 4: Run test**

Run: `pytest tests/unit/test_fish_audio_service.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add inference/fish_audio_service.py tests/unit/test_fish_audio_service.py
git commit -m "perf: use requests.Session for HTTP connection reuse"
```

---

### Task 8: Fix volume mount to :ro in API container

**Files:**
- Modify: `compose.yaml:82`

- [ ] **Step 1: Change mount to read-only**

Replace line 82:
```yaml
      - fish-tts:/models_fish-s2-pro
```

With:
```yaml
      - fish-tts:/models_fish-s2-pro:ro
```

- [ ] **Step 2: Commit**

```bash
git add compose.yaml
git commit -m "fix: mount fish-tts volume as read-only in API container"
```

---

### Task 9: Add missing env vars to worker

**Files:**
- Modify: `compose.yaml:135-147`

- [ ] **Step 1: Add SGLANG_BASE_URL and FISH_REF_AUDIO_DIR**

Add after line 135:
```yaml
      - SGLANG_BASE_URL=http://fish-sglang:8000
      - FISH_REF_AUDIO_DIR=/voices
```

- [ ] **Step 2: Commit**

```bash
git add compose.yaml
git commit -m "fix: add missing env vars to worker for Fish Audio"
```

---

### Task 10: Remove unused imports

**Files:**
- Modify: `inference/fish_audio_service.py:8-11`

- [ ] **Step 1: Remove unused imports**

Remove lines 8-11:
```python
import gc
import base64
import json
```

- [ ] **Step 2: Verify no other usage**

Run: `grep -n "gc\|base64\|json" inference/fish_audio_service.py`
Expected: No matches (except if used elsewhere - verify and keep if needed)

- [ ] **Step 3: Commit**

```bash
git add inference/fish_audio_service.py
git commit -m "refactor: remove unused imports from fish_audio_service"
```

---

## Chunk 4: Enable CUDA graphs (Optional)

### Task 11: Re-enable CUDA graphs

**Files:**
- Modify: `Dockerfile.sglang:220-230`

- [ ] **Step 1: Change disable_cuda_graph back to false**

Replace lines 220-230 with:
```python
stages = Path('/opt/venv/lib/python3.12/site-packages/sglang_omni/models/fishaudio_s2_pro/pipeline/stages.py')
text = stages.read_text(encoding='utf-8')
old = '        disable_cuda_graph=True,\n'
new = '        disable_cuda_graph=False,\n'
if old not in text:
    raise SystemExit('Expected disable_cuda_graph snippet not found in stages.py')
stages.write_text(text.replace(old, new, 1), encoding='utf-8')
```

- [ ] **Step 2: Build and test**

Run: `docker build -f Dockerfile.sglang -t fish-sglang-test . && docker compose up -d fish-sglang`
Test endpoint and verify no errors

- [ ] **Step 3: If stable, commit; if issues, revert**

```bash
# If successful
git add Dockerfile.sglang
git commit -m "perf: re-enable CUDA graphs for improved decode performance"

# If issues - revert only the disable_cuda_graph change
sed -i 's/disable_cuda_graph=False/disable_cuda_graph=True/' Dockerfile.sglang
git add Dockerfile.sglang
git commit -m "perf: keep CUDA graphs disabled (compatibility issues)"
```

---

## Verification Commands

After completing all tasks, run these checks:

```bash
# Format code
black --line-length 100 .
isort --profile black .

# Type checking
mypy .

# Linting
flake8 .

# Run tests
pytest tests/unit/ tests/integration/ -v

# Docker build test
docker compose build
```

---

## Plan Summary

| Chunk | Tasks | Priority |
|-------|-------|----------|
| 1 | 1-3 | High (blocking) |
| 2 | 4-6 | Medium (performance) |
| 3 | 7-10 | Low (quality) |
| 4 | 11 | Optional |

**Total: 11 tasks**
