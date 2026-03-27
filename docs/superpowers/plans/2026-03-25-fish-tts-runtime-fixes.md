# Fish TTS Runtime Fix Plan

> **For agentic workers:** REQUIRED: Use superpowers:subagent-driven-development (if subagents available) or superpowers:executing-plans to implement this plan. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Fix two runtime failures in `fish-sglang` that prevent TTS and voice-cloning requests from completing:
1. Flash Attention kernel unavailable on this GPU — partially mitigated but not fully guarded
2. `torchcodec` unavailable inside the container — breaks reference-audio preprocessing for voice cloning

**Background:** The `Dockerfile.sglang` already:
- Switched the default attention backend from `fa3` → `triton`
- Added `FISH_FORCE_SDPA_KVCACHE` env-var support with a try/except SDPA fallback in `modeling.py`
- Disabled CUDA graphs via `stages.py` patch

What is **not yet done**:
- `FISH_FORCE_SDPA_KVCACHE` defaults to `false`, meaning flash-attn-kvcache is still tried first and the try/except only catches one specific error string — other flash-attn failure modes propagate uncaught
- No torchcodec fallback exists anywhere in the container; sglang-omni's reference-audio pipeline tries to import/use torchcodec and fails silently or hard on this system

**Non-goals:**
- Do not rearchitect the sglang-omni pipeline; patch minimally in-place inside `Dockerfile.sglang`
- Do not touch application-layer code in `inference/fish_audio_service.py` — the service wrapper is correct
- Do not change the Qwen or Whisper paths

---

## Validation Summary

| Recommendation | Status | Evidence |
|---|---|---|
| Patch Flash Attention path | Partially done | `Dockerfile.sglang` has `FISH_FORCE_SDPA_KVCACHE` + try/except, but default is `false` and catch is too narrow |
| Patch torchcodec path | Not done | No torchcodec patch in `Dockerfile.sglang`; `requirements.txt` explicitly excludes it for the app container with comment noting it fails |
| Rebuild + validate | Not done | No changes made yet |

---

## File Structure

| File | Responsibility |
|------|----------------|
| `Dockerfile.sglang` | Add torchcodec fallback patch; harden the flash-attn fallback |
| `compose.yaml` | Set `FISH_FORCE_SDPA_KVCACHE=true` for the `fish-sglang` service to skip flash-attn-kvcache entirely |

---

## Chunk 1: Harden the Flash Attention Fallback

### Task 1: Force SDPA path via environment variable in compose.yaml

The `FISH_FORCE_SDPA_KVCACHE` flag already controls which code path is taken in `modeling.py`. The safest fix on a GPU where flash-attn-kvcache kernels are unavailable is to set it to `true` in `compose.yaml` rather than relying on the runtime try/except catch, which only guards one specific error string.

**Files:**
- Modify: `compose.yaml`

- [ ] **Step 1: Read the fish-sglang environment block in compose.yaml**

```bash
Read compose.yaml fish-sglang.environment section
```

- [ ] **Step 2: Add FISH_FORCE_SDPA_KVCACHE=true to fish-sglang environment**

In `compose.yaml`, under `fish-sglang.environment`, add:

```yaml
- FISH_FORCE_SDPA_KVCACHE=true
```

This skips the flash_attn_kvcache_op call entirely and uses the SDPA manual implementation. It costs a small amount of throughput on GPUs that do have working flash-attn, but eliminates the failure mode.

- [ ] **Step 3: Broaden the fallback catch in Dockerfile.sglang (optional hardening)**

The current try/except in `modeling.py` only catches:
```python
if "no kernel image is available for execution on the device" not in str(exc):
    raise
```

This misses other flash-attn failures such as `CUDA error: no kernel image`, CUBLAS errors, or import-time failures. Consider catching the exception and logging, then falling through to SDPA for all `RuntimeError` types when `y is None`. However, since `FISH_FORCE_SDPA_KVCACHE=true` bypasses the try/except entirely, this is optional and only needed if you ever want to re-enable flash-attn opportunistically on different hardware.

> **Note:** If this service is ever deployed without the `FISH_FORCE_SDPA_KVCACHE=true` env var (e.g., a different compose file or k8s manifest), the narrow catch becomes the only guard again. Broaden it then.

- [ ] **Step 4: Commit and restart (no rebuild needed)**

`FISH_FORCE_SDPA_KVCACHE` is read at runtime by the already-patched `modeling.py`, so no image rebuild is required — only a container restart.

```bash
git add compose.yaml
git commit -m "fix: force SDPA KV-cache path for fish-sglang on this GPU"
docker compose up -d fish-sglang
```

---

## Chunk 2: Patch the torchcodec Reference Audio Path

### Task 2: Add a torchcodec → soundfile fallback in the fish-sglang image

The sglang-omni pipeline uses `torchcodec` (or a torch-based audio loader) to decode reference audio files before encoding them as mel features. On this system, `torchcodec` is unavailable — `libtorch.so` is not on `LD_LIBRARY_PATH` inside the container, or the torchcodec wheels are not built for this CUDA version.

The fix is to patch `sglang-omni`'s audio loading in `Dockerfile.sglang` to fall back to `soundfile` + `numpy` when `torchcodec` or its torch audio path fails, similar to the approach already used in the main app container.

**Files:**
- Modify: `Dockerfile.sglang`

- [ ] **Step 0: Ensure the fish-sglang image is built**

The discovery steps below require a runnable `fish-sglang:local` image. Check whether it already exists:

```bash
docker image inspect fish-sglang:local --format '{{.Id}}' 2>/dev/null && echo "Image exists" || echo "Image not found"
```

If not found, build it first (this will take several minutes):

```bash
docker compose build fish-sglang
```

- [ ] **Step 1: Identify the torchcodec call site in the sglang-omni pipeline**

Inspect the installed package inside the container to find where reference audio is decoded.

The image tag is `fish-sglang:local` — this is the default tag Docker Compose assigns when no explicit `image:` field is set in `compose.yaml`.

```bash
docker run --rm --entrypoint bash fish-sglang:local -c \
  "grep -rn 'torchcodec\|torchaudio\|AudioDecoder\|audio_decoder' \
   /opt/venv/lib/python3.12/site-packages/sglang_omni/models/fishaudio_s2_pro/ \
   --include='*.py' -l"
```

Expected locations based on sglang-omni architecture:
- `pipeline/stages.py` — already patched for cuda_graph; may also load audio
- `preprocessing/audio.py` or similar — dedicated audio preprocessing helper

- [ ] **Step 2: Read the relevant audio loading file(s)**

Once the file is identified:

```bash
docker run --rm --entrypoint bash fish-sglang:local -c \
  "cat /opt/venv/lib/python3.12/site-packages/sglang_omni/models/fishaudio_s2_pro/pipeline/stages.py"
```

Look for patterns like:
- `from torchcodec.decoders import AudioDecoder`
- `torchaudio.load`
- `torch.ops.torchcodec`

- [ ] **Step 3: Write a Dockerfile.sglang patch for the audio loader**

Add a `RUN python - <<'PY'` block in `Dockerfile.sglang` that patches the identified file. Shape of the patch depends on what Step 1/2 finds. Two likely cases:

**Case A — direct torchcodec import:**
```python
from pathlib import Path

target = Path('/opt/venv/lib/python3.12/site-packages/sglang_omni/models/fishaudio_s2_pro/REPLACE_WITH_FILE.py')
text = target.read_text(encoding='utf-8')

old = 'from torchcodec.decoders import AudioDecoder\n'
new = (
    'try:\n'
    '    from torchcodec.decoders import AudioDecoder\n'
    '    _TORCHCODEC_AVAILABLE = True\n'
    'except (ImportError, OSError):\n'
    '    _TORCHCODEC_AVAILABLE = False\n'
)
if old not in text:
    raise SystemExit('Expected torchcodec import not found — patch needs updating')
target.write_text(text.replace(old, new, 1), encoding='utf-8')
```

**Case B — torchaudio.load call that fails:**
```python
old = 'waveform, sample_rate = torchaudio.load(audio_path)\n'
new = (
    'try:\n'
    '    waveform, sample_rate = torchaudio.load(audio_path)\n'
    'except Exception:\n'
    '    import soundfile as sf, numpy as np, torch\n'
    '    arr, sample_rate = sf.read(audio_path, dtype="float32", always_2d=True)\n'
    '    waveform = torch.from_numpy(arr.T)\n'
)
```

> **Important:** The exact patch content MUST be determined by reading the actual file in Step 1/2. The shapes above are representative only. Copy the exact line(s) verbatim from Step 2's output — including all whitespace and indentation — into the `old` string before committing. A whitespace mismatch will cause `if old not in text` to silently pass the guard and leave the file unpatched.

- [ ] **Step 4: Ensure soundfile is installed in the fish-sglang image**

In `Dockerfile.sglang`, add `soundfile` to the pip install if it is not already present via the sglang-omni `[s2pro]` extras:

```bash
docker run --rm --entrypoint bash fish-sglang:local -c \
  "python -c 'import soundfile; print(soundfile.__version__)'"
```

If not present, add to the install block:

```dockerfile
RUN uv pip install soundfile
```

Place this after the existing `uv pip install ".[s2pro]"` line.

- [ ] **Step 5: Rebuild the image and spot-check the patch**

```bash
docker compose build fish-sglang
docker run --rm --entrypoint bash fish-sglang:local -c \
  "python -c 'from sglang_omni.models.fishaudio_s2_pro.pipeline import stages; print(\"OK\")'"
```

- [ ] **Step 6: Commit**

```bash
git add Dockerfile.sglang
git commit -m "fix: patch torchcodec reference audio path in fish-sglang image"
```

---

## Chunk 3: Rebuild and Validate with Live Requests

### Task 3: Full rebuild and live-request validation

**Files:**
- None — operational steps only

- [ ] **Step 1: Rebuild and restart (only needed if Chunk 2 changed Dockerfile.sglang)**

If Chunk 2 produced `Dockerfile.sglang` changes, rebuild first. If Chunk 2 found no torchcodec issue and made no changes, skip the build — the container is already running from the Chunk 1 restart.

```bash
docker compose build fish-sglang
docker compose up -d fish-sglang
```

- [ ] **Step 2: Watch startup logs for previous error signatures**

```bash
docker logs -f fish-sglang 2>&1 | grep -E "torchcodec|flash_attn|ERROR|CRITICAL|no kernel"
```

Expect: no torchcodec import errors, no flash-attn kernel errors on startup.

- [ ] **Step 3: Wait for health**

```bash
docker compose ps fish-sglang  # wait until healthy
```

- [ ] **Step 4: Plain TTS request**

```bash
curl -s -X POST http://localhost:18001/v1/audio/speech \
  -H "Content-Type: application/json" \
  -d '{"input": "Hello from Fish TTS."}' \
  --output /tmp/fish_plain.wav && echo "Plain TTS OK"
```

- [ ] **Step 5: Voice cloning request (reference audio)**

Pick any `.wav` file from `./voices/`. Before running, verify that `compose.yaml` mounts `./voices` into the `fish-sglang` container (look for a `volumes:` entry like `- ./voices:/voices`). If the mount path differs, update the `audio_path` value accordingly.

```bash
REF_FILE=$(ls ./voices/*.wav | head -1)
curl -s -X POST http://localhost:18001/v1/audio/speech \
  -H "Content-Type: application/json" \
  -d "{\"input\": \"Clone voice test.\", \"references\": [{\"audio_path\": \"/voices/$(basename $REF_FILE)\"}]}" \
  --output /tmp/fish_cloned.wav && echo "Voice clone OK"
```

If this request previously failed with a torchcodec error, it should now succeed.

- [ ] **Step 6: End-to-end via API**

Port `18000` is the app API (`api` service); port `18001` is the direct sglang endpoint (`fish-sglang` service). Steps 4–5 above test sglang directly; this step tests the full stack through the app layer.

```bash
curl -s -X POST http://localhost:18000/v1/tts/s2-pro/sync \
  -H "Content-Type: application/json" \
  -d '{"text": "End to end Fish TTS test."}' \
  --output /tmp/fish_e2e.wav && echo "E2E OK"
```

- [ ] **Step 7: Rollback (if needed)**

If the image fails to build or the container crashes after Chunk 2 changes:
```bash
git revert HEAD  # revert the Dockerfile.sglang patch commit
docker compose build fish-sglang
docker compose up -d fish-sglang
```

- [ ] **Step 8: Commit (if no Dockerfile.sglang changes were needed in chunk 2)**

If Chunk 2 produced no code changes (torchcodec was not actually the issue), document the finding and close the chunk as resolved-without-change. If changes were made, they were committed in Chunk 2.

---

## Acceptance Criteria

- [ ] `fish-sglang` container starts and passes its health check without flash-attn kernel errors in logs
- [ ] `POST /v1/audio/speech` with plain text returns a valid WAV file
- [ ] `POST /v1/audio/speech` with `references` (reference audio path) returns a valid WAV file
- [ ] `POST /v1/tts/s2-pro/sync` via the API returns a valid WAV file
- [ ] No `torchcodec` import or decode errors appear in `fish-sglang` logs
- [ ] `FISH_FORCE_SDPA_KVCACHE=true` is set in `compose.yaml` for `fish-sglang`

---

## Notes

- The `FISH_FORCE_SDPA_KVCACHE` SDPA fallback code path already exists in `Dockerfile.sglang` — this plan activates it by default rather than trusting the runtime try/except
- Chunk 2 Step 1 is a discovery step; the exact patch text depends on what sglang-omni actually does. Do not write the patch until you have read the actual source
- If `soundfile` is already present via `[s2pro]` extras, skip the explicit install step
- The plain TTS path does not use reference audio, so it is unaffected by the torchcodec issue — test it first to confirm the flash-attn fix before attempting voice cloning
