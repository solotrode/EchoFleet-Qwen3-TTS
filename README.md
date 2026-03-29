# EchoFleet-Qwen3-TTS Multi-GPU Docker (CUDA 12.8 / PyTorch cu128)

This repo (renamed to **EchoFleet-Qwen3-TTS**) containerizes Qwen3‑based TTS and provides a **FastAPI** service plus a **Redis + RQ** worker stack intended for **multi‑GPU** TTS generation and (optional) Whisper‑based scoring.

Fork notice: This project is a fork and extension of the upstream Qwen3‑TTS repository. The original project is available at https://github.com/QwenLM/Qwen3-TTS — please see that repository for upstream model sources and official releases.

What’s working today:
- Docker image builds on CUDA **12.8.1** with **Python 3.12** and **PyTorch 2.7.0+cu128**.
- Compose stack boots: **Redis**, **API** container, **dedicated worker** container.
- Health endpoint confirms torch/CUDA import and GPU visibility.
- Multi‑GPU dispatch and per‑GPU worker processes: dispatcher splits multi‑candidate jobs into per‑GPU candidate tasks and workers pinned to specific GPUs process them concurrently.
- ASR (Whisper) scoring pipeline integrated for candidate ranking.

What’s working now (end-to-end):
- Synchronous TTS endpoints (legacy):
  - `POST /v1/tts/voice-clone`
  - `POST /v1/tts/custom-voice`
  - `POST /v1/tts/voice-design`
- Asynchronous job pipeline (worker): submit with `/v1/tts/voice-clone/submit` to enqueue jobs processed by the `qwen3-worker` container.
- Multi‑candidate generation with per‑GPU parallelism and ASR ranking (select best candidate automatically).
- Web UI: Gradio service on host port `18860`.
 - Synchronous TTS endpoints (legacy):
   - `POST /v1/tts/voice-clone`
   - `POST /v1/tts/custom-voice`
   - `POST /v1/tts/voice-design`
 - Asynchronous job pipeline (worker): submit with `/v1/tts/voice-clone/submit` to enqueue jobs processed by the `echofleet-qwen3-worker` container.
 - Multi‑candidate generation with per‑GPU parallelism and ASR ranking (select best candidate automatically).
 - Web UI: Gradio service on host port `18860`.

## Performance (measured)
- Baseline (pre‑parallel): 1 GPU, 4 candidates — 13 minutes 6 seconds.
- Parallel result: 4 GPUs, 4 candidates — 6 minutes 45 seconds.
- Target: 4 GPUs, 4 candidates — ≤ 6 minutes (further tuning pending).

What is scaffolded / planned (not fully implemented yet):
- The Redis/RQ worker pipeline exists, but TTS requests are currently served synchronously by the API.
- Whisper scoring exists as a module but isn’t wired into the worker pipeline yet.

---

## Fish S2 Pro Container Lazy Start/Stop (Fish TTS Docker Lifecycle)

The Fish S2 Pro TTS backend is provisioned as a separate Docker container (`fish-sglang`) and is managed by the main API container to conserve GPU VRAM and system resources. By default, **the Fish container is NOT running** until the first S2 Pro TTS request is received.

**How lazy start/stop works**:
- When a Fish TTS request is made to `/v1/tts/s2-pro/sync`, the API attempts to start the `fish-sglang` Docker container (if not already running).
- The controller waits for the container and health check to turn `healthy` before proceeding.
- If no Fish requests are received for the configured idle period (default: 300 seconds), the container is automatically stopped to free VRAM and host resources.
- **Manual unload:** Whenever the Fish S2 Pro model is explicitly unloaded via `/v1/models/unload?model_type=s2-pro`, the API will also stop the Fish container.

**Key configuration variables (see compose.yaml, settings.py):**
- `FISH_IDLE_UNLOAD_SECONDS`: Idle duration before fish-sglang container is auto-stopped (default: 300)
- `FISH_STARTUP_TIMEOUT_SECONDS`: How long to wait for startup/health (default: 300)
- `FISH_STOP_TIMEOUT_SECONDS`: Timeout for graceful stop (default: 30)
- `FISH_SGLANG_CONTAINER_NAME`: Name of the container (default: fish-sglang)
- `FISH_DOCKER_SOCKET_PATH`: Path to the Docker socket (default: /var/run/docker.sock)
- `HOST_DOCKER_GID`: Group to access Docker socket; must match host Docker group

**Operational Notes / Troubleshooting:**
- The API container mounts the Docker socket and must run with a group matching the Docker host for socket access; otherwise, startup or stop will fail (see `group_add` in compose.yaml).
- `GET /v1/models/status` shows the health and state of the Fish service **and** its container.
- Container lifecycle actions are logged with detail in `/logs/echofleet-qwen3-tts.jsonl`.

**Examples:**
- Force unload and stop: `curl -X POST http://localhost:18000/v1/models/unload?model_type=s2-pro`
- Check container status: `curl http://localhost:18000/v1/models/status` (see `fish_container_state`)
- If the Fish S2 Pro backend fails to start due to Docker permissions, check your user/groups and Docker Desktop/Engine config.

---

- GPU inference stability (tokenizer-on-CPU regression): [docs/GPU_INFERENCE_RUNBOOK.md](docs/GPU_INFERENCE_RUNBOOK.md)
- Model unloading via API (free VRAM on demand): [docs/MODEL_UNLOAD.md](docs/MODEL_UNLOAD.md)

## Chunking configuration

Long input texts can cause excessive generation times or GPU memory pressure. The service implements sentence-aware chunking for TTS inputs. Configure the preferred chunk size (characters) via the `TTS_CHUNK_MAX_CHARS` environment variable. Default: `1000`.

Example `.env` entry:

```ini
# Preferred maximum characters per TTS chunk (split on sentence boundaries)
TTS_CHUNK_MAX_CHARS=1000
```

## Quickstart (Docker Compose)

### 1) Prerequisites

- NVIDIA GPU + driver installed on the host.
- Docker with NVIDIA GPU support.
  - On Windows this typically means Docker Desktop with the WSL2 backend and NVIDIA CUDA support enabled.
- Disk space: these models and CUDA wheels are large.

### 2) (Recommended) Download models into `./models/`

If you already have the model folders under `./models/`, you can skip this.

This repo includes a helper script:

```powershell
# Download a single repo
python scripts/download_model.py --repo Qwen/Qwen3-TTS-12Hz-1.7B-Base

# Download a batch list (one repo id per line)
python scripts/download_model.py --file models_to_download.txt
```

Minimum required for all endpoints:
- `Qwen/Qwen3-TTS-Tokenizer-12Hz`
- `Qwen/Qwen3-TTS-12Hz-1.7B-Base`
- `Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice`
- `Qwen/Qwen3-TTS-12Hz-1.7B-VoiceDesign`

Models are mounted into containers at `/models`.

### 3) Build and start

```powershell
docker compose build
docker compose up -d
```

### 4) Verify health

PowerShell:

```powershell
Invoke-RestMethod -Uri http://localhost:18000/v1/health | ConvertTo-Json -Depth 6
```

Expected output includes:
- `cuda_available: true`
- `gpu_count: 4` (or however many GPUs you have)
- `pytorch_version: 2.7.0+cu128`

### 5) Open API docs

- Swagger UI: http://localhost:18000/docs
- ReDoc: http://localhost:18000/redoc

---

## API Examples (curl / PowerShell)

Below are example invocations for each public endpoint. Replace `localhost:18000` with your host/port if different.

- Health

```bash
curl -s http://localhost:18000/v1/health
```

PowerShell:

```powershell
Invoke-RestMethod -Uri http://localhost:18000/v1/health
```

- Voice clone (synchronous)

Example using a hosted reference audio URL:

```bash
curl -X POST http://localhost:18000/v1/tts/voice-clone \
  -H "Content-Type: application/json" \
  -d '{
    "text": "Hello world",
    "language": "en",
    "ref_audio": "https://example.com/speaker.wav",
    "ref_text": "This is the reference audio.",
    "x_vector_only_mode": false
  }'
```

Example using a local WAV file encoded as base64 (bash):

```bash
REF_B64=$(base64 -w0 sample.wav)
curl -X POST http://localhost:18000/v1/tts/voice-clone \
  -H "Content-Type: application/json" \
  -d "{\"text\":\"Hello world\",\"language\":\"en\",\"ref_audio\":\"data:audio/wav;base64,$REF_B64\",\"ref_text\":\"Reference text\",\"x_vector_only_mode\":false}"
```

PowerShell (base64 encode then POST):

```powershell
$b = [Convert]::ToBase64String([IO.File]::ReadAllBytes('sample.wav'))
$body = @{ text = 'Hello world'; language = 'en'; ref_audio = "data:audio/wav;base64,$b"; ref_text = 'Reference text'; x_vector_only_mode = $false } | ConvertTo-Json
Invoke-RestMethod -Uri http://localhost:18000/v1/tts/voice-clone -Method Post -Body $body -ContentType 'application/json'
```

### Multi-candidate generation (ASR-ranking)

Request multiple generation candidates and optionally return all candidates with their scores. The service will (optionally) run ASR scoring and pick the best candidate when `return_all_candidates` is false.

Example (bash) requesting 4 candidates and asking to receive all candidates' metadata:

```bash
REF_B64=$(base64 -w0 sample.wav)
curl -X POST http://localhost:18000/v1/tts/voice-clone \
  -H "Content-Type: application/json" \
  -d "{\"text\":\"Hello world\",\"language\":\"en\",\"ref_audio\":\"data:audio/wav;base64,$REF_B64\",\"ref_text\":\"Reference text\",\"x_vector_only_mode\":false,\"num_candidates\":4,\"return_all_candidates\":true}"
```

If you only want the best candidate selected by ASR scoring, set `return_all_candidates` to `false` (default) and the response will contain the chosen `job_id`/`audio_base64` for the top candidate.


- Voice clone (enqueue)

```bash
curl -X POST http://localhost:18000/v1/tts/voice-clone/submit \
  -H "Content-Type: application/json" \
  -d '{"text":"Hello","language":"en","ref_audio":"https://example.com/s.wav","ref_text":"Ref","x_vector_only_mode":false}'
```

Response: `{"job_id":"<hex>"}`. Check job status:

```bash
curl http://localhost:18000/v1/jobs/<job_id>
```

- Custom voice

```bash
curl -X POST http://localhost:18000/v1/tts/custom-voice \
  -H "Content-Type: application/json" \
  -d '{"text":"Hello world","language":"en","speaker":"Vivian","instruct":"Warm, friendly"}'
```

- Voice design

The service now supports both asynchronous submission (recommended) and a synchronous endpoint.

Async (submit + poll):

```bash
# Submit job (returns job id)
curl -s -X POST http://localhost:18000/v1/tts/voice-design \
  -H "Content-Type: application/json" \
  -d '{"text":"Hello world","language":"en","instruct":"Bright female voice, playful"}' | jq

# Poll job status (replace <job_id> with the returned id)
curl -s http://localhost:18000/v1/jobs/<job_id> | jq

# When status == "done", download the WAV
curl -O http://localhost:18000/v1/audio/<job_id>.wav
```

Synchronous (legacy behavior):

```bash
# This endpoint blocks until generation completes and returns the audio inline
curl -s -X POST http://localhost:18000/v1/tts/voice-design/sync \
  -H "Content-Type: application/json" \
  -d '{"text":"Hello world","language":"en","instruct":"Bright female voice, playful"}' | jq
```

- Fish Audio S2 Pro

The API wrapper accepts a flat request body so clients like n8n do not need to
build Fish's nested `references` structure themselves.

Plain TTS:

```bash
curl -X POST http://localhost:18000/v1/tts/s2-pro/sync \
  -H "Content-Type: application/json" \
  -d '{"text":"Hello world","language":"en"}'
```

Voice cloning with top-level reference fields:

```bash
curl -X POST http://localhost:18000/v1/tts/s2-pro/sync \
  -H "Content-Type: application/json" \
  -d '{"text":"Hello world","language":"en","ref_audio":"voices/sample.wav","ref_text":"Reference transcript"}'
```

- Unload models (free GPU memory)

Unload all models:

```bash
curl -X POST http://localhost:18000/v1/models/unload
```

Unload a single model (e.g. `voice-design`):

```bash
curl -X POST "http://localhost:18000/v1/models/unload?model_type=voice-design"
```

- Download generated audio (new)

Once a job completes (either synchronous TTS response or queued job), you can retrieve the final WAV via a file-download endpoint:

```bash
curl -O http://localhost:18000/v1/audio/<job_id>.wav
```

Behavior:
- If the worker persisted a WAV to disk, the endpoint streams that file.
- If only `audio_base64` is stored in Redis, the endpoint returns the decoded WAV bytes.
- If the job is not yet `done`, the endpoint returns HTTP 202 (Accepted).

Notes:
- For `voice-clone`, `ref_audio` may be a local path (when the container can access it), a URL, or a data URL with base64 audio (`data:audio/wav;base64,...`).
- If `x_vector_only_mode` is `true`, `ref_text` may be omitted (lower quality). If `false`, provide `ref_text` for best results.
- All TTS endpoints return a JSON payload containing `audio_base64`, `sample_rate`, `duration_seconds`, and `model`.

---

## Services and ports

Compose file: [compose.yaml](compose.yaml)

- **API** (`qwen3-tts`)
  - Host `18000` → container `8000` (FastAPI)
- **Web UI** (`qwen3-ui`)
  - Host `18860` → container `7860` (Gradio)
- **Redis** (`redis`)
  - Host `16379` → container `6379`
- **Worker** (`qwen3-worker`)
  - No ports exposed; processes jobs from Redis
 - **API** (`echofleet-qwen3-tts`)
  - Host `18000` → container `8000` (FastAPI)
 - **Web UI** (`echofleet-qwen3-ui`)
  - Host `18860` → container `7860` (Gradio)
 - **Redis** (`redis`)
  - Host `16379` → container `6379`
 - **Worker** (`echofleet-qwen3-worker`)
  - No ports exposed; processes jobs from Redis

---

## Configuration

Environment template: [.env.example](.env.example)

Key variables:
- `MODEL_CACHE_DIR` (container): defaults to `/models` (mounted from `./models`)
- `OUTPUT_DIR` (container): defaults to `/workspace/outputs` (mounted from `./outputs`)
- `TTS_GPUS`: GPU IDs for TTS scheduling (example: `0,1,2,3`)
- `WHISPER_DEVICE_POOL`: GPU IDs for Whisper (example: `0,1,2,3`)
- `WHISPER_MODEL`: HuggingFace model id (preferred) or shorthand
  - Preferred: `openai/whisper-large-v3-turbo`
  - Shorthand supported: `large-v3-turbo`

Settings loader: [config/settings.py](config/settings.py)

Voices folder (optional):

- You can store reusable reference audio files under a `voices/` directory at the repo root and mount it into containers. Files should use the simple format `voices/<speaker>.wav` (for example `voices/john.wav`).
- Recommended format: WAV (PCM), mono, 24000 Hz. The container also accepts data URLs (`data:audio/wav;base64,...`) and remote URLs.
- If you plan to use container-local reference files, add this volume to `compose.yaml`:

```yaml
  echofleet-qwen3-tts:
    volumes:
      - ./voices:/voices:ro
```


---

## Implementation overview

### Runtime architecture

- **FastAPI** app: [api/main.py](api/main.py)
  - `/v1/health` checks Redis and reports torch/CUDA status.
  - `/v1/tts/*` endpoints are listed as “coming soon”.
- **Redis** provides a shared queue/state store.
- **Worker** runs the background worker process defined in `worker.run` (see [compose.yaml](compose.yaml)).

### CUDA / PyTorch pinning

- The image is built from `nvidia/cuda:12.8.1-devel-ubuntu22.04`.
- PyTorch is pinned to `torch==2.7.0+cu128` (and matching torchvision/torchaudio).
- Constraints file prevents dependency-driven downgrades:
  - [constraints-cu128-py312.txt](constraints-cu128-py312.txt)

Docker build: [Dockerfile](Dockerfile)

### Whisper (HuggingFace)

- Whisper is implemented via **Transformers** (not the `openai-whisper` Python package).
- The transcriber wrapper lives at:
  - [inference/whisper_service.py](inference/whisper_service.py)

Note: The module is present so the worker can use it for scoring later; it is not currently invoked by the API.

---

## Web UI

The Gradio UI runs as a dedicated compose service and calls the FastAPI endpoints.
Open: http://localhost:18860

---

## Troubleshooting

### PowerShell `curl` pitfalls

In Windows PowerShell, `curl` is often an alias for `Invoke-WebRequest`.
Use one of these instead:

```powershell
# PowerShell-native
Invoke-RestMethod -Uri http://localhost:18000/v1/health

# Or the real curl binary if available
curl.exe -f http://localhost:18000/v1/health
```

### GPU not detected inside containers

If `/v1/health` shows `cuda_available: false`:
- Confirm Docker is running with NVIDIA GPU support.
- Confirm your host drivers are installed and working.
- Confirm you’re using Linux containers (WSL2 backend on Windows).

### Port already in use

Edit host ports in [compose.yaml](compose.yaml) if `18000`, `18860`, or `16379` conflicts with another service.

---

## Development

Dev requirements: [requirements-dev.txt](requirements-dev.txt)

Suggested local tooling:

```powershell
python -m pip install -r requirements-dev.txt
python -m pytest
```
