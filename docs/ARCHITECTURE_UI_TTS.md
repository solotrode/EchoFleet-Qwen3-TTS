% Expanded architecture and in-depth guide (long-form)

# Architecture: UI → API → Worker / TTS — Deep Dive for Engineers

Overview
- This document is an exhaustive guide intended for junior and intermediate engineers who need to understand, maintain, and extend the TTS pipeline implemented across the UI (front-end), the FastAPI-based backend, the model loading and GPU worker/pool, and the audio utility functions. It contains conceptual explanations, code mapping, step-by-step traces through the system, debugging playbooks, instrumentation advice, CI/test examples, and best-practice recommendations for handling user input and GPU resources.

How to use this document
- If you are new to the project: read the "High-level flow" section first, then skim "Key code paths" to connect concepts to files.
- If you need to debug: follow the "Debugging & triage" checklist in Part E and then inspect the files listed in the Quick file map.
- If you are adding features or tests: see Part F (tests) and Part H (example client + integration tests).

Terminology used in this document
- Input payload: JSON (or multipart) data posted by the UI containing `text`, `ref_audio`, and other options.
- ref_audio: user-provided reference audio that can be uploaded as bytes, encoded as a data URL, or presented as a long base64 string.
- model: the TTS model loaded from pre-trained weights (Hugging Face style expected), used to synthesize audio.
- engine: inference wrapper that orchestrates tokenization, conditioning, and `model.generate()`.

SECTION 1 — High-level flow (detailed)

This section expands the one-line description into a comprehensive narrative with motivations and constraints.

1.1 Motivation and constraints
- Goal: Produce high-quality, low-latency TTS using GPU-accelerated models while ensuring user inputs are safely handled, the system is debuggable, and logs are not overwhelmed by user-provided binary data.
- Constraints:
  - Keep CPU work minimal (no silent CPU fallback for GPU-capable operations).
  - Avoid writing user audio payloads to disk whenever possible — use in-memory buffers.
  - Ensure model loading is cached and multi-GPU aware.
  - Log sufficiently to be actionable, but never leak full user payloads in logs.

1.2 Narrative flow (step-by-step with reasons)
- UI action: user clicks generate — the UI must first validate client-side that `text` is present and `ref_audio` is not excessively large (client-side limits improve UX but server must be authoritative).
- Network transfer: the client sends the request. The backend must be resilient to slow clients (use timeouts) and reject overly large bodies early.
- Input handling: backend decodes `ref_audio` using heuristics to differentiate URLs, data, and paths.
- Preprocessing: audio is converted to a `torch.Tensor`, coerced to mono, and resampled to `target_sr` (default 24kHz) using GPU resampling when possible.
- Model interaction: the ModelManager ensures the model is loaded with the correct device mapping and dtype. The engine passes the preprocessed audio and the text to the model for generation.
- Postprocessing: output is transformed to WAV bytes in-memory and returned to the client as base64.

SECTION 2 — Key code paths and mapping to files

This section lists the most important files and the functions you will interact with when performing typical tasks.

2.1 API / request handling (api/main.py)
- Responsibilities:
  - Registering routes and middleware.
  - Wiring dependency injection for the `ModelManager` and other shared services.
  - Handling request validation and formatting responses.
- Typical functions to inspect:
  - `create_app()` — constructs `FastAPI` and attaches middleware.
  - Route handler for `/v1/tts/voice-clone` — check request handling logic and the shape of `VoiceCloneRequest`.

2.2 Model loading and cache (models/loader.py)
- Responsibilities:
  - Provide `ModelManager` with caching semantics and a registry of known models.
  - Convert raw string identifiers to model objects via `from_pretrained`.
  - Expose unload utilities to free GPU memory.
- Things to look for:
  - How `device_map` is computed (automatic vs explicit mapping).
  - How dtype is selected (`bfloat16` vs `float16`) and how it's passed to `from_pretrained`.

2.3 GPU pool and worker coordination (models/gpu_pool.py)
- Responsibilities:
  - Serialize and schedule inference calls to GPUs.
  - Maintain locks or semaphores to prevent concurrent OOMs on single GPUs.
  - Sanitize file-paths and input strings to avoid Errno 36 and log flooding.

2.4 Inference engine (inference/voice_clone.py)
- Responsibilities:
  - Tokenization and text preprocessing.
  - Speaker embedding extraction from `ref_audio`.
  - Orchestration of the call to `model.generate(...)` with proper device placement.

2.5 Audio utilities (utils/audio_utils.py)
- Responsibilities:
  - Accept and normalize `ref_audio` payloads of multiple shapes.
  - Perform GPU-first resampling where supported.
  - Serialize tensors to WAV bytes via `BytesIO`.
  - Mask/truncate logging of user inputs.

SECTION 3 — Function deep dives (practical walkthroughs)

3.1 load_audio() – full control-flow and examples
- Pseudocode (annotated):

    def load_audio(source, target_sr=None):
        if source is None: raise FileNotFoundError

        # numpy/torch branch
        if isinstance(source, tuple) and isinstance(source[0], np.ndarray):
            arr, sr = source
            tensor = torch.from_numpy(arr).float()
            if tensor.dim() > 1: tensor = tensor.mean(dim=-1)
            if target_sr and sr != target_sr:
                tensor = resample_audio(tensor, sr, target_sr)
                sr = target_sr
            if torch.cuda.is_available(): try move to cuda
            return tensor, int(sr)

        # bytes branch
        if isinstance(source, (bytes, bytearray)):
            data_bytes = bytes(source)
            bio = io.BytesIO(data_bytes)
            audio, sr = sf.read(bio, dtype='float32')
            process and return

        # string branch — data URL / base64 / URL / filesystem
        if isinstance(source, str):
            src = source.strip()
            if src.startswith('data:'):
                header, b64 = src.split(',', 1)
                data_bytes = base64.b64decode(b64)
            elif re.match(base64_regex) and len(src) > threshold:
                data_bytes = base64.b64decode(src)
            elif src.startswith('http://') or src.startswith('https://'):
                resp = requests.get(src, timeout=...)
                resp.raise_for_status()
                data_bytes = resp.content
            else:
                try: exists = os.path.exists(src)
                except OSError: raise AudioProcessingError('Invalid path')
                if not exists: raise FileNotFoundError
                audio, sr = sf.read(src)

        # when we have bytes
        if data_bytes is not None:
            bio = io.BytesIO(data_bytes)
            audio, sr = sf.read(bio)
            convert to tensor, resample, move to cuda
            return tensor, int(sr)

3.2 resample_audio() – device semantics and pitfalls
- The function aims to resample audio on GPU using `torchaudio.transforms.Resample` when possible. Key pitfalls:
  - `Resample` may not implement `.to(device)` depending on `torchaudio` version.
  - Even if `.to()` exists, it may still run CPU kernels depending on the compiled backend.
  - The project policy is to prefer GPU resampling; if GPU isn't possible, raise `AudioProcessingError` instead of silently falling back to CPU. This makes failures explicit for the caller and avoids hidden performance regressions.

3.3 Inference orchestration – what to check if generation fails
- Check devices: ensure both model and tensors are on the same `cuda` device.
- For OOMs:
  - log `torch.cuda.memory_summary()`
  - attempt to free caches if appropriate but prefer explicit failure messages.

SECTION 4 — Logging, privacy, and input sanitization (detailed policies)

4.1 Masking policy and helper patterns
- Always call `_mask_str(user_string)` before logging any user-provided content. `_mask_str` returns the whole string only if it's below a configured threshold (120 characters by default) and otherwise returns a truncated representation showing the first N and last M characters and the total length.

4.2 File access policy
- Never write or read arbitrary user-provided strings as file paths. The only acceptable filesystem paths are those validated by the server (e.g., paths created by the server itself or paths that match a known safe directory pattern). For user content, decode to bytes and process with `BytesIO`.

4.3 HTTP fetching and size limits
- When accepting URLs in `ref_audio`, call `requests.get` with a size-limited stream pattern or raise if content-length exceeds a configured maximum (e.g., 10MB). Use `stream=True` and read only up to the configured bytes to avoid memory exhaustion.

4.4 Error and exception patterns
- User errors (bad input): return 4xx and explain what was wrong.
- Internal errors: return 500 generic message. Log the stacktrace but ensure sensitive strings are masked.

SECTION 5 — Debugging playbook (concrete steps to reproduce and triage common issues)

5.1 Reproducing a 500 while avoiding log pollution
1) Ensure `_mask_str` is used in `utils` and `api` modules.
2) Use a small WAV file (1 second) in tests to avoid generating huge base64 payloads.
3) Start the backend with logs redirected to file and run the test call from a client.

Example commands (PowerShell):

```powershell
# start services with logs in a file
docker compose up --build > backend-stdout.log 2>&1

# run a local python client that posts small data
python examples/client_small.py

# inspect logs for 'Traceback' and other error keywords
Select-String -Path backend-stdout.log -Pattern "Traceback", "ERROR", "Exception" -Context 0,20
```

5.2 Interpreting stack traces
- Focus on the first user-code frame — that is likely where the exception originated (e.g., `utils/audio_utils.py` or `inference/voice_clone.py`).
- If the message contains masked placeholders like `<truncated 12345 chars>`, the user payload was present but not fully logged — this is expected and desired.

SECTION 6 — Testing matrix and example test cases (expanded)

6.1 Unit tests to implement (detailed)
- `test_audio_utils_load_bytes`:
  - assert `load_audio(bytes)` returns (tensor, sr)
- `test_audio_utils_load_base64`:
  - base64-encode a small WAV and assert `load_audio(base64_string)` returns (tensor, sr)
- `test_load_audio_path_raises_on_long_string`:
  - pass a very long string and assert it does not cause OSError propagation but raises `AudioProcessingError` or `FileNotFoundError`.

6.2 Integration tests (detailed)
- `test_voice_clone_e2e`:
  - Mock `ModelManager.get_model` with a lightweight model that returns a predictable tensor, call the API endpoint using `TestClient`, and assert the returned base64 decodes to a WAV with expected sample rate.

6.3 Continuous integration recommendations
- Separate unit and integration stages; run integration tests with mocked models unless you have GPU runners.

SECTION 7 — Performance, resource management, and operational considerations

7.1 Memory and GPU usage patterns
- Prefer `low_cpu_mem_usage=True` during model load.
- Consider using `torch.inference_mode()` if available during generation to reduce memory usage.

7.2 Concurrency model and request queuing
- For each GPU compute node, run a fixed number of worker processes equal to the number of GPUs, or use a queue that dispatches work to worker processes pinned to GPUs.

7.3 Instrumentation and observability (metrics to collect)
- Model load time, model memory footprint, generation latencies (p50/p95/p99), request sizes, and rejection rates for oversized payloads.

SECTION 8 — Example clients and reproducible snippets (robust versions)

8.1 Python example client (with size guard)

```python
import base64
import requests

def make_payload(text, wav_path, max_size=5_000_000):
    with open(wav_path, 'rb') as f:
        data = f.read()
    if len(data) > max_size:
        raise ValueError('Reference audio too large')
    return {
        'text': text,
        'ref_audio': 'data:audio/wav;base64,' + base64.b64encode(data).decode('ascii'),
        'ref_text': text
    }

resp = requests.post('http://localhost:8000/v1/tts/voice-clone', json=make_payload('hi', 'test_1s.wav'))
print(resp.status_code)
```

8.2 TestClient integration test sample (pytest style)

```python
from fastapi.testclient import TestClient
from api.main import app
import base64, io
import soundfile as sf

client = TestClient(app)

def test_tts_returns_wav():
    with open('test_1s.wav','rb') as f:
        data = f.read()
    payload = {'text':'Hello','ref_audio':'data:audio/wav;base64,'+base64.b64encode(data).decode('ascii'),'ref_text':'Hello'}
    r = client.post('/v1/tts/voice-clone', json=payload)
    assert r.status_code == 200
    j = r.json()
    wav = base64.b64decode(j['audio_data'])
    arr, sr = sf.read(io.BytesIO(wav))
    assert sr == 24000
```

SECTION 9 — Example failures and concrete fixes (cookbook)

9.1 Symptom: POST returns 500; logs are flooded with base64 payloads
Cause: code somewhere logs raw user string (print, json.dumps or logger call without masking)
Fix: search for `print(` and `json.dumps(` in the repo and replace with masked logging or remove prints; audit `logger.*` calls and ensure `_mask_str` is used for user inputs.

9.2 Symptom: Errno 36 "File name too long"
Cause: base64/data URL string passed to `os.path.exists` or `open`
Fix: add detection logic to `load_audio` to decode base64/data URLs before any filesystem operation. Wrap `os.path.exists` in try/except OSError and raise `AudioProcessingError` if it hits OSError.

9.3 Symptom: Resampling silently falls back to CPU and is slow
Cause: `torchaudio` version may not support GPU resampling or resampler was not moved to device
Fix: ensure `resample_audio` attempts `.to(device)` and, on failure, raise `AudioProcessingError` (or explicitly document allowed CPU fallback if acceptable). Pin `torchaudio` to a version that supports GPU resampling for your production environment.

SECTION 10 — Deployment and operational tips

10.1 Docker Compose considerations
- When building the runtime image, ensure the runtime has CUDA-enabled base images and the matching `torch` + `torchaudio` wheels for your CUDA version.
- Keep `MAX_JOBS` small to avoid parallel compilation during image build.

10.2 GPU provisioning
- For multi-GPU instances, verify `nvidia-smi` and that the container has the correct device access (use `--gpus` or device mapping in Compose).

10.3 Logging and retention
- Configure log rotation and retention in Compose or container runtime. Do not let logs grow unbounded.

Appendix A — glossary and references

- BytesIO: Python's in-memory bytes buffer.
- base64: encoding for binary data used in JSON payloads.
- sf / soundfile: Python library using libsndfile for audio IO.
- torchaudio: PyTorch audio processing library; used here for Resample transforms.

Appendix B — quick file map

- api/main.py — FastAPI wiring and endpoints
- api/schemas.py — request/response validation
- models/loader.py — ModelManager, loading, caching
- models/gpu_pool.py — worker/gpu coordination and path sanitization
- inference/voice_clone.py — inference orchestration
- utils/audio_utils.py — load_audio, resample_audio, serialization, _mask_str

Closing notes and next steps

- I expanded this architecture document substantially to provide detailed guidance for junior engineers, including actionable debugging steps and tests.
- Next optional actions I can perform for you:
  1) Annotate `api/main.py`, `utils/audio_utils.py`, and `models/gpu_pool.py` with inline comments and exact line references mapping this doc to the code.
  2) Add the client example and integration tests to `examples/` and `tests/` and run the unit/integration tests locally.
  3) Add a CI job skeleton and sample GitHub Actions workflow for running unit tests and static checks.

End of document.

