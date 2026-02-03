# Qwen3‑TTS Parallel GPU Architecture and Implementation Guide

## 1. Purpose and Scope
This document explains, in detail, how multi‑GPU parallelization is designed and implemented for Qwen3‑TTS in this repository. It documents the end‑to‑end execution model, the dispatch strategy, how candidates and chunks are handled, how results are collected and scored, and the practical requirements that make the system operate correctly in a Docker‑based deployment. The goal is to use all available GPUs to reduce wall‑time for multi‑candidate jobs while keeping each candidate’s generation deterministic and stable.

## 2. High‑Level Goals
1. Use all available GPUs concurrently.
2. Avoid global locks that serialize work.
3. Keep one model instance per GPU to avoid repeated model load times.
4. Ensure that per‑candidate chunking stays on a single GPU to avoid cross‑device transfers.
5. Preserve correctness while achieving the target runtime improvement.

## 3. Baseline, Target, and Result
- Baseline (pre‑parallel): One GPU, four candidates: **13 minutes 6 seconds**.
- Target: Four GPUs, four candidates: **6 minutes or less**.
- Result (parallelized): Four GPUs, four candidates: **6 minutes 45 seconds**.

## 4. Deployment Layout
### 4.1 Containers
The system runs in Docker with multiple services defined in [compose.yaml](compose.yaml):
- **qwen3-tts**: API server (Uvicorn + FastAPI).
- **qwen3-worker**: Dedicated job worker process for background jobs.
- **redis**: Job queue and state.
- **qwen3-ui**: Optional UI client.

### 4.2 Environment Settings
Key environment variables (see [config/settings.py](config/settings.py)):
- `MODEL_CACHE_DIR=/models`
- `OUTPUT_DIR=/workspace/outputs`
- `TTS_GPUS=0,1,2,3` (example)
- `TTS_PREFERRED_GPU=0`
- `TTS_WORKER_CONCURRENCY=4` (worker processes)
- `WHISPER_DEVICE_POOL=0,1,2,3`
- `TTS_CHUNK_MAX_CHARS` (controls chunking)

### 4.3 Model and Output Paths
- Models are mounted read‑only at `/models` via `qwen3_tts_models` volume.
- Whisper models are mounted at `/stt_models`.
- Generated audio and metadata are written under `/workspace/outputs`.

### 4.4 Python Path
The container sets `PYTHONPATH=/app` (see [Dockerfile](Dockerfile)), and the repository code is mounted at `/app`.

## 5. Execution Model Overview
The parallelization model is a two‑stage pipeline:
1. **Dispatch**: A dispatcher receives a job and splits it into candidate tasks. Each candidate is assigned to a GPU‑specific queue.
2. **Worker‑per‑GPU**: A dedicated worker process, pinned to a specific GPU, pulls candidate tasks from its own queue and generates audio.
3. **Aggregation and Scoring**: After all candidates are generated, a scoring task is dispatched (to a chosen GPU) to transcribe and score candidates and finalize job output.

This separation ensures GPUs are utilized concurrently across candidates, while each worker stays deterministic and stable.

## 6. Detailed Architecture
### 6.1 Redis Queues
The architecture relies on Redis lists:
- `tts:jobs`: main job ingress queue (jobs submitted by API).
- `tts:jobs:cuda:0`, `tts:jobs:cuda:1`, ...: per‑GPU queues for candidate tasks.
- `tts:commands`: administrative commands (e.g., unload models).
- `tts:job:<job_id>:candidates`: candidate results collected for scoring.

### 6.2 Job Payload Structure
Each job contains fields such as:
- `job_id`
- `text`
- `language`
- `ref_audio`, `ref_text`
- `num_candidates`
- `return_all_candidates`
- `model_type` (e.g., `base`)

Candidate tasks extend the payload with:
- `task = "candidate"`
- `candidate_id`
- `assigned_gpu`

Score tasks use:
- `task = "score"`

## 7. Dispatcher Design
### 7.1 Responsibilities
The dispatcher is responsible for:
- Reading jobs from the global queue (`tts:jobs`).
- Writing the job’s payload into Redis for later scoring.
- Assigning candidates across GPU queues in round‑robin order.
- Enqueuing a score task when all candidates are done.

### 7.2 Dispatch Logic
1. `blpop("tts:jobs")` receives a job.
2. Dispatcher stores `tts:job:<job_id>:payload`.
3. If `num_candidates <= 1` or model_type != base, it sends the job to the first GPU queue.
4. If `num_candidates > 1`, it creates one candidate task per candidate and distributes them across GPU queues.

### 7.3 Concurrency Behavior
This ensures that, for 4 candidates and 4 GPUs, each GPU receives exactly one candidate. Each candidate’s chunking remains local to its worker/GPU.

## 8. Worker‑per‑GPU Design
### 8.1 Worker Lifecycle
Workers are spawned in [worker/run.py](worker/run.py):
- One process per GPU.
- Each process is pinned to a device (e.g., `cuda:2`).
- Each worker listens only to its assigned per‑GPU queue.

### 8.1.1 GPU Assignment (Detailed)
The worker startup script reads the GPU list from `TTS_GPUS`, converts each GPU ID to a CUDA device string, and then launches one process per GPU. Each process receives:
1. Its worker index (`worker_id`)
2. Its assigned device string (`cuda:N`)
3. Its GPU‑specific queue name (`tts:jobs:cuda:N`)

This makes the binding explicit and stable. There is **no device hopping** within a worker.

#### Key Code Path (Simplified Extract)
```python
# worker/run.py (simplified)
gpu_ids = settings.tts_gpu_list()          # e.g., [0,1,2,3]
gpu_devices = [f"cuda:{gid}" for gid in gpu_ids]

for i in range(concurrency):
	device = gpu_devices[i % len(gpu_devices)]
	queue_name = f"tts:jobs:{device}"
	p = multiprocessing.Process(
		target=job_worker_loop,
		args=(i, device, queue_name),
		name=f"tts-job-worker-{i}",
	)
	p.start()
```

### 8.1.2 Dispatcher‑to‑Worker Routing
The dispatcher routes candidate tasks to the GPU‑specific queues. It uses a round‑robin mapping so candidates are distributed across available GPUs:

```python
# worker/run.py (simplified)
for i in range(num_candidates):
	target = gpu_devices[i % len(gpu_devices)]
	cand_job = dict(job)
	cand_job["task"] = "candidate"
	cand_job["candidate_id"] = i
	cand_job["assigned_gpu"] = target
	redis_client.rpush(f"tts:jobs:{target}", json.dumps(cand_job))
```

### 8.1.3 Worker Binding Inside the Job Loop
Each worker constructs its own `QwenTTSServer` instance with the assigned device. This instance is reused for all tasks handled by that worker, so the model stays on the correct GPU.

```python
# api/main.py (simplified)
def job_worker_loop(worker_id: int, device: str, queue_name: str) -> None:
	tts_server = QwenTTSServer(settings, assigned_device=device)
	while True:
		item = redis_client.blpop([queue_name, "tts:commands"], timeout=5)
		...
		if job.get("task") == "candidate":
			_generate_candidate(tts_server, ...)
```

### 8.1.4 GPU Assignment Guarantees
These steps guarantee that:
- A worker **always** uses the GPU it was assigned at startup.
- A candidate **always** runs on the GPU of its assigned worker.
- A candidate’s chunks **never** migrate to a different GPU mid‑generation.

### 8.1.5 Concurrency and GPU Utilization
With 4 GPUs and 4 candidates:
- Worker 0 on `cuda:0` handles candidate 0
- Worker 1 on `cuda:1` handles candidate 1
- Worker 2 on `cuda:2` handles candidate 2
- Worker 3 on `cuda:3` handles candidate 3

All four candidates are generated in parallel across GPUs, while each candidate’s chunks remain sequential within the worker.

### 8.2 Candidate Task Execution
When a worker receives a candidate task:
1. It creates or reuses a `QwenTTSServer` instance bound to the worker’s GPU.
2. It calls `_generate_candidate()` with the candidate’s inputs.
3. The WAV output is written to `/workspace/outputs/<job_id>/candidates/`.
4. Candidate metadata is pushed to `tts:job:<job_id>:candidates`.
5. The worker increments `completed_candidates` in the job hash.

### 8.3 Score Task Execution
When the score task is received:
1. The worker fetches the original payload from Redis.
2. It gathers all candidate results from `tts:job:<job_id>:candidates`.
3. It uses Whisper to transcribe each candidate and compute scores.
4. It selects the best candidate and writes the final job results.

## 9. Model Loading Details
### 9.1 Local Model Paths
The model loader in [inference/qwen_tts_service.py](inference/qwen_tts_service.py) checks:
- `/models/Qwen/<model-dir>`
- `/models/<model-dir>`

### 9.2 Configuration Loading
The loader uses:
- `AutoConfig.from_pretrained(..., local_files_only=True)`
- `AutoProcessor.from_pretrained(..., local_files_only=True)`

It injects:
- `config.dtype`
- `config._attn_implementation`

### 9.3 Model Creation
The model is built via:
- `Qwen3TTSForConditionalGeneration.from_pretrained(...)`
- wrapped in `Qwen3TTSModel`

### 9.4 Device Placement
The loader moves the model to the assigned GPU via `.to(device, dtype)` and avoids `device_map`. This ensures deterministic single‑GPU placement.

## 10. Tokenizer Handling
### 10.1 Tokenizer is Not a Module
The speech tokenizer is not a `torch.nn.Module`, so it is not moved automatically.

### 10.2 Explicit Device Move
The loader explicitly moves the tokenizer’s internal model to the GPU and sets `speech_tokenizer.device` to keep decoding on the GPU.

### 10.3 Enforcement
During generation, the tokenizer device is checked and enforced before prompt creation.

## 11. Chunking Behavior
### 11.1 Chunk Size
Chunk size is controlled by `TTS_CHUNK_MAX_CHARS` and input text length.

### 11.2 Sequential Chunking Per Candidate
Chunks for a candidate are generated sequentially on the same GPU. This avoids cross‑device transfers and keeps prompt states local.

### 11.3 Parallelism Level
Parallelism is achieved across candidates, not within a candidate.

## 12. GPU Parallelism Strategy
### 12.1 Candidate‑Level Parallelism
The primary parallelization unit is the candidate. Each candidate is assigned to a separate GPU when available.

### 12.2 No Cross‑GPU Synchronization
Workers do not share model instances and do not transfer tensors between GPUs during generation.

### 12.3 Score Aggregation
Scoring can occur on a selected GPU using Whisper, or can be distributed in future enhancements.

## 13. Redis Job State
Each job has a hash:
- `status`: queued / running / done / failed
- `completed_candidates`: count of finished candidates
- `num_candidates`: expected candidates
- `audio_base64`, `wav_path`, etc.

Candidate results are stored separately, allowing aggregation without re‑generation.

## 14. Error Handling
### 14.1 Candidate Failure
If a candidate fails after retries, it is omitted. If all candidates fail, the job fails.

### 14.2 Score Failure
If scoring fails, job is marked failed with error details.

### 14.3 Model Load Failures
Model load errors are surfaced early and logged with device and path info.

## 15. Performance Considerations
### 15.1 Avoid Global Locks
No global GPU lease lock is used. Each worker owns its GPU.

### 15.2 Avoid Process Pickle of Models
Models remain in the worker process, and candidate tasks are not executed in subprocesses.

### 15.3 Minimize Reloads
Models stay resident in GPU memory across jobs unless explicitly unloaded.

## 16. Example Execution (4 GPUs, 4 Candidates)
- Candidate 0 runs on GPU0, chunks 1→N sequentially.
- Candidate 1 runs on GPU1, chunks 1→N sequentially.
- Candidate 2 runs on GPU2, chunks 1→N sequentially.
- Candidate 3 runs on GPU3, chunks 1→N sequentially.
All candidates run concurrently across GPUs.

## 17. Example Execution (2 GPUs, 4 Candidates)
- Candidate 0 on GPU0
- Candidate 1 on GPU1
- Candidate 2 on GPU0
- Candidate 3 on GPU1
This halves total wall time relative to a single GPU.

## 18. Worker Concurrency
`TTS_WORKER_CONCURRENCY` should be set to the number of GPUs available, which determines the number of worker processes.

## 19. Queue Isolation
Workers only read from their own queues. This prevents two workers from accidentally processing the same candidate.

## 20. Scoring Strategy
The scoring stage is currently centralized onto a selected GPU to reduce complexity. It can be parallelized later if Whisper inference becomes a bottleneck.

## 21. Output Artifacts
Candidate WAVs and final results are written under `/workspace/outputs/<job_id>/` for inspection and download.

## 22. API Interaction
The API returns job IDs and allows polling via `/v1/jobs/<job_id>` and downloading via `/v1/audio/<job_id>.wav`.

## 23. Observability
Logs include:
- Candidate generation start/end
- Chunk timings
- Model load and device placement
- Candidate scoring

These logs are written to `/logs/qwen3-worker.jsonl` and `/logs/qwen3-tts.jsonl`.

## 24. Known Local Constraints
- Models must be present in `/models` for `local_files_only=True`.
- Docker must expose GPUs and NVIDIA runtime must be available.
- `TTS_GPUS` controls which devices are used.

## 25. Mapping to Source Files
- Worker dispatch and GPU binding: [worker/run.py](worker/run.py)
- Job processing and scoring: [api/main.py](api/main.py)
- Model loading and tokenizer handling: [inference/qwen_tts_service.py](inference/qwen_tts_service.py)
- Settings and env wiring: [config/settings.py](config/settings.py)
- Container runtime: [compose.yaml](compose.yaml), [Dockerfile](Dockerfile)

## 26. Summary
This GPU‑parallel architecture distributes candidate generation across GPUs in parallel, maintains per‑GPU model ownership, and centralizes scoring for simplicity. It is designed to meet the 6‑minute target for 4 candidates on 4 GPUs while preserving correctness and stability.
