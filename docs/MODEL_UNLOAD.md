# Model Unload Implementation

## Overview

The `/v1/models/unload` endpoint allows you to free GPU memory by unloading cached TTS models when no jobs are running.

## Architecture

### Problem
- Models are loaded in the **worker** container (`qwen3-worker`), not the API container (`qwen3-tts`)
- The API container has `TTS_WORKER_CONCURRENCY=0`, so it only queues jobs to Redis
- Direct unload calls in the API container had no effect on worker memory

### Solution
- API sends unload commands to workers via a Redis command queue (`tts:commands`)
- Workers poll both `tts:jobs` (for generation work) and `tts:commands` (for admin commands)
- Worker executes unload and writes result back to Redis
- API waits for result and returns it to the caller

## Flow

```
Client -> POST /v1/models/unload
   |
   v
API checks Redis for running jobs
   |
   ├─> If jobs running: return {"status":"blocked","running_jobs":[...]}
   |
   v
API pushes command to tts:commands queue
   |
   v
Worker pops command from tts:commands
   |
   v
Worker calls tts_server.unload_model(model_type)
   |
   ├─> Deletes cached model objects
   ├─> Calls torch.cuda.synchronize()
   ├─> Runs gc.collect() 3x
   └─> Calls torch.cuda.empty_cache()
   |
   v
Worker writes result to tts:cmd:{cmd_id}
   |
   v
API polls for result (30s timeout)
   |
   v
API returns {"status":"ok","unloaded":[...]} to client
```

## Usage

### Unload all models

```bash
curl -X POST http://localhost:18000/v1/models/unload
```

```powershell
Invoke-RestMethod -Uri http://localhost:18000/v1/models/unload -Method Post
```

**Response (success):**
```json
{
  "status": "ok",
  "unloaded": [
    ["voice-design", "cuda:0"],
    ["base", "cuda:1"]
  ]
}
```

**Response (blocked):**
```json
{
  "status": "blocked",
  "running_jobs": ["a8c1d3ff8f504fed8a0bb0d81548fdc0"]
}
```

### Unload specific model type

```bash
curl -X POST "http://localhost:18000/v1/models/unload?model_type=voice-design"
```

**Query parameters:**
- `model_type` (optional): One of `base`, `custom-voice`, `voice-design`. If omitted, unloads all models.

## Implementation Details

### Redis Keys

- **Command queue:** `tts:commands` (list)
  - Worker blocks on this queue with `blpop(["tts:jobs", "tts:commands"], timeout=5)`
  - Commands are JSON payloads: `{"cmd_id": "...", "type": "unload", "model_type": null}`

- **Command result:** `tts:cmd:{cmd_id}` (hash)
  - Status: `done` or `error`
  - Result: JSON array of unloaded `[model_type, device]` tuples
  - Expires after 60 seconds

### Safety Checks

1. **Running job detection:** API scans `tts:job:*` keys for `status` in `["running", "processing"]`
2. **Blocking behavior:** If any jobs are active, unload is blocked and job IDs are returned
3. **Timeout:** API waits up to 30 seconds for worker to process the command

### Memory Cleanup

The worker's `unload_model()` performs:

1. Pop model from cache (`self._models`)
2. Delete model object
3. `torch.cuda.synchronize()` — wait for all CUDA operations to complete
4. `gc.collect()` — run garbage collection 3 times
5. `torch.cuda.empty_cache()` — release cached GPU memory allocator blocks

## Diagnostic Endpoint

Check what models are currently loaded:

```bash
curl http://localhost:18000/v1/models/status
```

**Response:**
```json
{
  "cached_models": [
    {
      "model_type": "voice-design",
      "device": "cuda:0",
      "param_device": "cuda:0"
    }
  ],
  "gpu_memory": [
    {
      "gpu_id": 0,
      "allocated_gb": 3.42,
      "reserved_gb": 3.98
    }
  ],
  "model_count": 1
}
```

**Note:** This endpoint reports status from the **API container**. Since the API has `TTS_WORKER_CONCURRENCY=0`, it will show empty `cached_models`. To check worker memory, use `nvidia-smi` or container logs.

## Background Auto-Unload

The worker also runs an idle-unload watcher that automatically unloads models after `TTS_UNLOAD_IDLE_SECONDS` (default: 300) of inactivity.

This background process uses the same `unload_model()` implementation and respects the idle timeout, so models are only unloaded after the configured period with no generation activity.

## Environment Variables

- `TTS_UNLOAD_IDLE_SECONDS` (default: `300`) — seconds of idle time before automatic unload
- `TTS_WORKER_CONCURRENCY` (default: `0` in API, `1` in worker) — number of in-process worker threads

## Troubleshooting

### Unload returns `[]` but GPU memory still high

- Check which container has models loaded:
  ```bash
  docker logs qwen3-worker --tail 50 | grep "Loading Qwen3-TTS"
  ```
- Verify the worker is processing commands:
  ```bash
  docker logs qwen3-worker --tail 50 | grep "unload"
  ```

### Unload times out (504 error)

- Worker may be stuck processing a long job
- Check worker logs for errors:
  ```bash
  docker logs qwen3-worker --tail 100
  ```

### Models immediately re-load after unload

- Check if there's a queued job in Redis:
  ```bash
  redis-cli -p 16379 LLEN tts:jobs
  ```
- The worker will load models on-demand when processing new jobs

## See Also

- [GPU Inference Runbook](GPU_INFERENCE_RUNBOOK.md) — debugging GPU device placement issues
- [Architecture](ARCHITECTURE_UI_TTS.md) — system design overview
