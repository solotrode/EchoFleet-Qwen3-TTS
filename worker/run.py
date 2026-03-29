"""Dedicated background worker runner.

This runs the existing Redis list-based job worker loop (`tts:jobs`) in a
standalone process, so the FastAPI server can remain responsive and we avoid
RQ pubsub idle disconnect behavior.
"""

from __future__ import annotations

import faulthandler
import json
import multiprocessing
import signal
import threading
import time

# Set the start method to 'spawn' for CUDA safety with multiprocessing
# This must be done before any CUDA-related imports or operations.
# It ensures that child processes get a clean CUDA context.
try:
    multiprocessing.set_start_method("spawn", force=True)
    print("--- Multiprocessing start method set to 'spawn' ---")
except RuntimeError:
    # The start method can only be set once. If it's already set, we can ignore the error.
    pass

from api.main import (
    _decode_candidate_state,
    _job_candidate_state_key,
    _job_candidate_terminal_key,
    _load_job_payload,
    _maybe_enqueue_score_task,
    _set_candidate_state,
    job_worker_loop,
    logger,
    redis_client,
    settings,
)


def _select_rescue_device(gpu_devices: list[str], excluded_device: str | None = None) -> str:
    """Select the least-loaded healthy GPU queue for a rescued candidate."""
    candidates = [device for device in gpu_devices if device != excluded_device]
    if not candidates:
        candidates = list(gpu_devices)
    if not candidates:
        return "cpu"

    def _queue_depth(device: str) -> int:
        try:
            return int(redis_client.llen(f"tts:jobs:{device}"))
        except Exception:
            return 0

    return min(candidates, key=_queue_depth)


def _rescue_stale_candidate_jobs(gpu_devices: list[str]) -> None:
    """Requeue candidate work that has sat unstarted in a dead GPU queue."""
    stale_seconds = max(1, int(getattr(settings, "tts_candidate_stale_seconds", 900)))
    max_rescue_attempts = max(0, int(getattr(settings, "tts_candidate_rescue_attempts", 2)))
    now = time.time()

    for key in redis_client.scan_iter(match="tts:job:*"):
        if key.count(":") != 2:
            continue

        try:
            job = redis_client.hgetall(key)
            if not job or job.get("status") != "running":
                continue

            job_id = key.rsplit(":", 1)[-1]
            expected = int(job.get("num_candidates", 1) or 1)
            settled = int(job.get("settled_candidates", 0) or 0)
            if expected <= 1 or settled >= expected:
                continue

            payload = _load_job_payload(job_id)
            if not payload:
                continue

            state_entries = redis_client.hgetall(_job_candidate_state_key(job_id))
            terminal_entries = redis_client.hgetall(_job_candidate_terminal_key(job_id))

            for candidate_id in range(expected):
                candidate_key = str(candidate_id)
                if terminal_entries.get(candidate_key):
                    continue

                state = _decode_candidate_state(state_entries.get(candidate_key, ""))
                status = str(state.get("status") or "queued")
                if status not in {"queued", "requeued"}:
                    continue

                updated_at = float(state.get("updated_at") or 0.0)
                if updated_at and (now - updated_at) < stale_seconds:
                    continue

                rescue_count = int(state.get("rescue_count") or 0)
                assigned_gpu = str(state.get("assigned_gpu") or "")
                if rescue_count >= max_rescue_attempts:
                    if redis_client.hsetnx(
                        _job_candidate_terminal_key(job_id),
                        candidate_key,
                        "failed",
                    ):
                        redis_client.hincrby(f"tts:job:{job_id}", "failed_candidates", 1)
                        settled_now = redis_client.hincrby(
                            f"tts:job:{job_id}",
                            "settled_candidates",
                            1,
                        )
                        _set_candidate_state(
                            job_id,
                            candidate_id,
                            status="failed",
                            assigned_gpu=assigned_gpu or "unassigned",
                            updated_at=now,
                            rescue_count=rescue_count,
                            reason="stale_queue_timeout",
                        )
                        if settled_now >= expected:
                            _maybe_enqueue_score_task(job_id, expected)
                    continue

                rescue_device = _select_rescue_device(gpu_devices, excluded_device=assigned_gpu)
                cand_job = dict(payload)
                cand_job["task"] = "candidate"
                cand_job["candidate_id"] = candidate_id
                cand_job["assigned_gpu"] = rescue_device
                cand_job["rescue_count"] = rescue_count + 1
                redis_client.rpush(f"tts:jobs:{rescue_device}", json.dumps(cand_job))
                _set_candidate_state(
                    job_id,
                    candidate_id,
                    status="requeued",
                    assigned_gpu=rescue_device,
                    updated_at=now,
                    rescue_count=rescue_count + 1,
                    previous_gpu=assigned_gpu,
                )
                logger.warning(
                    "Requeued stale candidate to a healthy GPU",
                    extra={
                        "job_id": job_id,
                        "candidate_id": candidate_id,
                        "previous_gpu": assigned_gpu,
                        "rescue_gpu": rescue_device,
                        "rescue_count": rescue_count + 1,
                    },
                )
        except Exception:
            logger.exception("Failed to rescue stale candidate jobs")


def dispatch_loop(gpu_devices: list[str]) -> None:
    """Dispatch incoming jobs to per-GPU queues for parallel candidate generation."""
    logger.info("Starting job dispatcher", extra={"gpu_devices": gpu_devices})

    while True:
        try:
            item = redis_client.blpop(["tts:jobs"], timeout=5)
            if not item:
                _rescue_stale_candidate_jobs(gpu_devices)
                continue

            _, payload_json = item
            job = json.loads(payload_json)
            job_id = job.get("job_id")
            if not job_id:
                continue

            num_candidates = int(job.get("num_candidates", 1))
            model_type = job.get("model_type", "base")

            redis_client.hset(
                f"tts:job:{job_id}",
                mapping={
                    "status": "running",
                    "started_at": str(time.time()),
                    "num_candidates": str(num_candidates),
                    "completed_candidates": "0",
                    "failed_candidates": "0",
                    "settled_candidates": "0",
                },
            )
            redis_client.set(f"tts:job:{job_id}:payload", json.dumps(job))

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
        except Exception:
            logger.exception("Dispatcher error; continuing")
            time.sleep(1)


def main() -> None:
    """Start one or more job worker threads and block forever."""
    try:
        faulthandler.enable()
        faulthandler.register(signal.SIGUSR1, all_threads=True)
        logger.info("Enabled faulthandler SIGUSR1 process dumps")
    except Exception:
        logger.warning("Failed to enable faulthandler SIGUSR1 process dumps", exc_info=True)

    raw_concurrency = int(getattr(settings, "tts_worker_concurrency", 1))
    concurrency = max(1, raw_concurrency)

    logger.info("Starting dedicated job worker process", extra={"concurrency": concurrency})

    # Assign GPUs to worker processes in a round-robin fashion.
    gpu_ids = settings.tts_gpu_list()
    if not gpu_ids:
        # If no GPUs are specified, run all workers on CPU.
        # This is mainly for testing or CPU-only environments.
        logger.warning("No TTS_GPUS configured; all workers will run on CPU.")
        gpu_devices = ["cpu"]
    else:
        gpu_devices = [f"cuda:{gid}" for gid in gpu_ids]

    processes: list[multiprocessing.Process] = []

    dispatcher = threading.Thread(
        target=dispatch_loop,
        args=(gpu_devices,),
        daemon=True,
        name="tts-job-dispatcher",
    )
    dispatcher.start()
    for i in range(concurrency):
        # Assign a device (e.g., 'cuda:0' or 'cpu') to each worker.
        device = gpu_devices[i % len(gpu_devices)]
        queue_name = f"tts:jobs:{device}"
        p = multiprocessing.Process(
            target=job_worker_loop,
            args=(i, device, queue_name),
            daemon=False,
            name=f"tts-job-worker-{i}",
        )
        p.start()
        processes.append(p)

    for p in processes:
        p.join()


if __name__ == "__main__":
    main()
