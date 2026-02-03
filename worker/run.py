"""Dedicated background worker runner.

This runs the existing Redis list-based job worker loop (`tts:jobs`) in a
standalone process, so the FastAPI server can remain responsive and we avoid
RQ pubsub idle disconnect behavior.
"""

from __future__ import annotations

import faulthandler
import multiprocessing
import signal
import json
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

from api.main import job_worker_loop, logger, redis_client, settings


def dispatch_loop(gpu_devices: list[str]) -> None:
    """Dispatch incoming jobs to per-GPU queues for parallel candidate generation."""
    logger.info("Starting job dispatcher", extra={"gpu_devices": gpu_devices})

    while True:
        try:
            item = redis_client.blpop(["tts:jobs"], timeout=5)
            if not item:
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
                redis_client.rpush(f"tts:jobs:{target}", json.dumps(cand_job))
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
