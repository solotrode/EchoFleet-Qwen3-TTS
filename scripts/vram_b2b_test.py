"""Back-to-back TTS VRAM monitor.

Runs two sequential async TTS jobs and samples GPU memory/utilization for all visible GPUs.

This is designed to help correlate:
- Host GPU numbering
- Container-visible GPU ordering (nvidia-smi index)
- PyTorch CUDA device indices (torch.cuda)

Configuration is via environment variables (works well in Docker).

Environment variables:
- API_BASE_URL: Base URL for the API (default: http://qwen3-tts:8000)
- REF_WAV: Reference wav path inside the container (default: /workspace/outputs/ref.wav)
- SAMPLE_INTERVAL_S: GPU sample interval during job2 (default: 1.5)
- JOB_POLL_INTERVAL_S: Status polling interval (default: 1.5)
- MAX_JOB_SECONDS: Max seconds to wait for a job to finish (default: 1200)

Exit codes:
- 0: Success
- 2: Missing reference audio
- 3: API error
"""

from __future__ import annotations

import logging
import os
import subprocess
import time
from dataclasses import dataclass
from typing import Any

import requests

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class Settings:
    """Runtime configuration."""

    api_base_url: str
    ref_wav: str
    sample_interval_s: float
    job_poll_interval_s: float
    max_job_seconds: int


def _settings_from_env() -> Settings:
    return Settings(
        api_base_url=os.environ.get("API_BASE_URL", "http://qwen3-tts:8000"),
        ref_wav=os.environ.get("REF_WAV", "/workspace/outputs/ref.wav"),
        sample_interval_s=float(os.environ.get("SAMPLE_INTERVAL_S", "1.5")),
        job_poll_interval_s=float(os.environ.get("JOB_POLL_INTERVAL_S", "1.5")),
        max_job_seconds=int(os.environ.get("MAX_JOB_SECONDS", "1200")),
    )


def _run(cmd: list[str]) -> str:
    return subprocess.check_output(cmd, text=True, stderr=subprocess.STDOUT)


def gpu_inventory() -> list[dict[str, Any]]:
    """Return static GPU identity info from nvidia-smi."""
    out = _run(
        [
            "nvidia-smi",
            "--query-gpu=index,uuid,name,memory.total",
            "--format=csv,noheader,nounits",
        ]
    )
    gpus: list[dict[str, Any]] = []
    for line in out.strip().splitlines():
        parts = [p.strip() for p in line.split(",")]
        if len(parts) < 4:
            continue
        gpus.append(
            {
                "index": int(parts[0]),
                "uuid": parts[1],
                "name": parts[2],
                "mem_total_mib": int(parts[3]),
            }
        )
    return gpus


def sample_gpus() -> list[dict[str, Any]]:
    """Return current memory/utilization samples for all GPUs from nvidia-smi."""
    out = _run(
        [
            "nvidia-smi",
            "--query-gpu=index,uuid,memory.used,memory.total,utilization.gpu,utilization.memory,temperature.gpu",
            "--format=csv,noheader,nounits",
        ]
    )
    samples: list[dict[str, Any]] = []
    for line in out.strip().splitlines():
        parts = [p.strip() for p in line.split(",")]
        if len(parts) < 7:
            continue
        samples.append(
            {
                "index": int(parts[0]),
                "uuid": parts[1],
                "mem_used_mib": int(parts[2]),
                "mem_total_mib": int(parts[3]),
                "util_gpu_pct": int(parts[4]),
                "util_mem_pct": int(parts[5]),
                "temp_c": int(parts[6]),
            }
        )
    return samples


def log_torch_map() -> None:
    """Log PyTorch CUDA device list to help correlate indexing."""
    logger.info("Torch CUDA map (may differ from nvidia-smi indices)")
    logger.info("CUDA_VISIBLE_DEVICES=%s", os.environ.get("CUDA_VISIBLE_DEVICES"))
    try:
        import torch

        logger.info("torch.cuda.is_available=%s", torch.cuda.is_available())
        if not torch.cuda.is_available():
            return
        n = torch.cuda.device_count()
        logger.info("torch.cuda.device_count=%s", n)
        for i in range(n):
            props = torch.cuda.get_device_properties(i)
            logger.info(
                "torch_index=%s name=%s cc=%s.%s total_mem=%.0fMiB",
                i,
                props.name,
                props.major,
                props.minor,
                props.total_memory / 1024 / 1024,
            )
    except Exception:
        logger.exception("Failed to read torch cuda device info")


def submit_job(settings: Settings, text: str) -> tuple[str, str]:
    payload = {
        "text": text,
        "language": "en",
        "ref_audio": settings.ref_wav,
        "ref_text": text,
        "num_candidates": 1,
    }
    resp = requests.post(
        f"{settings.api_base_url}/v1/tts/voice-clone",
        json=payload,
        timeout=30,
    )
    resp.raise_for_status()
    data = resp.json()
    return data["job_id"], f"{settings.api_base_url}{data['status_url']}"


def get_status(status_url: str) -> dict[str, Any]:
    resp = requests.get(status_url, timeout=15)
    resp.raise_for_status()
    return resp.json()


def wait_done(settings: Settings, status_url: str, job_name: str) -> dict[str, Any]:
    start = time.time()
    while True:
        st = get_status(status_url)
        status = st.get("status")
        if status in ("done", "failed"):
            return st
        if time.time() - start > settings.max_job_seconds:
            raise TimeoutError(
                f"Timed out waiting for {job_name} after {settings.max_job_seconds}s"
            )
        time.sleep(settings.job_poll_interval_s)


def log_sample(t0: float, tag: str) -> list[dict[str, Any]]:
    t = time.time() - t0
    samples = sorted(sample_gpus(), key=lambda d: d["index"])
    for g in samples:
        logger.info(
            "t=+%.1fs %-12s gpu=%s uuid=%s mem=%s/%sMiB util=%s%% temp=%sC",
            t,
            tag,
            g["index"],
            g["uuid"],
            g["mem_used_mib"],
            g["mem_total_mib"],
            g["util_gpu_pct"],
            g["temp_c"],
        )
    return samples


def main() -> int:
    logging.basicConfig(level=os.environ.get("LOG_LEVEL", "INFO"))
    settings = _settings_from_env()

    if not os.path.exists(settings.ref_wav):
        logger.error("Missing REF_WAV at %s", settings.ref_wav)
        return 2

    try:
        inv = gpu_inventory()
    except Exception:
        logger.exception("Failed to read nvidia-smi inventory")
        return 3

    logger.info("nvidia-smi inventory")
    for g in inv:
        logger.info(
            "nvsmi_index=%s uuid=%s name=%s total=%sMiB",
            g["index"],
            g["uuid"],
            g["name"],
            g["mem_total_mib"],
        )

    log_torch_map()

    t0 = time.time()
    logger.info("Baseline sample")
    baseline = {g["index"]: g["mem_used_mib"] for g in log_sample(t0, "baseline")}

    try:
        logger.info("Job 1 submit")
        job1, url1 = submit_job(settings, "VRAM back-to-back test job 1")
        logger.info("Job1 submitted: %s", job1)
        st1 = wait_done(settings, url1, "job1")
        logger.info("Job1 finished: status=%s", st1.get("status"))

        time.sleep(2)
        logger.info("Post-job1 VRAM")
        after1_samples = log_sample(t0, "after_job1")
        after1 = {g["index"]: g["mem_used_mib"] for g in after1_samples}

        logger.info("Job 2 submit (monitoring all GPUs)")
        job2, url2 = submit_job(settings, "VRAM back-to-back test job 2")
        logger.info("Job2 submitted: %s", job2)

        peak2: dict[int, int] = {g["index"]: 0 for g in inv}
        start2 = time.time()
        while True:
            st2 = get_status(url2)
            status2 = st2.get("status", "?")
            tag = f"job2_{status2}"
            smp = log_sample(t0, tag)
            for g in smp:
                peak2[g["index"]] = max(peak2.get(g["index"], 0), g["mem_used_mib"])

            if status2 in ("done", "failed"):
                logger.info("Job2 finished: status=%s", status2)
                break

            if time.time() - start2 > settings.max_job_seconds:
                raise TimeoutError(f"Timed out monitoring job2 after {settings.max_job_seconds}s")

            time.sleep(settings.sample_interval_s)

        time.sleep(2)
        logger.info("Post-job2 VRAM")
        after2 = {g["index"]: g["mem_used_mib"] for g in log_sample(t0, "after_job2")}

        logger.info("Summary (MiB)")
        for g in sorted([x["index"] for x in inv]):
            b = baseline.get(g)
            a1 = after1.get(g)
            a2 = after2.get(g)
            p2 = peak2.get(g)
            logger.info(
                "gpu=%s baseline=%s after1=%s peak2=%s after2=%s delta(base->after1)=%s delta(after1->peak2)=%s",
                g,
                b,
                a1,
                p2,
                a2,
                (a1 - b) if (b is not None and a1 is not None) else "NA",
                (p2 - a1) if (p2 is not None and a1 is not None) else "NA",
            )

    except requests.RequestException:
        logger.exception("API request failed")
        return 3
    except Exception:
        logger.exception("VRAM back-to-back test failed")
        return 3

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
