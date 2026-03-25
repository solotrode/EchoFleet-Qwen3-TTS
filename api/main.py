"""FastAPI server for Qwen3-TTS Multi-GPU service.

This module provides the main FastAPI application with health check
and placeholder endpoints. Full TTS endpoints will be added in Phase 4.
"""
import asyncio
import base64
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from typing import Any, Dict
import sys
from typing import Optional
from concurrent.futures import ThreadPoolExecutor, wait
from datetime import datetime, timezone
import shutil

from utils.logging import get_logger
from config.settings import get_settings
from api.schemas import (
    AudioResponse,
    # Keep S2ProRequest local to the module that needs it; add import here so the
    # schema isn't imported at package import time unnecessarily.
    CustomVoiceRequest,
    EnhancedAudioResponse,
    ErrorResponse,
    JobSubmitResponse,
    VoiceCloneRequest,
    VoiceDesignRequest,
)
from api.schemas import S2ProRequest
# Lazy-import friendly fallbacks: importing heavy inference modules at import
# time causes test collection and lightweight imports to fail when optional
# dependencies are missing. Provide graceful fallbacks so the module can be
# imported in test environments; functions that actually need these classes
# will raise clear runtime errors when used.
try:
    from inference.qwen_tts_service import QwenTTSServer, wav_to_wav_bytes
except Exception:
    QwenTTSServer = None

    def wav_to_wav_bytes(wav, sr):
        raise RuntimeError("inference.qwen_tts_service not available in this environment")

try:
    from inference.accuracy_scorer import AccuracyScorer
except Exception:
    AccuracyScorer = None

try:
    from inference.whisper_service import WhisperTranscriberHF, WhisperTranscription
except Exception:
    WhisperTranscriberHF = None

    class WhisperTranscription:  # minimal stub used for isinstance checks
        def __init__(self, text: str = ""):
            self.text = text
from fastapi import Request
from fastapi.responses import JSONResponse, FileResponse, Response
import os
import uuid
import json
from redis import Redis
import threading
import time
import gc
import torch

logger = get_logger(__name__)
settings = get_settings()
# Each worker process will have its own TTS server instance.
# This avoids sharing state between processes.
# tts_server = QwenTTSServer(settings)
redis_client = Redis(host=settings.redis_host, port=settings.redis_port, decode_responses=True)



_SYNC_TTS_SERVER: Optional[QwenTTSServer] = None
_SYNC_TTS_LOCK = threading.Lock()


def _gpu_queue_name(device: str) -> str:
    """Return the Redis queue name for a specific GPU device."""
    return f"tts:jobs:{device}"


def _select_score_device() -> str:
    """Select a GPU device for scoring tasks."""
    gpu_ids = settings.tts_gpu_list()
    preferred = settings.tts_preferred_gpu
    if preferred in gpu_ids:
        return f"cuda:{preferred}"
    if gpu_ids:
        return f"cuda:{gpu_ids[0]}"
    return "cpu"


def _load_job_payload(job_id: str) -> Dict[str, Any]:
    """Load a stored job payload from Redis."""
    raw = redis_client.get(f"tts:job:{job_id}:payload")
    if not raw:
        return {}
    try:
        return json.loads(raw)
    except Exception:
        logger.warning("Failed to decode job payload", extra={"job_id": job_id})
        return {}


def get_sync_tts_server() -> QwenTTSServer:
    """Return a singleton TTS server for synchronous API endpoints.

    The API service is configured with TTS_WORKER_CONCURRENCY=0, so it should
    only handle synchronous requests. We lazily create a local server here to
    avoid loading models at import time.
    """
    global _SYNC_TTS_SERVER
    if _SYNC_TTS_SERVER is None:
        with _SYNC_TTS_LOCK:
            if _SYNC_TTS_SERVER is None:
                device = "cpu"
                if torch.cuda.is_available():
                    device = f"cuda:{settings.tts_preferred_gpu}"
                _SYNC_TTS_SERVER = QwenTTSServer(settings, assigned_device=device)
                logger.info("Initialized sync TTS server", extra={"device": device})
    return _SYNC_TTS_SERVER


# S2 Pro singleton accessor
_SYNC_FISH_AUDIO_SERVICE = None
_SYNC_FISH_AUDIO_LOCK = threading.Lock()
_FISH_IDLE_WATCHER_THREAD: Optional[threading.Thread] = None
_FISH_IDLE_WATCHER_STOP = threading.Event()


def get_sync_fish_audio_service():
    """Return a singleton FishAudioService for sync S2 Pro calls.

    This lazily imports the `inference.fish_audio_service` module so that the
    package can be imported in test environments where optional runtime
    dependencies may be missing.
    """
    global _SYNC_FISH_AUDIO_SERVICE
    if _SYNC_FISH_AUDIO_SERVICE is None:
        with _SYNC_FISH_AUDIO_LOCK:
            if _SYNC_FISH_AUDIO_SERVICE is None:
                try:
                    from inference.fish_audio_service import FishAudioService

                    _SYNC_FISH_AUDIO_SERVICE = FishAudioService()
                except Exception as e:
                    logger.exception("Failed to initialize FishAudioService: %s", e)
                    raise
    return _SYNC_FISH_AUDIO_SERVICE


def _run_fish_idle_watcher() -> None:
    """Unload the sync Fish backend after the configured idle timeout."""
    poll_interval = 5.0
    idle_unload_seconds = int(getattr(settings, "tts_unload_idle_seconds", 300))

    while not _FISH_IDLE_WATCHER_STOP.wait(timeout=poll_interval):
        if idle_unload_seconds <= 0:
            continue

        svc = _SYNC_FISH_AUDIO_SERVICE
        if svc is None:
            continue

        try:
            unloaded = svc.unload_idle_models(idle_seconds=idle_unload_seconds)
            if unloaded:
                logger.info("Unloaded idle Fish backend", extra={"unloaded": unloaded})
        except Exception:
            logger.exception("Failed to unload idle Fish backend")


# s2-pro endpoint is registered after app is created below


def _generate_candidate(
    tts_server: QwenTTSServer,
    text: str,
    language: Optional[str],
    ref_audio: str,
    ref_text: Optional[str],
    x_vector_only_mode: bool,
    job_id: str,
    candidate_id: int,
    max_retries: int = 2,
) -> Optional[Dict[str, Any]]:
    """Generate a single candidate with the TTS server and persist the WAV to disk.

    Returns a dict with candidate metadata or None on failure after retries.
    """
    last_error = None

    for attempt in range(max_retries + 1):
        try:
            # Per-candidate generation timing telemetry
            candidate_gen_start = time.time()
            wav, sr = tts_server.generate_voice_clone(
                text=text,
                language=language,
                ref_audio=ref_audio,
                ref_text=ref_text,
                x_vector_only_mode=x_vector_only_mode,
            )
            candidate_gen_end = time.time()

            wav_bytes = wav_to_wav_bytes(wav, sr)
            candidate_dir = os.path.join(settings.output_dir, job_id, "candidates")
            os.makedirs(candidate_dir, exist_ok=True)
            wav_path = os.path.join(candidate_dir, f"candidate_{candidate_id}.wav")
            with open(wav_path, "wb") as f:
                f.write(wav_bytes)
                try:
                    f.flush()
                    os.fsync(f.fileno())
                except Exception:
                    # Best-effort flush; don't treat as generation failure
                    logger.warning("Failed to fsync candidate file", extra={"job_id": job_id, "candidate_id": candidate_id, "path": wav_path})

            logger.info("candidate_saved", extra={"job_id": job_id, "candidate_id": candidate_id, "wav_path": wav_path})

            audio_b64 = base64.b64encode(wav_bytes).decode("utf-8")
            duration = float(len(wav) / sr) if hasattr(wav, "__len__") else 0.0

            if attempt > 0:
                logger.info(
                    f"Candidate {candidate_id} succeeded after {attempt} retries",
                    extra={"job_id": job_id, "candidate_id": candidate_id, "attempt": attempt}
                )

            result = {
                "candidate_id": candidate_id,
                "wav_path": wav_path,
                "audio_base64": audio_b64,
                "sample_rate": sr,
                "duration_seconds": duration,
                "tts_gpu": None,
                "timings": {
                    "generation_start": candidate_gen_start,
                    "generation_end": candidate_gen_end,
                    "generation_duration": round(candidate_gen_end - candidate_gen_start, 3),
                },
            }
            # NOTE: Do NOT aggressively unload GPU memory here. Keeping the
            # TTS model resident across candidate generation avoids repeated
            # model reloads and large unload/load delays. A single unload is
            # performed later in the job lifecycle before loading Whisper.
            return result
        except RuntimeError as e:
            error_msg = str(e)
            last_error = error_msg

            # Check for tensor size mismatch (common issue with reference audio)
            if "size of tensor" in error_msg and "must match" in error_msg:
                logger.warning(
                    f"Candidate {candidate_id} failed with tensor size mismatch (attempt {attempt + 1}/{max_retries + 1})",
                    extra={"job_id": job_id, "candidate_id": candidate_id, "error": error_msg}
                )
                if attempt < max_retries:
                    # Small delay before retry
                    time.sleep(0.5 * (attempt + 1))
                    continue
            else:
                logger.exception(f"Candidate {candidate_id} generation failed with RuntimeError")
                break
        except Exception as e:
            last_error = str(e)
            logger.exception(f"Candidate {candidate_id} generation failed (attempt {attempt + 1}/{max_retries + 1})")
            if attempt < max_retries:
                time.sleep(0.5 * (attempt + 1))
                continue
            break

    logger.error(
        f"Candidate {candidate_id} failed after all retries",
        extra={"job_id": job_id, "candidate_id": candidate_id, "last_error": last_error}
    )
    return None


def _score_for_audit_log(score: Dict[str, Any], max_preview_chars: int = 160) -> Dict[str, Any]:
    """Reduce a score dict to audit-safe fields for logging.

    Args:
        score: Score dict returned by AccuracyScorer.
        max_preview_chars: Max characters to include for transcription preview.

    Returns:
        A dict safe to log.
    """
    transcribed_text = str(score.get("transcribed_text") or "")
    preview = transcribed_text[:max_preview_chars]
    if len(transcribed_text) > max_preview_chars:
        preview = preview + "...(truncated)"

    return {
        "accuracy_score": score.get("accuracy_score"),
        "word_error_rate": score.get("word_error_rate"),
        "char_error_rate": score.get("char_error_rate"),
        "duration_seconds": score.get("duration_seconds"),
        "reference_words": score.get("reference_words"),
        "transcribed_words": score.get("transcribed_words"),
        "transcribed_text_preview": preview,
    }


def _utc_now_iso() -> str:
    """Return current UTC time in ISO-8601 with a Z suffix."""
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _score_candidates(job_id: str, candidates: list[Dict[str, Any]], job: Dict[str, Any]) -> None:
    """Score candidates and finalize a job in Redis.

    Args:
        job_id: Job identifier.
        candidates: Candidate results from workers.
        job: Original job payload.
    """
    stt_devices = settings.whisper_device_list()
    transcribers: Dict[str, WhisperTranscriberHF] = {}
    for dev in stt_devices:
        try:
            transcribers[dev] = WhisperTranscriberHF(settings.whisper_model_id(), device=dev)
        except Exception:
            logger.exception("Failed to init transcriber on %s", dev)

    scorer = AccuracyScorer()
    scored: list[Dict[str, Any]] = []

    for cand in candidates:
        dev = stt_devices[cand["candidate_id"] % max(1, len(stt_devices))]
        trans = transcribers.get(dev)
        if not trans:
            continue
        try:
            transcription = trans.transcribe(cand["wav_path"])
            trans_text = (
                transcription.text
                if isinstance(transcription, WhisperTranscription)
                else transcription.get("text", "")
                if isinstance(transcription, dict)
                else str(transcription)
            )
            duration = cand.get("duration_seconds", 0.0)
            score = scorer.score_candidate(
                reference=job.get("text"),
                transcription=trans_text,
                duration=duration,
            )
            scored.append({
                "candidate_id": cand["candidate_id"],
                "wav_path": cand["wav_path"],
                "audio_base64": cand["audio_base64"],
                "sample_rate": cand["sample_rate"],
                "duration_seconds": duration,
                "tts_gpu": cand.get("tts_gpu"),
                "timings": cand.get("timings", {}),
                "stt_device": dev,
                "score": score,
            })
        except Exception:
            logger.exception("Transcription/scoring failed for candidate %s", cand.get("candidate_id"))

    if not scored:
        raise RuntimeError("All transcriptions failed")

    # Unload Whisper models after scoring
    for dev, transcriber in transcribers.items():
        try:
            transcriber.unload()
            logger.info("Whisper model unloaded", extra={"device": dev, "job_id": job_id})
        except Exception as exc:
            logger.warning("Failed to unload Whisper", extra={"device": dev, "job_id": job_id, "error": str(exc)})

    scored.sort(
        key=lambda c: (
            -float((c.get("score") or {}).get("accuracy_score") or 0.0),
            float((c.get("score") or {}).get("duration_seconds") or 0.0),
        )
    )
    best = scored[0]

    result_path = ""
    try:
        result_payload: Dict[str, Any] = {
            "job_id": job_id,
            "completed_at": _utc_now_iso(),
            "num_candidates": int(job.get("num_candidates", len(scored))),
            "winner_candidate_id": best.get("candidate_id"),
            "winner_score": _score_for_audit_log(best.get("score") or {}),
            "candidates": [
                {
                    "candidate_id": c.get("candidate_id"),
                    "tts_gpu": c.get("tts_gpu"),
                    "stt_device": c.get("stt_device"),
                    "wav_path": c.get("wav_path"),
                    "sample_rate": c.get("sample_rate"),
                    "duration_seconds": c.get("duration_seconds"),
                    "score": _score_for_audit_log(c.get("score") or {}),
                }
                for c in scored
            ],
        }
        result_path = _write_job_result_file(job_id, result_payload)
        logger.info("Wrote per-job result summary", extra={"job_id": job_id, "path": result_path})
    except Exception:
        logger.warning("Failed to write per-job result summary", exc_info=True)

    redis_client.hset(
        f"tts:job:{job_id}",
        mapping={
            "status": "done",
            "audio_base64": best.get("audio_base64"),
            "sample_rate": best.get("sample_rate"),
            "duration_seconds": best.get("duration_seconds"),
            "wav_path": best.get("wav_path"),
            "download_url": f"/v1/audio/{job_id}.wav",
            "best_candidate": json.dumps(best.get("score")),
            "all_candidates": json.dumps(scored) if bool(job.get("return_all_candidates", False)) else "",
            "num_candidates_generated": len(scored),
            "winner_candidate_id": str(best.get("candidate_id")),
            "result_path": result_path,
            "completed_at": str(time.time()),
        },
    )


def _write_job_result_file(job_id: str, payload: Dict[str, Any]) -> str:
    """Write a per-job result summary JSON file.

    This intentionally stores only per-job results (candidate scores + winner)
    and not the full service-wide audit trail.

    Args:
        job_id: Job identifier.
        payload: JSON-serializable dict.

    Returns:
        Absolute path to the written JSON file.
    """
    job_dir = os.path.join(settings.output_dir, job_id)
    os.makedirs(job_dir, exist_ok=True)
    out_path = os.path.join(job_dir, "result.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
    return out_path


def _cleanup_old_job_artifacts(output_dir: str, retention_days: int) -> None:
    """Delete job artifacts older than retention_days.

    Deletes per-job directories under OUTPUT_DIR and any root-level wav files
    that might have been written by older code paths.
    """
    if retention_days <= 0:
        return

    now = time.time()
    cutoff = now - float(retention_days) * 86400.0

    try:
        entries = list(os.scandir(output_dir))
    except FileNotFoundError:
        return

    for entry in entries:
        try:
            mtime = entry.stat().st_mtime
        except Exception:
            continue

        if mtime >= cutoff:
            continue

        try:
            if entry.is_dir(follow_symlinks=False):
                shutil.rmtree(entry.path, ignore_errors=True)
                logger.info(
                    "Deleted expired job directory",
                    extra={"path": entry.path, "retention_days": retention_days},
                )
            elif entry.is_file(follow_symlinks=False) and entry.name.lower().endswith(".wav"):
                os.remove(entry.path)
                logger.info(
                    "Deleted expired wav artifact",
                    extra={"path": entry.path, "retention_days": retention_days},
                )
        except Exception:
            logger.warning("Failed to delete expired artifact", exc_info=True)


def _enqueue_voice_clone_job(request: VoiceCloneRequest) -> JobSubmitResponse:
    """Enqueue a voice-clone job and return metadata for polling.

    Args:
        request: Voice clone request payload.

    Returns:
        JobSubmitResponse containing job_id and URLs.

    Raises:
        ValueError: If required reference fields are missing.
    """
    if not request.x_vector_only_mode:
        if not request.ref_audio or not str(request.ref_audio).strip():
            raise ValueError("ref_audio is required for voice cloning unless x_vector_only_mode=True")
        if not request.ref_text or not str(request.ref_text).strip():
            raise ValueError("ref_text is required for voice cloning unless x_vector_only_mode=True")

    job_id = uuid.uuid4().hex
    payload = {
        "job_id": job_id,
        "text": request.text,
        "language": request.language,
        "ref_audio": request.ref_audio,
        "ref_text": request.ref_text,
        "x_vector_only_mode": request.x_vector_only_mode,
        "num_candidates": request.num_candidates,
        "return_all_candidates": request.return_all_candidates,
    }

    redis_client.hset(
        f"tts:job:{job_id}",
        mapping={
            "status": "queued",
            "created_at": str(time.time()),
            "num_candidates": str(int(request.num_candidates or 1)),
        },
    )
    redis_client.rpush("tts:jobs", json.dumps(payload))
    return JobSubmitResponse(
        job_id=job_id,
        status_url=f"/v1/jobs/{job_id}",
        audio_url=f"/v1/audio/{job_id}.wav",
        status="queued",
    )


def _enqueue_voice_design_job(request: "VoiceDesignRequest") -> JobSubmitResponse:
    """Enqueue a voice-design job and return metadata for polling.

    Args:
        request: VoiceDesignRequest payload.

    Returns:
        JobSubmitResponse containing job_id and URLs.
    """
    job_id = uuid.uuid4().hex
    payload = {
        "job_id": job_id,
        "text": request.text,
        "language": request.language,
        "instruct": request.instruct,
        "model_type": "voice-design",
    }

    redis_client.hset(
        f"tts:job:{job_id}",
        mapping={
            "status": "queued",
            "created_at": str(time.time()),
            "model_type": "voice-design",
        },
    )
    redis_client.rpush("tts:jobs", json.dumps(payload))
    return JobSubmitResponse(
        job_id=job_id,
        status_url=f"/v1/jobs/{job_id}",
        audio_url=f"/v1/audio/{job_id}.wav",
        status="queued",
    )

app = FastAPI(
    title="Qwen3-TTS Multi-GPU API",
    description="Multi-model text-to-speech API with voice cloning, custom voices, and voice design",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
)

# Global request size limit (bytes). Can be overridden by env `MAX_REQUEST_SIZE`.
MAX_REQUEST_SIZE = int(os.getenv("MAX_REQUEST_SIZE", str(1_000_000)))


@app.middleware("http")
async def limit_request_size(request: Request, call_next):
    """Reject requests with bodies larger than MAX_REQUEST_SIZE to avoid
    log flooding and resource exhaustion.
    """
    # Fast path: honor Content-Length header when present
    cl = request.headers.get("content-length")
    try:
        if cl and int(cl) > MAX_REQUEST_SIZE:
            return JSONResponse(status_code=413, content={"detail": "Payload too large"})
    except Exception:
        # If header is malformed, fall through to reading body
        pass

    body = await request.body()
    if len(body) > MAX_REQUEST_SIZE:
        return JSONResponse(status_code=413, content={"detail": "Payload too large"})

    # Store body back onto the request so downstream can read it
    request._body = body
    return await call_next(request)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure appropriately for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.on_event("startup")
async def startup_event():
    """Initialize resources on startup."""
    logger.info("Starting Qwen3-TTS Multi-GPU API server")
    logger.info(f"Redis: {settings.redis_host}:{settings.redis_port}")
    logger.info(f"TTS GPUs: {settings.tts_gpu_list()}")
    logger.info(f"Model cache: {settings.model_cache_dir}")
    logger.info(f"Output directory: {settings.output_dir}")

    # The `qwen3-worker` service is responsible for initializing executors
    # and running jobs. The API server (`qwen3-tts`) only handles requests
    # and does not need to start its own workers or executors.
    raw_concurrency = int(getattr(settings, "tts_worker_concurrency", 1))
    if raw_concurrency > 0:
        logger.info(
            "Job workers are managed by the dedicated 'qwen3-worker' service, not the API server.",
            extra={"concurrency": raw_concurrency},
        )
    else:
        logger.info("In-process job workers disabled.", extra={"concurrency": raw_concurrency})

    global _FISH_IDLE_WATCHER_THREAD
    if _FISH_IDLE_WATCHER_THREAD is None or not _FISH_IDLE_WATCHER_THREAD.is_alive():
        _FISH_IDLE_WATCHER_STOP.clear()
        _FISH_IDLE_WATCHER_THREAD = threading.Thread(
            target=_run_fish_idle_watcher,
            name="fish-idle-watcher",
            daemon=True,
        )
        _FISH_IDLE_WATCHER_THREAD.start()


def job_worker_loop(worker_id: int, device: str, queue_name: str) -> None:
    """Background worker loop that processes jobs from Redis list `tts:jobs`.

    This runs in a dedicated process, assigned to a specific GPU. It blocks on
    `blpop` and executes generation synchronously, storing results in a

    Args:
        worker_id: The integer ID of this worker process.
        device: The device this worker should use (e.g., 'cuda:0' or 'cpu').
    """
    # Each worker process gets its own TTS server instance and its own executor.
    tts_server = QwenTTSServer(settings, assigned_device=device)

    # Create a dedicated executor for this worker process.
    # NOTE: The underlying TTS model is not thread-safe for concurrent generation
    # within the same process, so we serialize candidate generation per worker.
    max_sub_workers = 1
    executor = ThreadPoolExecutor(max_workers=max_sub_workers)

    logger.info(
        "Starting job worker loop",
        extra={
            "worker_id": worker_id,
            "device": device,
            "queue": queue_name,
            "sub_workers": max_sub_workers,
        },
    )

    last_job_time = time.time()
    idle_unload_seconds = int(getattr(settings, "tts_unload_idle_seconds", 300))

    while True:
        try:
            # Check for both jobs and commands (use blpop with multiple keys)
            item = redis_client.blpop([queue_name, "tts:commands"], timeout=5)

            # Check for idle models to unload
            if idle_unload_seconds > 0:
                idle_time = time.time() - last_job_time
                if idle_time >= idle_unload_seconds:
                    try:
                        unloaded = tts_server.unload_idle_models(idle_seconds=idle_unload_seconds)
                        if unloaded:
                            logger.info(
                                "Unloaded idle models",
                                extra={"worker_id": worker_id, "device": device, "unloaded": unloaded},
                            )
                    except Exception as e:
                        logger.warning(f"Failed to unload idle models: {e}")
                    finally:
                        last_job_time = time.time()

            if not item:
                continue

            queue_name, payload_json = item

            # Handle command queue
            if queue_name == "tts:commands":
                try:
                    cmd = json.loads(payload_json)
                    cmd_id = cmd.get("cmd_id")
                    cmd_type = cmd.get("type")

                    if cmd_type == "unload":
                        model_type = cmd.get("model_type")
                        logger.info(f"Worker received unload command: {cmd_id}, model_type={model_type}")
                        unloaded = tts_server.unload_model(model_type)
                        redis_client.hset(
                            f"tts:cmd:{cmd_id}",
                            mapping={
                                "status": "done",
                                "result": json.dumps(unloaded),
                                "completed_at": str(time.time()),
                            }
                        )
                        redis_client.expire(f"tts:cmd:{cmd_id}", 60)
                        logger.info(f"Worker unload complete: {unloaded}")
                except Exception as e:
                    logger.exception("Failed to process command")
                    if cmd_id:
                        redis_client.hset(
                            f"tts:cmd:{cmd_id}",
                            mapping={"status": "error", "error": str(e)}
                        )
                continue

            # Handle job queue
            job = json.loads(payload_json)
            job_id = job.get("job_id")
            if not job_id:
                continue

            task_type = job.get("task", "job")

            # Candidate-only task: generate audio and store result for scoring.
            if task_type == "candidate":
                candidate_id = int(job.get("candidate_id", 0))
                num_candidates = int(job.get("num_candidates", 1))
                result = _generate_candidate(
                    tts_server,
                    job.get("text"),
                    job.get("language"),
                    job.get("ref_audio"),
                    job.get("ref_text"),
                    job.get("x_vector_only_mode", False),
                    job_id,
                    candidate_id,
                )
                if result:
                    redis_client.rpush(f"tts:job:{job_id}:candidates", json.dumps(result))
                    completed = redis_client.hincrby(
                        f"tts:job:{job_id}",
                        "completed_candidates",
                        1,
                    )
                    expected = int(job.get("num_candidates", 1))
                    if completed >= expected and expected > 1:
                        if redis_client.setnx(f"tts:job:{job_id}:score_enqueued", "1"):
                            score_device = _select_score_device()
                            score_task = {"task": "score", "job_id": job_id}
                            redis_client.rpush(_gpu_queue_name(score_device), json.dumps(score_task))
                last_job_time = time.time()
                continue

            # Score-only task: gather candidates and finalize job.
            if task_type == "score":
                job_payload = _load_job_payload(job_id)
                if not job_payload:
                    redis_client.hset(
                        f"tts:job:{job_id}",
                        mapping={"status": "failed", "error": "missing_job_payload"},
                    )
                    continue

                cand_items = redis_client.lrange(f"tts:job:{job_id}:candidates", 0, -1)
                candidates = []
                for item_json in cand_items:
                    try:
                        candidates.append(json.loads(item_json))
                    except Exception:
                        logger.warning("Failed to decode candidate payload", extra={"job_id": job_id})

                if not candidates:
                    redis_client.hset(
                        f"tts:job:{job_id}",
                        mapping={"status": "failed", "error": "no_candidates"},
                    )
                    continue

                try:
                    _score_candidates(job_id=job_id, candidates=candidates, job=job_payload)
                except Exception as exc:
                    logger.exception("Scoring failed", extra={"job_id": job_id})
                    redis_client.hset(
                        f"tts:job:{job_id}",
                        mapping={"status": "failed", "error": str(exc)[:200]},
                    )
                last_job_time = time.time()
                continue

            # Log full job text for diagnostic purposes
            job_text = job.get("text", "")
            logger.info(
                "Worker claimed job",
                extra={
                    "job_id": job_id,
                    "text_len": len(job_text),
                    "text_start": job_text[:200] if job_text else "",
                    "text_end": job_text[-200:] if len(job_text) > 200 else job_text
                }
            )

            key = f"tts:job:{job_id}"
            # Record that the job has been claimed by a worker and initialize timings
            redis_client.hset(key, mapping={"status": "running", "started_at": str(time.time())})
            timings: Dict[str, float] = {}
            timings["claimed_at"] = time.time()

            try:
                # Job type and candidate generation parameters
                model_type = job.get("model_type", "base")
                num_candidates = int(job.get("num_candidates", 1))
                return_all = bool(job.get("return_all_candidates", False))

                # Handle model-type specific single-shot jobs (voice-design)
                if model_type == "voice-design":
                    # Voice-design produces a single output; ignore candidate parameters
                    # Instrument: record generation start/end
                    timings["generation_start"] = time.time()
                    wav, sr = tts_server.generate_voice_design(
                        text=job.get("text"),
                        language=job.get("language"),
                        instruct=job.get("instruct"),
                    )
                    timings["generation_end"] = time.time()
                    wav_bytes = wav_to_wav_bytes(wav, sr)
                    out_path = ""
                    try:
                        out_dir = settings.output_dir
                        os.makedirs(out_dir, exist_ok=True)
                        out_path = os.path.join(out_dir, f"{job_id}.wav")
                        with open(out_path, "wb") as f:
                            f.write(wav_bytes)
                        logger.info("Wrote generated WAV to disk", extra={"job_id": job_id, "path": out_path})
                        timings["postprocess_write_wav_at"] = time.time()
                    except Exception:
                        logger.warning("Failed to write WAV to disk", exc_info=True)

                    result_path = ""
                    try:
                        result_payload: Dict[str, Any] = {
                            "job_id": job_id,
                            "completed_at": _utc_now_iso(),
                            "num_candidates": 1,
                            "timings": {
                                "claimed_at": timings.get("claimed_at"),
                                "generation_start": timings.get("generation_start"),
                                "generation_end": timings.get("generation_end"),
                                "postprocess_write_wav_at": timings.get("postprocess_write_wav_at"),
                            },
                            "winner_candidate_id": 0,
                            "candidates": [
                                {
                                    "candidate_id": 0,
                                    "wav_path": out_path,
                                    "sample_rate": sr,
                                    "duration_seconds": float(len(wav) / sr) if hasattr(wav, "__len__") else 0.0,
                                    "score": None,
                                }
                            ],
                        }
                        result_path = _write_job_result_file(job_id, result_payload)
                        logger.info(
                            "Wrote per-job result summary",
                            extra={"job_id": job_id, "path": result_path},
                        )
                    except Exception:
                        logger.warning("Failed to write per-job result summary", exc_info=True)

                    audio_b64 = base64.b64encode(wav_bytes).decode("utf-8")
                    duration = float(len(wav) / sr) if hasattr(wav, "__len__") else 0.0
                    download_url = f"/v1/audio/{job_id}.wav"
                    timings["completed_at"] = time.time()
                    redis_client.hset(key, mapping={
                        "status": "done",
                        "audio_base64": audio_b64,
                        "sample_rate": sr,
                        "duration_seconds": duration,
                        "wav_path": out_path if 'out_path' in locals() else "",
                        "result_path": result_path,
                        "download_url": download_url,
                        "completed_at": str(time.time()),
                        "timings": json.dumps(timings),
                    })
                    continue

                # Single-candidate path
                if num_candidates <= 1:
                    timings["generation_start"] = time.time()
                    wav, sr = tts_server.generate_voice_clone(
                        text=job.get("text"),
                        language=job.get("language"),
                        ref_audio=job.get("ref_audio"),
                        ref_text=job.get("ref_text"),
                        x_vector_only_mode=job.get("x_vector_only_mode", False),
                    )
                    timings["generation_end"] = time.time()
                    wav_bytes = wav_to_wav_bytes(wav, sr)
                    out_path = ""
                    try:
                        out_dir = settings.output_dir
                        os.makedirs(out_dir, exist_ok=True)
                        out_path = os.path.join(out_dir, f"{job_id}.wav")
                        with open(out_path, "wb") as f:
                            f.write(wav_bytes)
                        logger.info("Wrote generated WAV to disk", extra={"job_id": job_id, "path": out_path})
                    except Exception:
                        logger.warning("Failed to write WAV to disk", exc_info=True)

                    result_path = ""
                    try:
                        result_payload: Dict[str, Any] = {
                            "job_id": job_id,
                            "completed_at": _utc_now_iso(),
                            "num_candidates": 1,
                            "winner_candidate_id": 0,
                            "candidates": [
                                {
                                    "candidate_id": 0,
                                    "wav_path": out_path,
                                    "sample_rate": sr,
                                    "duration_seconds": float(len(wav) / sr) if hasattr(wav, "__len__") else 0.0,
                                    "score": None,
                                }
                            ],
                        }
                        result_path = _write_job_result_file(job_id, result_payload)
                        logger.info(
                            "Wrote per-job result summary",
                            extra={"job_id": job_id, "path": result_path},
                        )
                    except Exception:
                        logger.warning("Failed to write per-job result summary", exc_info=True)

                    audio_b64 = base64.b64encode(wav_bytes).decode("utf-8")
                    duration = float(len(wav) / sr) if hasattr(wav, "__len__") else 0.0
                    download_url = f"/v1/audio/{job_id}.wav"
                    timings["postprocess_write_wav_at"] = timings.get("postprocess_write_wav_at", time.time())
                    timings["completed_at"] = time.time()
                    redis_client.hset(key, mapping={
                        "status": "done",
                        "audio_base64": audio_b64,
                        "sample_rate": sr,
                        "duration_seconds": duration,
                        "wav_path": out_path if 'out_path' in locals() else "",
                        "result_path": result_path,
                        "download_url": download_url,
                        "completed_at": str(time.time()),
                        "timings": json.dumps(timings),
                    })
                else:
                    # Multi-candidate generation
                    candidates = []
                    futures = []
                    candidate_dir = os.path.join(settings.output_dir, job_id, "candidates")
                    os.makedirs(candidate_dir, exist_ok=True)

                    # Instrument generation timing for multi-candidate
                    timings["generation_start"] = time.time()
                    # Spawn generation tasks
                    for i in range(num_candidates):
                        futures.append(executor.submit(
                            _generate_candidate,
                            tts_server,
                            job.get("text"),
                            job.get("language"),
                            job.get("ref_audio"),
                            job.get("ref_text"),
                            job.get("x_vector_only_mode", False),
                            job_id,
                            i,
                        ))

                    gen_timeout = None
                    if float(getattr(settings, "job_timeout", 0) or 0) > 0:
                        gen_timeout = float(settings.job_timeout)

                    done, not_done = wait(futures, timeout=gen_timeout)
                    for fut in done:
                        try:
                            res = fut.result()
                        except Exception:
                            res = None
                        if res:
                            candidates.append(res)

                    # Best-effort: cancel any remaining tasks to avoid runaway CPU/GPU work.
                    for fut in not_done:
                        try:
                            fut.cancel()
                        except Exception:
                            pass

                    if not_done and gen_timeout is not None:
                        raise TimeoutError("Candidate generation timed out")

                    timings["generation_end"] = time.time()
                    if not candidates:
                        raise RuntimeError("All candidates failed to generate")
                    
                    # Unload Qwen TTS models BEFORE loading Whisper only when
                    # there are no additional jobs queued. Deferring unload while
                    # jobs remain prevents repeated unload/load cycles during
                    # retries and keeps TTS resident for subsequent candidates.
                    try:
                        queued_jobs = 0
                        try:
                            queued_jobs = redis_client.llen("tts:jobs")
                        except Exception:
                            logger.exception("Failed to check queued jobs; proceeding with unload check", extra={"job_id": job_id})

                        if queued_jobs > 0:
                            logger.info(
                                "Deferring TTS unload because jobs are queued",
                                extra={"job_id": job_id, "queued_jobs": queued_jobs},
                            )
                            unloaded_models = []
                        else:
                            # No queued jobs — safe to unload before loading Whisper
                            logger.info("Unloading Qwen TTS models before Whisper loading", extra={"job_id": job_id})
                            unloaded_models = tts_server.unload_model(model_type=None)
                            logger.info(
                                "Qwen TTS models unloaded",
                                extra={"job_id": job_id, "unloaded_count": len(unloaded_models)}
                            )
                    except Exception as e:
                        logger.warning(f"Failed to unload TTS models: {e}", exc_info=True)
                    
                    # Additional GPU memory cleanup after model unload
                    if torch.cuda.is_available():
                        logger.info("Performing GPU memory cleanup after TTS unload", extra={"job_id": job_id})
                        for device_id in range(torch.cuda.device_count()):
                            with torch.cuda.device(device_id):
                                torch.cuda.synchronize()
                                torch.cuda.empty_cache()
                        for _ in range(3):
                            gc.collect()
                        for device_id in range(torch.cuda.device_count()):
                            with torch.cuda.device(device_id):
                                torch.cuda.empty_cache()
                        logger.info("GPU memory cleanup completed", extra={"job_id": job_id})

                    # Prepare STT transcribers
                    stt_devices = settings.whisper_device_list()
                    transcribers = {}
                    for dev in stt_devices:
                        try:
                            transcribers[dev] = WhisperTranscriberHF(settings.whisper_model_id(), device=dev)
                        except Exception:
                            logger.exception("Failed to init transcriber on %s", dev)

                    scorer = AccuracyScorer()
                    stt_futures = []
                    for cand in candidates:
                        dev = stt_devices[cand["candidate_id"] % max(1, len(stt_devices))]
                        trans = transcribers.get(dev)
                        if trans:
                            stt_futures.append((cand, executor.submit(trans.transcribe, cand["wav_path"]), dev))

                    scored = []
                    timings["scoring_start"] = time.time()
                    
                    # Log reference text for comparison
                    ref_text = job.get("text", "")
                    logger.info(
                        "Reference text for scoring",
                        extra={
                            "job_id": job_id,
                            "reference_text_len": len(ref_text),
                            "reference_text": ref_text
                        }
                    )
                    
                    for cand, sf, dev in stt_futures:
                        try:
                            transcription = sf.result(timeout=120)
                            # transcription may be WhisperTranscription or dict
                            trans_text = transcription.text if isinstance(transcription, WhisperTranscription) else transcription.get("text", "") if isinstance(transcription, dict) else str(transcription)
                            duration = cand.get("duration_seconds", 0.0)
                            score = scorer.score_candidate(reference=job.get("text"), transcription=trans_text, duration=duration)
                            scored.append({
                                "candidate_id": cand["candidate_id"],
                                "wav_path": cand["wav_path"],
                                "audio_base64": cand["audio_base64"],
                                "sample_rate": cand["sample_rate"],
                                "duration_seconds": duration,
                                "tts_gpu": cand.get("tts_gpu", None),
                                "timings": cand.get("timings", {}),
                                "stt_device": dev,
                                "score": score,
                            })

                            logger.info(
                                "Candidate scored",
                                extra={
                                    "job_id": job_id,
                                    "candidate_id": cand.get("candidate_id"),
                                    "stt_device": dev,
                                    "tts_gpu": cand.get("tts_gpu"),
                                    "score": _score_for_audit_log(score),
                                },
                            )
                            
                            # Log full transcribed text for diagnosis
                            logger.info(
                                "Candidate transcription",
                                extra={
                                    "job_id": job_id,
                                    "candidate_id": cand.get("candidate_id"),
                                    "transcribed_text_len": len(trans_text),
                                    "transcribed_text": trans_text,
                                    "audio_duration_sec": duration
                                }
                            )
                        except Exception:
                            logger.exception("Transcription/scoring failed for candidate %s", cand.get("candidate_id"))

                    timings["scoring_end"] = time.time()
                    if not scored:
                        raise RuntimeError("All transcriptions failed")
                    
                    # CRITICAL: Unload Whisper models immediately after scoring completes
                    # This frees GPU memory before final job processing
                    logger.info("Unloading Whisper models after scoring", extra={"job_id": job_id})
                    for dev, transcriber in transcribers.items():
                        try:
                            transcriber.unload()
                            logger.info(f"Whisper model unloaded from {dev}", extra={"job_id": job_id})
                        except Exception as e:
                            logger.warning(f"Failed to unload Whisper from {dev}: {e}")
                    
                    # Clear transcriber references
                    transcribers.clear()
                    
                    # Additional GPU memory cleanup after Whisper unload
                    if torch.cuda.is_available():
                        logger.info("GPU cleanup after Whisper unload", extra={"job_id": job_id})
                        for device_id in range(torch.cuda.device_count()):
                            with torch.cuda.device(device_id):
                                torch.cuda.synchronize()
                                torch.cuda.empty_cache()
                        for _ in range(3):
                            gc.collect()
                        logger.info("GPU cleanup completed", extra={"job_id": job_id})

                    # Select best candidate by accuracy then shortest duration
                    scored.sort(
                        key=lambda c: (
                            -float((c.get("score") or {}).get("accuracy_score") or 0.0),
                            float((c.get("score") or {}).get("duration_seconds") or 0.0),
                        )
                    )
                    best = scored[0]

                    logger.info(
                        "Selected best candidate",
                        extra={
                            "job_id": job_id,
                            "best_candidate_id": best.get("candidate_id"),
                            "best_score": _score_for_audit_log(best.get("score") or {}),
                            "num_candidates_scored": len(scored),
                        },
                    )

                    out_path = best.get("wav_path", "")
                    audio_b64 = best.get("audio_base64")
                    download_url = f"/v1/audio/{job_id}.wav"

                    result_path = ""
                    try:
                        result_payload: Dict[str, Any] = {
                            "job_id": job_id,
                            "completed_at": _utc_now_iso(),
                            "num_candidates": int(num_candidates),
                            "timings": {
                                "claimed_at": timings.get("claimed_at"),
                                "generation_start": timings.get("generation_start"),
                                "generation_end": timings.get("generation_end"),
                                "scoring_start": timings.get("scoring_start"),
                                "scoring_end": timings.get("scoring_end"),
                                "postprocess_write_result_at": time.time(),
                            },
                            "winner_candidate_id": best.get("candidate_id"),
                            "winner_score": _score_for_audit_log(best.get("score") or {}),
                            "candidates": [
                                {
                                    "candidate_id": c.get("candidate_id"),
                                    "tts_gpu": c.get("tts_gpu"),
                                    "stt_device": c.get("stt_device"),
                                    "wav_path": c.get("wav_path"),
                                    "sample_rate": c.get("sample_rate"),
                                    "duration_seconds": c.get("duration_seconds"),
                                    "score": _score_for_audit_log(c.get("score") or {}),
                                }
                                for c in scored
                            ],
                        }
                        result_path = _write_job_result_file(job_id, result_payload)
                        logger.info(
                            "Wrote per-job result summary",
                            extra={"job_id": job_id, "path": result_path},
                        )
                    except Exception:
                        logger.warning("Failed to write per-job result summary", exc_info=True)
                    redis_client.hset(key, mapping={
                        "status": "done",
                        "audio_base64": audio_b64,
                        "sample_rate": best.get("sample_rate"),
                        "duration_seconds": best.get("duration_seconds"),
                        "wav_path": out_path,
                        "download_url": download_url,
                        "best_candidate": json.dumps(best.get("score")),
                        "all_candidates": json.dumps(scored) if return_all else "",
                        "num_candidates_generated": len(scored),
                        "winner_candidate_id": str(best.get("candidate_id")),
                        "result_path": result_path,
                        "completed_at": str(time.time()),
                    })
                
                # CRITICAL: Free all GPU memory after job completes
                # This prevents VRAM accumulation across sequential jobs
                if torch.cuda.is_available():
                    logger.info("Performing aggressive GPU memory cleanup after job completion", extra={"job_id": job_id})
                    for device_id in range(torch.cuda.device_count()):
                        with torch.cuda.device(device_id):
                            torch.cuda.synchronize()
                            torch.cuda.empty_cache()
                    # Multiple GC passes to ensure thorough cleanup
                    for _ in range(3):
                        gc.collect()
                    # Final cache clear after GC
                    for device_id in range(torch.cuda.device_count()):
                        with torch.cuda.device(device_id):
                            torch.cuda.empty_cache()
                    logger.info("GPU memory cleanup completed", extra={"job_id": job_id})
                    last_job_time = time.time()

            except Exception as e:
                logger.exception("Job failed", exc_info=e)
                redis_client.hset(key, mapping={"status": "failed", "error": str(e)[:200], "completed_at": str(time.time())})
                
                # Clean up GPU memory even on failure
                if torch.cuda.is_available():
                    for device_id in range(torch.cuda.device_count()):
                        with torch.cuda.device(device_id):
                            torch.cuda.synchronize()
                            torch.cuda.empty_cache()
                    gc.collect()

        except Exception:
            # Prevent worker thread from dying on unexpected errors
            logger.exception("Job worker encountered an error; continuing")
            time.sleep(1)


@app.on_event("shutdown")
async def shutdown_event():
    """Cleanup resources on shutdown."""
    logger.info("Shutting down Qwen3-TTS API server")
    _FISH_IDLE_WATCHER_STOP.set()
    svc = _SYNC_FISH_AUDIO_SERVICE
    if svc is not None:
        try:
            svc.unload_model("s2-pro")
        except Exception:
            logger.exception("Failed to unload Fish backend during shutdown")


@app.get("/")
async def root() -> Dict[str, Any]:
    """API root endpoint.

    Returns a short document describing available endpoints.
    """
    return {
        "service": "Qwen3-TTS Multi-GPU API",
        "version": "1.0.0",
        "status": "online",
        "endpoints": [
            "GET /v1/health",
            "GET /docs",
            "POST /v1/tts/voice-clone",
            "POST /v1/tts/voice-clone/sync",
            "POST /v1/tts/custom-voice",
            "POST /v1/tts/voice-design",
        ],
    }


@app.post(
    "/v1/tts/voice-clone",
    status_code=202,
    response_model=JobSubmitResponse,
    responses={400: {"model": ErrorResponse}, 500: {"model": ErrorResponse}},
    tags=["TTS"],
    summary="Voice clone (async submit)",
)
async def tts_voice_clone(request: VoiceCloneRequest) -> JobSubmitResponse:
    """Submit a voice-clone job and return a job id.

    This endpoint is intentionally non-blocking to avoid HTTP timeouts.
    Poll `/v1/jobs/{job_id}` until status is `done`, then download
    `/v1/audio/{job_id}.wav`.
    """
    try:
        return _enqueue_voice_clone_job(request)
    except (ValueError, FileNotFoundError) as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:
        logger.exception("Failed to enqueue voice-clone job")
        raise HTTPException(status_code=500, detail="Internal error") from exc


@app.post(
    "/v1/tts/voice-clone/sync",
    response_model=EnhancedAudioResponse,
    responses={400: {"model": ErrorResponse}, 500: {"model": ErrorResponse}},
    tags=["TTS"],
    summary="Voice clone (sync; may be slow)",
)
async def tts_voice_clone_sync(request: VoiceCloneRequest) -> EnhancedAudioResponse:
    """Generate speech with voice cloning using the Base model (synchronous)."""
    try:
        tts_server = get_sync_tts_server()
        # Validate ref_audio early to avoid passing empty values into audio loaders
        if not request.x_vector_only_mode:
            if not request.ref_audio or not str(request.ref_audio).strip():
                raise ValueError("ref_audio is required for voice cloning unless x_vector_only_mode=True")
            if not request.ref_text or not str(request.ref_text).strip():
                raise ValueError("ref_text is required for voice cloning unless x_vector_only_mode=True")

        num_candidates = int(request.num_candidates or 1)

        # Single-candidate fast path
        if num_candidates <= 1:
            wav, sr = tts_server.generate_voice_clone(
                text=request.text,
                language=request.language,
                ref_audio=request.ref_audio,
                ref_text=request.ref_text,
                x_vector_only_mode=request.x_vector_only_mode,
            )
            wav_bytes = wav_to_wav_bytes(wav, sr)
            audio_b64 = base64.b64encode(wav_bytes).decode("utf-8")
            duration = float(len(wav) / sr) if hasattr(wav, "__len__") else 0.0

            # Persist output and create a job record so callers can download later
            job_id = uuid.uuid4().hex
            out_path = ""
            try:
                out_dir = settings.output_dir
                os.makedirs(out_dir, exist_ok=True)
                out_path = os.path.join(out_dir, f"{job_id}.wav")
                with open(out_path, "wb") as f:
                    f.write(wav_bytes)
                logger.info("Wrote generated WAV to disk (sync)", extra={"job_id": job_id, "path": out_path})
            except Exception:
                logger.warning("Failed to persist WAV to disk for synchronous request", exc_info=True)

            download_url = f"/v1/audio/{job_id}.wav"
            redis_client.hset(f"tts:job:{job_id}", mapping={
                "status": "done",
                "audio_base64": audio_b64,
                "sample_rate": sr,
                "duration_seconds": duration,
                "wav_path": out_path,
                "download_url": download_url,
                "completed_at": str(time.time()),
            })

            return EnhancedAudioResponse(
                audio_base64=audio_b64,
                sample_rate=sr,
                duration_seconds=duration,
                format="wav",
                model="base",
                job_id=job_id,
                download_url=download_url,
                best_candidate_score=None,
                all_candidates=None,
                num_candidates_generated=1,
            )

        # Multi-candidate synchronous processing
        job_id = uuid.uuid4().hex
        candidate_dir = os.path.join(settings.output_dir, job_id, "candidates")
        os.makedirs(candidate_dir, exist_ok=True)

        candidates = []
        sync_timeout = None
        if float(getattr(settings, "job_timeout", 0) or 0) > 0:
            sync_timeout = float(settings.job_timeout)

        start_time = time.time()
        for i in range(num_candidates):
            if sync_timeout is not None and (time.time() - start_time) > sync_timeout:
                raise HTTPException(status_code=504, detail="Candidate generation timed out")
            res = _generate_candidate(
                tts_server,
                request.text,
                request.language,
                request.ref_audio,
                request.ref_text,
                request.x_vector_only_mode,
                job_id,
                i,
            )
            if res:
                candidates.append(res)

        if not candidates:
            raise HTTPException(status_code=500, detail="All candidates failed to generate")

        # Transcribe and score
        stt_devices = settings.whisper_device_list()
        transcribers = {}
        for dev in stt_devices:
            try:
                transcribers[dev] = WhisperTranscriberHF(settings.whisper_model_id(), device=dev)
            except Exception:
                logger.exception("Failed to init transcriber on %s", dev)

        scorer = AccuracyScorer()
        scored = []
        for cand in candidates:
            dev = stt_devices[cand["candidate_id"] % max(1, len(stt_devices))]
            trans = transcribers.get(dev)
            if not trans:
                continue
            try:
                transcription = trans.transcribe(cand["wav_path"])
                trans_text = (
                    transcription.text
                    if isinstance(transcription, WhisperTranscription)
                    else transcription.get("text", "")
                    if isinstance(transcription, dict)
                    else str(transcription)
                )
                score = scorer.score_candidate(
                    reference=request.text,
                    transcription=trans_text,
                    duration=cand.get("duration_seconds", 0.0),
                )
                scored.append({
                    "candidate_id": cand["candidate_id"],
                    "wav_path": cand["wav_path"],
                    "audio_base64": cand["audio_base64"],
                    "sample_rate": cand["sample_rate"],
                    "duration_seconds": cand.get("duration_seconds", 0.0),
                    "tts_gpu": cand.get("tts_gpu", None),
                    "timings": cand.get("timings", {}),
                    "stt_device": dev,
                    "score": score,
                })

                logger.info(
                    "Candidate scored (sync)",
                    extra={
                        "job_id": job_id,
                        "candidate_id": cand.get("candidate_id"),
                        "stt_device": dev,
                        "tts_gpu": cand.get("tts_gpu"),
                        "score": _score_for_audit_log(score),
                    },
                )
            except Exception:
                logger.exception("Transcription/scoring failed for candidate %s", cand.get("candidate_id"))

        if not scored:
            raise HTTPException(status_code=500, detail="All transcriptions failed")

            scored.sort(
                key=lambda c: (
                    -float((c.get("score") or {}).get("accuracy_score") or 0.0),
                    float((c.get("score") or {}).get("duration_seconds") or 0.0),
                )
            )
        best = scored[0]

        logger.info(
            "Selected best candidate (sync)",
            extra={
                "job_id": job_id,
                "best_candidate_id": best.get("candidate_id"),
                "best_score": _score_for_audit_log(best.get("score") or {}),
                "num_candidates_scored": len(scored),
            },
        )

        # Persist best
        out_path = best.get("wav_path", "")
        audio_b64 = best.get("audio_base64")
        download_url = f"/v1/audio/{job_id}.wav"
        redis_client.hset(f"tts:job:{job_id}", mapping={
            "status": "done",
            "audio_base64": audio_b64,
            "sample_rate": best.get("sample_rate"),
            "duration_seconds": best.get("duration_seconds"),
            "wav_path": out_path,
            "download_url": download_url,
            "best_candidate": json.dumps(best.get("score")),
            "all_candidates": json.dumps(scored) if request.return_all_candidates else "",
            "num_candidates_generated": len(scored),
            "completed_at": str(time.time()),
        })

        return EnhancedAudioResponse(
            audio_base64=audio_b64,
            sample_rate=best.get("sample_rate"),
            duration_seconds=best.get("duration_seconds"),
            format="wav",
            model="base",
            job_id=job_id,
            download_url=download_url,
            best_candidate_score=best.get("score"),
            all_candidates=[c.get("score") for c in scored] if request.return_all_candidates else None,
            num_candidates_generated=len(scored),
        )
    except (ValueError, FileNotFoundError) as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except HTTPException as http_exc:
        # Re-raise HTTP exceptions as-is to preserve status and detail
        raise http_exc
    except Exception as exc:
        # Avoid logging huge user-supplied payloads embedded in exception messages
        def _masked_exc(e: Exception, max_len: int = 300) -> Exception:
            msg = str(e)
            if len(msg) > max_len:
                msg = msg[:max_len] + "...(truncated)"
            return type(e)(msg)

        masked = _masked_exc(exc)
        logger.exception("Voice-clone request failed", exc_info=(type(exc), masked, exc.__traceback__))
        raise HTTPException(status_code=500, detail=f"Internal error") from exc


@app.post(
    "/v1/tts/voice-clone/submit",
    status_code=202,
    response_model=JobSubmitResponse,
)
async def tts_voice_clone_submit(request: VoiceCloneRequest) -> JobSubmitResponse:
    """Enqueue a voice-clone job and return a job id immediately."""
    try:
        return _enqueue_voice_clone_job(request)
    except (ValueError, FileNotFoundError) as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:
        logger.exception("Failed to enqueue job")
        raise HTTPException(status_code=500, detail="Internal error") from exc


@app.get("/v1/jobs/{job_id}")
async def get_job_status(job_id: str):
    key = f"tts:job:{job_id}"
    if not redis_client.exists(key):
        raise HTTPException(status_code=404, detail="Job not found")
    job = redis_client.hgetall(key)
    return job


@app.get("/v1/audio/{job_id}.wav")
async def download_audio(job_id: str):
    """Download generated WAV for a completed job.

    Returns 202 if the job exists but is not yet completed.
    If the job recorded a `wav_path` the file is streamed from disk. If not,
    but `audio_base64` is present, the bytes are returned directly.
    """
    key = f"tts:job:{job_id}"
    if not redis_client.exists(key):
        raise HTTPException(status_code=404, detail="Job not found")

    job = redis_client.hgetall(key)
    status = job.get("status", "unknown")

    # If WAV was persisted to disk, prefer serving the file
    wav_path = job.get("wav_path") or ""
    if wav_path:
        import os

        if os.path.exists(wav_path):
            return FileResponse(wav_path, media_type="audio/wav", filename=f"{job_id}.wav")

    # Otherwise, if we have base64 bytes in Redis, stream them
    audio_b64 = job.get("audio_base64")
    if audio_b64:
        try:
            wav_bytes = base64.b64decode(audio_b64)
            return Response(content=wav_bytes, media_type="audio/wav", headers={"Content-Disposition": f"attachment; filename={job_id}.wav"})
        except Exception:
            raise HTTPException(status_code=500, detail="Failed to decode audio bytes")

    # Not ready
    if status != "done":
        raise HTTPException(status_code=202, detail=f"Job not ready (status={status})")

    raise HTTPException(status_code=404, detail="Audio not available")


@app.post("/v1/models/unload")
async def unload_models(model_type: Optional[str] = None):
    """Unload cached models from GPU. If `model_type` is omitted, unload all models."""
    try:
        # If any jobs are currently running, block unload and report those jobs.
        running_jobs = []
        try:
            for key in redis_client.scan_iter(match="tts:job:*"):
                try:
                    status = redis_client.hget(key, "status")
                except Exception:
                    status = None
                if status and status.lower() in ("running", "processing"):
                    job_id = key.split(":", 2)[-1]
                    running_jobs.append(job_id)
        except Exception:
            logger.exception("Failed to scan Redis for running jobs while handling unload request")

        if running_jobs:
            return {"status": "blocked", "running_jobs": running_jobs}

        fish_unloaded = []
        if model_type in (None, "s2-pro"):
            svc = _SYNC_FISH_AUDIO_SERVICE
            if svc is not None:
                fish_unloaded = svc.unload_model("s2-pro")

            if model_type == "s2-pro":
                return {"status": "ok", "unloaded": fish_unloaded}

        # Send unload command to worker via Redis
        cmd_id = uuid.uuid4().hex
        cmd_payload = {
            "cmd_id": cmd_id,
            "type": "unload",
            "model_type": model_type,
        }
        
        logger.info(f"API sending unload command to worker: {cmd_id}")
        redis_client.rpush("tts:commands", json.dumps(cmd_payload))
        
        # Wait for worker to process command (poll for result)
        timeout = 30  # seconds
        start_time = time.time()
        while time.time() - start_time < timeout:
            result = redis_client.hgetall(f"tts:cmd:{cmd_id}")
            if result and result.get("status") == "done":
                unloaded = json.loads(result.get("result", "[]"))
                combined_unloaded = list(unloaded) + list(fish_unloaded)
                logger.info(f"Worker unload completed: {combined_unloaded}")
                return {"status": "ok", "unloaded": combined_unloaded}
            elif result and result.get("status") == "error":
                error = result.get("error", "Unknown error")
                logger.error(f"Worker unload failed: {error}")
                raise HTTPException(status_code=500, detail=f"Worker error: {error}")
            
            await asyncio.sleep(0.5)
        
        # Timeout
        logger.warning(f"Unload command {cmd_id} timed out")
        raise HTTPException(status_code=504, detail="Unload command timed out")
        
    except HTTPException:
        raise
    except Exception as e:
        logger.exception("Failed to unload models")
        raise HTTPException(status_code=500, detail=str(e)) from e


@app.get("/v1/models/status")
async def models_status():
    """Diagnostic endpoint: show cached models and GPU memory."""
    try:
        tts_server = get_sync_tts_server()
        cached_models = []
        with tts_server._lock:
            for key, model in tts_server._models.items():
                model_type, device = key
                # Try to get parameter device
                param_device = None
                try:
                    if hasattr(model, "model") and hasattr(model.model, "parameters"):
                        param_device = str(next(model.model.parameters()).device)
                    elif hasattr(model, "parameters"):
                        param_device = str(next(model.parameters()).device)
                except Exception:
                    param_device = "unknown"
                
                cached_models.append({
                    "model_type": model_type,
                    "device": device,
                    "param_device": param_device,
                })

        fish_service = _SYNC_FISH_AUDIO_SERVICE
        fish_status = fish_service.status() if fish_service is not None else {
            "model_type": "s2-pro",
            "loaded": False,
            "backend": "remote",
            "initialized": False,
        }
        
        gpu_memory = []
        if torch.cuda.is_available():
            for i in range(torch.cuda.device_count()):
                allocated = torch.cuda.memory_allocated(i) / (1024**3)
                reserved = torch.cuda.memory_reserved(i) / (1024**3)
                gpu_memory.append({
                    "gpu_id": i,
                    "allocated_gb": round(allocated, 2),
                    "reserved_gb": round(reserved, 2),
                })
        
        return {
            "cached_models": cached_models,
            "fish_backend": fish_status,
            "gpu_memory": gpu_memory,
            "model_count": len(cached_models),
        }
    except Exception as e:
        logger.exception("Failed to get model status")
        raise HTTPException(status_code=500, detail=str(e)) from e


@app.post(
    "/v1/tts/custom-voice",
    response_model=AudioResponse,
    responses={400: {"model": ErrorResponse}, 500: {"model": ErrorResponse}},
    tags=["TTS"],
    summary="Custom voice (CustomVoice model)",
)
async def tts_custom_voice(request: CustomVoiceRequest) -> AudioResponse:
    """Generate speech using the CustomVoice model."""
    try:
        tts_server = get_sync_tts_server()
        wav, sr = tts_server.generate_custom_voice(
            text=request.text,
            language=request.language,
            speaker=request.speaker,
            instruct=request.instruct,
        )
        wav_bytes = wav_to_wav_bytes(wav, sr)
        audio_b64 = base64.b64encode(wav_bytes).decode("utf-8")
        duration = float(len(wav) / sr) if hasattr(wav, "__len__") else 0.0

        job_id = uuid.uuid4().hex
        out_path = ""
        try:
            out_dir = settings.output_dir
            os.makedirs(out_dir, exist_ok=True)
            out_path = os.path.join(out_dir, f"{job_id}.wav")
            with open(out_path, "wb") as f:
                f.write(wav_bytes)
            logger.info("Wrote generated WAV to disk (sync)", extra={"job_id": job_id, "path": out_path})
        except Exception:
            logger.warning("Failed to persist WAV to disk for synchronous request", exc_info=True)

        download_url = f"/v1/audio/{job_id}.wav"
        redis_client.hset(f"tts:job:{job_id}", mapping={
            "status": "done",
            "audio_base64": audio_b64,
            "sample_rate": sr,
            "duration_seconds": duration,
            "wav_path": out_path,
            "download_url": download_url,
            "completed_at": str(time.time()),
        })

        return AudioResponse(
            audio_base64=audio_b64,
            sample_rate=sr,
            duration_seconds=duration,
            format="wav",
            model="custom-voice",
            job_id=job_id,
            download_url=download_url,
        )
    except (ValueError, FileNotFoundError) as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:
        def _masked_exc(e: Exception, max_len: int = 300) -> Exception:
            msg = str(e)
            if len(msg) > max_len:
                msg = msg[:max_len] + "...(truncated)"
            return type(e)(msg)

        masked = _masked_exc(exc)
        logger.exception("Custom-voice request failed", exc_info=(type(exc), masked, exc.__traceback__))
        raise HTTPException(status_code=500, detail=f"Internal error") from exc


@app.post(
    "/v1/tts/voice-design/sync",
    response_model=AudioResponse,
    responses={400: {"model": ErrorResponse}, 500: {"model": ErrorResponse}},
    tags=["TTS"],
    summary="Voice design (VoiceDesign model) - synchronous",
)
async def tts_voice_design_sync(request: VoiceDesignRequest) -> AudioResponse:
    """Synchronous voice-design generation (keeps previous behavior)."""
    try:
        tts_server = get_sync_tts_server()
        wav, sr = tts_server.generate_voice_design(
            text=request.text,
            language=request.language,
            instruct=request.instruct,
        )
        wav_bytes = wav_to_wav_bytes(wav, sr)
        audio_b64 = base64.b64encode(wav_bytes).decode("utf-8")
        duration = float(len(wav) / sr) if hasattr(wav, "__len__") else 0.0

        job_id = uuid.uuid4().hex
        out_path = ""
        try:
            out_dir = settings.output_dir
            os.makedirs(out_dir, exist_ok=True)
            out_path = os.path.join(out_dir, f"{job_id}.wav")
            with open(out_path, "wb") as f:
                f.write(wav_bytes)
            logger.info("Wrote generated WAV to disk (sync)", extra={"job_id": job_id, "path": out_path})
        except Exception:
            logger.warning("Failed to persist WAV to disk for synchronous request", exc_info=True)

        download_url = f"/v1/audio/{job_id}.wav"
        redis_client.hset(f"tts:job:{job_id}", mapping={
            "status": "done",
            "audio_base64": audio_b64,
            "sample_rate": sr,
            "duration_seconds": duration,
            "wav_path": out_path,
            "download_url": download_url,
            "completed_at": str(time.time()),
        })

        return AudioResponse(
            audio_base64=audio_b64,
            sample_rate=sr,
            duration_seconds=duration,
            format="wav",
            model="voice-design",
            job_id=job_id,
            download_url=download_url,
        )
    except (ValueError, FileNotFoundError) as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:
        def _masked_exc(e: Exception, max_len: int = 300) -> Exception:
            msg = str(e)
            if len(msg) > max_len:
                msg = msg[:max_len] + "...(truncated)"
            return type(e)(msg)

        masked = _masked_exc(exc)
        logger.exception("Voice-design request failed", exc_info=(type(exc), masked, exc.__traceback__))
        raise HTTPException(status_code=500, detail=f"Internal error") from exc


@app.post(
    "/v1/tts/s2-pro/sync",
    response_model=AudioResponse,
    responses={400: {"model": ErrorResponse}, 500: {"model": ErrorResponse}, 503: {"model": ErrorResponse}},
    tags=["TTS"],
    summary="Fish Audio S2 Pro plain TTS (sync)",
)
async def tts_s2_pro_sync(request: S2ProRequest) -> AudioResponse:
    """Synchronous plain-TTS endpoint backed by Fish Audio S2 Pro.

    This is intentionally narrow: accepts plain text and optional language,
    calls a synchronous FishAudioService.generate, persists the WAV, writes a
    minimal Redis job record, and returns an `AudioResponse` with
    `model="s2-pro"`.
    """
    try:
        svc = get_sync_fish_audio_service()

        # Generate waveform - run in executor to avoid blocking event loop
        loop = asyncio.get_event_loop()
        wav, sr = await loop.run_in_executor(None, svc.generate, request.text, request.language)

        # Serialize to WAV bytes
        wav_bytes = wav_to_wav_bytes(wav, int(sr))

        audio_b64 = base64.b64encode(wav_bytes).decode("utf-8")
        duration = float(len(wav) / sr) if hasattr(wav, "__len__") else 0.0

        # Persist output and create a job record so callers can download later
        job_id = uuid.uuid4().hex
        out_path = ""
        try:
            out_dir = settings.output_dir
            os.makedirs(out_dir, exist_ok=True)
            out_path = os.path.join(out_dir, f"{job_id}.wav")
            with open(out_path, "wb") as f:
                f.write(wav_bytes)
            logger.info("Wrote generated WAV to disk (s2-pro sync)", extra={"job_id": job_id, "path": out_path})
        except Exception:
            logger.warning("Failed to persist WAV to disk for s2-pro synchronous request", exc_info=True)

        download_url = f"/v1/audio/{job_id}.wav"
        redis_client.hset(f"tts:job:{job_id}", mapping={
            "status": "done",
            "audio_base64": audio_b64,
            "sample_rate": int(sr),
            "duration_seconds": duration,
            "wav_path": out_path,
            "download_url": download_url,
            "completed_at": str(time.time()),
        })

        return AudioResponse(
            audio_base64=audio_b64,
            sample_rate=int(sr),
            duration_seconds=duration,
            format="wav",
            model="s2-pro",
            job_id=job_id,
            download_url=download_url,
        )

    except FileNotFoundError as e:
        # Model files unavailable
        logger.exception("S2 Pro model unavailable")
        raise HTTPException(status_code=503, detail=str(e)) from e
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e)) from e
    except NotImplementedError as e:
        # Stub/implementation not ready
        logger.exception("S2 Pro generate not implemented")
        raise HTTPException(status_code=500, detail=str(e)) from e
    except Exception as e:
        logger.exception("S2 Pro synchronous request failed")
        raise HTTPException(status_code=500, detail="Internal error") from e


@app.post(
    "/v1/tts/voice-design",
    status_code=202,
    response_model=JobSubmitResponse,
)
async def tts_voice_design(request: VoiceDesignRequest) -> JobSubmitResponse:
    """Enqueue a voice-design job and return immediately with a job id."""
    try:
        return _enqueue_voice_design_job(request)
    except (ValueError, FileNotFoundError) as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception:
        logger.exception("Failed to enqueue voice-design job")
        raise HTTPException(status_code=500, detail="Internal error") from exc


@app.get("/v1/health")
async def health_check() -> Dict[str, Any]:
    """Health check endpoint for container orchestration.
    
    Returns:
        Health status with system information.
    """
    import torch
    
    try:
        # Check Redis connectivity
        from redis import Redis
        redis_conn = Redis(
            host=settings.redis_host,
            port=settings.redis_port,
            socket_connect_timeout=2
        )
        redis_healthy = redis_conn.ping()
    except Exception as e:
        logger.warning(f"Redis health check failed: {e}")
        redis_healthy = False
    
    return {
        "status": "healthy" if redis_healthy else "degraded",
        "redis": "connected" if redis_healthy else "disconnected",
        "cuda_available": torch.cuda.is_available(),
        "gpu_count": torch.cuda.device_count() if torch.cuda.is_available() else 0,
        "python_version": sys.version.split()[0],
        "pytorch_version": torch.__version__,
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        app,
        host=settings.api_host,
        port=settings.api_port,
        log_level=settings.log_level.lower(),
    )

