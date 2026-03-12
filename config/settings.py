"""Centralized configuration using environment variables.

This module provides a single Settings class and a get_settings() helper
to load and cache configuration from environment variables or a .env file.
"""
from __future__ import annotations

from functools import lru_cache
from typing import List
import os


class Settings:
    """Lightweight settings loader from environment variables.

    This reads configuration from environment variables with sensible
    defaults for the Qwen3-TTS multi-GPU service.
    """

    def __init__(self) -> None:
        # Redis configuration
        self.redis_host: str = os.getenv("REDIS_HOST", "redis")
        self.redis_port: int = int(os.getenv("REDIS_PORT", "6379"))

        # Model directories
        self.model_cache_dir: str = os.getenv("MODEL_CACHE_DIR", "/models")
        self.output_dir: str = os.getenv("OUTPUT_DIR", "/workspace/outputs")
        
        # GPU configuration
        self.tts_gpus: str = os.getenv("TTS_GPUS", "0,1,2,3")
        # Preferred GPU is used when idle; others are used only when there are
        # concurrent requests (i.e., preferred GPU is busy).
        self.tts_preferred_gpu: int = int(os.getenv("TTS_PREFERRED_GPU", "0"))
        # Maximum concurrent jobs per GPU (default 1 for large models).
        self.tts_gpu_capacity: int = int(os.getenv("TTS_GPU_CAPACITY", "1"))
        # Number of in-process Redis job worker threads.
        # Set to >1 to allow multiple queued jobs to execute concurrently and
        # utilize multiple GPUs.
        self.tts_worker_concurrency: int = int(os.getenv("TTS_WORKER_CONCURRENCY", "1"))
        self.gpu_memory_threshold_mb: int = int(os.getenv("GPU_MEMORY_THRESHOLD_MB", "1024"))
        
        # Model variants to load
        self.enable_base_model: bool = os.getenv("ENABLE_BASE_MODEL", "true").lower() == "true"
        self.enable_custom_voice: bool = os.getenv("ENABLE_CUSTOM_VOICE", "true").lower() == "true"
        self.enable_voice_design: bool = os.getenv("ENABLE_VOICE_DESIGN", "true").lower() == "true"
        
        # Model dtype and device settings
        self.default_dtype: str = os.getenv("DEFAULT_DTYPE", "bfloat16")
        self.device_map: str = os.getenv("DEVICE_MAP", "auto")

        # Attention implementation for TTS model loading.
        # Options:
        # - "auto" (default): prefer FlashAttention2 when available, else fall back to SDPA
        # - "sdpa" or "eager"
        # - "flash_attention_2"/"fa2"/"flash": force FA2 when available, else fall back to SDPA
        self.tts_attn_impl: str = os.getenv("TTS_ATTN_IMPL", "auto")

        # TTS generation controls
        self.tts_sample_rate: int = int(os.getenv("TTS_SAMPLE_RATE", "24000"))
        
        # DEPRECATED: These token estimation settings are no longer used.
        # We now rely only on the model's inherent max_new_tokens from generation_config.json.
        # Keeping them for backward compatibility but they have no effect.
        self.tts_tokens_per_second: float = float(os.getenv("TTS_TOKENS_PER_SECOND", "12"))
        self.tts_words_per_second: float = float(os.getenv("TTS_WORDS_PER_SECOND", "2.6"))
        self.tts_max_output_seconds: int = int(os.getenv("TTS_MAX_OUTPUT_SECONDS", "120"))
        self.tts_max_new_tokens: int = int(os.getenv("TTS_MAX_NEW_TOKENS", "2048"))
        self.tts_min_new_tokens: int = int(os.getenv("TTS_MIN_NEW_TOKENS", "96"))
        
        # Chunking behavior for long texts: preferred max characters per chunk.
        # This is the ONLY limit we apply (along with the model's inherent max_new_tokens).
        # Splits occur at sentence boundaries; single sentences larger than this
        # value will be returned as a single chunk.
        self.tts_chunk_max_chars: int = int(os.getenv("TTS_CHUNK_MAX_CHARS", "1000"))
        
        # API configuration
        self.api_host: str = os.getenv("API_HOST", "0.0.0.0")
        self.api_port: int = int(os.getenv("API_PORT", "8000"))
        self.gradio_port: int = int(os.getenv("GRADIO_PORT", "7860"))
        
        # Whisper configuration
        # We use HuggingFace/Transformers Whisper models (repo IDs), but also accept
        # shorthand values like "large-v3-turbo" for convenience.
        self.whisper_model: str = os.getenv("WHISPER_MODEL", "openai/whisper-large-v3-turbo")
        self.whisper_device_pool: str = os.getenv("WHISPER_DEVICE_POOL", "0,1,2,3")
        # Local path where Whisper models may be pre-downloaded (e.g. mounted volume)
        self.whisper_model_path: str = os.getenv("WHISPER_MODEL_PATH", "/stt_models/whisper-large-v3-turbo")
        
        # Job processing limits
        self.max_text_length: int = int(os.getenv("MAX_TEXT_LENGTH", "5000"))
        self.max_candidates: int = int(os.getenv("MAX_CANDIDATES", "10"))
        self.default_candidates: int = int(os.getenv("DEFAULT_CANDIDATES", "1"))
        self.job_timeout: int = int(os.getenv("JOB_TIMEOUT", "3600"))
        
        # Cleanup and retention
        self.retention_hours: int = int(os.getenv("RETENTION_HOURS", "24"))
        self.cleanup_schedule: str = os.getenv("CLEANUP_SCHEDULE", "0:00")
        # Unload models from GPU after a period of inactivity (seconds)
        self.tts_unload_idle_seconds: int = int(os.getenv("TTS_UNLOAD_IDLE_SECONDS", "300"))

        # Retention policies (days)
        # - log_retention_days controls how long rotated log files are kept under /logs.
        # - job_artifact_retention_days controls how long per-job artifacts under OUTPUT_DIR are kept.
        self.log_retention_days: int = int(os.getenv("LOG_RETENTION_DAYS", "10"))
        self.job_artifact_retention_days: int = int(os.getenv("JOB_ARTIFACT_RETENTION_DAYS", "10"))
        
        # Logging
        self.log_level: str = os.getenv("LOG_LEVEL", "INFO")
        self.structured_logging: bool = os.getenv("STRUCTURED_LOGGING", "true").lower() == "true"

        # Optional file logging (in addition to stdout)
        # When enabled, logs are written to LOG_FILE_PATH inside the container.
        # In compose.yaml this is typically a bind mount to ./logs.
        self.log_to_file: bool = os.getenv("LOG_TO_FILE", "false").lower() == "true"
        self.log_file_path: str = os.getenv("LOG_FILE_PATH", "")

        # Fish Audio S2 Pro model configuration
        # Local path where the S2 Pro model is stored (e.g. mounted Docker volume)
        self.s2_pro_model_dir: str = os.getenv("S2_PRO_MODEL_DIR", "/models/fish-s2-pro")
        # VCS repo id for reference (not used at runtime, for docs)
        self.s2_pro_repo_id: str = os.getenv("S2_PRO_REPO_ID", "fishaudio/s2-pro")
        # Preferred device for S2 Pro (e.g. "cuda:0" or "cpu")
        self.s2_pro_device: str = os.getenv("S2_PRO_DEVICE", "cuda:0")

    def tts_gpu_list(self) -> List[int]:
        """Return TTS GPUs as a list of ints.
        
        Example:
            >>> settings.tts_gpus = "0,1,2,3"
            >>> settings.tts_gpu_list()
            [0, 1, 2, 3]
        """
        try:
            parts = [p.strip() for p in self.tts_gpus.split(",") if p.strip()]
            return [int(p) for p in parts]
        except Exception:
            return [0, 1, 2, 3]
    
    def whisper_device_list(self) -> List[str]:
        """Return Whisper devices as a list of CUDA device strings.
        
        Example:
            >>> settings.whisper_device_pool = "0,1,2,3"
            >>> settings.whisper_device_list()
            ['cuda:0', 'cuda:1', 'cuda:2', 'cuda:3']
        """
        try:
            parts = [p.strip() for p in self.whisper_device_pool.split(",") if p.strip()]
            return [f"cuda:{int(p)}" for p in parts]
        except Exception:
            return ["cuda:0", "cuda:1", "cuda:2", "cuda:3"]

    def whisper_model_id(self) -> str:
        """Return a HuggingFace model repo id for Whisper.

        This supports both full repo IDs and a few common shorthand names.

        Returns:
            HuggingFace repo id, e.g. "openai/whisper-large-v3-turbo".
        """
        raw = (self.whisper_model or "").strip()
        if "/" in raw:
            return raw

        shorthand_map = {
            "large-v3-turbo": "openai/whisper-large-v3-turbo",
            "large-v3": "openai/whisper-large-v3",
            "large": "openai/whisper-large",
            "medium": "openai/whisper-medium",
            "small": "openai/whisper-small",
            "base": "openai/whisper-base",
            "tiny": "openai/whisper-tiny",
        }
        return shorthand_map.get(raw, f"openai/whisper-{raw}")

    def whisper_local_path(self) -> str:
        """Return a local filesystem path for a Whisper model when available.

        This is intended to point at a mounted volume like `/stt_models/whisper-large-v3-turbo`.
        """
        return (self.whisper_model_path or "/stt_models/whisper-large-v3-turbo")


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Return a cached Settings instance.

    Use this helper across modules to avoid repeated environment parsing.
    
    Example:
        >>> from config.settings import get_settings
        >>> settings = get_settings()
        >>> print(settings.redis_host)
        'redis'
    """
    return Settings()


__all__ = ["Settings", "get_settings"]
