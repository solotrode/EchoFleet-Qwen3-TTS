"""Custom exceptions for Qwen3-TTS service.

This module defines project-specific exceptions following the pattern
established in AGENTS.md.
"""


class Qwen3TTSError(Exception):
    """Base exception for all Qwen3-TTS errors."""

    pass


class ModelNotFoundError(Qwen3TTSError):
    """Raised when requested model is not found or not loaded."""

    pass


class ModelLoadError(Qwen3TTSError):
    """Raised when model loading fails."""

    pass


class AudioProcessingError(Qwen3TTSError):
    """Raised when audio processing fails."""

    pass


class CUDAOutOfMemoryError(Qwen3TTSError):
    """Raised when GPU memory is exhausted."""

    pass


class InferenceError(Qwen3TTSError):
    """Raised when TTS inference fails."""

    pass


class TranscriptionError(Qwen3TTSError):
    """Raised when Whisper transcription fails."""

    pass


class ValidationError(Qwen3TTSError):
    """Raised when request validation fails."""

    pass


class JobNotFoundError(Qwen3TTSError):
    """Raised when job ID is not found in Redis."""

    pass


class GPUNotAvailableError(Qwen3TTSError):
    """Raised when no GPUs are available for processing."""

    pass


__all__ = [
    "Qwen3TTSError",
    "ModelNotFoundError",
    "ModelLoadError",
    "AudioProcessingError",
    "CUDAOutOfMemoryError",
    "InferenceError",
    "TranscriptionError",
    "ValidationError",
    "JobNotFoundError",
    "GPUNotAvailableError",
]
