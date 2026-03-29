"""Inference utilities and model wrappers.

This package intentionally avoids importing heavyweight dependencies (e.g.
PyTorch/CUDA) at import time so lightweight helpers can be unit tested on the
host machine.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

__all__ = ["QwenTTSServer", "WhisperTranscriberHF", "WhisperTranscription"]

if TYPE_CHECKING:
    from inference.qwen_tts_service import QwenTTSServer
    from inference.whisper_service import WhisperTranscriberHF, WhisperTranscription


def __getattr__(name: str) -> Any:
    if name == "QwenTTSServer":
        from inference.qwen_tts_service import QwenTTSServer as _QwenTTSServer

        return _QwenTTSServer
    if name == "WhisperTranscriberHF":
        from inference.whisper_service import (
            WhisperTranscriberHF as _WhisperTranscriberHF,
        )

        return _WhisperTranscriberHF
    if name == "WhisperTranscription":
        from inference.whisper_service import (
            WhisperTranscription as _WhisperTranscription,
        )

        return _WhisperTranscription
    raise AttributeError(name)
