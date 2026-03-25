"""Fish Audio S2 Pro service wrapper.

Calls the SGLang server (fish-sglang) which runs the Fish Audio S2 Pro model.
Supports both plain TTS and voice cloning via reference audio.

Unlike Qwen, the actual GPU-resident model lives in the remote SGLang process.
This wrapper mirrors the same lazy-load and unload contract at the application
layer so the API can treat both backends consistently.
"""
from typing import Any, Optional, Tuple
import os
import threading
import time

import numpy as np
import requests

from config.settings import get_settings
from utils.logging import get_logger

logger = get_logger(__name__)


class FishAudioService:
    """Fish Audio S2 Pro wrapper calling SGLang server."""

    def __init__(self) -> None:
        self._settings = get_settings()
        self._lock = threading.Lock()
        self._model_dir = getattr(self._settings, "s2_pro_model_dir", "/models/s2-pro")
        self._sglang_url = getattr(self._settings, "sglang_base_url", "http://fish-sglang:8000")
        self._ref_audio_dir = getattr(self._settings, "fish_ref_audio_dir", "/ref_audio")
        self._session = requests.Session()
        self._loaded = False
        self._last_activity = time.monotonic()

    def __del__(self) -> None:
        if hasattr(self, "_session"):
            self._session.close()

    @property
    def model_dir(self) -> str:
        return self._model_dir

    @property
    def is_loaded(self) -> bool:
        with self._lock:
            return self._loaded

    @property
    def last_activity(self) -> float:
        with self._lock:
            return self._last_activity

    def _mark_loaded(self) -> None:
        with self._lock:
            self._loaded = True
            self._last_activity = time.monotonic()

    def _touch_activity(self) -> None:
        with self._lock:
            self._last_activity = time.monotonic()

    def _load_model(self) -> None:
        """Mark the remote Fish backend as loaded after a readiness check.

        The S2 Pro model is hosted by the external SGLang server, so there is no
        in-process torch model to initialize here. We still mirror the lazy-load
        contract by verifying the backend is reachable the first time it is used.
        """
        with self._lock:
            if self._loaded:
                self._last_activity = time.monotonic()
                return

        if not os.path.exists(self._model_dir):
            raise FileNotFoundError(f"S2 Pro model directory not found: {self._model_dir}")

        health_endpoint = f"{self._sglang_url}/v1/models"
        response = self._session.get(health_endpoint, timeout=30)
        response.raise_for_status()

        logger.info(
            "Fish Audio backend marked loaded",
            extra={"sglang_url": self._sglang_url, "model_dir": self._model_dir},
        )
        self._mark_loaded()

    def generate(
        self,
        text: str,
        language: Optional[str] = None,
        reference_audio: Optional[str] = None,
        reference_text: Optional[str] = None,
    ) -> Tuple[np.ndarray, int]:
        """Generate waveform for the given text.

        Args:
            text: Input text to synthesize.
            language: Optional language hint (e.g., "English", "Chinese").
            reference_audio: Optional path to reference audio file for voice cloning.
            reference_text: Transcript of the reference audio.

        Returns:
            Tuple of (audio_array, sample_rate).

        Raises:
            requests.RequestException: If SGLang API call fails.
        """
        if not self.is_loaded:
            self._load_model()
        else:
            self._touch_activity()

        endpoint = f"{self._sglang_url}/v1/audio/speech"

        payload: dict = {"input": text}
        if language:
            payload["language"] = language

        if reference_audio:
            ref_path = reference_audio
            if not os.path.isabs(ref_path):
                ref_path = os.path.join(self._ref_audio_dir, ref_path)

            if not os.path.exists(ref_path):
                raise FileNotFoundError(f"Reference audio not found: {ref_path}")

            refs = [{"audio_path": ref_path}]
            if reference_text:
                refs[0]["text"] = reference_text
            payload["references"] = refs

        response = self._session.post(endpoint, json=payload, timeout=300)
        response.raise_for_status()

        audio_bytes = response.content
        wav_data = self._parse_wav(audio_bytes)
        self._touch_activity()
        return wav_data

    def unload_model(self, model_type: Optional[str] = None) -> list[Tuple[str, str]]:
        """Unload the cached Fish backend handle from this process.

        This mirrors the Qwen app-layer contract. It does not terminate the
        remote SGLang server process because that process owns the actual model
        memory and currently exposes no unload endpoint in this stack.
        """
        if model_type not in (None, "s2-pro"):
            return []

        with self._lock:
            if not self._loaded:
                return []
            self._loaded = False
            self._last_activity = time.monotonic()

        logger.info("Fish Audio backend marked unloaded", extra={"sglang_url": self._sglang_url})
        return [("s2-pro", "remote")]

    def unload_idle_models(self, idle_seconds: Optional[int] = None) -> list[Tuple[str, str]]:
        """Unload the Fish backend handle after a period of inactivity."""
        if idle_seconds is None:
            idle_seconds = int(getattr(self._settings, "tts_unload_idle_seconds", 300))

        with self._lock:
            if not self._loaded:
                return []
            if time.monotonic() - self._last_activity < float(idle_seconds):
                return []

        return self.unload_model("s2-pro")

    def status(self) -> dict[str, Any]:
        with self._lock:
            return {
                "model_type": "s2-pro",
                "loaded": self._loaded,
                "backend": "remote",
                "sglang_url": self._sglang_url,
                "model_dir": self._model_dir,
                "last_activity": self._last_activity,
            }

    def _parse_wav(self, wav_bytes: bytes) -> Tuple[np.ndarray, int]:
        """Parse WAV bytes to numpy array.

        Args:
            wav_bytes: Raw WAV file content.

        Returns:
            Tuple of (audio_array, sample_rate).
        """
        import io
        import wave

        with wave.open(io.BytesIO(wav_bytes), "rb") as wf:
            sample_rate = wf.getframerate()
            n_frames = wf.getnframes()
            audio_data = wf.readframes(n_frames)
            if wf.getsampwidth() == 2:
                audio = np.frombuffer(audio_data, dtype=np.int16).astype(np.float32) / 32768.0
            elif wf.getsampwidth() == 4:
                audio = np.frombuffer(audio_data, dtype=np.int32).astype(np.float32) / 2147483648.0
            else:
                audio = np.frombuffer(audio_data, dtype=np.uint8).astype(np.float32) / 255.0
            return audio, sample_rate

    def cleanup_vram(self) -> None:
        """Aggressive GPU cleanup (best-effort).

        Note: VRAM is managed by the SGLang server, not this service.
        Mark the app-layer handle unloaded so the next request re-checks and
        re-initializes the backend lazily.
        """
        self.unload_model("s2-pro")
