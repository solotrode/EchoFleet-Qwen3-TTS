"""Fish Audio S2 Pro service wrapper.

Calls the SGLang server (fish-sglang) which runs the Fish Audio S2 Pro model.
Supports both plain TTS and voice cloning via reference audio.
"""
from typing import Optional, Tuple
import threading
import os

import numpy as np
import requests

from config.settings import get_settings


class FishAudioService:
    """Fish Audio S2 Pro wrapper calling SGLang server."""

    def __init__(self) -> None:
        self._settings = get_settings()
        self._lock = threading.Lock()
        self._model_dir = getattr(self._settings, "s2_pro_model_dir", "/models/s2-pro")
        self._sglang_url = getattr(self._settings, "sglang_base_url", "http://fish-sglang:8000")
        self._ref_audio_dir = getattr(self._settings, "fish_ref_audio_dir", "/ref_audio")
        self._session = requests.Session()

    def __del__(self) -> None:
        if hasattr(self, "_session"):
            self._session.close()

    @property
    def model_dir(self) -> str:
        return self._model_dir

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
        return wav_data

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
        This is a no-op but kept for interface consistency.
        """
        pass
