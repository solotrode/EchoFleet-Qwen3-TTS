"""Fish Audio S2 Pro service wrapper (stub).

This is a narrow, lazy-loading wrapper around the Fish Audio S2 Pro inference
library. The actual integration is deferred until the upstream Python package
and API shape are verified. For now `generate` raises NotImplementedError so
the endpoint and tests can mock this service without importing heavy
dependencies at import time.
"""
from typing import Optional, Tuple
import threading
import os
import gc

import numpy as np

from config.settings import get_settings


class FishAudioService:
    """Lazy-loading Fish Audio S2 Pro wrapper.

    The real implementation should load the verified Fish Audio model and
    expose a `generate(text, language=None) -> (numpy.ndarray, int)` method.
    """

    def __init__(self) -> None:
        self._settings = get_settings()
        self._lock = threading.Lock()
        self._model = None
        self._model_dir = getattr(self._settings, "s2_pro_model_dir", "/models/s2-pro")
        self._device = getattr(self._settings, "s2_pro_device", None)

    @property
    def model_dir(self) -> str:
        return self._model_dir

    def _load_model(self) -> None:
        """Perform minimal model availability checks.

        The real loader should download or load the S2 Pro model. Here we only
        verify the configured model directory exists to provide a useful error
        (FileNotFoundError) when the operator hasn't placed model files.
        """
        if not os.path.exists(self._model_dir):
            raise FileNotFoundError(f"S2 Pro model directory not found: {self._model_dir}")
        # Real model loading happens here in a future implementation.
        self._model = object()

    def generate(self, text: str, language: Optional[str] = None) -> Tuple[np.ndarray, int]:
        """Generate waveform for the given text.

        Raises:
            FileNotFoundError: if model files are missing
            NotImplementedError: until the real Fish Audio package is wired in
        """
        if self._model is None:
            with self._lock:
                if self._model is None:
                    self._load_model()

        # For initial smoke testing we return a deterministic 1s 24kHz 440Hz sine
        # wave. This lets the endpoint, storage, and Redis plumbing be tested
        # without the real Fish Audio package installed.
        import numpy as _np

        sr = 24000
        t = _np.linspace(0, 1.0, int(sr * 1.0), endpoint=False)
        wav = 0.2 * _np.sin(2 * _np.pi * 440.0 * t).astype(_np.float32)
        return wav, sr

    def cleanup_vram(self) -> None:
        """Aggressive GPU cleanup (best-effort).

        Mirrors cleanup used elsewhere in this repo.
        """
        try:
            import torch

            if torch.cuda.is_available():
                for device_id in range(torch.cuda.device_count()):
                    with torch.cuda.device(device_id):
                        torch.cuda.synchronize()
                        torch.cuda.empty_cache()
        except Exception:
            # If torch isn't available, ignore silently — cleanup is best-effort.
            pass

        for _ in range(3):
            gc.collect()
