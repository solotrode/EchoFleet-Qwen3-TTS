"""HuggingFace/Transformers-based Whisper transcription.

We intentionally do not depend on the `openai-whisper` Python package because it
pins `triton<3`, which conflicts with PyTorch 2.7 on Linux (torch requires
triton==3.x).

This module loads the Whisper model from HuggingFace Hub (or cache) and provides
a small wrapper used by workers for transcription/scoring.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional

import torch
from transformers import (
    AutoModelForSpeechSeq2Seq,
    AutoProcessor,
    pipeline,
)

from utils.logging import get_logger

logger = get_logger(__name__)


@dataclass(frozen=True)
class WhisperTranscription:
    """A single transcription result."""

    text: str
    language: Optional[str] = None
    raw: Optional[Dict[str, Any]] = None


class WhisperTranscriberHF:
    """Whisper transcriber using HuggingFace Transformers.

    Args:
        model_id: HuggingFace model repo id, e.g. "openai/whisper-large-v3-turbo".
        device: Torch device string, e.g. "cuda:0" or "cpu".
        dtype: Torch dtype to load the model weights.

    Note:
        This class keeps the ASR pipeline in memory; create one per GPU if you want
        true parallelism.
    """

    def __init__(
        self,
        model_id: str,
        device: str = "cuda:0",
        dtype: torch.dtype = torch.float16,
    ) -> None:
        self._model_id = model_id
        self._device = device
        self._dtype = dtype

        device_index = self._device_index(device)
        logger.info(
            "Initializing Whisper HF transcriber",
            extra={"model_id": model_id, "device": device, "dtype": str(dtype)},
        )

        processor = AutoProcessor.from_pretrained(model_id)
        model = AutoModelForSpeechSeq2Seq.from_pretrained(
            model_id,
            dtype=dtype,
            low_cpu_mem_usage=True,
        )

        if device_index is not None:
            model.to(f"cuda:{device_index}")

        # transformers.pipeline handles chunking + preprocessing.
        # device: -1 for CPU, otherwise CUDA device index.
        self._asr = pipeline(
            task="automatic-speech-recognition",
            model=model,
            tokenizer=processor.tokenizer,
            feature_extractor=processor.feature_extractor,
            device=-1 if device_index is None else device_index,
        )

    @staticmethod
    def _device_index(device: str) -> Optional[int]:
        if device == "cpu":
            return None
        if device.startswith("cuda:"):
            try:
                return int(device.split(":", 1)[1])
            except Exception:
                return 0
        if device == "cuda":
            return 0
        return None

    def transcribe(
        self,
        audio_path: str,
        language: Optional[str] = None,
        chunk_length_s: int = 30,
        batch_size: int = 8,
    ) -> WhisperTranscription:
        """Transcribe an audio file.

        Args:
            audio_path: Path to an audio file (wav/mp3/etc).
            language: Optional language hint (e.g., "en").
            chunk_length_s: Chunk length for long audio.
            batch_size: Pipeline batch size.

        Returns:
            Transcription result.

        Raises:
            RuntimeError: If transcription fails.
        """
        try:
            asr_kwargs: Dict[str, Any] = {
                "chunk_length_s": chunk_length_s,
                "batch_size": batch_size,
                "return_timestamps": False,
            }
            if language:
                asr_kwargs["generate_kwargs"] = {"language": language}

            result = self._asr(audio_path, **asr_kwargs)

            # transformers returns a dict with at least {"text": ...}
            text = (result.get("text") if isinstance(result, dict) else None) or ""
            return WhisperTranscription(text=text.strip(), language=language, raw=result)

        except Exception as exc:
            logger.exception(
                "Whisper transcription failed",
                extra={"audio_path": audio_path, "model_id": self._model_id},
            )
            raise RuntimeError(f"Whisper transcription failed: {exc}") from exc

    def unload(self) -> None:
        """Unload Whisper model and free GPU memory.

        Performs aggressive cleanup to ensure GPU memory is released:
        - Deletes ASR pipeline
        - Runs garbage collection multiple times
        - Empties CUDA cache multiple times
        - Synchronizes CUDA device
        """
        import gc

        logger.info(
            "Unloading Whisper model", extra={"model_id": self._model_id, "device": self._device}
        )

        # Delete the ASR pipeline
        try:
            del self._asr
        except Exception as e:
            logger.warning(f"Failed to delete ASR pipeline: {e}")

        # Synchronize CUDA if on GPU
        if self._device != "cpu" and torch.cuda.is_available():
            try:
                device_idx = self._device_index(self._device)
                if device_idx is not None:
                    torch.cuda.synchronize(device_idx)
            except Exception as e:
                logger.warning(f"CUDA synchronize failed: {e}")

        # Multiple rounds of garbage collection
        for _ in range(5):
            gc.collect()

        # Empty CUDA cache multiple times if on GPU
        if self._device != "cpu" and torch.cuda.is_available():
            device_idx = self._device_index(self._device)
            if device_idx is not None:
                try:
                    with torch.cuda.device(device_idx):
                        for _ in range(3):
                            torch.cuda.empty_cache()
                            gc.collect()
                except Exception as e:
                    logger.warning(f"CUDA empty_cache failed for {self._device}: {e}")

        logger.info(
            "Whisper model unloaded", extra={"model_id": self._model_id, "device": self._device}
        )
