"""
Accuracy scoring utilities for candidate generation.

Provides normalization (via whisper-normalizer) and WER/CER scoring (via jiwer).
"""
from __future__ import annotations

from typing import Dict, Any
import logging

from jiwer import wer, cer

# Try to import whisper-normalizer robustly; API changed between versions.
try:
    from whisper_normalizer import BasicTextNormalizer  # preferred
    _normalizer = BasicTextNormalizer()
except Exception:
    try:
        import whisper_normalizer as _wn

        # Common alternative entrypoints
        if hasattr(_wn, "BasicTextNormalizer"):
            _normalizer = _wn.BasicTextNormalizer()
        elif hasattr(_wn, "WhisperNormalizer"):
            _normalizer = _wn.WhisperNormalizer()
        elif hasattr(_wn, "basic_normalizer"):
            _normalizer = _wn.basic_normalizer()
        elif hasattr(_wn, "normalize"):
            _normalizer = _wn.normalize
        else:
            raise ImportError("no known normalizer in whisper_normalizer")
    except Exception:
        # Fallback: simple lowercase function
        _normalizer = lambda s: (s or "").lower()

logger = logging.getLogger(__name__)


def normalize_text(text: str) -> str:
    """Normalize text for fair WER/CER comparison.

    Uses whisper-normalizer to handle punctuation, numbers, and common
    normalization steps used in ASR evaluation.
    """
    if text is None:
        return ""
    try:
        normalized = _normalizer(text)
    except Exception as exc:
        logger.debug("whisper-normalizer failed, falling back to simple normalize: %s", exc)
        normalized = text.lower()

    # Collapse whitespace
    return " ".join(normalized.split())


class AccuracyScorer:
    """Scores a candidate transcription against a reference using WER/CER.

    Methods return a dict with accuracy_score (1 - WER), word_error_rate, char_error_rate,
    and normalized strings for debugging.
    """

    def score_candidate(self, reference: str, transcription: str, duration: float) -> Dict[str, Any]:
        ref_clean = normalize_text(reference or "")
        trans_clean = normalize_text(transcription or "")

        try:
            word_error_rate = wer(ref_clean, trans_clean)
        except Exception as exc:
            logger.warning("WER calculation failed: %s", exc)
            word_error_rate = 1.0

        try:
            char_error_rate = cer(ref_clean, trans_clean)
        except Exception as exc:
            logger.warning("CER calculation failed: %s", exc)
            char_error_rate = 1.0

        accuracy = max(0.0, 1.0 - word_error_rate)

        return {
            "accuracy_score": accuracy,
            "word_error_rate": word_error_rate,
            "char_error_rate": char_error_rate,
            "duration_seconds": duration,
            "reference_text": ref_clean,
            "transcribed_text": trans_clean,
            "reference_words": len(ref_clean.split()),
            "transcribed_words": len(trans_clean.split()),
        }
