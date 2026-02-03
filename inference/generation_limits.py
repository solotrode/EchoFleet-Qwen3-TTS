"""Helpers for controlling Qwen3-TTS generation length.

These helpers are intentionally torch-free so they can be unit tested on the
host machine without requiring the full CUDA/PyTorch stack.
"""

from __future__ import annotations


def estimate_max_new_tokens(
    text: str | None,
    *,
    tokens_per_second: float,
    words_per_second: float,
    max_output_seconds: int,
    max_new_tokens: int,
    min_new_tokens: int,
) -> int:
    """Estimate a reasonable `max_new_tokens` for Qwen3-TTS.

    Qwen3-TTS "12Hz" models produce roughly `tokens_per_second` acoustic tokens
    per second of audio. If `max_new_tokens` is set too high, the model can
    generate minutes of output (often trailing silence) before hitting the cap.

    Args:
        text: Input text to synthesize.
        tokens_per_second: Approximate acoustic token rate (e.g., 12 for 12Hz models).
        words_per_second: Estimated speaking rate used to convert text length to seconds.
        max_output_seconds: Hard cap on output seconds.
        max_new_tokens: Hard cap on `max_new_tokens`.
        min_new_tokens: Floor for `max_new_tokens` to avoid truncating short texts.

    Returns:
        Estimated `max_new_tokens`, clamped to [min_new_tokens, max_new_tokens].
    """
    safe_text = (text or "").strip()
    word_count = len(safe_text.split()) if safe_text else 0

    safe_wps = max(0.5, float(words_per_second))
    est_seconds = (word_count / safe_wps) if word_count else 3.0

    # Add padding for pauses/breathing.
    est_seconds = max(3.0, est_seconds + 2.0)
    est_seconds = min(float(max_output_seconds), est_seconds)

    safe_tps = max(1.0, float(tokens_per_second))
    est_tokens = int(est_seconds * safe_tps)

    est_tokens = max(int(min_new_tokens), est_tokens)
    est_tokens = min(int(max_new_tokens), est_tokens)
    return est_tokens
