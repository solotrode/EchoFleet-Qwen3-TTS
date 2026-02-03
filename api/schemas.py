"""Pydantic schemas for the FastAPI service."""

from __future__ import annotations

from typing import Literal, Optional, List

from pydantic import BaseModel, Field, validator


class ErrorResponse(BaseModel):
    """Standard error payload."""

    error: str = Field(..., description="Error type")
    message: str = Field(..., description="Human-readable message")
    detail: Optional[str] = Field(default=None, description="Optional detail")


class AudioResponse(BaseModel):
    """Audio response for synchronous TTS endpoints."""

    audio_base64: str = Field(..., description="Base64-encoded WAV")
    sample_rate: int = Field(..., description="Sample rate in Hz")
    duration_seconds: float = Field(..., description="Audio duration in seconds")
    format: Literal["wav"] = Field(default="wav")
    model: str = Field(..., description="Model type used (base/custom-voice/voice-design)")
    job_id: Optional[str] = Field(default=None, description="Job id when persisted")
    download_url: Optional[str] = Field(default=None, description="URL to download persisted WAV")


class VoiceCloneRequest(BaseModel):
    """Request payload for voice cloning."""

    text: str = Field(..., min_length=1, max_length=5000)
    language: Optional[str] = Field(default=None, description="e.g. English, Chinese, Auto")

    ref_audio: str = Field(
        ...,
        description="Reference audio. Can be a local path, URL, or base64 string (per Qwen3-TTS).",
    )
    ref_text: Optional[str] = Field(
        default=None,
        description="Transcript for ref_audio (optional if x_vector_only_mode=True)",
    )
    x_vector_only_mode: bool = Field(
        default=False,
        description="If true, only use speaker embedding; ref_text can be omitted (lower quality)",
    )

    # Candidate generation fields
    num_candidates: int = Field(
        default=1,
        ge=1,
        le=12,
        description="Number of candidates to generate (1-12)",
    )
    return_all_candidates: bool = Field(
        default=False,
        description="If true, return all candidates with scores; otherwise return best only",
    )

    @validator("language", pre=True)
    def _map_short_language_codes(cls, v: Optional[str]) -> Optional[str]:
        if not v:
            return v
        mapping = {
            "en": "english",
            "eng": "english",
            "zh": "chinese",
            "cn": "chinese",
            "ja": "japanese",
            "jp": "japanese",
            "ko": "korean",
            "kr": "korean",
        }
        key = str(v).strip().lower()
        return mapping.get(key, v)


class CustomVoiceRequest(BaseModel):
    """Request payload for custom voice TTS."""

    text: str = Field(..., min_length=1, max_length=5000)
    language: Optional[str] = Field(default=None, description="e.g. English, Chinese, Auto")
    speaker: str = Field(..., description="CustomVoice speaker, e.g. Vivian, Ryan")
    instruct: Optional[str] = Field(default=None, description="Optional style instruction")


class VoiceDesignRequest(BaseModel):
    """Request payload for voice design TTS."""

    text: str = Field(..., min_length=1, max_length=5000)
    language: Optional[str] = Field(default=None, description="e.g. English, Chinese, Auto")
    instruct: str = Field(..., min_length=1, description="Natural-language voice description")


class CandidateScore(BaseModel):
    """Scoring metrics for a single candidate."""

    candidate_id: int
    accuracy_score: float = Field(..., description="1 - WER (0-1, higher is better)")
    word_error_rate: float = Field(..., description="WER (0-1, lower is better)")
    char_error_rate: float = Field(..., description="CER (0-1, lower is better)")
    duration_seconds: float
    reference_words: int
    transcribed_words: int
    transcribed_text: str
    tts_gpu: int = Field(..., description="GPU used for TTS generation")
    stt_device: str = Field(..., description="Device used for transcription")


class EnhancedAudioResponse(AudioResponse):
    """Enhanced audio response including candidate scoring metadata."""

    best_candidate_score: Optional[CandidateScore] = Field(
        default=None, description="Scoring metrics for the returned (best) candidate"
    )
    all_candidates: Optional[List[CandidateScore]] = Field(
        default=None, description="All candidate scores (if return_all_candidates=True)"
    )
    num_candidates_generated: Optional[int] = Field(default=None, description="Total candidates generated")


class JobSubmitResponse(BaseModel):
    """Response payload for async job submission."""

    job_id: str = Field(..., description="Job id to poll")
    status_url: str = Field(..., description="URL to poll job status")
    audio_url: str = Field(..., description="URL to download audio when ready")
    status: Literal["queued"] = Field(default="queued")
