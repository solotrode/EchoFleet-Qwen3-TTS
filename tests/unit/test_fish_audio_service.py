import io
import os
import time
import wave

import requests
from unittest.mock import MagicMock, patch

from inference.fish_audio_service import FishAudioService


def _wav_bytes(sample_rate: int = 24000, frame_count: int = 16) -> bytes:
    buf = io.BytesIO()
    with wave.open(buf, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)
        wf.writeframes(b"\x00\x00" * frame_count)
    return buf.getvalue()

def test_fish_audio_service_uses_session():
    svc = FishAudioService()
    assert hasattr(svc, "_session")
    assert isinstance(svc._session, requests.Session)


def test_fish_audio_service_lazy_loads_once():
    svc = FishAudioService()
    svc._model_dir = "/fake/model"

    mock_health = MagicMock()
    mock_health.raise_for_status.return_value = None

    mock_speech = MagicMock()
    mock_speech.raise_for_status.return_value = None
    mock_speech.content = _wav_bytes()

    with patch("inference.fish_audio_service.os.path.exists", return_value=True):
        svc._session.get = MagicMock(return_value=mock_health)
        svc._session.post = MagicMock(return_value=mock_speech)

        svc.generate("hello")
        svc.generate("again")

    assert svc._session.get.call_count == 1
    assert svc._session.post.call_count == 2
    assert svc.is_loaded is True


def test_fish_audio_service_flattens_reference_fields_to_upstream_payload():
    svc = FishAudioService()
    svc._model_dir = "/fake/model"

    mock_health = MagicMock()
    mock_health.raise_for_status.return_value = None

    mock_speech = MagicMock()
    mock_speech.raise_for_status.return_value = None
    mock_speech.content = _wav_bytes()

    with patch("inference.fish_audio_service.os.path.exists", return_value=True):
        svc._session.get = MagicMock(return_value=mock_health)
        svc._session.post = MagicMock(return_value=mock_speech)

        svc.generate(
            "hello",
            language="english",
            reference_audio="voices/sample.wav",
            reference_text="Reference text",
        )

    svc._session.post.assert_called_once_with(
        "http://fish-sglang:8000/v1/audio/speech",
        json={
            "input": "hello",
            "language": "english",
            "references": [
                {
                    "audio_path": os.path.join("/ref_audio", "voices/sample.wav"),
                    "text": "Reference text",
                }
            ],
        },
        timeout=300,
    )


def test_fish_audio_service_unload_model_marks_unloaded():
    svc = FishAudioService()
    svc._loaded = True

    unloaded = svc.unload_model("s2-pro")

    assert unloaded == [("s2-pro", "remote")]
    assert svc.is_loaded is False


def test_fish_audio_service_unload_idle_models_respects_timeout():
    svc = FishAudioService()
    svc._loaded = True
    svc._last_activity = time.monotonic() - 10

    unloaded = svc.unload_idle_models(idle_seconds=1)

    assert unloaded == [("s2-pro", "remote")]
    assert svc.is_loaded is False
