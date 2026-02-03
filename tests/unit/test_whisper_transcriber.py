import types

import pytest

import inference.whisper_service as ws


class DummyProcessor:
    def __init__(self):
        self.tokenizer = object()
        self.feature_extractor = object()


class DummyModel:
    pass


def dummy_pipeline(task, model, tokenizer, feature_extractor, device):
    def fn(audio_path):
        return {"text": "dummy transcription", "language": "en"}

    return fn


@pytest.fixture(autouse=True)
def patch_hf(monkeypatch):
    # Patch HF constructors used by WhisperTranscriberHF
    monkeypatch.setattr(ws, "AutoProcessor", types.SimpleNamespace(from_pretrained=lambda *a, **k: DummyProcessor()))
    monkeypatch.setattr(ws, "AutoModelForSpeechSeq2Seq", types.SimpleNamespace(from_pretrained=lambda *a, **k: DummyModel()))
    monkeypatch.setattr(ws, "pipeline", dummy_pipeline)
    yield


def test_whisper_transcriber_hf_cpu():
    transcriber = ws.WhisperTranscriberHF(model_id="local/test", device="cpu", dtype=None)
    res = transcriber.transcribe("fake.wav")
    assert res.text == "dummy transcription"
    assert res.language == "en"
