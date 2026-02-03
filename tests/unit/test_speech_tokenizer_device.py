from __future__ import annotations

from dataclasses import dataclass

import pytest

torch = pytest.importorskip("torch")

from inference.qwen_tts_service import _ensure_speech_tokenizer_on_device


class _InnerTorchModule(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.linear = torch.nn.Linear(1, 1)
        self.last_to_kwargs: dict | None = None

    def to(self, *args, **kwargs):  # type: ignore[override]
        self.last_to_kwargs = dict(kwargs)
        return super().to(*args, **kwargs)


@dataclass
class _SpeechTokenizer:
    model: _InnerTorchModule
    device: torch.device


@dataclass
class _HFModel:
    speech_tokenizer: _SpeechTokenizer


@dataclass
class _Wrapper:
    model: _HFModel


def test_ensure_speech_tokenizer_on_device_moves_inner_model_and_sets_device() -> None:
    inner = _InnerTorchModule()
    tok = _SpeechTokenizer(model=inner, device=torch.device("cpu"))
    wrapper = _Wrapper(model=_HFModel(speech_tokenizer=tok))

    _ensure_speech_tokenizer_on_device(wrapper, device="cpu", dtype=torch.float32)
    assert tok.device == torch.device("cpu")

    # We can't guarantee CUDA in unit tests, so validate that we set .device and attempted to move.
    _ensure_speech_tokenizer_on_device(wrapper, device="cuda:0", dtype=torch.float16)
    assert tok.device == torch.device("cuda:0")
    assert inner.last_to_kwargs is not None
    assert inner.last_to_kwargs.get("device") == "cuda:0"
    assert inner.last_to_kwargs.get("dtype") == torch.float16


def test_ensure_speech_tokenizer_on_device_is_noop_when_missing() -> None:
    class _NoTokenizer:
        pass

    _ensure_speech_tokenizer_on_device(_NoTokenizer(), device="cuda:0", dtype=torch.float16)
