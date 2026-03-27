import asyncio
import pytest
import threading
import sys
from types import ModuleType
from unittest.mock import patch, MagicMock, AsyncMock


def _install_fake_redis_module() -> None:
    if "redis" in sys.modules:
        return

    fake_redis = ModuleType("redis")

    class FakeRedis:
        def __init__(self, *args, **kwargs):
            pass

        def hset(self, *args, **kwargs):
            return None

        def hget(self, *args, **kwargs):
            return None

        def hgetall(self, *args, **kwargs):
            return {}

        def get(self, *args, **kwargs):
            return None

        def rpush(self, *args, **kwargs):
            return None

        def scan_iter(self, *args, **kwargs):
            return iter(())

    fake_redis.Redis = FakeRedis
    sys.modules["redis"] = fake_redis


def _install_fake_torch_module() -> None:
    if "torch" in sys.modules:
        return

    fake_torch = ModuleType("torch")

    class FakeCuda:
        @staticmethod
        def is_available():
            return False

        @staticmethod
        def device_count():
            return 0

        @staticmethod
        def memory_allocated(_device):
            return 0

        @staticmethod
        def memory_reserved(_device):
            return 0

        @staticmethod
        def empty_cache():
            return None

        @staticmethod
        def synchronize():
            return None

    fake_torch.cuda = FakeCuda()
    sys.modules["torch"] = fake_torch


def _import_test_app():
    _install_fake_redis_module()
    _install_fake_torch_module()
    from api.main import app

    return app

@pytest.mark.asyncio
async def test_s2_pro_sync_endpoint_uses_executor():
    """Verify svc.generate() is called via run_in_executor to avoid blocking event loop."""
    app = _import_test_app()
    from httpx import AsyncClient
    
    mock_wav = MagicMock()
    mock_wav.__len__ = lambda self: 48000
    
    with patch("api.main.get_sync_fish_audio_service") as mock_svc_factory:
        mock_svc = MagicMock()
        mock_svc.generate = MagicMock(return_value=(mock_wav, 24000))
        mock_svc_factory.return_value = mock_svc

        with patch("api.main.wav_to_wav_bytes", return_value=b"RIFF"):
            loop = asyncio.get_running_loop()

            # Patch run_in_executor on the live loop used by the endpoint.
            with patch.object(loop, "run_in_executor", AsyncMock(return_value=(mock_wav, 24000))) as mock_run_in_executor:
                async with AsyncClient(app=app, base_url="http://test") as client:
                    response = await client.post(
                        "/v1/tts/s2-pro/sync",
                        json={"text": "Hello world"}
                    )

                assert response.status_code == 200
                mock_run_in_executor.assert_called_once()


@pytest.mark.asyncio
async def test_s2_pro_sync_endpoint_passes_flat_reference_fields():
    app = _import_test_app()
    from httpx import AsyncClient

    mock_wav = MagicMock()
    mock_wav.__len__ = lambda self: 48000

    with patch("api.main.get_sync_fish_audio_service") as mock_svc_factory:
        mock_svc = MagicMock()
        mock_svc.generate = MagicMock(return_value=(mock_wav, 24000))
        mock_svc_factory.return_value = mock_svc

        with patch("api.main.wav_to_wav_bytes", return_value=b"RIFF"):
            loop = asyncio.get_running_loop()

            with patch.object(loop, "run_in_executor", AsyncMock(return_value=(mock_wav, 24000))) as mock_run_in_executor:
                async with AsyncClient(app=app, base_url="http://test") as client:
                    response = await client.post(
                        "/v1/tts/s2-pro/sync",
                        json={
                            "text": "Hello world",
                            "language": "en",
                            "ref_audio": "voices/sample.wav",
                            "ref_text": "Reference text",
                        },
                    )

                assert response.status_code == 200
                mock_run_in_executor.assert_called_once_with(
                    None,
                    mock_svc.generate,
                    "Hello world",
                    "english",
                    "voices/sample.wav",
                    "Reference text",
                )


@pytest.mark.asyncio
async def test_models_unload_handles_s2_pro_locally():
    app = _import_test_app()
    from httpx import AsyncClient

    mock_svc = MagicMock()
    mock_svc.unload_model.return_value = [("s2-pro", "remote")]

    with patch("api.main._SYNC_FISH_AUDIO_SERVICE", mock_svc):
        async with AsyncClient(app=app, base_url="http://test") as client:
            response = await client.post("/v1/models/unload?model_type=s2-pro")

    assert response.status_code == 200
    assert response.json()["unloaded"] == [["s2-pro", "remote"]]
    mock_svc.unload_model.assert_called_once_with("s2-pro")


@pytest.mark.asyncio
async def test_models_status_includes_fish_backend():
    app = _import_test_app()
    from httpx import AsyncClient

    mock_qwen = MagicMock()
    mock_qwen._lock = threading.Lock()
    mock_qwen._models = {}

    mock_fish = MagicMock()
    mock_fish.status.return_value = {
        "model_type": "s2-pro",
        "loaded": True,
        "backend": "remote",
        "sglang_url": "http://fish-sglang:8000",
    }

    with patch("api.main.get_sync_tts_server", return_value=mock_qwen):
        with patch("api.main._SYNC_FISH_AUDIO_SERVICE", mock_fish):
            async with AsyncClient(app=app, base_url="http://test") as client:
                response = await client.get("/v1/models/status")

    assert response.status_code == 200
    assert response.json()["fish_backend"]["loaded"] is True
