import asyncio
import sys
import threading
from types import ModuleType
from unittest.mock import AsyncMock, MagicMock, patch

import pytest


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
def test_models_unload_handles_s2_pro_locally():
    app = _import_test_app()
    from httpx import AsyncClient

    mock_svc = MagicMock()
    mock_svc.unload_model.return_value = [("s2-pro", "remote")]

    mock_controller = MagicMock()
    # Simulate container was running and is stopped successfully
    mock_controller.stop_if_running.return_value = True

    with patch("api.main._SYNC_FISH_AUDIO_SERVICE", mock_svc):
        with patch("api.main.get_fish_sglang_controller", return_value=mock_controller):
            async with AsyncClient(app=app, base_url="http://test") as client:
                response = await client.post("/v1/models/unload?model_type=s2-pro")

    assert response.status_code == 200
    assert response.json()["unloaded"] == [["s2-pro", "remote"]]
    assert response.json()["container_stopped"] is True
    mock_svc.unload_model.assert_called_once_with("s2-pro")
    mock_controller.stop_if_running.assert_called_once()

@pytest.mark.asyncio
def test_models_status_includes_fish_backend():
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
