import pytest
import asyncio
from unittest.mock import patch, MagicMock, AsyncMock

@pytest.mark.asyncio
async def test_s2_pro_sync_endpoint_uses_executor():
    """Verify svc.generate() is called via run_in_executor to avoid blocking event loop."""
    from api.main import app
    from httpx import AsyncClient
    
    mock_wav = MagicMock()
    mock_wav.__len__ = lambda self: 48000
    
    with patch("api.main.get_sync_fish_audio_service") as mock_svc_factory:
        mock_svc = MagicMock()
        mock_svc.generate = MagicMock(return_value=(mock_wav, 24000))
        mock_svc_factory.return_value = mock_svc
        
        # Patch run_in_executor to verify it's called (not direct svc.generate)
        with patch("asyncio.get_event_loop") as mock_loop:
            mock_loop_instance = MagicMock()
            mock_loop_instance.run_in_executor = MagicMock(return_value=(mock_wav, 24000))
            mock_loop.return_value = mock_loop_instance
            
            async with AsyncClient(app=app, base_url="http://test") as client:
                response = await client.post(
                    "/v1/tts/s2-pro/sync",
                    json={"text": "Hello world"}
                )
            
            # Verify run_in_executor was called (not direct svc.generate)
            mock_loop_instance.run_in_executor.assert_called_once()
