import pytest
import requests
from inference.fish_audio_service import FishAudioService

def test_fish_audio_service_uses_session():
    svc = FishAudioService()
    assert hasattr(svc, "_session")
    assert isinstance(svc._session, requests.Session)
