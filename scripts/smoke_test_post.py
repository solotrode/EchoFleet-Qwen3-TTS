import base64
import io
import json
import sys
import numpy as np
import soundfile as sf
import requests
try:
    from utils.logging import get_logger
except Exception:
    # If script is executed directly, ensure repo root is on sys.path
    import pathlib
    repo_root = str(pathlib.Path(__file__).resolve().parents[1])
    if repo_root not in sys.path:
        sys.path.insert(0, repo_root)
    from utils.logging import get_logger

logger = get_logger(__name__)

API_URL = "http://localhost:18000/v1/tts/voice-clone"

# Generate 0.5s sine at 24000 Hz
sr = 24000
t = np.linspace(0, 0.5, int(0.5 * sr), endpoint=False)
wave = 0.1 * np.sin(2 * np.pi * 440 * t).astype(np.float32)

buf = io.BytesIO()
sf.write(buf, wave, sr, format="WAV")
buf.seek(0)

b64 = base64.b64encode(buf.read()).decode("ascii")

payload = {
    "text": "Testing smoke",
    "ref_audio": b64,
    "ref_text": "test",
    # Use a model-expected language label to bypass mapping issues.
    "language": "english",
    "x_vector_only_mode": False,
    "output_format": "wav"
}

try:
    resp = requests.post(API_URL, json=payload, timeout=120)
    logger.info("smoke_test: response", extra={"status_code": resp.status_code})
    if resp.headers.get('content-type','').startswith('application/json'):
        data = resp.json()
        logger.info("smoke_test: json keys", extra={"keys": list(data.keys())})
        if 'audio_base64' in data:
            logger.info("smoke_test: audio size", extra={"audio_base64_len": len(data['audio_base64'])})
    else:
        logger.warning("smoke_test: non-json response", extra={"length": len(resp.content)})
    logger.debug(resp.text[:400])
except Exception as e:
    logger.exception("smoke_test: request failed")
    sys.exit(2)
