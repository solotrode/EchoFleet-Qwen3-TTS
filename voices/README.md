# voices/

Place reference audio files here for use with the API. Use the simple filename format:

- `voices/<speaker>.wav`

Guidelines:
- Preferred format: WAV, PCM, mono, 24000 Hz (24 kHz). This minimizes preprocessing/resampling artifacts.
- MP3 is supported if ffmpeg/libsndfile are available inside the container, but WAV is recommended.
- Provide at least 3 seconds of clean speech; 5–30s is ideal for better speaker characterization.

Usage examples (inside requests):
- Container-local path: `/voices/speaker.wav`
- Data URL (base64): `data:audio/wav;base64,...`
- Remote URL: `https://example.com/speaker.wav`

Security:
- Only place trusted audio in `voices/` — the files will be readable by the service.
- If mounting from the host, ensure permissions are set appropriately.
