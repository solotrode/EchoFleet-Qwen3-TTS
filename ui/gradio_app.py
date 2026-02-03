"""Gradio web UI for Qwen3-TTS API.

This UI is intentionally a separate process/service that calls the FastAPI
endpoints. That keeps the serving architecture clean (one API process, one UI
process) and avoids running multiple servers in a single container.
"""

from __future__ import annotations

import base64
import json
import os
import io
import urllib.error
import urllib.request
from utils.logging import get_logger
import socket
import time
from typing import Any, Dict, Optional, Tuple

import gradio as gr
import numpy as np
import soundfile as sf


def _api_base_url() -> str:
    return os.getenv("API_BASE_URL", "http://localhost:18000").rstrip("/")


def _post_json(path: str, payload: Dict[str, Any]) -> Dict[str, Any]:
    url = f"{_api_base_url()}{path}"
    data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(
        url,
        data=data,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=600) as resp:
            raw = resp.read().decode("utf-8")
            return json.loads(raw)
    except urllib.error.HTTPError as exc:
        body = exc.read().decode("utf-8") if exc.fp else ""
        raise RuntimeError(f"HTTP {exc.code} calling {url}: {body}") from exc
    except urllib.error.URLError as e:
        raise RuntimeError(f"Network error calling {url}: {e}") from e
    except (BrokenPipeError, socket.error, OSError) as e:
        raise RuntimeError(f"Connection failed when calling {url}: {e}") from e


def _decode_audio_base64(audio_b64: str) -> Tuple[int, np.ndarray]:
    wav_bytes = base64.b64decode(audio_b64)
    bio = io.BytesIO(wav_bytes)
    wav, sr = sf.read(bio, dtype="float32")
    if wav.ndim > 1:
        wav = wav[:, 0]
    # Return (sample_rate, array) for Gradio `Audio` with `type='numpy'`
    return int(sr), wav.astype("float32")


def _file_to_base64(path: str) -> str:
    # Enforce client-side upload size limit to avoid sending huge base64 blobs
    try:
        max_mb = int(os.getenv("MAX_UPLOAD_MB", "10"))
    except Exception:
        max_mb = 10
    size = os.path.getsize(path)
    if size > (max_mb * 1024 * 1024):
        raise ValueError(f"File too large for upload ({size} bytes > {max_mb} MB)")
    with open(path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")


def voice_clone(
    text: str,
    language: str,
    ref_audio_file: Optional[str],
    ref_audio_url: str,
    ref_text: str,
    x_vector_only_mode: bool,
) -> Tuple[Tuple[int, np.ndarray], str]:
    if ref_audio_file:
        ref_audio = _file_to_base64(ref_audio_file)
    else:
        ref_audio = ref_audio_url.strip()
    # If user pasted a very long data URL or base64, refuse client-side
    try:
        if isinstance(ref_audio, str) and (ref_audio.startswith("data:") or ("base64" in ref_audio and len(ref_audio) > 1024)):
            raise ValueError("Ref audio appears to be a large data URL; please upload a file instead (or shorten the data).")
    except Exception:
        pass

    # Client-side validation: prevent sending empty ref_audio when not using x_vector_only_mode
    if not x_vector_only_mode and not ref_audio:
        raise ValueError("Reference audio required unless x_vector_only_mode=True")

    payload: Dict[str, Any] = {
        "text": text,
        "language": (language or None),
        "ref_audio": ref_audio,
        "ref_text": (ref_text or None),
        "x_vector_only_mode": x_vector_only_mode,
    }

    # Submit job for asynchronous processing and poll for result
    try:
        submit = _post_json("/v1/tts/voice-clone/submit", payload)
    except Exception as e:
        return None, f"Error submitting job: {e}"

    job_id = submit.get("job_id")
    if not job_id:
        return None, "Failed to obtain job id from server"

    # Poll for job completion (respect UI timeout). Poll interval 2s.
    timeout_seconds = 600
    poll_interval = 2
    elapsed = 0
    while elapsed < timeout_seconds:
        try:
            status = _post_json(f"/v1/jobs/{job_id}", {}) if False else None
            # Use urllib directly for GET
            url = f"{_api_base_url()}/v1/jobs/{job_id}"
            req = urllib.request.Request(url, method="GET")
            with urllib.request.urlopen(req, timeout=10) as resp:
                raw = resp.read().decode("utf-8")
                job = json.loads(raw)

            if job.get("status") == "done":
                try:
                    wav = _decode_audio_base64(job["audio_base64"])
                except Exception as e:
                    return None, f"Error decoding audio: {e}"
                meta = f"sr={job.get('sample_rate')} duration={float(job.get('duration_seconds',0)):.2f}s"
                return wav, meta

            if job.get("status") == "failed":
                return None, f"Job failed: {job.get('error', 'unknown error')}"

        except urllib.error.HTTPError as exc:
            body = exc.read().decode("utf-8") if exc.fp else ""
            return None, f"HTTP {exc.code} polling job: {body}"
        except Exception as e:
            # transient network issues — wait and retry
            pass

        time.sleep(poll_interval)
        elapsed += poll_interval

    return None, f"Job timed out after {timeout_seconds}s"


def custom_voice(
    text: str,
    language: str,
    speaker: str,
    instruct: str,
) -> Tuple[Tuple[int, np.ndarray], str]:
    payload: Dict[str, Any] = {
        "text": text,
        "language": (language or None),
        "speaker": speaker,
        "instruct": (instruct or None),
    }

    try:
        data = _post_json("/v1/tts/custom-voice", payload)
    except Exception as e:
        return None, f"Error calling API: {e}"

    try:
        wav = _decode_audio_base64(data["audio_base64"])
    except Exception as e:
        return None, f"Error decoding audio: {e}"

    meta = f"sr={data['sample_rate']} duration={data['duration_seconds']:.2f}s"
    return wav, meta


def voice_design(
    text: str,
    language: str,
    instruct: str,
) -> Tuple[Tuple[int, np.ndarray], str]:
    payload: Dict[str, Any] = {
        "text": text,
        "language": (language or None),
        "instruct": instruct,
    }

    try:
        data = _post_json("/v1/tts/voice-design", payload)
    except Exception as e:
        return None, f"Error calling API: {e}"

    try:
        wav = _decode_audio_base64(data["audio_base64"])
    except Exception as e:
        return None, f"Error decoding audio: {e}"

    meta = f"sr={data['sample_rate']} duration={data['duration_seconds']:.2f}s"
    return wav, meta


def build_ui() -> gr.Blocks:
    with gr.Blocks(title="Qwen3-TTS") as demo:
        gr.Markdown(
            "# Qwen3-TTS Web UI\n"
            "This UI calls the FastAPI service. Set `API_BASE_URL` to point at it."
        )

        with gr.Tab("Voice Clone"):
            text = gr.Textbox(label="Text", lines=3)
            language = gr.Textbox(label="Language (optional)", value="English")
            ref_audio_file = gr.File(label="Reference audio file (optional)", type="filepath")
            ref_audio_url = gr.Textbox(
                label="Reference audio URL (used if no file)",
                value="https://qianwen-res.oss-cn-beijing.aliyuncs.com/Qwen3-TTS-Repo/clone.wav",
            )
            ref_text = gr.Textbox(
                label="Reference transcript (optional if x_vector_only_mode)",
                value=(
                    "Okay. Yeah. I resent you. I love you. I respect you. "
                    "But you know what? You blew it! And thanks to you."
                ),
                lines=2,
            )
            x_vector_only_mode = gr.Checkbox(label="x_vector_only_mode", value=False)
            btn = gr.Button("Generate")
            out_audio = gr.Audio(label="Output", type="numpy")
            out_meta = gr.Textbox(label="Info")
            btn.click(
                voice_clone,
                inputs=[text, language, ref_audio_file, ref_audio_url, ref_text, x_vector_only_mode],
                outputs=[out_audio, out_meta],
            )

        with gr.Tab("Custom Voice"):
            text = gr.Textbox(label="Text", lines=3)
            language = gr.Textbox(label="Language (optional)", value="English")
            speaker = gr.Dropdown(
                label="Speaker",
                choices=[
                    "Vivian",
                    "Serena",
                    "Uncle_Fu",
                    "Dylan",
                    "Eric",
                    "Ryan",
                    "Aiden",
                    "Ono_Anna",
                    "Sohee",
                ],
                value="Ryan",
            )
            instruct = gr.Textbox(label="Instruct (optional)", value="Very happy.")
            btn = gr.Button("Generate")
            out_audio = gr.Audio(label="Output", type="numpy")
            out_meta = gr.Textbox(label="Info")
            btn.click(custom_voice, inputs=[text, language, speaker, instruct], outputs=[out_audio, out_meta])

        with gr.Tab("Voice Design"):
            text = gr.Textbox(label="Text", lines=3)
            language = gr.Textbox(label="Language (optional)", value="English")
            instruct = gr.Textbox(
                label="Voice description",
                lines=3,
                value="Speak in an incredulous tone, with a hint of panic.",
            )
            btn = gr.Button("Generate")
            out_audio = gr.Audio(label="Output", type="numpy")
            out_meta = gr.Textbox(label="Info")
            btn.click(voice_design, inputs=[text, language, instruct], outputs=[out_audio, out_meta])

    return demo


def main() -> None:
    demo = build_ui()
    demo.queue(default_concurrency_limit=2)
    demo.launch(server_name="0.0.0.0", server_port=7860)


if __name__ == "__main__":
    main()
