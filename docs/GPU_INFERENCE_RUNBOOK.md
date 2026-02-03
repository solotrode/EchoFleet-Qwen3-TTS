# GPU Inference Runbook (Qwen3-TTS)

## Symptoms

- Jobs show `status=running` but GPUs are mostly idle.
- Voice-design latency regresses dramatically (e.g. ~90s → 400s+).
- Worker logs indicate CUDA is available, but CPU utilization is high.

## Root Cause: speech tokenizer stuck on CPU

Qwen3-TTS uses a **speech tokenizer decoder** (12Hz) to convert generated codes into waveform.
That tokenizer is exposed as `model.model.speech_tokenizer`.

Important detail:
- `speech_tokenizer` is **not** a `torch.nn.Module`.
- If you load the model on CPU and later do `model.model.to("cuda:0")`, PyTorch does **not**
  automatically move `speech_tokenizer.model`.
- If `speech_tokenizer.device` stays CPU, decode becomes CPU-bound and dominates runtime.

This repo addresses that by explicitly moving the speech tokenizer to the leased GPU during
model initialization.

Implementation: [inference/qwen_tts_service.py](../inference/qwen_tts_service.py)

## What “healthy” looks like

- Worker log line `Set model device` shows a CUDA device (e.g. `cuda:1`).
- Worker log line `Set speech tokenizer device` also shows the same CUDA device.
- During generation you see sustained GPU utilization during decode.

## Quick Checks

Inside the worker container:

- Confirm CUDA works:
  - `python -c "import torch; print(torch.cuda.is_available(), torch.cuda.device_count())"`
- Confirm model + tokenizer devices match (at runtime, via logs):
  - Look for `Set model device` and `Set speech tokenizer device`.

## Avoiding regressions

- If you ever change model loading to use CPU-first placement, ensure you also move:
  - `model.model.speech_tokenizer.model.to(cuda_device)`
  - `model.model.speech_tokenizer.device = torch.device(cuda_device)`

If you use `device_map` at load time instead, verify that the tokenizer is also loaded with
an equivalent `device_map`.
