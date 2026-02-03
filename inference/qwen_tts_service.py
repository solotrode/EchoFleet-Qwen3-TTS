"""Qwen3-TTS model loading and inference helpers.

This module provides a minimal, production-friendly wrapper around Qwen3-TTS
that:
- Loads models lazily from the locally mounted `/models` directory.
- Caches model instances in-process.
- Chooses a GPU from `TTS_GPUS` (simple round-robin) for each model type.

Note:
    This is a synchronous, in-process implementation intended to get the API
    endpoints working end-to-end. The RQ/Redis multi-GPU job pipeline can be
    layered on top once the API contract is stable.
"""

from __future__ import annotations

import threading
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional, Tuple

import soundfile as sf
import torch
import time

from config.settings import Settings
from qwen_tts import Qwen3TTSModel
from utils.logging import get_logger
from utils.audio_utils import load_audio, concat_wavs
from utils.text_chunker import chunk_text
from utils.gpu_pool import GpuLeasePool
from inference.generation_limits import estimate_max_new_tokens
import gc

logger = get_logger(__name__)


_TRANSFORMERS_LOG_FILTERS_INSTALLED = False


def _install_transformers_log_filters() -> None:
    """Install targeted log filters for excessively noisy third-party warnings."""

    global _TRANSFORMERS_LOG_FILTERS_INSTALLED
    if _TRANSFORMERS_LOG_FILTERS_INSTALLED:
        return

    class _DropFlashAttnDtypeWarning(logging.Filter):
        def filter(self, record: logging.LogRecord) -> bool:
            msg = record.getMessage()
            if "Flash Attention 2 without specifying a torch dtype" in msg:
                return False
            return True

    logging.getLogger("transformers.modeling_utils").addFilter(_DropFlashAttnDtypeWarning())
    _TRANSFORMERS_LOG_FILTERS_INSTALLED = True


_install_transformers_log_filters()


def _ensure_speech_tokenizer_on_device(model: object, device: str, dtype: torch.dtype) -> None:
    """Best-effort move of Qwen3-TTS speech tokenizer to the target device.

    Why this exists:
        `Qwen3TTSModel` exposes a `speech_tokenizer` that is *not* a
        `torch.nn.Module`, so `model.model.to(device)` will not recurse into it.
        If it remains on CPU, the 12Hz tokenizer decode path becomes CPU-bound
        even when the main model is on CUDA.

    Args:
        model: A `Qwen3TTSModel` instance (or compatible wrapper).
        device: Target device string (e.g. "cuda:0" or "cpu").
        dtype: Target dtype for the tokenizer model weights.
    """
    from utils.logging import get_logger
    logger = get_logger(__name__)
    
    if device == "cpu":
        return

    speech_tokenizer = None
    # Preferred: wrapper has `.model` which holds the HF model.
    try:
        if hasattr(model, "model") and hasattr(model.model, "speech_tokenizer"):
            speech_tokenizer = model.model.speech_tokenizer
        elif hasattr(model, "speech_tokenizer"):
            speech_tokenizer = model.speech_tokenizer
    except Exception:
        speech_tokenizer = None

    if speech_tokenizer is None:
        logger.warning("Speech tokenizer not found on model")
        return

    # Log current tokenizer device state
    try:
        inner = getattr(speech_tokenizer, "model", None)
        current_device = None
        if inner is not None and hasattr(inner, "parameters"):
            current_device = str(next(inner.parameters()).device)
        logger.info(
            "Speech tokenizer device check before move",
            extra={
                "current_device": current_device,
                "target_device": device,
                "has_inner_model": inner is not None
            }
        )
    except Exception as e:
        logger.warning(f"Failed to check tokenizer device: {e}")

    # Best-effort short-circuit: avoid re-moving if it is already on the target device.
    try:
        inner = getattr(speech_tokenizer, "model", None)
        if inner is not None and hasattr(inner, "parameters"):
            inner_device = next(inner.parameters()).device
            if str(inner_device) == str(torch.device(device)):
                tok_dev = getattr(speech_tokenizer, "device", None)
                if tok_dev is None or str(tok_dev) == str(torch.device(device)):
                    logger.info(
                        "Speech tokenizer already on target device - skipping move",
                        extra={"device": device}
                    )
                    return
    except Exception:
        # If inspection fails, continue with best-effort moves below.
        pass

    logger.info(
        "Moving speech tokenizer to device",
        extra={"target_device": device, "dtype": str(dtype)}
    )

    # The tokenizer wrapper holds an internal torch module at `.model`.
    inner = getattr(speech_tokenizer, "model", None)
    if inner is not None and hasattr(inner, "to"):
        try:
            inner.to(device=device, dtype=dtype)
            logger.info("Moved tokenizer.model.to()", extra={"device": device})
        except TypeError:
            inner.to(device=device)
            logger.info("Moved tokenizer.model.to() without dtype", extra={"device": device})

    # Some versions expose `.to()` / `.cuda()` directly on the tokenizer wrapper.
    if hasattr(speech_tokenizer, "to"):
        try:
            speech_tokenizer.to(device=device)
            logger.info("Called tokenizer.to()", extra={"device": device})
        except Exception as e:
            logger.warning(f"Failed tokenizer.to(): {e}")
    if hasattr(speech_tokenizer, "cuda"):
        try:
            # Accept either "cuda:0" or "0" forms.
            gpu_id = int(str(device).split(":")[-1])
            speech_tokenizer.cuda(gpu_id)
            logger.info("Called tokenizer.cuda()", extra={"gpu_id": gpu_id})
        except Exception as e:
            logger.warning(f"Failed tokenizer.cuda(): {e}")

    # Qwen tokenizer wrapper uses `.device` to move inputs before decode.
    if hasattr(speech_tokenizer, "device"):
        try:
            speech_tokenizer.device = torch.device(device)
            logger.info("Set tokenizer.device attribute", extra={"device": device})
        except Exception as e:
            logger.warning(f"Failed to set tokenizer.device: {e}")
    
    # Log final device state
    try:
        inner = getattr(speech_tokenizer, "model", None)
        final_device = None
        if inner is not None and hasattr(inner, "parameters"):
            final_device = str(next(inner.parameters()).device)
        logger.info(
            "Speech tokenizer device after move",
            extra={"final_device": final_device, "target_device": device}
        )
    except Exception as e:
        logger.warning(f"Failed to verify final tokenizer device: {e}")

# Qwen3TTSModel.from_pretrained() performs global AutoConfig/AutoModel/AutoProcessor
# registrations. Those registrations are not thread-safe. When multiple worker
# threads attempt to load models concurrently (e.g., 4 queued jobs hitting 4 GPUs),
# it can leave partially-initialized modules on the `meta` device and crash later.
#
# Serialize all from_pretrained() calls to keep initialization deterministic.
_FROM_PRETRAINED_LOCK = threading.Lock()


@dataclass(frozen=True)
class AudioResult:
    """Generated audio and metadata."""

    wav: "torch.Tensor | list | object"  # kept flexible; qwen returns numpy/torch-like arrays
    sample_rate: int


class QwenTTSServer:
    """Lazy-loading Qwen3-TTS model server.

    Args:
        settings: Application settings.

    Raises:
        FileNotFoundError: If expected model directories are missing.
    """

    def __init__(self, settings: Settings, assigned_device: Optional[str] = None) -> None:
        self._settings = settings
        self._assigned_device = assigned_device  # Device for this specific worker instance
        self._lock = threading.Lock()
        self._loading: Dict[Tuple[str, str], threading.Event] = {}
        # Cache models per (model_type, device)
        self._models: Dict[Tuple[str, str], Qwen3TTSModel] = {}
        # Timestamp of last generation activity (monotonic time)
        self._last_activity: float = time.monotonic()

        # The GpuLeasePool is no longer needed here, as each worker has a dedicated device.
        self._gpu_pool: Optional[GpuLeasePool] = None


    def _torch_dtype(self) -> torch.dtype:
        # Use float32 by default for stability with tokenizer/conv ops
        # that may have float biases. This avoids runtime errors where
        # inputs are bfloat16 but some bias tensors remain float32.
        raw = (self._settings.default_dtype or "float32").lower()
        if raw in {"bf16", "bfloat16"}:
            return torch.bfloat16
        if raw in {"fp16", "float16"}:
            return torch.float16
        if raw in {"fp32", "float32"}:
            return torch.float32
        return torch.float32

    def _attention_implementation(self, dtype: torch.dtype) -> str:
        """Select an attention implementation that is safe for this runtime.

        Preference order:
            - If `TTS_ATTN_IMPL` is set to a concrete value, honor it (with safety checks).
            - If `TTS_ATTN_IMPL=auto` (default), prefer FlashAttention2 when importable.
            - Otherwise, fall back to SDPA for reliability.

        Notes:
            FlashAttention2 requires a compatible `flash_attn` build for your exact
            torch/CUDA/Python combo. If it's not importable, we must not request it.
        """
        requested = (self._settings.tts_attn_impl or "").strip().lower()

        if requested in {"sdpa", "eager"}:
            # If the operator explicitly requests SDPA/eager, do not treat it as an error.
            return requested

        # FlashAttention2 only applies to fp16/bf16.
        def _flash_attention2_if_available() -> str | None:
            if dtype not in (torch.float16, torch.bfloat16):
                return None
            try:
                import flash_attn  # noqa: F401

                return "flash_attention_2"
            except Exception:
                return None

        if requested in {"flash", "flash_attention_2", "fa2"}:
            attn = _flash_attention2_if_available()
            if attn is not None:
                return attn
            logger.error(
                "Falling back to SDPA because FlashAttention2 is not available",
                extra={
                    "tts_attn_impl": requested,
                    "dtype": str(dtype),
                    "fallback": "sdpa",
                    "reason": "flash_attn_not_available_or_incompatible",
                },
            )
            return "sdpa"

        if requested in {"", "auto"}:
            attn = _flash_attention2_if_available()
            if attn is not None:
                return attn
            logger.error(
                "Using SDPA because FlashAttention2 is not available",
                extra={
                    "tts_attn_impl": requested or "auto",
                    "dtype": str(dtype),
                    "selected": "sdpa",
                    "reason": "flash_attn_not_available_or_incompatible",
                },
            )
            return "sdpa"

        logger.error(
            "Unknown TTS_ATTN_IMPL value; falling back to SDPA",
            extra={"tts_attn_impl": requested, "fallback": "sdpa"},
        )
        return "sdpa"

    def _lease_device(self):
        """Return the assigned device for this worker instance."""
        from contextlib import nullcontext
        if self._assigned_device:
            return nullcontext(self._assigned_device)
        
        # Fallback to CPU if no device is assigned
        logger.warning("No assigned device for this worker, falling back to CPU.")
        return nullcontext("cpu")

    def _model_dir(self, model_type: str) -> Path:
        base = Path(self._settings.model_cache_dir)
        # Support two possible layouts for the mounted models volume:
        # 1) /models/Qwen/<model-dir>  (when models were copied under a Qwen folder)
        # 2) /models/<model-dir>       (when the model dirs were copied directly)
        qwen_dir = base / "Qwen"

        mapping = {
            "base": "Qwen3-TTS-12Hz-1.7B-Base",
            "custom-voice": "Qwen3-TTS-12Hz-1.7B-CustomVoice",
            "voice-design": "Qwen3-TTS-12Hz-1.7B-VoiceDesign",
        }
        if model_type not in mapping:
            raise ValueError(f"Unknown model type: {model_type}")

        candidate1 = qwen_dir / mapping[model_type]
        candidate2 = base / mapping[model_type]

        if candidate1.exists():
            return candidate1
        if candidate2.exists():
            return candidate2

        # Helpful error message listing the checked paths
        raise FileNotFoundError(
            "Model directory not found. Checked paths:\n"
            f"  - {candidate1}\n"
            f"  - {candidate2}\n"
            "Please place your model directories under one of those locations and mount to /models."
        )

    def get_model(self, model_type: str, device: str) -> Qwen3TTSModel:
        """Get or load a model instance for a model type on a specific device."""
        cache_key = (model_type, device)

        # Fast path: return cached model.
        with self._lock:
            cached = self._models.get(cache_key)
            if cached is not None:
                return cached

            # If another thread is already loading this exact model, wait.
            in_flight = self._loading.get(cache_key)
            if in_flight is not None:
                wait_event = in_flight
            else:
                wait_event = threading.Event()
                self._loading[cache_key] = wait_event
                in_flight = None

        if in_flight is not None:
            wait_event.wait()
            with self._lock:
                cached = self._models.get(cache_key)
                if cached is None:
                    raise RuntimeError(
                        f"Model load failed in another thread: model_type={model_type} device={device}"
                    )
                return cached

        dtype = self._torch_dtype()
        attn_impl = self._attention_implementation(dtype)
        model_path = str(self._model_dir(model_type))

        logger.info(
            "Loading Qwen3-TTS model",
            extra={
                "model_type": model_type,
                "device": device,
                "dtype": str(dtype),
                "attn_implementation": attn_impl,
                "path": model_path,
            },
        )

        # Avoid `device_map=` entirely here.
        # In some accelerate/transformers combos this can create meta tensors
        # and crash with "Cannot copy out of meta tensor" during dispatch.
        # We instead load on CPU (real tensors) and then move to the leased GPU.
        try:
            model = None
            load_kwargs = {
                "attn_implementation": attn_impl,
                "low_cpu_mem_usage": False,
                "local_files_only": True,
            }

            with _FROM_PRETRAINED_LOCK:
                # Transformers' FlashAttention2 dispatch checks consult `model.config.dtype`.
                # If the on-disk config lacks a dtype (common), Transformers warns once that
                # FA2 is being enabled "without specifying a torch dtype" even if we pass
                # dtype in `from_pretrained`. Pre-loading config and setting `config.dtype`
                # avoids that warning and keeps the dtype explicit.
                try:
                    from transformers import AutoConfig, AutoProcessor
                    from qwen_tts.core.models import (
                        Qwen3TTSConfig,
                        Qwen3TTSForConditionalGeneration,
                        Qwen3TTSProcessor,
                    )

                    AutoConfig.register("qwen3_tts", Qwen3TTSConfig)
                    AutoProcessor.register(Qwen3TTSConfig, Qwen3TTSProcessor)

                    cfg = AutoConfig.from_pretrained(model_path, local_files_only=True)
                    try:
                        cfg.dtype = dtype
                    except Exception:
                        pass
                    try:
                        cfg._attn_implementation = attn_impl
                    except Exception:
                        pass

                    processor = AutoProcessor.from_pretrained(
                        model_path,
                        fix_mistral_regex=True,
                        local_files_only=True,
                    )
                except Exception:
                    cfg = None
                    processor = None

                logger.info(
                    "Prepared config for model load",
                    extra={
                        "model_type": model_type,
                        "device": device,
                        "config_injected": cfg is not None,
                        "config_dtype": str(getattr(cfg, "dtype", None)) if cfg is not None else None,
                        "attn_implementation": attn_impl,
                    },
                )

                if cfg is None or processor is None:
                    raise RuntimeError(
                        "Failed to prepare Qwen3-TTS config/processor for model load; cannot continue"
                    )

                # Load the concrete model class directly so we can control config dtype/attn settings
                # before FlashAttention2 checks run. Then wrap it in the qwen_tts convenience wrapper.
                from qwen_tts.core.models import Qwen3TTSForConditionalGeneration

                model_core = Qwen3TTSForConditionalGeneration.from_pretrained(
                    model_path,
                    **({"config": cfg} if cfg is not None else {}),
                    dtype=dtype,
                    **load_kwargs,
                )

                model = Qwen3TTSModel(
                    model=model_core,
                    processor=processor,
                    generate_defaults=getattr(model_core, "generate_config", None),
                )

                # Ensure config reflects the dtype we intend to run with.
                # NOTE: In newer Transformers versions, `config.torch_dtype` is a deprecated
                # property that emits a warning; use `config.dtype` instead.
                for maybe_model in (model, getattr(model, "model", None)):
                    if maybe_model is None:
                        continue
                    cfg = getattr(maybe_model, "config", None)
                    if cfg is None:
                        continue
                    try:
                        cfg.dtype = dtype
                    except Exception:
                        pass

                # Move to target device (and update wrapper device) if requested.
                if device != "cpu" and torch.cuda.is_available():
                    # Use device context for placement instead of global set_device
                    gpu_id = int(str(device).split(":")[-1])

                    if hasattr(model, "model") and hasattr(model.model, "to"):
                        # The `with torch.cuda.device()` context is removed as it can cause
                        # conflicts in a 'spawn' multiprocessing context. The worker process's
                        # CUDA context is already configured. We just need to move the model.
                        try:
                            model.model.to(device=device, dtype=dtype)
                        except Exception:
                            # This might fail if already on the device, which is acceptable.
                            pass
                        try:
                            model.model.eval()
                        except Exception:
                            pass

                    # Some upstream/Transformers code paths can leave dispatch metadata on the
                    # model that results in multi-GPU execution (e.g., a stale hf_device_map).
                    # We do not use device_map in this project; ensure the model is treated as
                    # a single-device module after `.to(device)`.
                    try:
                        core = getattr(model, "model", None)
                        if core is not None:
                            if hasattr(core, "hf_device_map"):
                                hf_map = getattr(core, "hf_device_map", None)
                                if hf_map:
                                    logger.warning(
                                        "Clearing unexpected hf_device_map for single-device execution",
                                        extra={"model_type": model_type, "device": device, "hf_device_map": str(hf_map)[:500]},
                                    )
                                try:
                                    delattr(core, "hf_device_map")
                                except Exception:
                                    try:
                                        setattr(core, "hf_device_map", None)
                                    except Exception:
                                        pass
                    except Exception:
                        logger.exception("Failed to clear hf_device_map")

                    # Verify all parameters/buffers are on the intended device.
                    # If we detect stragglers on a different CUDA device, re-run `.to(device)` and log.
                    try:
                        core = getattr(model, "model", None)
                        if core is not None:
                            expected = str(torch.device(device))
                            seen: set[str] = set()
                            for p in core.parameters(recurse=True):
                                seen.add(str(p.device))
                                if len(seen) > 2:
                                    break
                            for b in core.buffers(recurse=True):
                                seen.add(str(b.device))
                                if len(seen) > 2:
                                    break

                            if expected not in seen or len(seen) > 1:
                                logger.warning(
                                    "Model parameters not uniformly on requested device; re-applying .to(device)",
                                    extra={
                                        "model_type": model_type,
                                        "requested_device": device,
                                        "expected": expected,
                                        "observed_devices": sorted(seen),
                                    },
                                )
                                core.to(device=device, dtype=dtype)
                                try:
                                    core.eval()
                                except Exception:
                                    pass

                                # Re-scan (limited) after corrective move.
                                seen2: set[str] = set()
                                for p in core.parameters(recurse=True):
                                    seen2.add(str(p.device))
                                    if len(seen2) > 2:
                                        break
                                for b in core.buffers(recurse=True):
                                    seen2.add(str(b.device))
                                    if len(seen2) > 2:
                                        break
                                logger.info(
                                    "Post-move device verification",
                                    extra={
                                        "model_type": model_type,
                                        "requested_device": device,
                                        "observed_devices": sorted(seen2),
                                    },
                                )
                    except Exception:
                        logger.exception("Failed during post-load device verification")

                    _ensure_speech_tokenizer_on_device(model, device=device, dtype=dtype)
                    try:
                        model.device = torch.device(device)
                    except Exception:
                        pass
                else:
                    try:
                        model.device = torch.device("cpu")
                    except Exception:
                        pass

        except Exception:
            # Unblock any waiters and re-raise.
            with self._lock:
                load_event = self._loading.pop(cache_key, None)
            if load_event is not None:
                load_event.set()
            raise

        # Log model device (determine from model parameters to be accurate)
        try:
            if hasattr(model, 'model') and hasattr(model.model, 'parameters'):
                model_param_device = next(model.model.parameters()).device
            elif hasattr(model, 'parameters'):
                model_param_device = next(model.parameters()).device
            else:
                model_param_device = torch.device('cpu')
            logger.info(
                "Set model device",
                extra={"model_type": model_type, "model_param_device": str(model_param_device)}
            )
        except Exception:
            model_param_device = torch.device('cpu')

        # Log tokenizer device separately (it is not a torch module).
        try:
            speech_tokenizer = None
            if hasattr(model, "model") and hasattr(model.model, "speech_tokenizer"):
                speech_tokenizer = model.model.speech_tokenizer
            elif hasattr(model, "speech_tokenizer"):
                speech_tokenizer = model.speech_tokenizer
            if speech_tokenizer is not None:
                tok_device = getattr(speech_tokenizer, "device", None)
                inner = getattr(speech_tokenizer, "model", None)
                if inner is not None and hasattr(inner, "parameters"):
                    inner_device = next(inner.parameters()).device
                else:
                    inner_device = None
                logger.info(
                    "Set speech tokenizer device",
                    extra={
                        "model_type": model_type,
                        "speech_tokenizer_device": str(tok_device) if tok_device is not None else None,
                        "speech_tokenizer_param_device": str(inner_device) if inner_device is not None else None,
                    },
                )
        except Exception:
            pass

        # DO NOT call model.to(device) - it conflicts with device_map and causes hanging!
        # The model is already placed on the device by device_map parameter.
        
        logger.info(
            "Model loaded successfully",
            extra={
                "model_type": model_type,
                "device": device,
            }
        )

        with self._lock:
            self._models[cache_key] = model
            load_event = self._loading.pop(cache_key, None)

        if load_event is not None:
            load_event.set()

        return model

    def unload_model(self, model_type: Optional[str] = None) -> list[Tuple[str, str]]:
        """Unload cached model(s) from GPU memory.

        If `model_type` is None, unload all cached models. Returns a list of
        (model_type, device) keys that were unloaded.
        """
        unloaded = []
        devices_affected = set()
        
        logger.info(
            "Unload requested",
            extra={"model_type": model_type, "cached_models": list(self._models.keys())}
        )
        
        with self._lock:
            keys = list(self._models.keys())
            for key in keys:
                mt, dev = key
                if model_type is None or mt == model_type:
                    try:
                        model = self._models.pop(key)
                        logger.info(f"Unloading model from cache: {key}")
                        
                        # Track which devices were affected
                        if dev != "cpu" and dev.startswith("cuda:"):
                            devices_affected.add(dev)
                        
                        # Delete the model reference immediately
                        try:
                            del model
                        except Exception as e:
                            logger.warning(f"Failed to delete model {key}: {e}")
                        
                        unloaded.append((mt, dev))
                    except KeyError:
                        continue

        # Aggressively clear GPU memory
        if torch.cuda.is_available() and devices_affected:
            # Synchronize all devices first
            for dev_str in devices_affected:
                try:
                    device_id = int(dev_str.split(":")[-1])
                    torch.cuda.synchronize(device_id)
                except Exception as e:
                    logger.warning(f"CUDA synchronize failed for {dev_str}: {e}")
            
            # Run garbage collection multiple times
            try:
                for i in range(5):
                    gc.collect()
            except Exception as e:
                logger.warning(f"GC failed: {e}")
            
            # Empty cache for each affected device multiple times
            for dev_str in devices_affected:
                try:
                    device_id = int(dev_str.split(":")[-1])
                    # Set device context and empty cache
                    with torch.cuda.device(device_id):
                        for i in range(3):
                            torch.cuda.empty_cache()
                            if i < 2:
                                gc.collect()
                    logger.info(f"CUDA cache emptied for {dev_str}")
                except Exception as e:
                    logger.warning(f"CUDA empty_cache failed for {dev_str}: {e}")
            
            # One final global empty cache
            try:
                torch.cuda.empty_cache()
            except Exception as e:
                logger.warning(f"Final CUDA empty_cache failed: {e}")
        else:
            try:
                gc.collect()
            except Exception:
                pass

        logger.info(
            "Unload complete",
            extra={"unloaded": unloaded, "remaining_models": list(self._models.keys())}
        )
        
        return unloaded

    def unload_idle_models(self, idle_seconds: Optional[int] = None) -> list[Tuple[str, str]]:
        """Unload models if no generation activity has occurred for `idle_seconds`.

        Returns list of unloaded models keys.
        """
        if idle_seconds is None:
            idle_seconds = int(getattr(self._settings, "tts_unload_idle_seconds", 300))

        now = time.monotonic()
        if now - self._last_activity < float(idle_seconds):
            return []

        return self.unload_model(None)

    @staticmethod
    def _ensure_language(language: Optional[str]) -> str:
        if not language:
            return "Auto"

        # Map common short codes to the model's expected language labels.
        lang = language.strip().lower()
        mapping = {
            "en": "english",
            "eng": "english",
            "english": "english",
            "zh": "chinese",
            "cn": "chinese",
            "chinese": "chinese",
            "ja": "japanese",
            "jp": "japanese",
            "japanese": "japanese",
            "ko": "korean",
            "kr": "korean",
            "korean": "korean",
        }

        return mapping.get(lang, language)

    def generate_voice_clone(
        self,
        *,
        text: str,
        language: Optional[str],
        ref_audio: Optional[str],
        ref_text: Optional[str],
        x_vector_only_mode: bool,
    ) -> Tuple[object, int]:
        with self._lease_device() as leased:
            if isinstance(leased, str):
                device = leased
            else:
                device = f"cuda:{leased.gpu_id}"

            model = self.get_model("base", device)

            # Re-enforce speech tokenizer device at generation time.
            # The speech tokenizer is not a torch module, and some upstream codepaths
            # can leave it on a different CUDA device than the main model.
            try:
                dtype = self._torch_dtype()
                _ensure_speech_tokenizer_on_device(model, device=device, dtype=dtype)
            except Exception:
                logger.exception("Failed to ensure speech tokenizer device before generation")

            # Preprocess reference audio.
            preproc_ref = None
            if ref_audio is not None:
                # load_audio accepts strings (paths/URLs/base64/data URLs) or bytes
                audio_obj, _sr = load_audio(ref_audio, target_sr=self._settings.tts_sample_rate)

                import numpy as _np  # local import to avoid top-level dep

                if isinstance(audio_obj, torch.Tensor):
                    audio_np = audio_obj.cpu().numpy().astype(_np.float32)
                    if not audio_np.flags["C_CONTIGUOUS"]:
                        audio_np = _np.ascontiguousarray(audio_np)
                    preproc_ref = (audio_np, int(_sr or self._settings.tts_sample_rate))
                elif isinstance(audio_obj, _np.ndarray):
                    audio_np = _np.asarray(audio_obj).astype(_np.float32)
                    if not audio_np.flags["C_CONTIGUOUS"]:
                        audio_np = _np.ascontiguousarray(audio_np)
                    preproc_ref = (audio_np, int(_sr or self._settings.tts_sample_rate))
                else:
                    preproc_ref = audio_obj
                    logger.warning("Unexpected audio type", extra={"type": str(type(audio_obj))})

            # Determine actual model device to ensure generation runs on same GPU
            try:
                if hasattr(model, "model") and hasattr(model.model, "parameters"):
                    model_param_device = next(model.model.parameters()).device
                elif hasattr(model, "parameters"):
                    model_param_device = next(model.parameters()).device
                else:
                    model_param_device = torch.device("cpu")
            except Exception:
                model_param_device = torch.device("cpu")

            # If the model is on CUDA, ensure tokenizer follows the actual model device.
            try:
                if str(model_param_device).startswith("cuda"):
                    dtype = self._torch_dtype()
                    _ensure_speech_tokenizer_on_device(
                        model,
                        device=str(model_param_device),
                        dtype=dtype,
                    )
            except Exception:
                logger.exception("Failed to align speech tokenizer with model_param_device")

            logger.info(
                "Starting voice clone generation",
                extra={
                    "language": language,
                    "cuda_available": torch.cuda.is_available(),
                    "model_param_device": str(model_param_device),
                    "lease_device": device,
                },
            )

            # Diagnostic: log speech tokenizer device if available
            try:
                speech_tokenizer = None
                if hasattr(model, "model") and hasattr(model.model, "speech_tokenizer"):
                    speech_tokenizer = model.model.speech_tokenizer
                elif hasattr(model, "speech_tokenizer"):
                    speech_tokenizer = model.speech_tokenizer

                tok_inner_dev = None
                tok_device_attr = None
                if speech_tokenizer is not None:
                    inner = getattr(speech_tokenizer, "model", None)
                    if inner is not None and hasattr(inner, "parameters"):
                        try:
                            tok_inner_dev = str(next(inner.parameters()).device)
                        except Exception:
                            tok_inner_dev = None
                    tok_device_attr = getattr(speech_tokenizer, "device", None)

                logger.info(
                    "Speech tokenizer device check",
                    extra={
                        "speech_tokenizer_param_device": tok_inner_dev,
                        "speech_tokenizer_device_attr": str(tok_device_attr) if tok_device_attr is not None else None,
                    },
                )
            except Exception:
                logger.exception("Failed to inspect speech tokenizer device")

            if str(model_param_device).startswith("cuda"):
                gpu_id = int(str(model_param_device).split(":")[-1])
                device_context = torch.cuda.device(gpu_id)
                active_gpu_id: int | None = gpu_id
            else:
                from contextlib import nullcontext

                device_context = nullcontext()
                active_gpu_id = None

            self._last_activity = time.monotonic()

            # CRITICAL: Ensure speech tokenizer is on the correct device BEFORE any GPU operations.
            # The create_voice_clone_prompt call below performs GPU-intensive work (audio encoding
            # + speaker embedding extraction) and the tokenizer must be on the target device.
            if active_gpu_id is not None:
                try:
                    dtype = self._torch_dtype()
                    _ensure_speech_tokenizer_on_device(model, device=device, dtype=dtype)
                    logger.info(
                        "Speech tokenizer device enforced before prompt creation",
                        extra={"device": device, "gpu_id": active_gpu_id}
                    )
                except Exception:
                    logger.exception("Failed to enforce speech tokenizer device before prompt creation")

            with torch.no_grad(), device_context:
                lang_for_model = self._ensure_language(language)

                # IMPORTANT: Upstream requires `(audio_np, sr)` for numpy waveform inputs.
                ref_for_model = None
                try:
                    if preproc_ref is not None:
                        import numpy as _np

                        if isinstance(preproc_ref, (list, tuple)) and len(preproc_ref) >= 2:
                            audio_np = preproc_ref[0]
                            sr_val = int(preproc_ref[1])
                        else:
                            audio_np = preproc_ref
                            sr_val = int(self._settings.tts_sample_rate)

                        if not isinstance(audio_np, _np.ndarray):
                            audio_np = _np.asarray(audio_np, dtype=_np.float32)
                        else:
                            audio_np = audio_np.astype(_np.float32)
                        if not audio_np.flags["C_CONTIGUOUS"]:
                            audio_np = _np.ascontiguousarray(audio_np)

                        ref_for_model = (audio_np, sr_val)
                    else:
                        ref_for_model = ref_audio
                except Exception:
                    if isinstance(preproc_ref, (list, tuple)) and len(preproc_ref) >= 2:
                        ref_for_model = (preproc_ref[0], int(preproc_ref[1]))
                    else:
                        ref_for_model = ref_audio

                # Get max_new_tokens from the model's generation_config.json if available.
                # This is the only limit we enforce - the model's inherent capability.
                max_new_tokens_total = None
                try:
                    import json

                    gen_cfg_path = self._model_dir("base") / "generation_config.json"
                    if gen_cfg_path.exists():
                        with open(gen_cfg_path, "r", encoding="utf-8") as fh:
                            gen_cfg = json.load(fh)
                        model_limit = gen_cfg.get("max_new_tokens")
                        if isinstance(model_limit, int) and model_limit > 0:
                            max_new_tokens_total = int(model_limit)
                            logger.info(
                                "Using model max_new_tokens from generation_config",
                                extra={"max_new_tokens": max_new_tokens_total}
                            )
                except Exception as e:
                    logger.warning(f"Could not read model generation_config: {e}")
                
                # If no model limit found, use a very high default (effectively unlimited)
                # Increased fallback budget to reduce unnecessary retries for long inputs.
                if max_new_tokens_total is None:
                    max_new_tokens_total = 16384
                    logger.info(
                        "No model limit found, using high default",
                        extra={"max_new_tokens": max_new_tokens_total}
                    )

                # Log the actual input text to verify it's not being truncated before chunking
                logger.info(
                    "Input text for chunking",
                    extra={
                        "text_length": len(text),
                        "text_preview_start": text[:200],
                        "text_preview_end": text[-200:] if len(text) > 200 else text
                    }
                )

                # Chunk text by sentence boundaries using configured limit
                chunks = chunk_text(text, max_chars=self._settings.tts_chunk_max_chars)
                
                logger.info(
                    "Text chunking complete",
                    extra={
                        "input_text_len": len(text),
                        "chunk_max_chars": self._settings.tts_chunk_max_chars,
                        "num_chunks": len(chunks),
                        "chunk_lengths": [len(c) for c in chunks],
                        "first_chunk_preview": chunks[0][:100] if chunks else None,
                        "last_chunk_preview": chunks[-1][:100] if chunks else None
                    }
                )

                if not chunks:
                    raise ValueError("Empty text after chunking")

                # If single chunk, run normally
                if len(chunks) == 1:
                    per_chunk_tokens = max_new_tokens_total
                    wavs, sr = model.generate_voice_clone(
                        text=text,
                        language=lang_for_model,
                        ref_audio=ref_for_model,
                        ref_text=ref_text,
                        x_vector_only_mode=x_vector_only_mode,
                        do_sample=True,
                        temperature=0.3,
                        max_new_tokens=per_chunk_tokens,
                    )

                    if not isinstance(wavs, (list, tuple)) or len(wavs) == 0:
                        raise RuntimeError("Model returned no audio")

                    return wavs[0], int(sr)

                # Multi-chunk: Calculate a reasonable max_new_tokens budget per chunk based on text length.
                # Qwen3-TTS generates ~12 tokens/second of audio at 24kHz, and typical speech is ~150 words/min = 2.5 words/sec.
                # A 500-char chunk (~100 words) should produce ~40 seconds of audio = ~480 tokens.
                # Add 50% safety margin and cap at model's max.
                logger.info(
                    "Starting multi-chunk generation",
                    extra={
                        "total_chunks": len(chunks),
                        "max_new_tokens_per_chunk": max_new_tokens_total,
                        "chunk_texts": [ch[:50] + "..." if len(ch) > 50 else ch for ch in chunks]
                    }
                )
                
                import numpy as _np
                chunk_outputs = []
                out_sr = None

                # DON'T cache voice clone prompt in multi-threaded environment!
                # The prompt contains CUDA tensors pinned to a specific GPU.
                # In a multi-threaded worker, threads share the same process memory
                # but each thread uses a different GPU. Reusing a cached prompt
                # from GPU 0 on threads using GPU 1/2/3 causes cross-GPU transfers
                # and massive slowdowns (6 minutes per chunk instead of 90 seconds).
                # 
                # Let each chunk build its own prompt on its assigned GPU.

                # Telemetry: record per-chunk start/end timestamps (epoch seconds)
                chunk_times: list[dict] = []

                for chunk_idx, ch in enumerate(chunks):
                    chunk_start = time.time()
                    
                    # Verify model/tokenizer device state before chunk generation
                    device_check = {}
                    try:
                        if hasattr(model, "model") and hasattr(model.model, "parameters"):
                            model_dev = str(next(model.model.parameters()).device)
                            device_check["model_device"] = model_dev
                        if hasattr(model, "model") and hasattr(model.model, "speech_tokenizer"):
                            tok = model.model.speech_tokenizer
                            tok_inner = getattr(tok, "model", None)
                            if tok_inner and hasattr(tok_inner, "parameters"):
                                tok_dev = str(next(tok_inner.parameters()).device)
                                device_check["tokenizer_device"] = tok_dev
                        if torch.cuda.is_available():
                            device_check["cuda_current"] = int(torch.cuda.current_device())
                    except Exception:
                        pass
                    
                    logger.info(
                        "Starting chunk generation",
                        extra={
                            "chunk_index": chunk_idx,
                            "chunk_text_len": len(ch),
                            "chunk_text_preview": ch[:60],
                            "chunk_start_time": chunk_start,
                            "lease_device": device,
                            **device_check
                        }
                    )

                    gen_kwargs = dict(
                        text=ch,
                        language=lang_for_model,
                        do_sample=True,  # Keep sampling for natural quality (official benchmarks use this)
                        temperature=0.9,  # Official benchmark default
                        top_p=1.0,  # Official benchmark default
                        top_k=50,  # Official benchmark default
                        repetition_penalty=1.05,  # Official benchmark default
                        max_new_tokens=max_new_tokens_total,
                    )
                    
                    # Estimate tokens needed for this chunk based on text length.
                    # Use 12 tokens/sec audio, ~150 words/min speech = 2.5 words/sec,
                    # and ~5 chars/word average. Add 2x safety margin.
                    estimated_audio_sec = (len(ch) / 5.0) / 2.5  # chars -> words -> seconds
                    estimated_tokens = int(estimated_audio_sec * 12 * 2.0)  # 2x safety margin
                    chunk_max_tokens = min(estimated_tokens, max_new_tokens_total)
                    chunk_max_tokens = max(chunk_max_tokens, 96)  # Minimum floor
                    
                    gen_kwargs["max_new_tokens"] = chunk_max_tokens
                    
                    logger.info(
                        "Chunk generation params",
                        extra={
                            "chunk_index": chunk_idx,
                            "text_len": len(ch),
                            "estimated_audio_sec": round(estimated_audio_sec, 1),
                            "max_new_tokens": chunk_max_tokens,
                        }
                    )
                    
                    # Always pass ref_audio/ref_text so each chunk builds its own prompt
                    # on the correct GPU (no cross-GPU tensor sharing)
                    gen_kwargs["ref_audio"] = ref_for_model
                    gen_kwargs["ref_text"] = ref_text
                    gen_kwargs["x_vector_only_mode"] = x_vector_only_mode

                    # CRITICAL: Verify tokenizer is still on correct device RIGHT before generation
                    try:
                        if hasattr(model, "model") and hasattr(model.model, "speech_tokenizer"):
                            tok = model.model.speech_tokenizer
                            tok_dev_attr = getattr(tok, "device", None)
                            tok_inner = getattr(tok, "model", None)
                            tok_inner_dev = None
                            if tok_inner and hasattr(tok_inner, "parameters"):
                                tok_inner_dev = str(next(tok_inner.parameters()).device)
                            logger.info(
                                "Tokenizer device immediately before generate",
                                extra={
                                    "chunk_index": chunk_idx,
                                    "expected_device": device,
                                    "tokenizer_attr_device": str(tok_dev_attr) if tok_dev_attr else None,
                                    "tokenizer_param_device": tok_inner_dev,
                                    "model_param_device": str(next(model.model.parameters()).device) if hasattr(model.model, "parameters") else None
                                }
                            )
                    except Exception as e:
                        logger.warning(f"Failed to check tokenizer device before generate: {e}")

                    wavs, sr = model.generate_voice_clone(**gen_kwargs)
                    chunk_end = time.time()
                    chunk_duration = chunk_end - chunk_start
                    chunk_times.append({"chunk_index": chunk_idx, "start": chunk_start, "end": chunk_end, "duration": round(chunk_duration, 3)})
                    
                    logger.info(
                        "Chunk generation completed",
                        extra={
                            "chunk_index": chunk_idx,
                            "duration_seconds": round(chunk_duration, 2),
                            "device": device
                        }
                    )

                    # Log what the model actually returned
                    try:
                        wavs_type = type(wavs).__name__
                        wavs_len = len(wavs) if isinstance(wavs, (list, tuple)) else "N/A"
                        logger.info(
                            "Model returned for chunk",
                            extra={
                                "chunk_index": chunk_idx,
                                "wavs_type": wavs_type,
                                "wavs_len": wavs_len,
                                "sr": sr
                            }
                        )
                    except Exception:
                        logger.exception("Failed to log model return type")

                    if not isinstance(wavs, (list, tuple)) or len(wavs) == 0:
                        raise RuntimeError(f"Model returned no audio for chunk {chunk_idx}")

                    # Extract the first element (primary audio segment for this chunk)
                    chunk_wav = wavs[0]
                    
                    # Log the raw type before conversion
                    try:
                        raw_type = type(chunk_wav).__name__
                        if isinstance(chunk_wav, (list, tuple)):
                            raw_details = f"list/tuple of {len(chunk_wav)} items"
                        elif isinstance(chunk_wav, torch.Tensor):
                            raw_details = f"tensor shape {chunk_wav.shape}"
                        elif isinstance(chunk_wav, _np.ndarray):
                            raw_details = f"ndarray shape {chunk_wav.shape}"
                        else:
                            raw_details = str(type(chunk_wav))
                        logger.info(
                            "Raw chunk wav type",
                            extra={"chunk_index": chunk_idx, "type": raw_type, "details": raw_details}
                        )
                    except Exception:
                        logger.exception("Failed to log raw chunk type")

                    # Normalize to 1-D numpy float32 array
                    try:
                        if isinstance(chunk_wav, torch.Tensor):
                            if chunk_wav.is_cuda:
                                chunk_wav = chunk_wav.cpu()
                            chunk_wav = chunk_wav.numpy()
                        elif isinstance(chunk_wav, (list, tuple)):
                            # If wavs[0] is itself a list/tuple, flatten it
                            parts = []
                            for item in chunk_wav:
                                if isinstance(item, torch.Tensor):
                                    parts.append(item.cpu().numpy() if item.is_cuda else item.numpy())
                                elif isinstance(item, _np.ndarray):
                                    parts.append(item)
                                else:
                                    parts.append(_np.asarray(item))
                            chunk_wav = _np.concatenate([p.reshape(-1) for p in parts])
                        elif not isinstance(chunk_wav, _np.ndarray):
                            chunk_wav = _np.asarray(chunk_wav)

                        # Ensure 1-D
                        if chunk_wav.ndim > 1:
                            chunk_wav = chunk_wav.reshape(-1)
                        
                        # Ensure float32
                        if chunk_wav.dtype != _np.float32:
                            chunk_wav = chunk_wav.astype(_np.float32)
                        
                        # Ensure contiguous
                        if not chunk_wav.flags["C_CONTIGUOUS"]:
                            chunk_wav = _np.ascontiguousarray(chunk_wav)
                            
                    except Exception:
                        logger.exception(f"Failed to normalize chunk {chunk_idx} audio")
                        raise RuntimeError(f"Chunk {chunk_idx} normalization failed")

                    # Validate the normalized chunk
                    chunk_samples = chunk_wav.size
                    if chunk_samples == 0:
                        logger.error(
                            "Chunk produced zero samples",
                            extra={"chunk_index": chunk_idx, "chunk_text": ch[:100]}
                        )
                        raise RuntimeError(f"Chunk {chunk_idx} produced no audio")
                    
                    logger.info(
                        "Chunk normalized and validated",
                        extra={
                            "chunk_index": chunk_idx,
                            "samples": chunk_samples,
                            "duration_sec": round(chunk_samples / sr, 3),
                            "dtype": str(chunk_wav.dtype),
                            "shape": chunk_wav.shape
                        }
                    )

                    chunk_outputs.append(chunk_wav)
                    out_sr = int(sr)

                # Final validation before concatenation
                logger.info(
                    "Concatenating chunks",
                    extra={
                        "total_chunks_collected": len(chunk_outputs),
                        "expected_chunks": len(chunks),
                        "chunk_samples": [c.size for c in chunk_outputs],
                        "chunk_times": chunk_times
                    }
                )
                
                if len(chunk_outputs) != len(chunks):
                    raise RuntimeError(
                        f"Chunk count mismatch: collected {len(chunk_outputs)} but expected {len(chunks)}"
                    )

                # Concatenate chunks with small silence padding to prevent audio artifacts
                # at chunk boundaries (prevents clicks/pops that confuse transcription)
                try:
                    # Add 100ms silence between chunks (2400 samples at 24kHz)
                    silence_samples = int(0.1 * out_sr)
                    silence = _np.zeros(silence_samples, dtype=_np.float32)
                    
                    parts = []
                    for i, chunk_wav in enumerate(chunk_outputs):
                        parts.append(chunk_wav)
                        # Add silence between chunks (but not after the last one)
                        if i < len(chunk_outputs) - 1:
                            parts.append(silence)
                    
                    full_wav = _np.concatenate(parts)
                    if not full_wav.flags["C_CONTIGUOUS"]:
                        full_wav = _np.ascontiguousarray(full_wav)
                    
                    total_samples = full_wav.size
                    expected_samples = sum(c.size for c in chunk_outputs)
                    
                    logger.info(
                        "Chunks concatenated successfully",
                        extra={
                            "total_samples": total_samples,
                            "expected_samples": expected_samples,
                            "duration_sec": round(total_samples / out_sr, 3),
                            "sample_rate": out_sr
                        }
                    )
                    
                    if total_samples != expected_samples:
                        logger.warning(
                            "Sample count mismatch after concatenation",
                            extra={"got": total_samples, "expected": expected_samples}
                        )
                    
                except Exception:
                    logger.exception("Failed to concatenate chunks")
                    raise RuntimeError("Chunk concatenation failed")

                return full_wav, int(out_sr)

    def generate_custom_voice(
        self,
        *,
        text: str,
        language: Optional[str],
        speaker: str,
        instruct: Optional[str],
    ) -> Tuple[object, int]:
        with self._lease_device() as leased:
            if isinstance(leased, str):
                device = leased
            else:
                device = f"cuda:{leased.gpu_id}"

            model = self.get_model("custom-voice", device)

            try:
                dtype = self._torch_dtype()
                _ensure_speech_tokenizer_on_device(model, device=device, dtype=dtype)
            except Exception:
                logger.exception("Failed to ensure speech tokenizer device before generation")

            try:
                if hasattr(model, "model") and hasattr(model.model, "parameters"):
                    model_param_device = next(model.model.parameters()).device
                elif hasattr(model, "parameters"):
                    model_param_device = next(model.parameters()).device
                else:
                    model_param_device = torch.device("cpu")
            except Exception:
                model_param_device = torch.device("cpu")

            if str(model_param_device).startswith("cuda"):
                gpu_id = int(str(model_param_device).split(":")[-1])
                device_context = torch.cuda.device(gpu_id)
            else:
                from contextlib import nullcontext

                device_context = nullcontext()

            self._last_activity = time.monotonic()

            with torch.no_grad(), device_context:
                wavs, sr = model.generate_custom_voice(
                    text=text,
                    language=self._ensure_language(language),
                    speaker=speaker,
                    instruct=instruct or "",
                )
                if not isinstance(wavs, (list, tuple)) or len(wavs) == 0:
                    raise RuntimeError("Model returned no audio")
                return wavs[0], int(sr)

    def generate_voice_design(
        self,
        *,
        text: str,
        language: Optional[str],
        instruct: str,
    ) -> Tuple[object, int]:
        with self._lease_device() as leased:
            if isinstance(leased, str):
                device = leased
            else:
                device = f"cuda:{leased.gpu_id}"

            model = self.get_model("voice-design", device)

            try:
                dtype = self._torch_dtype()
                _ensure_speech_tokenizer_on_device(model, device=device, dtype=dtype)
            except Exception:
                logger.exception("Failed to ensure speech tokenizer device before generation")

            try:
                if hasattr(model, "model") and hasattr(model.model, "parameters"):
                    model_param_device = next(model.model.parameters()).device
                elif hasattr(model, "parameters"):
                    model_param_device = next(model.parameters()).device
                else:
                    model_param_device = torch.device("cpu")
            except Exception:
                model_param_device = torch.device("cpu")

            if str(model_param_device).startswith("cuda"):
                gpu_id = int(str(model_param_device).split(":")[-1])
                device_context = torch.cuda.device(gpu_id)
            else:
                from contextlib import nullcontext

                device_context = nullcontext()

            logger.info(
                "Starting voice design generation",
                extra={
                    "language": language,
                    "cuda_available": torch.cuda.is_available(),
                    "model_param_device": str(model_param_device),
                    "lease_device": device,
                    "text_len": len(text or ""),
                    "instruct_len": len(instruct or ""),
                },
            )

            self._last_activity = time.monotonic()

            with torch.no_grad(), device_context:
                wavs, sr = model.generate_voice_design(
                    text=text,
                    language=self._ensure_language(language),
                    instruct=instruct,
                )
                if not isinstance(wavs, (list, tuple)) or len(wavs) == 0:
                    raise RuntimeError("Model returned no audio")
                return wavs[0], int(sr)


def wav_to_wav_bytes(wav: object, sample_rate: int) -> bytes:
    """Serialize a waveform to WAV bytes."""
    import io

    buf = io.BytesIO()

    # If we have a torch tensor, move it to CPU and convert once to numpy
    if isinstance(wav, torch.Tensor):
        try:
            if wav.is_cuda:
                wav = wav.cpu()
            wav = wav.numpy()
        except Exception:
            # Fallback: convert via .cpu().numpy() which may raise if already numpy-like
            try:
                wav = wav.cpu().numpy()
            except Exception:
                pass
    # If wav is list/tuple, flatten and concatenate pieces
    try:
        import numpy as _np

        if isinstance(wav, (list, tuple)):
            parts = []
            for p in wav:
                if isinstance(p, torch.Tensor):
                    try:
                        parts.append(p.cpu().numpy())
                    except Exception:
                        parts.append(_np.asarray(p))
                else:
                    parts.append(_np.asarray(p))
            if parts:
                wav = _np.concatenate([_p.reshape(-1) for _p in parts])
            else:
                wav = _np.array([], dtype=_np.float32)
        elif isinstance(wav, _np.ndarray):
            pass
        else:
            # Attempt to coerce other types
            wav = _np.asarray(wav)
    except Exception:
        # leave as-is and let soundfile attempt to handle it
        pass

    sf.write(buf, wav, sample_rate, format="WAV")
    return buf.getvalue()
