"""Audio processing utilities for Qwen3-TTS.

This module provides utilities for audio I/O, format conversion,
and preprocessing operations.
"""
import io
import os
import re
import base64
from typing import Union, Tuple, Optional
import requests
import numpy as np
import torch
import soundfile as sf

from utils.logging import get_logger
from utils.errors import AudioProcessingError
from typing import Iterable

logger = get_logger(__name__)

def concat_wavs(wav_list, sample_rate: Optional[int] = None):
    """Concatenate a list of waveforms (torch.Tensor or numpy.ndarray).

    Args:
        wav_list: Iterable of 1-D torch.Tensor or numpy.ndarray audio arrays.
        sample_rate: Optional sample rate to return; if None, inferred from context/user.

    Returns:
        Tuple of (numpy.ndarray, sample_rate or None).

    Notes:
        - This function preserves dtype float32 and returns a contiguous numpy array.
        - Caller is responsible for ensuring consistent sample rates across inputs.
    """
    arrays = []
    inferred_sr = sample_rate

    def _flatten_iterable(obj):
        """Yield atomic elements from potentially nested iterables (lists/tuples)."""
        if obj is None:
            return
        if isinstance(obj, (list, tuple)):
            for itm in obj:
                yield from _flatten_iterable(itm)
        else:
            yield obj

    # Flatten nested lists/tuples of segments first so the rest of the code
    # operates on atomic waveform-like objects (torch.Tensor / np.ndarray / scalar).
    flattened = []
    for item in wav_list:
        for atomic in _flatten_iterable(item):
            if atomic is None:
                continue
            flattened.append(atomic)

    for w in flattened:
        # Normalize common container types to numpy 1-D arrays.
        if isinstance(w, torch.Tensor):
            try:
                w = w.cpu().numpy()
            except Exception:
                w = np.asarray(w)
        elif isinstance(w, np.ndarray):
            # already ok
            pass
        else:
            # Last-resort coercion (covers scalars, lists of numbers)
            try:
                w = np.asarray(w)
            except Exception:
                w = np.array([], dtype=np.float32)

        # Ensure 1-D
        try:
            if getattr(w, 'ndim', 1) > 1:
                w = w.reshape(-1)
        except Exception:
            w = np.asarray(w).reshape(-1)

        # Cast to float32
        try:
            if w.dtype != np.float32:
                w = w.astype(np.float32)
        except Exception:
            w = np.asarray(w, dtype=np.float32)

        arrays.append(w)

        arrays.append(w)

    if not arrays:
        return np.array([], dtype=np.float32), inferred_sr

    try:
        out = np.concatenate(arrays)
    except Exception:
        # Fallback: progressively append
        out = arrays[0]
        for a in arrays[1:]:
            out = np.concatenate([out, a])

    if not out.flags["C_CONTIGUOUS"]:
        out = np.ascontiguousarray(out)

    return out, inferred_sr

def _mask_str(s: str, max_len: int = 120) -> str:
    """Return a truncated, safe representation of a potentially-large string.

    Keeps the start and end of the string and reports total length when truncated.
    """
    try:
        if not isinstance(s, str):
            return str(s)
    except Exception:
        return "<unrepresentable>"

    if len(s) <= max_len:
        return s
    head = s[:60]
    tail = s[-20:]
    return f"{head}...<truncated {len(s)} chars>...{tail}"


def load_audio(
    source: Union[str, bytes, np.ndarray, torch.Tensor, Tuple[np.ndarray, int]],
    target_sr: Optional[int] = None
) -> Tuple[torch.Tensor, int]:
    """Load audio from various sources and return (tensor, sample_rate).

    Supported `source` types:
      - file path (str)
      - http(s) URL (str)
      - data URL / base64 string (str starting with 'data:' or plain base64)
      - raw bytes (bytes)
      - numpy array or (numpy array, sample_rate) tuple
      - torch.Tensor

    Args:
        source: Audio source as described above.
        target_sr: If provided, resample to this sample rate.

    Returns:
        Tuple of (`torch.Tensor`, sample_rate).

    Raises:
        FileNotFoundError: If a file path is provided but does not exist.
        AudioProcessingError: For other audio-loading failures.
    """
    # Reject empty/blank inputs early
    if source is None:
        raise FileNotFoundError("No audio source provided")

    # Handle numpy/torch inputs directly
    if isinstance(source, tuple) and isinstance(source[0], (np.ndarray,)):
        arr, sr = source
        tensor = torch.from_numpy(arr).float()
        if tensor.dim() > 1:
            tensor = tensor.mean(dim=-1)
        if target_sr and sr != target_sr:
            tensor = resample_audio(tensor, sr, target_sr)
            sr = target_sr
        # Pad to frame boundaries
        tensor = pad_audio_to_frame_boundary(tensor)
        return tensor, int(sr)

    if isinstance(source, np.ndarray):
        sr = target_sr or 24000
        tensor = torch.from_numpy(source).float()
        if tensor.dim() > 1:
            tensor = tensor.mean(dim=-1)
        if target_sr:
            tensor = resample_audio(tensor, sr, target_sr)
            sr = target_sr
        # Pad to frame boundaries
        tensor = pad_audio_to_frame_boundary(tensor)
        return tensor, int(sr)

    if isinstance(source, torch.Tensor):
        sr = target_sr or 24000
        tensor = source.float()
        if tensor.dim() > 1:
            tensor = tensor.mean(dim=-1)
        if target_sr:
            tensor = resample_audio(tensor, sr, target_sr)
            sr = target_sr
        # Pad to frame boundaries
        tensor = pad_audio_to_frame_boundary(tensor)
        return tensor, int(sr)

    # At this point we expect a string path/URL/base64 or raw bytes
    # Normalize bytes input
    data_bytes: Optional[bytes] = None
    if isinstance(source, (bytes, bytearray)):
        data_bytes = bytes(source)

    if isinstance(source, str):
        src = source.strip()
        if not src:
            raise FileNotFoundError("Empty audio source string provided")

        # data URL (base64)
        if src.startswith("data:"):
            # data:<mime>;base64,<data>
            try:
                header, b64 = src.split(",", 1)
                data_bytes = base64.b64decode(b64)
            except Exception as e:
                logger.error("Invalid data URL for audio: %s", e)
                raise AudioProcessingError(f"Invalid data URL: {e}") from e

        # plain base64
        elif re.match(r"^[A-Za-z0-9+/=\n\r]+$", src) and len(src) > 100:
            # heuristic: long base64 string
            try:
                data_bytes = base64.b64decode(src)
            except Exception as e:
                logger.error("Invalid base64 audio string: %s", e)
                raise AudioProcessingError(f"Invalid base64 audio: {e}") from e

        # http/https URL
        elif src.startswith("http://") or src.startswith("https://"):
            try:
                resp = requests.get(src, timeout=15)
                resp.raise_for_status()
                data_bytes = resp.content
            except Exception as e:
                logger.error("Failed to fetch audio URL %s: %s", _mask_str(src), e)
                raise AudioProcessingError(f"Failed to fetch audio URL: {e}") from e

        else:
            # Treat as filesystem path
            try:
                exists = os.path.exists(src)
            except OSError as e:
                logger.warning("Path check failed for audio source: %s; error=%s", _mask_str(src), e)
                raise AudioProcessingError("Invalid audio source path") from e

            if not exists:
                raise FileNotFoundError(f"Audio file not found: {_mask_str(src)}")

            try:
                audio, sr = sf.read(src)
            except sf.LibsndfileError as e:
                logger.error("Failed to read audio file %s: %s", _mask_str(src), e)
                raise AudioProcessingError(f"Cannot read audio file: {e}") from e
            except Exception as e:
                logger.exception("Unexpected error loading audio %s", _mask_str(src))
                raise AudioProcessingError(f"Audio loading failed") from e

            try:
                audio_tensor = torch.from_numpy(audio).float()
                if audio_tensor.dim() > 1:
                    audio_tensor = audio_tensor.mean(dim=-1)
                if target_sr and sr != target_sr:
                    audio_tensor = resample_audio(audio_tensor, sr, target_sr)
                    sr = target_sr
                # Pad to frame boundaries to prevent encoder size mismatches
                audio_tensor = pad_audio_to_frame_boundary(audio_tensor)
                logger.debug("Loaded audio from path: %s, shape=%s, sr=%s", _mask_str(src), audio_tensor.shape, sr)
                return audio_tensor, int(sr)
            except Exception as e:
                logger.exception("Failed processing audio from %s", _mask_str(src))
                raise AudioProcessingError("Audio processing failed") from e

    # If we have bytes, read via soundfile from BytesIO
    if data_bytes is not None:
        try:
            bio = io.BytesIO(data_bytes)
            audio, sr = sf.read(bio, dtype="float32")
            audio_tensor = torch.from_numpy(audio).float()
            if audio_tensor.dim() > 1:
                audio_tensor = audio_tensor.mean(dim=-1)
            if target_sr and sr != target_sr:
                audio_tensor = resample_audio(audio_tensor, sr, target_sr)
                sr = target_sr
            # Pad to frame boundaries to prevent encoder size mismatches
            audio_tensor = pad_audio_to_frame_boundary(audio_tensor)
            logger.debug(f"Loaded audio from bytes, shape={audio_tensor.shape}, sr={sr}")
            return audio_tensor, int(sr)
        except sf.LibsndfileError as e:
            logger.error(f"Failed to read audio from bytes: {e}")
            raise AudioProcessingError(f"Cannot read audio bytes: {e}") from e
        except Exception as e:
            logger.exception("Unexpected error loading audio from bytes")
            raise AudioProcessingError("Audio loading failed") from e

    raise AudioProcessingError("Unsupported audio source type")


def save_audio(
    audio: Union[torch.Tensor, np.ndarray],
    file_path: str,
    sample_rate: int = 24000
) -> None:
    """Save audio tensor to file.
    
    Args:
        audio: Audio tensor or numpy array.
        file_path: Destination file path.
        sample_rate: Audio sample rate.
    
    Raises:
        AudioProcessingError: If saving fails.
    
    Example:
        >>> audio = torch.randn(24000)  # 1 second at 24kHz
        >>> save_audio(audio, "output.wav", sample_rate=24000)
    """
    try:
        # Ensure output directory exists
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        
        # Convert to numpy if needed (accept lists, tuples, torch tensors)
        original_audio = audio
        if isinstance(audio, torch.Tensor):
            audio = audio.cpu().numpy()
        elif not isinstance(audio, np.ndarray):
            # Covers list, tuple, and other sequence types
            # If it's a sequence of arrays or lists, concatenate them
            if isinstance(original_audio, Iterable):
                try:
                    audio, _ = concat_wavs(list(original_audio), sample_rate=sample_rate)
                except Exception:
                    audio = np.asarray(original_audio)
            else:
                audio = np.asarray(original_audio)

        # Ensure proper dtype
        if not hasattr(audio, 'dtype') or audio.dtype != np.float32:
            try:
                audio = audio.astype(np.float32)
            except Exception:
                audio = np.asarray(audio, dtype=np.float32)

        # Ensure proper shape (samples,) or (samples, channels)
        if audio.ndim == 1:
            pass  # Already correct shape
        elif audio.ndim == 2 and audio.shape[1] == 1:
            audio = audio.squeeze(1)
        elif audio.ndim == 0:
            # scalar -> convert to 1-sample array
            audio = np.atleast_1d(audio)

        sf.write(file_path, audio, sample_rate)
        logger.debug("Saved audio to %s", _mask_str(file_path))
    
    except Exception as e:
        logger.exception("Failed to save audio to %s", _mask_str(file_path))
        raise AudioProcessingError("Cannot save audio") from e


def resample_audio(
    audio: torch.Tensor,
    orig_sr: int,
    target_sr: int
) -> torch.Tensor:
    """Resample audio tensor to target sample rate.
    
    Args:
        audio: Audio tensor to resample.
        orig_sr: Original sample rate.
        target_sr: Target sample rate.
    
    Returns:
        Resampled audio tensor.
    """
    # Implementation continues below...


def pad_audio_to_frame_boundary(
    audio: torch.Tensor,
    frame_size: int = 320,
    hop_length: int = 160
) -> torch.Tensor:
    """Pad audio to align with encoder frame boundaries.
    
    This prevents tensor size mismatches in the speech tokenizer encoder
    by ensuring the audio length is compatible with the model's stride.
    
    Args:
        audio: Audio tensor to pad (shape: [samples] or [batch, samples]).
        frame_size: Size of each frame window (default 320 for 24kHz -> 12Hz).
        hop_length: Hop length between frames (default 160 for 50% overlap).
    
    Returns:
        Padded audio tensor aligned to frame boundaries.
    """
    if audio.dim() == 1:
        audio = audio.unsqueeze(0)
        squeeze_output = True
    else:
        squeeze_output = False
    
    batch_size, length = audio.shape
    
    # Calculate required padding to align to hop_length boundaries
    # The encoder processes audio in overlapping frames with stride=hop_length
    remainder = (length - frame_size) % hop_length
    if remainder != 0:
        pad_length = hop_length - remainder
        # Pad with zeros at the end
        audio = torch.nn.functional.pad(audio, (0, pad_length), mode='constant', value=0.0)
        logger.debug(f"Padded audio by {pad_length} samples for frame alignment (original: {length}, new: {audio.shape[-1]})")
    
    if squeeze_output:
        audio = audio.squeeze(0)
    
    return audio


def resample_audio(
    audio: torch.Tensor,
    orig_sr: int,
    target_sr: int
) -> torch.Tensor:
    """Resample audio tensor to target sample rate.
    
    Args:
        audio: Audio tensor to resample.
        orig_sr: Original sample rate.
        target_sr: Target sample rate.
    
    Returns:
        Resampled audio tensor.
    
    Example:
        >>> audio_16k = torch.randn(16000)
        >>> audio_24k = resample_audio(audio_16k, 16000, 24000)
        >>> audio_24k.shape
        torch.Size([24000])
    """
    if orig_sr == target_sr:
        return audio

    try:
        import torchaudio

        # Qwen3-TTS ultimately needs CPU numpy reference audio, so keep resampling
        # stable and deterministic on CPU to avoid CUDA/CPU kernel mismatches.
        orig_device = audio.device
        audio_cpu = audio.detach().to("cpu")

        resampler = torchaudio.transforms.Resample(orig_freq=orig_sr, new_freq=target_sr)
        resampled = resampler(audio_cpu)

        if str(orig_device).startswith("cuda"):
            return resampled.to(orig_device)
        return resampled

    except Exception as e:
        logger.error(f"torchaudio resampling failed: {e}")
        raise AudioProcessingError(f"Resampling failed: {e}") from e


def get_audio_duration(file_path: str) -> float:
    """Get audio file duration in seconds.
    
    Args:
        file_path: Path to audio file.
    
    Returns:
        Duration in seconds.
    
    Raises:
        AudioProcessingError: If duration cannot be determined.
    
    Example:
        >>> duration = get_audio_duration("audio.wav")
        >>> print(f"{duration:.2f} seconds")
    """
    try:
        info = sf.info(file_path)
        duration = info.duration
        logger.debug("Audio duration: %s = %.2fs", _mask_str(file_path), duration)
        return duration
    
    except Exception as e:
        logger.error("Failed to get duration for %s: %s", _mask_str(file_path), e)
        raise AudioProcessingError("Cannot determine audio duration") from e


def normalize_text(text: str) -> str:
    """Normalize text for comparison (used in accuracy scoring).
    
    Args:
        text: Input text to normalize.
    
    Returns:
        Normalized text (lowercase, no punctuation, single spaces).
    
    Example:
        >>> normalize_text("Hello,  World!")
        'hello world'
    """
    # Convert to lowercase
    text = text.lower()
    
    # Remove punctuation except apostrophes
    text = re.sub(r"[^\w\s']", " ", text)
    
    # Collapse multiple spaces
    text = re.sub(r"\s+", " ", text)
    
    # Strip leading/trailing whitespace
    text = text.strip()
    
    return text


def convert_audio_format(
    input_path: str,
    output_path: str,
    output_format: str = "wav",
    sample_rate: Optional[int] = None
) -> None:
    """Convert audio file to different format.
    
    Args:
        input_path: Input audio file path.
        output_path: Output audio file path.
        output_format: Output format ('wav', 'mp3', 'flac', etc.).
        sample_rate: Target sample rate. If None, keeps original.
    
    Raises:
        AudioProcessingError: If conversion fails.
    
    Example:
        >>> convert_audio_format("input.wav", "output.mp3", "mp3", 24000)
    """
    try:
        audio, sr = load_audio(input_path, target_sr=sample_rate)
        
        # Ensure output directory exists
        os.makedirs(os.path.dirname(output_path), exist_ok=True)

        # Save in target format
        sf.write(output_path, audio.numpy(), sr or 24000, format=output_format)

        logger.info("Converted %s -> %s (%s)", _mask_str(input_path), _mask_str(output_path), output_format)
    
    except Exception as e:
        logger.exception("Audio format conversion failed")
        raise AudioProcessingError("Conversion failed") from e


__all__ = [
    "load_audio",
    "save_audio",
    "resample_audio",
    "get_audio_duration",
    "normalize_text",
    "convert_audio_format",
]
