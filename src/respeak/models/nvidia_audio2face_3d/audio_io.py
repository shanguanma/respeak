"""Audio I/O helpers aligned with CosyVoice3Tts streaming format."""

from __future__ import annotations

import numpy as np

_DEFAULT_TARGET_SR = 16000


def resample(data: np.ndarray, original_sr: int, target_sr: int) -> np.ndarray:
    """Linear resample mono PCM (float32 in [-1, 1])."""
    if original_sr == target_sr or data.size == 0:
        return np.asarray(data, dtype=np.float32).reshape(-1)
    duration = data.shape[0] / original_sr
    target_length = int(duration * target_sr)
    return np.interp(
        np.linspace(0.0, duration, target_length, endpoint=False),
        np.linspace(0.0, duration, data.shape[0], endpoint=False),
        np.asarray(data, dtype=np.float32).reshape(-1),
    ).astype(np.float32)


def float_pcm_to_int16(speech: np.ndarray) -> np.ndarray:
    """Convert float32 mono PCM to int16 (same as CosyVoice3Tts)."""
    speech = np.asarray(speech, dtype=np.float32).reshape(-1)
    return (speech * 32768.0).astype(np.int16)


def int16_pcm_to_float(speech: np.ndarray) -> np.ndarray:
    """Convert int16 mono PCM to float32 in [-1, 1]."""
    return np.asarray(speech, dtype=np.int16).reshape(-1).astype(np.float32) / 32768.0


def encode_stream_pcm(
    speech: np.ndarray,
    *,
    source_sr: int,
    target_sr: int = _DEFAULT_TARGET_SR,
) -> np.ndarray:
    """Resample (if needed) and encode to int16 PCM for downstream audio I/O."""
    pcm_f32 = resample(speech, source_sr, target_sr)
    return float_pcm_to_int16(pcm_f32)
