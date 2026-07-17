"""Silero voice activity detection."""

from __future__ import annotations

from typing import Any

import numpy as np

from respeak.base import BaseModel

_DEFAULT_SAMPLING_RATE = 16000


class SileroVad(BaseModel):
    """Silero VAD wrapper returning speech probability for an audio chunk."""

    def __init__(
        self,
        backend: Any,
        *,
        sampling_rate: int = _DEFAULT_SAMPLING_RATE,
    ) -> None:
        self._backend = backend
        self.sampling_rate = sampling_rate

    @classmethod
    def from_pretrained(
        cls,
        *,
        onnx: bool = False,
        opset_version: int = 16,
        sampling_rate: int = _DEFAULT_SAMPLING_RATE,
        **kwargs: Any,
    ) -> SileroVad:
        """Load Silero VAD via ``silero_vad.load_silero_vad``.

        Extra kwargs are forwarded to ``load_silero_vad``.
        """
        from silero_vad import load_silero_vad

        backend = load_silero_vad(onnx=onnx, opset_version=opset_version, **kwargs)
        return cls(backend, sampling_rate=sampling_rate)

    def generate(
        self,
        input: Any,
        *,
        sampling_rate: int | None = None,
    ) -> float:
        """Return speech probability for an audio chunk."""
        audio = self._prepare_input(input)
        speech_prob = self._backend(audio, sampling_rate or self.sampling_rate)
        if hasattr(speech_prob, "item"):
            return float(speech_prob.item())
        if isinstance(speech_prob, np.ndarray):
            return float(speech_prob.item())
        return float(speech_prob)

    @staticmethod
    def _prepare_input(input: Any) -> Any:
        if not isinstance(input, np.ndarray):
            return input

        try:
            import torch
        except ImportError:
            return input.astype(np.float32, copy=False)

        return torch.from_numpy(input.astype(np.float32, copy=False))
