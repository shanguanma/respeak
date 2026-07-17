"""CosyVoice3 TTS with optional vLLM acceleration.

Vendors a minimal CosyVoice3 + Matcha inference stack under this package.
"""

from __future__ import annotations

import os
import sys
from collections.abc import Iterator
from pathlib import Path
from typing import Any
import queue
import numpy as np

from respeak.base import BaseModel

_DEFAULT_INSTRUCT_PREFIX = "You are a helpful assistant。<|endofprompt|>"
_DEFAULT_SPK_ID = "default"
_VLLM_REGISTERED = False
_PATHS_READY = False

_PKG_ROOT = Path(__file__).resolve().parent


def _apply_vllm_env() -> None:
    os.environ.setdefault("TORCHINDUCTOR_FALLBACK", "1")
    os.environ.setdefault("TORCHDYNAMO_DISABLE", "1")
    os.environ.setdefault("VLLM_USE_FLASH_ATTN", "0")
    os.environ.setdefault("VLLM_DISABLE_COMPILATION", "1")


def _ensure_local_paths() -> None:
    """Expose vendored ``cosyvoice`` / ``matcha`` as top-level imports (yaml needs them)."""
    global _PATHS_READY
    if _PATHS_READY:
        return
    root = str(_PKG_ROOT)
    if root not in sys.path:
        sys.path.insert(0, root)
    _PATHS_READY = True


def _register_vllm_cosyvoice() -> None:
    global _VLLM_REGISTERED
    if _VLLM_REGISTERED:
        return
    from vllm import ModelRegistry
    from cosyvoice.vllm.cosyvoice2 import CosyVoice2ForCausalLM

    ModelRegistry.register_model("CosyVoice2ForCausalLM", CosyVoice2ForCausalLM)
    _VLLM_REGISTERED = True


def _load_backend(
    model_dir: str,
    *,
    load_vllm: bool,
    load_trt: bool,
    fp16: bool,
    **kwargs: Any,
) -> Any:
    _apply_vllm_env()
    _ensure_local_paths()
    if load_vllm:
        _register_vllm_cosyvoice()
    from cosyvoice.cli.cosyvoice import AutoModel

    return AutoModel(
        model_dir=model_dir,
        load_vllm=load_vllm,
        load_trt=load_trt,
        fp16=fp16,
        **kwargs,
    )


def _speech_to_numpy(chunk: dict[str, Any], target_sr: int = 24000) -> np.ndarray:
    #cosyvoice3 output sample rate is 24000

    speech = chunk["tts_speech"][0]
    if hasattr(speech, "detach"):
        speech = speech.detach().cpu().numpy()
    #resample to target sample rate
    speech = resample(speech, original_sr=24000, target_sr=target_sr)
    #convert to int16, for  sending to audio interface
    speech = (speech * 32768).astype(np.int16)
    return speech

def resample(data, original_sr, target_sr):
    duration = data.shape[0] / original_sr
    target_length = int(duration * target_sr)
    resampled_data = np.interp(
        np.linspace(0.0, duration, target_length, endpoint=False),
        np.linspace(0.0, duration, data.shape[0], endpoint=False),
        data
    ).astype(np.float32)
    return resampled_data

class CosyVoice3Tts(BaseModel):
    """CosyVoice3 zero-shot TTS (vLLM-accelerated LLM decode when enabled)."""

    def __init__(
        self,
        backend: Any,
        *,
        zero_shot_spk_id: str = _DEFAULT_SPK_ID,
        instruct_prefix: str = _DEFAULT_INSTRUCT_PREFIX,
        sample_rate: int = 24000,
    ) -> None:
        self._backend = backend
        self.zero_shot_spk_id = zero_shot_spk_id
        self.instruct_prefix = instruct_prefix
        self.sample_rate = sample_rate

        #buffer for TTS
        self.TTS_buffer = np.array([], dtype=np.int16)
        #queue for responses
        self.Response_Queue = queue.Queue()

    @classmethod
    def from_pretrained(
        cls,
        model_dir: str,
        *,
        load_vllm: bool = True,
        load_trt: bool = False,
        fp16: bool = True,
        prompt_text: str | None = None,
        prompt_wav: str | None = None,
        zero_shot_spk_id: str = _DEFAULT_SPK_ID,
        instruct_prefix: str = _DEFAULT_INSTRUCT_PREFIX,
        warmup_text: str | None = None,
        save_spkinfo: bool = False,
        **kwargs: Any,
    ) -> CosyVoice3Tts:
        """Load CosyVoice3 from a local model directory (or ModelScope id)."""
        backend = _load_backend(
            model_dir,
            load_vllm=load_vllm,
            load_trt=load_trt,
            fp16=fp16,
            **kwargs,
        )
        model = cls(
            backend,
            zero_shot_spk_id=zero_shot_spk_id,
            instruct_prefix=instruct_prefix,
        )
        if prompt_text is not None and prompt_wav is not None:
            model.add_speaker(prompt_text, prompt_wav, spk_id=zero_shot_spk_id)
        if warmup_text:
            for _ in model.generate(warmup_text, stream=True):
                pass
        if save_spkinfo:
            model.save_spkinfo()
        return model

    def add_speaker(
        self,
        prompt_text: str,
        prompt_wav: str,
        *,
        spk_id: str | None = None,
        with_instruct_prefix: bool = True,
    ) -> bool:
        spk_id = spk_id or self.zero_shot_spk_id
        full_prompt = (
            f"{self.instruct_prefix}{prompt_text}" if with_instruct_prefix else prompt_text
        )
        ok = self._backend.add_zero_shot_spk(full_prompt, prompt_wav, spk_id)
        if not ok:
            raise RuntimeError(f"failed to add zero-shot speaker: {spk_id}")
        self.zero_shot_spk_id = spk_id
        return True

   

    def save_spkinfo(self) -> None:
        self._backend.save_spkinfo()

    def generate(
        self,
        input: str,
        *,
        stream: bool = True,
        zero_shot_spk_id: str | None = None,
        prompt_text: str = "",
        prompt_wav: str = "",
        speed: float = 1.0,
        return_dict: bool = False,
        target_sr: int = 24000,
        **kwargs: Any,
    ) -> Iterator[np.ndarray | dict[str, Any]] | np.ndarray | list[dict[str, Any]]:
        """Synthesize speech from text via zero-shot inference."""
        spk_id = self.zero_shot_spk_id if zero_shot_spk_id is None else zero_shot_spk_id
        outputs = self._backend.inference_zero_shot(
            input,
            prompt_text,
            prompt_wav,
            zero_shot_spk_id=spk_id,
            stream=stream,
            speed=speed,
            **kwargs,
        )

        if stream:
            return self._stream_generate(
                outputs,
                return_dict=return_dict,
                target_sr=target_sr,
            )

        chunks = list(
            self._stream_generate(outputs, return_dict=return_dict, target_sr=target_sr)
        )
        if return_dict:
            return chunks
        if not chunks:
            return np.zeros(0, dtype=np.int16)
        return np.concatenate(chunks)

    def _stream_generate(
        self,
        outputs: Any,
        *,
        return_dict: bool,
        target_sr: int,
    ) -> Iterator[np.ndarray | dict[str, Any]]:
        for chunk in outputs:
            if return_dict:
                yield chunk
                continue
            audio = _speech_to_numpy(chunk, target_sr=target_sr)
            self.TTS_buffer = np.concatenate([self.TTS_buffer, audio])
            self._emit_frames(target_sr)
            yield audio

    def _emit_frames(self, sample_rate: int) -> None:
        """Split buffered int16 PCM into 100ms frames and push to Response_Queue."""
        frame_length_ms = 100
        frame_samples = int(sample_rate * frame_length_ms / 1000)
        while frame_samples <= len(self.TTS_buffer):
            frame = self.TTS_buffer[:frame_samples]
            self.Response_Queue.put(frame.tobytes())
            self.TTS_buffer = self.TTS_buffer[frame_samples:]
