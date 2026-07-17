"""NVIDIA Audio2Face 3D: streaming audio -> ARKit blendshape weights."""

from __future__ import annotations

from typing import Any, Iterator

import numpy as np

from respeak.base import BaseModel
from respeak.models.nvidia_audio2face_3d.a2f_engine import A2FEngine

_DEFAULT_OUTPUT_FPS = 25
_DEFAULT_WINDOW_MS = 520


class NvidiaAudio2Face3D(BaseModel):
    """NVIDIA Audio2Face 3D wrapper producing ARKit blendshape weights.

    Expected model folder contents (NVIDIA Audio2Face 3D export)::

        network.onnx
        network_info.json
        model_data.npz
        bs_skin.npz
        bs_skin_config.json
    """

    def __init__(
        self,
        engine: A2FEngine,
        *,
        output_fps: int = _DEFAULT_OUTPUT_FPS,
        window_ms: int = _DEFAULT_WINDOW_MS,
    ) -> None:
        self._engine = engine
        self.output_fps = int(output_fps)
        self.window_ms = int(window_ms)

        self.sample_rate = engine.sample_rate
        self.buffer_len = engine.buffer_len
        self.window_samples = int(self.sample_rate * self.window_ms / 1000)
        if self.window_samples != self.buffer_len:
            # Prefer model config when it disagrees with the default window.
            self.window_samples = self.buffer_len
            self.window_ms = int(round(1000 * self.window_samples / self.sample_rate))

        self.half_window_samples = self.window_samples // 2
        self.step_samples = max(1, int(self.sample_rate / self.output_fps))

    @classmethod
    def from_pretrained(
        cls,
        model_folder: str,
        *,
        use_cuda: bool = True,
        num_streams: int = 1,
        enable_temporal_smoothing: bool = True,
        output_fps: int = _DEFAULT_OUTPUT_FPS,
        window_ms: int = _DEFAULT_WINDOW_MS,
        trt_cache_dir: str | None = None,
        **kwargs: Any,
    ) -> NvidiaAudio2Face3D:
        """Load an Audio2Face 3D model directory.

        Extra kwargs are ignored for forward-compat with other respeak loaders.
        """
        del kwargs
        engine = A2FEngine(
            model_folder=model_folder,
            use_cuda=use_cuda,
            num_streams=num_streams,
            enable_temporal_smoothing=enable_temporal_smoothing,
            trt_cache_dir=trt_cache_dir,
        )
        return cls(engine, output_fps=output_fps, window_ms=window_ms)

    def generate(
        self,
        input: np.ndarray,
        *,
        stream: bool = False,
        is_final: bool = False,
        session_idx: int = 0,
        return_debug_info: bool = False,
        return_dict: bool = False,
        pad_to: int | None = None,
    ) -> np.ndarray | dict[str, Any] | list[np.ndarray] | Iterator[np.ndarray]:
        """Infer ARKit blendshape weights from audio.

        Modes:
        - ``stream=False`` (default): treat ``input`` as one model window
          (``buffer_len`` samples, typically 8320 @ 16 kHz) and return one frame.
        - ``stream=True``: slide a 520 ms window over ``input`` at ``output_fps``
          and return a list of frames (or yield if caller iterates — returns list).

        Args:
            input: float32 mono audio in ``[-1, 1]``.
            stream: enable sliding-window streaming over a longer clip.
            is_final: when streaming, drain trailing half-window with zeros.
            session_idx: ONNX session index for multi-stream setups.
            return_debug_info: include timing / vertices (non-stream only).
            return_dict: return ``{"arkit_weights", "pose_names", ...}``.
            pad_to: optional pad/truncate weights to this length (e.g. 61).
        """
        audio = np.asarray(input, dtype=np.float32).reshape(-1)

        if stream:
            frames = list(
                self._stream_generate(
                    audio,
                    is_final=is_final,
                    session_idx=session_idx,
                    pad_to=pad_to,
                )
            )
            if return_dict:
                return {
                    "arkit_weights": frames,
                    "pose_names": self.get_arkit_pose_names(),
                    "fps": self.output_fps,
                }
            return frames

        result = self._engine.process_audio_chunk(
            audio,
            session_idx=session_idx,
            return_debug_info=return_debug_info,
        )
        weights = result["arkit_weights"]
        if pad_to is not None:
            weights = self.pad_weights(weights, pad_to)

        if return_dict or return_debug_info:
            out: dict[str, Any] = {
                "arkit_weights": weights,
                "pose_names": self.get_arkit_pose_names(),
            }
            if return_debug_info:
                out["timing"] = result.get("timing")
                out["vertices"] = result.get("vertices")
            return out
        return weights

    def _stream_generate(
        self,
        audio: np.ndarray,
        *,
        is_final: bool,
        session_idx: int,
        pad_to: int | None,
    ) -> Iterator[np.ndarray]:
        """Sliding-window streaming (no ROS): 520 ms window, step = 1/fps."""
        self.reset()

        window = np.zeros(self.window_samples, dtype=np.float32)
        cursor = 0
        first = True

        while True:
            needed = self.half_window_samples if first else self.step_samples
            remaining = len(audio) - cursor

            if remaining >= needed:
                chunk = audio[cursor : cursor + needed]
                cursor += needed
            elif is_final and remaining > 0:
                chunk = np.zeros(needed, dtype=np.float32)
                chunk[:remaining] = audio[cursor:]
                cursor = len(audio)
            elif is_final and remaining == 0:
                # Drain trailing future context (half window) with zeros.
                drain_steps = int(np.ceil(self.half_window_samples / self.step_samples))
                for _ in range(drain_steps):
                    window[:-self.step_samples] = window[self.step_samples :]
                    window[-self.step_samples :] = 0.0
                    result = self._engine.process_audio_chunk(
                        window, session_idx=session_idx
                    )
                    weights = result["arkit_weights"]
                    if pad_to is not None:
                        weights = self.pad_weights(weights, pad_to)
                    yield weights
                break
            else:
                break

            if first:
                window[self.half_window_samples :] = chunk
                first = False
            else:
                window[:-self.step_samples] = window[self.step_samples :]
                window[-self.step_samples :] = chunk

            result = self._engine.process_audio_chunk(window, session_idx=session_idx)
            weights = result["arkit_weights"]
            if pad_to is not None:
                weights = self.pad_weights(weights, pad_to)
            yield weights

    def reset(self) -> None:
        """Reset temporal smoothing for a new utterance."""
        self._engine.reset_temporal_state()

    def get_arkit_pose_names(self) -> list[str]:
        return self._engine.get_arkit_pose_names()

    @staticmethod
    def pad_weights(weights: np.ndarray, length: int = 61) -> np.ndarray:
        """Pad or truncate ARKit weights to a fixed length (e.g. Live Link 61)."""
        weights = np.asarray(weights, dtype=np.float32).reshape(-1)
        if weights.shape[0] == length:
            return weights
        out = np.zeros(length, dtype=np.float32)
        n = min(length, weights.shape[0])
        out[:n] = weights[:n]
        return out
