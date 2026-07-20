"""NVIDIA Audio2Face 3D: streaming audio -> ARKit blendshape weights."""

from __future__ import annotations

import queue
from typing import Any, Iterator

import numpy as np

from respeak.base import BaseModel
from respeak.models.nvidia_audio2face_3d.a2f_engine import (
    A2FEngine,
    ensure_cuda_library_path,
)
from respeak.models.nvidia_audio2face_3d.audio_io import encode_stream_pcm, int16_pcm_to_float

_DEFAULT_OUTPUT_FPS = 25
_DEFAULT_WINDOW_MS = 520
_DEFAULT_AUDIO_OUTPUT_OFFSET = 0.5
_DEFAULT_TARGET_SR = 16000
_DEFAULT_AUDIO_FRAME_MS = 100


class NvidiaAudio2Face3D(BaseModel):
    """NVIDIA Audio2Face 3D wrapper producing ARKit blendshape weights.

    Synced audio is emitted as **int16 PCM** (same format as ``CosyVoice3Tts``),
    optionally resampled via ``target_sr``. Buffered 100 ms frames are pushed to
    ``Response_Queue`` as ``bytes`` for playback / ROS-style consumers.

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
        audio_output_offset: float = _DEFAULT_AUDIO_OUTPUT_OFFSET,
        audio_frame_ms: int = _DEFAULT_AUDIO_FRAME_MS,
    ) -> None:
        self._engine = engine
        self.output_fps = int(output_fps)
        self.window_ms = int(window_ms)
        self.audio_output_offset = float(audio_output_offset)
        self.audio_frame_ms = int(audio_frame_ms)

        self.sample_rate = engine.sample_rate
        self.buffer_len = engine.buffer_len
        self.window_samples = int(self.sample_rate * self.window_ms / 1000)
        if self.window_samples != self.buffer_len:
            self.window_samples = self.buffer_len
            self.window_ms = int(round(1000 * self.window_samples / self.sample_rate))

        self.half_window_samples = self.window_samples // 2
        self.step_samples = max(1, int(self.sample_rate / self.output_fps))
        self.audio_output_samples = self.step_samples
        self.audio_output_start = int(
            self.half_window_samples
            + (self.window_samples - self.audio_output_samples)
            * (self.audio_output_offset - 0.5)
        )

        self.a2f_buffer = np.array([], dtype=np.int16)
        self.a2f_response_queue: queue.Queue[bytes] = queue.Queue()

    @classmethod
    def from_pretrained(
        cls,
        model_folder: str,
        *,
        use_cuda: bool = True,
        enable_tensorrt: bool = False,
        num_streams: int = 1,
        enable_temporal_smoothing: bool = True,
        output_fps: int = _DEFAULT_OUTPUT_FPS,
        window_ms: int = _DEFAULT_WINDOW_MS,
        audio_output_offset: float = _DEFAULT_AUDIO_OUTPUT_OFFSET,
        trt_cache_dir: str | None = None,
        **kwargs: Any,
    ) -> NvidiaAudio2Face3D:
        """Load an Audio2Face 3D model directory."""
        del kwargs
        if use_cuda:
            ensure_cuda_library_path()
        engine = A2FEngine(
            model_folder=model_folder,
            use_cuda=use_cuda,
            enable_tensorrt=enable_tensorrt,
            num_streams=num_streams,
            enable_temporal_smoothing=enable_temporal_smoothing,
            trt_cache_dir=trt_cache_dir,
        )
        return cls(
            engine,
            output_fps=output_fps,
            window_ms=window_ms,
            audio_output_offset=audio_output_offset,
        )

    @property
    def active_provider(self) -> str:
        return self._engine.active_provider

    def extract_synced_audio(self, window: np.ndarray) -> np.ndarray:
        """Extract the per-frame float32 audio slice aligned with blendshape output."""
        start = self.audio_output_start
        end = start + self.audio_output_samples
        if end <= window.shape[0]:
            return window[start:end].astype(np.float32, copy=True)
        out = np.zeros(self.audio_output_samples, dtype=np.float32)
        available = max(0, window.shape[0] - start)
        if available > 0:
            out[:available] = window[start : start + available]
        return out

    def _encode_output_audio(
        self,
        speech_f32: np.ndarray,
        *,
        target_sr: int,
    ) -> np.ndarray:
        return encode_stream_pcm(
            speech_f32,
            source_sr=self.sample_rate,
            target_sr=target_sr,
        )

    def _emit_response_frames(self, sample_rate: int) -> None:
        """Split buffered int16 PCM into fixed frames and push bytes to Response_Queue."""
        frame_samples = max(1, int(sample_rate * self.audio_frame_ms / 1000))
        while frame_samples <= len(self.a2f_buffer):
            frame = self.a2f_buffer[:frame_samples]
            self.a2f_response_queue.put(frame.tobytes())
            self.a2f_buffer = self.a2f_buffer[frame_samples:]

    def _buffer_output_audio(self, pcm: np.ndarray, *, target_sr: int) -> None:
        pcm = np.asarray(pcm, dtype=np.int16).reshape(-1)
        if pcm.size == 0:
            return
        self.a2f_buffer = np.concatenate([self.a2f_buffer, pcm])
        self._emit_response_frames(target_sr)

    def _make_frame(
        self,
        window: np.ndarray,
        weights: np.ndarray,
        *,
        pad_to: int | None,
        target_sr: int,
    ) -> dict[str, Any]:
        if pad_to is not None:
            weights = self.pad_weights(weights, pad_to)
        pcm = self._encode_output_audio(self.extract_synced_audio(window), target_sr=target_sr)
        self._buffer_output_audio(pcm, target_sr=target_sr)
        return {
            "audio": pcm,
            "arkit_weights": weights,
        }

    def generate(
        self,
        input: np.ndarray,
        *,
        stream: bool = False,
        is_final: bool = False,
        return_audio: bool = True,
        session_idx: int = 0,
        return_debug_info: bool = False,
        return_dict: bool = False,
        pad_to: int | None = None,
        target_sr: int = _DEFAULT_TARGET_SR,
    ) -> (
        np.ndarray
        | dict[str, Any]
        | list[np.ndarray]
        | list[dict[str, Any]]
        | Iterator[np.ndarray]
        | Iterator[dict[str, Any]]
    ):
        """Infer ARKit blendshape weights (and synced int16 audio) from input audio.

        Input may be float32 or int16 mono PCM. Output audio matches
        ``CosyVoice3Tts.generate(..., target_sr=...)``: **np.int16** chunks.

        When ``return_audio=True`` (default), each streaming frame is::

            {"audio": int16[...], "arkit_weights": float32[51 or pad_to]}

        ``Response_Queue`` receives 100 ms int16 PCM ``bytes`` frames (playback-ready).
        """
        audio = np.asarray(input).reshape(-1)
        if audio.dtype != np.float32:
            audio = int16_pcm_to_float(audio)

        if stream:
            frames = list(
                self._stream_generate(
                    audio,
                    is_final=is_final,
                    session_idx=session_idx,
                    pad_to=pad_to,
                    return_audio=return_audio,
                    target_sr=target_sr,
                )
            )
            if return_dict:
                out: dict[str, Any] = {
                    "frames": frames,
                    "pose_names": self.get_arkit_pose_names(),
                    "fps": self.output_fps,
                    "sample_rate": target_sr,
                }
                if return_audio and frames:
                    if isinstance(frames[0], dict):
                        out["audio"] = np.concatenate(
                            [f["audio"] for f in frames], axis=0
                        )
                        out["arkit_weights"] = [f["arkit_weights"] for f in frames]
                    else:
                        out["arkit_weights"] = frames
                else:
                    out["arkit_weights"] = frames
                return out

            return frames

        result = self._engine.process_audio_chunk(
            audio,
            session_idx=session_idx,
            return_debug_info=return_debug_info,
        )
        weights = result["arkit_weights"]
        if pad_to is not None:
            weights = self.pad_weights(weights, pad_to)

        if return_audio or return_dict or return_debug_info:
            out: dict[str, Any] = {
                "arkit_weights": weights,
                "pose_names": self.get_arkit_pose_names(),
                "sample_rate": target_sr,
            }
            if return_audio or return_dict:
                frame = self._make_frame(
                    audio, weights, pad_to=pad_to, target_sr=target_sr
                )
                out["audio"] = frame["audio"]
                out["arkit_weights"] = frame["arkit_weights"]
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
        return_audio: bool,
        target_sr: int,
    ) -> Iterator[np.ndarray | dict[str, Any]]:
        """Sliding-window streaming: 520 ms window, step = 1/fps."""
        self.reset()

        window = np.zeros(self.window_samples, dtype=np.float32)
        cursor = 0
        first = True

        def _emit(w: np.ndarray, weights: np.ndarray) -> np.ndarray | dict[str, Any]:
            if return_audio:
                return self._make_frame(
                    w, weights, pad_to=pad_to, target_sr=target_sr
                )
            if pad_to is not None:
                return self.pad_weights(weights, pad_to)
            return weights

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
                drain_steps = int(np.ceil(self.half_window_samples / self.step_samples))
                for _ in range(drain_steps):
                    window[:-self.step_samples] = window[self.step_samples :]
                    window[-self.step_samples :] = 0.0
                    result = self._engine.process_audio_chunk(
                        window, session_idx=session_idx
                    )
                    yield _emit(window, result["arkit_weights"])
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
            yield _emit(window, result["arkit_weights"])

    def reset(self) -> None:
        """Reset temporal smoothing and audio output buffers for a new utterance."""
        self._engine.reset_temporal_state()
        self.a2f_buffer = np.array([], dtype=np.int16)
        while not self.a2f_response_queue.empty():
            try:
                self.a2f_response_queue.get_nowait()
            except queue.Empty:
                break

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
