"""Audio2Face engine: audio chunk -> ARKit blendshape weights.

Pipeline:
1. ONNX inference: audio -> PCA coefficients
2. Mesh reconstruction: PCA coefficients -> mesh vertices
3. Blendshape optimization: mesh vertices -> ARKit blendshape weights
"""

from __future__ import annotations

import json
import threading
import time
from pathlib import Path
from typing import Any, Optional

import numpy as np


class A2FEngine:
    """End-to-end Audio2Face inference engine."""

    def __init__(
        self,
        model_folder: str | Path,
        *,
        use_cuda: bool = True,
        num_streams: int = 1,
        enable_temporal_smoothing: bool = True,
        trt_cache_dir: str | Path | None = None,
    ) -> None:
        self.model_folder = Path(model_folder)
        self.use_cuda = use_cuda
        self.num_streams = max(1, int(num_streams))
        self.enable_temporal_smoothing = enable_temporal_smoothing
        self.trt_cache_dir = (
            Path(trt_cache_dir)
            if trt_cache_dir is not None
            else self.model_folder / "trt_cache"
        )

        self._load_config()
        self._load_onnx_model()
        self._load_model_data()
        self._load_blendshape_data()
        self._precompute_optimization_matrices()

        self.last_arkit_weights: Optional[np.ndarray] = None
        self.state_lock = threading.Lock()

    def _load_config(self) -> None:
        with open(self.model_folder / "network_info.json", encoding="utf-8") as f:
            config = json.load(f)

        self.network_info = config.get("params", {})
        self.audio_params = config.get("audio_params", {})

        with open(self.model_folder / "bs_skin_config.json", encoding="utf-8") as f:
            self.bs_config = json.load(f)

        self.blendshape_params = self.bs_config.get("blendshape_params", {})

    def _load_onnx_model(self) -> None:
        try:
            import onnxruntime as ort
        except ImportError as exc:
            raise ImportError(
                "onnxruntime is required for NvidiaAudio2Face3D. "
                'Install with: pip install "respeak-ai[a2f]"'
            ) from exc

        model_path = self.model_folder / "network.onnx"
        providers: list[Any] = []

        if self.use_cuda:
            self.trt_cache_dir.mkdir(parents=True, exist_ok=True)
            providers.append(
                (
                    "TensorrtExecutionProvider",
                    {
                        "trt_max_workspace_size": 4 * 1024 * 1024 * 1024,
                        "trt_fp16_enable": True,
                        "trt_engine_cache_enable": True,
                        "trt_engine_cache_path": str(self.trt_cache_dir),
                    },
                )
            )
            providers.append(
                (
                    "CUDAExecutionProvider",
                    {
                        "device_id": 0,
                        "arena_extend_strategy": "kNextPowerOfTwo",
                        "gpu_mem_limit": 2 * 1024 * 1024 * 1024,
                        "cudnn_conv_algo_search": "EXHAUSTIVE",
                    },
                )
            )

        providers.append("CPUExecutionProvider")

        sess_options = ort.SessionOptions()
        sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        sess_options.intra_op_num_threads = 4
        sess_options.inter_op_num_threads = 4

        self.ort_sessions = []
        for _ in range(self.num_streams):
            session = ort.InferenceSession(
                str(model_path),
                sess_options=sess_options,
                providers=providers,
            )
            self.ort_sessions.append(session)

        self.input_names = [inp.name for inp in self.ort_sessions[0].get_inputs()]
        self.output_names = [out.name for out in self.ort_sessions[0].get_outputs()]

    def _load_model_data(self) -> None:
        model_data = np.load(self.model_folder / "model_data.npz")
        shapes_matrix_skin = model_data["shapes_matrix_skin"]
        self.shapes_mean_skin = model_data["shapes_mean_skin"]

        shapes_matrix_flat = shapes_matrix_skin.reshape(shapes_matrix_skin.shape[0], -1)
        self.shapes_matrix_skin_T = np.ascontiguousarray(
            shapes_matrix_flat.T, dtype=np.float32
        )
        self.shapes_mean_skin_flat = np.ascontiguousarray(
            self.shapes_mean_skin.flatten(), dtype=np.float32
        )

    def _load_blendshape_data(self) -> None:
        bs_data = np.load(self.model_folder / "bs_skin.npz")
        self.bs_neutral = bs_data["neutral"]
        self.bs_poseNames = bs_data["poseNames"].astype(str)
        self.bs_deltas_matrix = np.array(
            [bs_data[name] for name in self.bs_poseNames[1:]]
        )

    def _precompute_optimization_matrices(self) -> None:
        self.D = self.bs_deltas_matrix.reshape(self.bs_deltas_matrix.shape[0], -1).T

        strength_l2 = self.blendshape_params.get("strengthL2regularization", 0.5)
        strength_temporal = self.blendshape_params.get("strengthTemporalSmoothing", 0.3)

        q = self.D.T @ self.D
        q += strength_l2 * np.eye(q.shape[0])
        if self.enable_temporal_smoothing:
            q += strength_temporal * np.eye(q.shape[0])

        self.Q_inverse = np.linalg.inv(q)
        self.strengthL1 = self.blendshape_params.get("strengthL1regularization", 0.5)
        self.strengthTemporal = (
            strength_temporal if self.enable_temporal_smoothing else 0.0
        )

    def _onnx_inference(self, audio_chunk: np.ndarray, session_idx: int = 0) -> np.ndarray:
        buffer_len = int(self.audio_params.get("buffer_len", 8320))

        if len(audio_chunk) != buffer_len:
            audio_padded = np.zeros(buffer_len, dtype=np.float32)
            audio_padded[: min(len(audio_chunk), buffer_len)] = audio_chunk[:buffer_len]
        else:
            audio_padded = np.asarray(audio_chunk, dtype=np.float32)

        inputs = {
            "input": audio_padded.reshape(1, 1, -1).astype(np.float32),
            "emotion": np.zeros((1, 1, 26), dtype=np.float32),
        }
        session = self.ort_sessions[session_idx % len(self.ort_sessions)]
        outputs = session.run(self.output_names, inputs)

        raw_output = outputs[0][0, 0]
        num_shapes_skin = int(self.network_info.get("num_shapes_skin", 272))
        return raw_output[:num_shapes_skin]

    def _reconstruct_mesh(self, skin_coeffs: np.ndarray) -> np.ndarray:
        coeffs_f32 = skin_coeffs.astype(np.float32, copy=False)
        vertices_flat = self.shapes_mean_skin_flat + np.dot(
            self.shapes_matrix_skin_T, coeffs_f32
        )
        return vertices_flat.reshape(-1, 3)

    def _optimize_arkit(
        self,
        vertices: np.ndarray,
        last_weights: Optional[np.ndarray],
    ) -> np.ndarray:
        delta = vertices - self.bs_neutral
        delta_flat = delta.flatten()

        b = -2.0 * (delta_flat @ self.D)
        b += self.strengthL1 * np.ones(b.shape)
        if last_weights is not None and self.enable_temporal_smoothing:
            b += -2.0 * self.strengthTemporal * last_weights

        weights = -0.5 * (self.Q_inverse @ b)
        return np.clip(weights, 0.0, 1.0)

    def process_audio_chunk(
        self,
        audio_chunk: np.ndarray,
        *,
        session_idx: int = 0,
        return_debug_info: bool = False,
    ) -> dict[str, Any]:
        """Run the full pipeline on one audio window.

        Returns:
            dict with ``arkit_weights`` and optional ``timing`` / ``vertices``.
        """
        timing: dict[str, float] | None = {} if return_debug_info else None

        if return_debug_info:
            t0 = time.perf_counter()
        skin_coeffs = self._onnx_inference(audio_chunk, session_idx)
        if return_debug_info and timing is not None:
            timing["onnx_inference"] = (time.perf_counter() - t0) * 1000
            t0 = time.perf_counter()

        vertices = self._reconstruct_mesh(skin_coeffs)
        if return_debug_info and timing is not None:
            timing["mesh_reconstruction"] = (time.perf_counter() - t0) * 1000
            t0 = time.perf_counter()

        with self.state_lock:
            last_weights = self.last_arkit_weights

        arkit_weights = self._optimize_arkit(vertices, last_weights)

        with self.state_lock:
            self.last_arkit_weights = arkit_weights

        if return_debug_info and timing is not None:
            timing["arkit_optimization"] = (time.perf_counter() - t0) * 1000

        result: dict[str, Any] = {"arkit_weights": arkit_weights}
        if return_debug_info:
            result["timing"] = timing
            result["vertices"] = vertices
        return result

    def reset_temporal_state(self) -> None:
        """Reset temporal smoothing state for a new utterance."""
        with self.state_lock:
            self.last_arkit_weights = None

    def get_arkit_pose_names(self) -> list[str]:
        """ARKit pose names corresponding to weight indices (neutral excluded)."""
        return list(self.bs_poseNames[1:])

    @property
    def buffer_len(self) -> int:
        return int(self.audio_params.get("buffer_len", 8320))

    @property
    def sample_rate(self) -> int:
        return int(self.audio_params.get("samplerate", 16000))
