"""Audio2Face engine: audio chunk -> ARKit blendshape weights.

Pipeline:
1. ONNX inference: audio -> PCA coefficients
2. Mesh reconstruction: PCA coefficients -> mesh vertices
3. Blendshape optimization: mesh vertices -> ARKit blendshape weights
"""

from __future__ import annotations

import ctypes.util
import json
import logging
import os
import site
import threading
import time
from pathlib import Path
from typing import Any, Optional

import numpy as np

logger = logging.getLogger(__name__)


def _nvidia_pip_lib_dirs() -> list[str]:
    """Return ``site-packages/nvidia/*/lib`` dirs shipped by CUDA pip wheels."""
    nvidia_roots: list[Path] = []
    try:
        import nvidia as _nvidia_pkg  # type: ignore

        nvidia_roots.append(Path(_nvidia_pkg.__file__).resolve().parent)
    except Exception:  # noqa: BLE001
        pass

    try:
        for p in site.getsitepackages():
            nvidia_roots.append(Path(p) / "nvidia")
    except Exception:  # noqa: BLE001
        pass
    try:
        user = site.getusersitepackages()
        if user:
            nvidia_roots.append(Path(user) / "nvidia")
    except Exception:  # noqa: BLE001
        pass

    dirs: list[str] = []
    seen: set[str] = set()
    for nvidia in nvidia_roots:
        if not nvidia.is_dir():
            continue
        for lib_dir in sorted(nvidia.glob("*/lib")):
            if not lib_dir.is_dir():
                continue
            key = str(lib_dir)
            if key in seen:
                continue
            seen.add(key)
            dirs.append(key)
    return dirs


def _tensorrt_pip_lib_dirs() -> list[str]:
    """Return ``site-packages/tensorrt_libs`` when the ``tensorrt`` pip wheel is installed."""
    dirs: list[str] = []
    seen: set[str] = set()
    candidates: list[Path] = []
    try:
        for p in site.getsitepackages():
            candidates.append(Path(p) / "tensorrt_libs")
    except Exception:  # noqa: BLE001
        pass
    try:
        user = site.getusersitepackages()
        if user:
            candidates.append(Path(user) / "tensorrt_libs")
    except Exception:  # noqa: BLE001
        pass

    for lib_dir in candidates:
        if not lib_dir.is_dir():
            continue
        key = str(lib_dir)
        if key in seen:
            continue
        seen.add(key)
        dirs.append(key)
    return dirs


def _runtime_lib_dirs() -> list[str]:
    """CUDA (nvidia-*) and TensorRT (tensorrt_libs) directories for ORT GPU EPs."""
    dirs: list[str] = []
    seen: set[str] = set()
    for lib_dir in _nvidia_pip_lib_dirs() + _tensorrt_pip_lib_dirs():
        if lib_dir in seen:
            continue
        seen.add(lib_dir)
        dirs.append(lib_dir)
    return dirs


def _preload_runtime_shared_libs(lib_dirs: list[str]) -> None:
    """``dlopen`` pip-shipped CUDA / TensorRT libs with ``RTLD_GLOBAL`` for ORT EPs."""
    priority_substrings = (
        "libcudart.so",
        "libnvrtc.so",
        "libcublasLt.so",
        "libcublas.so",
        "libcurand.so",
        "libcufft.so",
        "libcusparse.so",
        "libcudnn.so",
        "libnvJitLink.so",
        "libnvjitlink.so",
        "libnvinfer.so",
        "libnvinfer_plugin.so",
        "libnvonnxparser.so",
    )

    candidates: list[Path] = []
    for lib_dir in lib_dirs:
        candidates.extend(sorted(Path(lib_dir).glob("lib*.so*")))

    ordered: list[Path] = []
    seen: set[str] = set()
    for needle in priority_substrings:
        for path in candidates:
            key = str(path)
            if key in seen or needle not in path.name:
                continue
            seen.add(key)
            ordered.append(path)
    for path in candidates:
        key = str(path)
        if key in seen:
            continue
        seen.add(key)
        ordered.append(path)

    for path in ordered:
        try:
            ctypes.CDLL(str(path), mode=ctypes.RTLD_GLOBAL)
        except OSError:
            continue


def ensure_cuda_library_path() -> list[str]:
    """Expose pip-installed CUDA / TensorRT libs to the dynamic linker.

    ``onnxruntime-gpu`` depends on ``nvidia-*`` wheels; TensorRT EP additionally
    needs the ``tensorrt`` pip package (``site-packages/tensorrt_libs``). Neither
    is always on ``LD_LIBRARY_PATH`` until we prepend / preload here.
    """
    lib_dirs = _runtime_lib_dirs()
    if not lib_dirs:
        return []

    current = os.environ.get("LD_LIBRARY_PATH", "")
    current_parts = [p for p in current.split(":") if p]
    prepend = [d for d in lib_dirs if d not in current_parts]
    if prepend:
        os.environ["LD_LIBRARY_PATH"] = ":".join(prepend + current_parts)

    _preload_runtime_shared_libs(lib_dirs)
    return prepend


def tensorrt_runtime_available() -> bool:
    """True when TensorRT runtime (``libnvinfer``) can be resolved."""
    ensure_cuda_library_path()
    if ctypes.util.find_library("nvinfer") is not None:
        return True
    for lib_dir in _tensorrt_pip_lib_dirs():
        for name in ("libnvinfer.so.10", "libnvinfer.so.8", "libnvinfer.so"):
            if (Path(lib_dir) / name).is_file():
                try:
                    ctypes.CDLL(str(Path(lib_dir) / name), mode=ctypes.RTLD_GLOBAL)
                    return True
                except OSError:
                    continue
    for name in ("libnvinfer.so.10", "libnvinfer.so.8", "libnvinfer.so"):
        try:
            ctypes.CDLL(name, mode=ctypes.RTLD_GLOBAL)
            return True
        except OSError:
            continue
    return False


class A2FEngine:
    """End-to-end Audio2Face inference engine."""

    def __init__(
        self,
        model_folder: str | Path,
        *,
        use_cuda: bool = True,
        enable_tensorrt: bool = False,
        num_streams: int = 1,
        enable_temporal_smoothing: bool = True,
        trt_cache_dir: str | Path | None = None,
    ) -> None:
        self.model_folder = Path(model_folder)
        self.use_cuda = use_cuda
        self.enable_tensorrt = enable_tensorrt
        self.num_streams = max(1, int(num_streams))
        self.enable_temporal_smoothing = enable_temporal_smoothing
        self.trt_cache_dir = (
            Path(trt_cache_dir)
            if trt_cache_dir is not None
            else self.model_folder / "trt_cache"
        )
        self.active_provider = "CPUExecutionProvider"

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

    @staticmethod
    def _provider_name(provider: Any) -> str:
        return provider[0] if isinstance(provider, tuple) else str(provider)

    def _cuda_provider(self) -> tuple[str, dict[str, Any]]:
        return (
            "CUDAExecutionProvider",
            {
                "device_id": 0,
                "arena_extend_strategy": "kNextPowerOfTwo",
                "gpu_mem_limit": 2 * 1024 * 1024 * 1024,
                "cudnn_conv_algo_search": "EXHAUSTIVE",
            },
        )

    def _tensorrt_provider(self) -> tuple[str, dict[str, Any]]:
        self.trt_cache_dir.mkdir(parents=True, exist_ok=True)
        return (
            "TensorrtExecutionProvider",
            {
                "trt_max_workspace_size": 4 * 1024 * 1024 * 1024,
                "trt_fp16_enable": True,
                "trt_engine_cache_enable": True,
                "trt_engine_cache_path": str(self.trt_cache_dir),
            },
        )

    def _provider_attempts(self, ort: Any) -> list[list[Any]]:
        """Build provider try-order: TRT+CUDA -> CUDA -> CPU.

        Important: if TensorRT is requested but its native libs are missing, ORT
        may fall all the way back to CPU and skip CUDA. Probe ``libnvinfer``
        first and only then include TensorRT.
        """
        available = set(ort.get_available_providers())
        attempts: list[list[Any]] = []

        want_trt = (
            self.use_cuda
            and self.enable_tensorrt
            and "TensorrtExecutionProvider" in available
        )
        if want_trt and not tensorrt_runtime_available():
            logger.warning(
                "TensorRT requested but libnvinfer was not found; "
                "falling back to CUDAExecutionProvider. "
                "Install TensorRT or omit --tensorrt / enable_tensorrt=True."
            )
            want_trt = False

        if want_trt and "CUDAExecutionProvider" in available:
            attempts.append(
                [self._tensorrt_provider(), self._cuda_provider(), "CPUExecutionProvider"]
            )

        if self.use_cuda and "CUDAExecutionProvider" in available:
            attempts.append([self._cuda_provider(), "CPUExecutionProvider"])

        attempts.append(["CPUExecutionProvider"])

        # De-duplicate identical lists while preserving order.
        unique: list[list[Any]] = []
        seen: set[tuple[str, ...]] = set()
        for providers in attempts:
            key = tuple(self._provider_name(p) for p in providers)
            if key in seen:
                continue
            seen.add(key)
            unique.append(providers)
        return unique

    def _create_sessions(
        self,
        ort: Any,
        model_path: Path,
        sess_options: Any,
        providers: list[Any],
    ) -> list[Any]:
        sessions = []
        for _ in range(self.num_streams):
            sessions.append(
                ort.InferenceSession(
                    str(model_path),
                    sess_options=sess_options,
                    providers=providers,
                )
            )
        return sessions

    def _load_onnx_model(self) -> None:
        # Must run before importing/using GPU EPs that dlopen cuBLAS etc.
        ensure_cuda_library_path()

        try:
            import onnxruntime as ort
        except ImportError as exc:
            raise ImportError(
                "onnxruntime is required for NvidiaAudio2Face3D. "
                'Install with: pip install "respeak-ai[a2f]" '
                '(use onnxruntime-gpu for CUDA)'
            ) from exc

        model_path = self.model_folder / "network.onnx"
        sess_options = ort.SessionOptions()
        sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        sess_options.intra_op_num_threads = 4
        sess_options.inter_op_num_threads = 4

        last_error: Exception | None = None
        self.ort_sessions = []
        for providers in self._provider_attempts(ort):
            requested = [self._provider_name(p) for p in providers]
            try:
                sessions = self._create_sessions(ort, model_path, sess_options, providers)
            except Exception as exc:  # noqa: BLE001 - ORT raises various init errors
                last_error = exc
                logger.warning("ONNX provider attempt %s failed: %s", requested, exc)
                continue

            active = sessions[0].get_providers()[0]
            # If we asked for a GPU EP but ORT bound CPU, try the next attempt.
            wanted_gpu = [p for p in requested if p != "CPUExecutionProvider"]
            if wanted_gpu and active == "CPUExecutionProvider":
                logger.warning(
                    "Requested %s but session bound to CPU; trying next provider list",
                    wanted_gpu,
                )
                continue

            self.ort_sessions = sessions
            self.active_provider = active
            break

        if not self.ort_sessions:
            raise RuntimeError(
                f"Failed to create ONNX Runtime session for {model_path}"
            ) from last_error

        self.input_names = [inp.name for inp in self.ort_sessions[0].get_inputs()]
        self.output_names = [out.name for out in self.ort_sessions[0].get_outputs()]

    def _load_model_data(self) -> None:
        """Load model data from npz file."""
        """
        Load model data from npz file.
        The npz file is downloaded from https://huggingface.co/nvidia/Audio2Face-3D-v2.3.1-Claire
        >>> import numpy as np
        >>> model_data = np.load("openFaceBot/models_hub/audio2face-3d-v2.3.1-claire/model_data.npz")
        >>> model_data["shapes_matrix_skin"].shape
        (140, 61520, 3)
        >>> model_data["shapes_mean_skin"].shape
        (61520, 3)

        Returns:
            dict with ``shapes_matrix_skin`` and ``shapes_mean_skin``.
        """
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
        """Load blendshape data from npz file.
        >>> bs_data= np.load("openFaceBot/models_hub/audio2face-3d-v2.3.1-claire/bs_skin.npz")
        >>> bs_data['neutral']
        array([[-2.16354132e-02,  1.27395966e+02, -1.54198265e+01],
            [-2.88138092e-02,  1.29121948e+02, -1.48939502e+00],
            [ 1.14790440e+01,  1.29663345e+02, -5.59380102e+00],
            ...,
            [ 9.97686195e+00,  1.30085632e+02, -9.21193027e+00],
            [ 9.99506187e+00,  1.30101410e+02, -8.46189404e+00],
            [ 9.99708843e+00,  1.30100403e+02, -8.83994007e+00]], dtype=float32)
        >>> bs_data['neutral'].shape
        (61520, 3)
        >>> bs_data['poseNames'].astype(str)
        array(['neutral', 'eyeBlinkLeft', 'eyeLookDownLeft', 'eyeLookInLeft',
            'eyeLookOutLeft', 'eyeLookUpLeft', 'eyeSquintLeft', 'eyeWideLeft',
            'eyeBlinkRight', 'eyeLookDownRight', 'eyeLookInRight',
            'eyeLookOutRight', 'eyeLookUpRight', 'eyeSquintRight',
            'eyeWideRight', 'jawForward', 'jawLeft', 'jawRight', 'jawOpen',
            'mouthClose', 'mouthFunnel', 'mouthPucker', 'mouthLeft',
            'mouthRight', 'mouthSmileLeft', 'mouthSmileRight',
            'mouthFrownLeft', 'mouthFrownRight', 'mouthDimpleLeft',
            'mouthDimpleRight', 'mouthStretchLeft', 'mouthStretchRight',
            'mouthRollLower', 'mouthRollUpper', 'mouthShrugLower',
            'mouthShrugUpper', 'mouthPressLeft', 'mouthPressRight',
            'mouthLowerDownLeft', 'mouthLowerDownRight', 'mouthUpperUpLeft',
            'mouthUpperUpRight', 'browDownLeft', 'browDownRight',
            'browInnerUp', 'browOuterUpLeft', 'browOuterUpRight', 'cheekPuff',
            'cheekSquintLeft', 'cheekSquintRight', 'noseSneerLeft',
            'noseSneerRight', 'tongueOut'], dtype='<U19')
        >>>  np.array([bs_data[name] for name in bs_data['poseNames'].astype(str)[1:]]).shape
        File "<stdin>", line 1
            np.array([bs_data[name] for name in bs_data['poseNames'].astype(str)[1:]]).shape
        IndentationError: unexpected indent
        >>> np.array([bs_data[name] for name in bs_data['poseNames'].astype(str)[1:]]).shape
        (52, 61520, 3)

        >>> np.array([bs_data[name] for name in bs_data['poseNames'].astype(str)]).shape
        (53, 61520, 3)
        """
        bs_data = np.load(self.model_folder / "bs_skin.npz")
        self.bs_neutral = bs_data["neutral"]
        self.bs_poseNames = bs_data["poseNames"].astype(str)
        self.bs_deltas_matrix = np.array(
            [bs_data[name] for name in self.bs_poseNames[1:]]
        )

    def _precompute_optimization_matrices(self) -> None:
        """Precompute optimization matrices."""
        """
        Precompute optimization matrices.

        Returns:
            dict with ``D``, ``Q_inverse``, ``strengthL1``, and ``strengthTemporal``.
        """
        self.D = self.bs_deltas_matrix.reshape(self.bs_deltas_matrix.shape[0], -1).T

        strength_l2 = self.blendshape_params.get("strengthL2regularization", 0.5)
        strength_temporal = self.blendshape_params.get("strengthTemporalSmoothing", 0.3)

        q = self.D.T @ self.D
        q += strength_l2 * np.eye(q.shape[0])
        if self.enable_temporal_smoothing:
            q += strength_temporal * np.eye(q.shape[0])

        # Q_inverse is the inverse of the matrix Q
        self.Q_inverse = np.linalg.inv(q)
        self.strengthL1 = self.blendshape_params.get("strengthL1regularization", 0.5)
        self.strengthTemporal = (
            strength_temporal if self.enable_temporal_smoothing else 0.0
        )

    def _onnx_inference(self, audio_chunk: np.ndarray, session_idx: int = 0) -> np.ndarray:
        """Run ONNX inference on one audio chunk."""
        """
        Run ONNX inference on one audio chunk.

        Returns:
            np.ndarray with shape (num_shapes_skin,).
        """
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

        # 1. ONNX inference
        if return_debug_info:
            t0 = time.perf_counter()
        skin_coeffs = self._onnx_inference(audio_chunk, session_idx)
        if return_debug_info and timing is not None:
            timing["onnx_inference"] = (time.perf_counter() - t0) * 1000
            t0 = time.perf_counter()

        # 2. Mesh reconstruction
        vertices = self._reconstruct_mesh(skin_coeffs)
        if return_debug_info and timing is not None:
            timing["mesh_reconstruction"] = (time.perf_counter() - t0) * 1000
            t0 = time.perf_counter()

        # 3. ARKit optimization
        with self.state_lock:
            last_weights = self.last_arkit_weights

        arkit_weights = self._optimize_arkit(vertices, last_weights)

        # 4. Update temporal state
        with self.state_lock:
            self.last_arkit_weights = arkit_weights

        if return_debug_info and timing is not None:
            timing["arkit_optimization"] = (time.perf_counter() - t0) * 1000

        # 5. Return result
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
