"""Testing suite for NvidiaAudio2Face3D (transformers-style)."""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, patch

import numpy as np

from respeak.models.nvidia_audio2face_3d import NvidiaAudio2Face3D
from respeak.models.nvidia_audio2face_3d.a2f_engine import A2FEngine
from respeak.models.nvidia_audio2face_3d.modeling_nvidia_audio2face_3d import (
    NvidiaAudio2Face3D as NvidiaAudio2Face3DModel,
)
from respeak.models.nvidia_audio2face_3d.ue5_blendshape_renderer import (
    Ue5BlendshapeRenderer,
)
from tests.test_modeling_common import ModelTesterMixin

_NUM_POSES = 51
_BUFFER_LEN = 8320
_NUM_SHAPES = 8
_NUM_VERTS = 16


def _write_tiny_model_folder(root: Path) -> Path:
    """Create a minimal on-disk A2F model layout for engine unit tests."""
    root.mkdir(parents=True, exist_ok=True)

    network_info = {
        "params": {"num_shapes_skin": _NUM_SHAPES},
        "audio_params": {"buffer_len": _BUFFER_LEN, "samplerate": 16000},
    }
    (root / "network_info.json").write_text(json.dumps(network_info), encoding="utf-8")

    bs_config = {
        "blendshape_params": {
            "numPoses": _NUM_POSES,
            "strengthL2regularization": 0.5,
            "strengthTemporalSmoothing": 0.3,
            "strengthL1regularization": 0.5,
        }
    }
    (root / "bs_skin_config.json").write_text(json.dumps(bs_config), encoding="utf-8")
    (root / "network.onnx").write_bytes(b"fake-onnx")

    shapes_matrix_skin = np.zeros((_NUM_SHAPES, _NUM_VERTS, 3), dtype=np.float32)
    shapes_mean_skin = np.zeros((_NUM_VERTS, 3), dtype=np.float32)
    np.savez(
        root / "model_data.npz",
        shapes_matrix_skin=shapes_matrix_skin,
        shapes_mean_skin=shapes_mean_skin,
    )

    pose_names = np.array(["neutral"] + [f"pose_{i}" for i in range(_NUM_POSES)])
    payload: dict[str, Any] = {
        "neutral": np.zeros((_NUM_VERTS, 3), dtype=np.float32),
        "poseNames": pose_names,
    }
    for name in pose_names[1:]:
        payload[str(name)] = np.zeros((_NUM_VERTS, 3), dtype=np.float32)
        payload[str(name)][0, 0] = 1.0
    np.savez(root / "bs_skin.npz", **payload)
    return root


class _FakeOrtSession:
    def __init__(self) -> None:
        self._inputs = [MagicMock(name="input"), MagicMock(name="emotion")]
        self._inputs[0].name = "input"
        self._inputs[1].name = "emotion"
        self._outputs = [MagicMock(name="output")]
        self._outputs[0].name = "output"

    def get_inputs(self):
        return self._inputs

    def get_outputs(self):
        return self._outputs

    def get_providers(self):
        return ["CPUExecutionProvider"]

    def run(self, _output_names, inputs):
        audio = inputs["input"]
        assert audio.shape[-1] == _BUFFER_LEN
        coeffs = np.zeros((1, 1, 301), dtype=np.float32)
        coeffs[0, 0, :_NUM_SHAPES] = 0.1
        return [coeffs]


class NvidiaAudio2Face3DModelTester:
    def __init__(
        self,
        parent: unittest.TestCase,
        buffer_len: int = _BUFFER_LEN,
        num_poses: int = _NUM_POSES,
        sample_rate: int = 16000,
        output_fps: int = 25,
    ) -> None:
        self.parent = parent
        self.buffer_len = buffer_len
        self.num_poses = num_poses
        self.sample_rate = sample_rate
        self.output_fps = output_fps

    def prepare_config_and_inputs(self) -> tuple[dict[str, Any], dict[str, Any]]:
        config = {
            "model_folder": "/tmp/fake-a2f",
            "use_cuda": False,
            "output_fps": self.output_fps,
        }
        inputs_dict = {
            "input": np.zeros(self.buffer_len, dtype=np.float32),
        }
        return config, inputs_dict

    def get_fake_engine(self) -> MagicMock:
        engine = MagicMock(spec=A2FEngine)
        engine.sample_rate = self.sample_rate
        engine.buffer_len = self.buffer_len
        engine.audio_params = {
            "buffer_len": self.buffer_len,
            "samplerate": self.sample_rate,
        }
        engine.get_arkit_pose_names.return_value = [f"pose_{i}" for i in range(self.num_poses)]

        def _process(audio_chunk, session_idx=0, return_debug_info=False):
            weights = np.linspace(0.0, 1.0, self.num_poses, dtype=np.float32)
            out: dict[str, Any] = {"arkit_weights": weights}
            if return_debug_info:
                out["timing"] = {"onnx_inference": 1.0}
                out["vertices"] = np.zeros((4, 3), dtype=np.float32)
            return out

        engine.process_audio_chunk.side_effect = _process
        return engine

    def get_model(self, **overrides: Any) -> NvidiaAudio2Face3D:
        config, _ = self.prepare_config_and_inputs()
        config.update(overrides)
        return NvidiaAudio2Face3D(
            self.get_fake_engine(),
            output_fps=config["output_fps"],
        )

    def create_and_check_from_pretrained(self, config: dict[str, Any]) -> NvidiaAudio2Face3D:
        fake_engine = self.get_fake_engine()
        with patch(
            "respeak.models.nvidia_audio2face_3d.modeling_nvidia_audio2face_3d.A2FEngine",
            return_value=fake_engine,
        ) as mock_cls:
            model = NvidiaAudio2Face3D.from_pretrained(
                config["model_folder"],
                use_cuda=config["use_cuda"],
                output_fps=config["output_fps"],
            )
            mock_cls.assert_called_once()
        self.parent.assertIsInstance(model, NvidiaAudio2Face3D)
        self.parent.assertEqual(model.output_fps, config["output_fps"])
        return model

    def create_and_check_generate(
        self,
        model: NvidiaAudio2Face3D,
        inputs_dict: dict[str, Any],
    ) -> np.ndarray:
        weights = model.generate(**inputs_dict, return_audio=False)
        self.parent.assertIsInstance(weights, np.ndarray)
        self.parent.assertEqual(weights.shape, (self.num_poses,))
        model._engine.process_audio_chunk.assert_called_once()
        return weights


class NvidiaAudio2Face3DModelTest(ModelTesterMixin, unittest.TestCase):
    all_model_classes = (NvidiaAudio2Face3DModel,)

    def setUp(self) -> None:
        self.model_tester = NvidiaAudio2Face3DModelTester(self)

    def test_config_and_inputs(self):
        config, inputs_dict = self.model_tester.prepare_config_and_inputs()
        self.assertEqual(config["output_fps"], 25)
        self.assertEqual(inputs_dict["input"].dtype, np.float32)
        self.assertEqual(len(inputs_dict["input"]), _BUFFER_LEN)

    def test_from_pretrained(self):
        config, _ = self.model_tester.prepare_config_and_inputs()
        self.model_tester.create_and_check_from_pretrained(config)

    def test_generate(self):
        _, inputs_dict = self.model_tester.prepare_config_and_inputs()
        model = self.model_tester.get_model()
        self.model_tester.create_and_check_generate(model, inputs_dict)

    def test_generate_return_dict(self):
        model = self.model_tester.get_model()
        out = model.generate(np.zeros(_BUFFER_LEN, dtype=np.float32), return_dict=True)
        self.assertIn("arkit_weights", out)
        self.assertIn("audio", out)
        self.assertIn("pose_names", out)
        self.assertEqual(len(out["pose_names"]), _NUM_POSES)
        self.assertEqual(out["audio"].dtype, np.int16)
        self.assertEqual(out["sample_rate"], 16000)
        self.assertEqual(out["audio"].shape[0], model.step_samples)

    def test_generate_accepts_int16_input(self):
        model = self.model_tester.get_model()
        pcm = (np.linspace(-0.5, 0.5, _BUFFER_LEN) * 32768.0).astype(np.int16)
        out = model.generate(pcm, return_dict=True)
        self.assertEqual(out["audio"].dtype, np.int16)

    def test_generate_synced_audio_stream(self):
        model = self.model_tester.get_model()
        audio = np.linspace(-0.2, 0.2, 16000, dtype=np.float32)
        frames = model.generate(audio, stream=True, is_final=True)
        self.assertGreater(len(frames), 0)
        self.assertIsInstance(frames[0], dict)
        self.assertIn("audio", frames[0])
        self.assertIn("arkit_weights", frames[0])
        self.assertEqual(frames[0]["audio"].dtype, np.int16)
        self.assertEqual(frames[0]["audio"].shape[0], model.step_samples)
        self.assertEqual(frames[0]["arkit_weights"].shape[0], _NUM_POSES)
        synced = np.concatenate([frame["audio"] for frame in frames], axis=0)
        self.assertEqual(synced.dtype, np.int16)
        self.assertGreater(float(np.abs(synced).max()), 0.0)

    def test_response_queue_emits_100ms_frames(self):
        model = self.model_tester.get_model()
        audio = np.linspace(-0.3, 0.3, 32000, dtype=np.float32)
        model.generate(audio, stream=True, is_final=True, target_sr=16000)
        queue_bytes = 0
        while not model.Response_Queue.empty():
            chunk = model.Response_Queue.get_nowait()
            self.assertIsInstance(chunk, bytes)
            self.assertEqual(len(chunk), 3200)  # 100 ms @ 16 kHz int16
            queue_bytes += len(chunk)
        self.assertGreater(queue_bytes, 0)

    def test_generate_stream_return_dict(self):
        model = self.model_tester.get_model()
        audio = np.zeros(8000, dtype=np.float32)
        out = model.generate(audio, stream=True, is_final=True, return_dict=True)
        self.assertIn("frames", out)
        self.assertIn("audio", out)
        self.assertIn("arkit_weights", out)
        self.assertEqual(out["sample_rate"], 16000)
        self.assertEqual(out["audio"].dtype, np.int16)
        self.assertEqual(out["audio"].shape[0], len(out["frames"]) * model.step_samples)

    def test_generate_pad_to(self):
        model = self.model_tester.get_model()
        out = model.generate(np.zeros(_BUFFER_LEN, dtype=np.float32), pad_to=61)
        self.assertIsInstance(out, dict)
        self.assertEqual(out["arkit_weights"].shape, (61,))
        self.assertTrue(np.all(out["arkit_weights"][51:] == 0.0))

    def test_generate_stream(self):
        model = self.model_tester.get_model()
        audio = np.zeros(16000, dtype=np.float32)
        frames = model.generate(audio, stream=True, is_final=True, return_audio=False)
        self.assertIsInstance(frames, list)
        self.assertGreater(len(frames), 0)
        self.assertEqual(frames[0].shape, (_NUM_POSES,))
        model._engine.reset_temporal_state.assert_called()

    def test_pad_weights_static(self):
        weights = np.ones(51, dtype=np.float32)
        padded = NvidiaAudio2Face3D.pad_weights(weights, 61)
        self.assertEqual(padded.shape, (61,))
        self.assertEqual(float(padded[0]), 1.0)
        self.assertEqual(float(padded[60]), 0.0)

    def test_reset(self):
        model = self.model_tester.get_model()
        model.reset()
        model._engine.reset_temporal_state.assert_called_once()


class A2FEngineTinyModelTest(unittest.TestCase):
    """Exercise A2FEngine against a tiny fake model + mocked ONNX Runtime."""

    def test_process_audio_chunk_with_fake_onnx(self):
        with tempfile.TemporaryDirectory() as tmp:
            model_dir = _write_tiny_model_folder(Path(tmp) / "a2f")

            fake_ort = MagicMock()
            fake_ort.SessionOptions.return_value = MagicMock()
            fake_ort.GraphOptimizationLevel.ORT_ENABLE_ALL = 99
            fake_ort.get_available_providers.return_value = ["CPUExecutionProvider"]
            fake_ort.InferenceSession.return_value = _FakeOrtSession()

            with patch.dict("sys.modules", {"onnxruntime": fake_ort}):
                engine = A2FEngine(
                    model_folder=model_dir,
                    use_cuda=False,
                    num_streams=1,
                    enable_temporal_smoothing=True,
                )
                audio = np.zeros(engine.buffer_len, dtype=np.float32)
                result = engine.process_audio_chunk(audio, return_debug_info=True)

            self.assertIn("arkit_weights", result)
            self.assertEqual(result["arkit_weights"].shape, (_NUM_POSES,))
            self.assertTrue(np.all(result["arkit_weights"] >= 0.0))
            self.assertTrue(np.all(result["arkit_weights"] <= 1.0))
            self.assertEqual(len(engine.get_arkit_pose_names()), _NUM_POSES)
            self.assertIn("timing", result)


class Ue5BlendshapeRendererTest(unittest.TestCase):
    def test_pack_message_prefix(self):
        payload = b"hello"
        packed = Ue5BlendshapeRenderer.pack_message(payload)
        self.assertEqual(len(packed), 8 + len(payload))
        self.assertEqual(int.from_bytes(packed[:8], "big"), len(payload))

    def test_to_55_weights_from_51(self):
        weights = np.ones(51, dtype=np.float32)
        out = Ue5BlendshapeRenderer.to_55_weights(
            Ue5BlendshapeRenderer.to_61_weights(weights)
        )
        self.assertEqual(out.shape, (55,))
        self.assertEqual(float(out[0]), 1.0)

    def test_build_frame_payload_json(self):
        renderer = Ue5BlendshapeRenderer(output_fps=25)
        payload = renderer.build_frame_payload(np.ones(51, dtype=np.float32))
        data = json.loads(payload)
        self.assertIn("Audio2Face", data)
        self.assertEqual(len(data["Audio2Face"]["Facial"]["Names"]), 55)

    def test_stream_frames_with_fake_socket(self):
        class FakeSocket:
            def __init__(self) -> None:
                self.sent: list[bytes] = []

            def sendall(self, data: bytes) -> None:
                self.sent.append(data)

            def close(self) -> None:
                return None

        renderer = Ue5BlendshapeRenderer(output_fps=25)
        renderer._socket = FakeSocket()
        sent = renderer.stream_frames(
            [{"arkit_weights": np.zeros(51, dtype=np.float32)}],
            realtime=False,
        )
        self.assertEqual(sent, 1)
        assert renderer._socket is not None
        self.assertEqual(len(renderer._socket.sent), 2)


if __name__ == "__main__":
    unittest.main()
