"""Testing suite for SileroVad (transformers-style)."""

from __future__ import annotations

import sys
import types
import unittest
from typing import Any
from unittest.mock import MagicMock, patch

import numpy as np

from respeak.models.silerovad import SileroVad
from tests.test_modeling_common import ModelTesterMixin


def _patch_silero_vad(fake_backend: MagicMock):
    """Install a stub ``silero_vad`` package so tests do not need real weights."""
    fake_silero_vad = types.ModuleType("silero_vad")
    fake_silero_vad.load_silero_vad = MagicMock(return_value=fake_backend)
    return patch.dict(sys.modules, {"silero_vad": fake_silero_vad}), fake_silero_vad


class _FakeTensorScalar:
    def __init__(self, value: float) -> None:
        self.value = value

    def item(self) -> float:
        return self.value


class SileroVadModelTester:
    """Builds tiny dummy inputs / a fake Silero backend for unit tests."""

    def __init__(
        self,
        parent: unittest.TestCase,
        sampling_rate: int = 16000,
        chunk_samples: int = 512,
        fake_speech_prob: float = 0.73,
    ) -> None:
        self.parent = parent
        self.sampling_rate = sampling_rate
        self.chunk_samples = chunk_samples
        self.fake_speech_prob = fake_speech_prob

    def prepare_config_and_inputs(self) -> tuple[dict[str, Any], dict[str, Any]]:
        config = {
            "onnx": False,
            "opset_version": 16,
            "sampling_rate": self.sampling_rate,
        }
        inputs_dict = {
            "input": np.zeros(self.chunk_samples, dtype=np.float32),
        }
        return config, inputs_dict

    def get_fake_backend(self) -> MagicMock:
        return MagicMock(return_value=_FakeTensorScalar(self.fake_speech_prob))

    def get_model(self, **overrides: Any) -> SileroVad:
        config, _ = self.prepare_config_and_inputs()
        config.update(overrides)
        return SileroVad(
            self.get_fake_backend(),
            sampling_rate=config["sampling_rate"],
        )

    def create_and_check_from_pretrained(self, config: dict[str, Any]) -> SileroVad:
        fake_backend = self.get_fake_backend()
        ctx, fake_silero_vad = _patch_silero_vad(fake_backend)
        with ctx:
            model = SileroVad.from_pretrained(
                onnx=config["onnx"],
                opset_version=config["opset_version"],
                sampling_rate=config["sampling_rate"],
            )
            fake_silero_vad.load_silero_vad.assert_called_once_with(
                onnx=config["onnx"],
                opset_version=config["opset_version"],
            )
        self.parent.assertIsInstance(model, SileroVad)
        self.parent.assertEqual(model.sampling_rate, config["sampling_rate"])
        return model

    def create_and_check_generate(
        self,
        model: SileroVad,
        inputs_dict: dict[str, Any],
    ) -> float:
        speech_prob = model.generate(**inputs_dict)
        self.parent.assertIsInstance(speech_prob, float)
        self.parent.assertEqual(speech_prob, self.fake_speech_prob)

        model._backend.assert_called_once()
        call_args = model._backend.call_args.args
        self.parent.assertEqual(call_args[1], model.sampling_rate)
        self.parent.assertEqual(len(call_args[0]), len(inputs_dict["input"]))
        return speech_prob

    def create_and_check_generate_override_sampling_rate(
        self,
        model: SileroVad,
        inputs_dict: dict[str, Any],
    ) -> None:
        model.generate(**inputs_dict, sampling_rate=8000)
        model._backend.assert_called_once()
        self.parent.assertEqual(model._backend.call_args.args[1], 8000)


class SileroVadModelTest(ModelTesterMixin, unittest.TestCase):
    all_model_classes = (SileroVad,)

    def setUp(self) -> None:
        self.model_tester = SileroVadModelTester(self)

    def test_config_and_inputs(self):
        config, inputs_dict = self.model_tester.prepare_config_and_inputs()
        self.assertEqual(config["sampling_rate"], 16000)
        self.assertIsInstance(inputs_dict["input"], np.ndarray)
        self.assertEqual(inputs_dict["input"].dtype, np.float32)

    def test_from_pretrained(self):
        config, _ = self.model_tester.prepare_config_and_inputs()
        self.model_tester.create_and_check_from_pretrained(config)

    def test_generate(self):
        _, inputs_dict = self.model_tester.prepare_config_and_inputs()
        model = self.model_tester.get_model()
        self.model_tester.create_and_check_generate(model, inputs_dict)

    def test_generate_override_sampling_rate(self):
        _, inputs_dict = self.model_tester.prepare_config_and_inputs()
        model = self.model_tester.get_model()
        self.model_tester.create_and_check_generate_override_sampling_rate(model, inputs_dict)

    def test_generate_numpy_scalar(self):
        model = SileroVad(MagicMock(return_value=np.array(0.42)), sampling_rate=16000)
        speech_prob = model.generate(np.zeros(512, dtype=np.float32))
        self.assertEqual(speech_prob, 0.42)

    def test_generate_converts_numpy_to_torch_tensor(self):
        fake_torch = types.ModuleType("torch")
        fake_torch.from_numpy = MagicMock(side_effect=lambda array: ("tensor", array))
        backend = MagicMock(return_value=_FakeTensorScalar(0.9))
        model = SileroVad(backend, sampling_rate=16000)
        audio = np.zeros(512, dtype=np.float64)

        with patch.dict(sys.modules, {"torch": fake_torch}):
            speech_prob = model.generate(audio)

        self.assertEqual(speech_prob, 0.9)
        fake_torch.from_numpy.assert_called_once()
        converted = fake_torch.from_numpy.call_args.args[0]
        self.assertEqual(converted.dtype, np.float32)
        backend.assert_called_once_with(("tensor", converted), 16000)


if __name__ == "__main__":
    unittest.main()
