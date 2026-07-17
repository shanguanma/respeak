"""Testing suite for CosyVoice3Tts (transformers-style)."""

from __future__ import annotations

import sys
import types
import unittest
from typing import Any
from unittest.mock import MagicMock, patch

import numpy as np

from respeak.models.cosyvoice3_tts import CosyVoice3Tts
from tests.test_modeling_common import ModelTesterMixin


def _install_fake_vllm_stack(fake_backend: MagicMock):
    """Stub vllm + vendored cosyvoice entrypoints (no GPU / weights)."""
    fake_vllm = types.ModuleType("vllm")
    fake_vllm.ModelRegistry = MagicMock()

    fake_cv2 = types.ModuleType("cosyvoice.vllm.cosyvoice2")
    fake_cv2.CosyVoice2ForCausalLM = object

    fake_cli = types.ModuleType("cosyvoice.cli.cosyvoice")
    fake_cli.AutoModel = MagicMock(return_value=fake_backend)

    modules = {
        "vllm": fake_vllm,
        "cosyvoice": types.ModuleType("cosyvoice"),
        "cosyvoice.vllm": types.ModuleType("cosyvoice.vllm"),
        "cosyvoice.vllm.cosyvoice2": fake_cv2,
        "cosyvoice.cli": types.ModuleType("cosyvoice.cli"),
        "cosyvoice.cli.cosyvoice": fake_cli,
    }
    return patch.dict(sys.modules, modules), fake_vllm, fake_cli


class CosyVoice3TtsModelTester:
    """Builds dummy configs / a fake CosyVoice backend for unit tests."""

    def __init__(
        self,
        parent: unittest.TestCase,
        model_dir: str = "dummy/tts",
        sample_rate: int = 24000,
        chunk_samples: int = 4800,
        zero_shot_spk_id: str = "my_spk",
        fake_text: str = "你好，欢迎使用本系统。",
    ) -> None:
        self.parent = parent
        self.model_dir = model_dir
        self.sample_rate = sample_rate
        self.chunk_samples = chunk_samples
        self.zero_shot_spk_id = zero_shot_spk_id
        self.fake_text = fake_text

    def prepare_config_and_inputs(self) -> tuple[dict[str, Any], dict[str, Any]]:
        config = {
            "model_dir": self.model_dir,
            "load_vllm": True,
            "load_trt": False,
            "fp16": True,
            "prompt_text": "近日，除了葛洲坝股价下跌外，其余三家均有不同程度的上涨。",
            "prompt_wav": "dummy/prompt.wav",
            "zero_shot_spk_id": self.zero_shot_spk_id,
        }
        inputs_dict = {"input": self.fake_text, "stream": True}
        return config, inputs_dict

    def prepare_config_and_inputs_for_common(self) -> tuple[dict[str, Any], dict[str, Any]]:
        return self.prepare_config_and_inputs()

    def _fake_chunk(self, scale: float = 1.0) -> dict[str, Any]:
        speech = MagicMock()
        speech.detach.return_value.cpu.return_value.numpy.return_value = (
            np.ones(self.chunk_samples, dtype=np.float32) * scale
        )
        return {"tts_speech": [speech]}

    def get_fake_backend(self, n_chunks: int = 2) -> MagicMock:
        backend = MagicMock()
        backend.sample_rate = self.sample_rate
        backend.add_zero_shot_spk.return_value = True
        backend.inference_zero_shot.return_value = iter(
            [self._fake_chunk(i + 1) for i in range(n_chunks)]
        )
        return backend

    def get_model(self, **overrides: Any) -> CosyVoice3Tts:
        config, _ = self.prepare_config_and_inputs()
        config.update(overrides)
        return CosyVoice3Tts(
            self.get_fake_backend(),
            zero_shot_spk_id=config["zero_shot_spk_id"],
            sample_rate=self.sample_rate,
        )

    def create_and_check_from_pretrained(self, config: dict[str, Any]) -> CosyVoice3Tts:
        fake_backend = self.get_fake_backend()
        ctx, fake_vllm, fake_cli = _install_fake_vllm_stack(fake_backend)
        import respeak.models.cosyvoice3_tts.modeling_cosyvoice3_tts as mod

        mod._VLLM_REGISTERED = False
        mod._PATHS_READY = False
        with ctx:
            model = CosyVoice3Tts.from_pretrained(
                config["model_dir"],
                load_vllm=config["load_vllm"],
                load_trt=config["load_trt"],
                fp16=config["fp16"],
                prompt_text=config["prompt_text"],
                prompt_wav=config["prompt_wav"],
                zero_shot_spk_id=config["zero_shot_spk_id"],
            )
            fake_vllm.ModelRegistry.register_model.assert_called_once()
            fake_cli.AutoModel.assert_called_once()
            call_kwargs = fake_cli.AutoModel.call_args.kwargs
            self.parent.assertEqual(call_kwargs["model_dir"], config["model_dir"])
            self.parent.assertTrue(call_kwargs["load_vllm"])
            self.parent.assertFalse(call_kwargs["load_trt"])
            fake_backend.add_zero_shot_spk.assert_called_once()

        self.parent.assertIsInstance(model, CosyVoice3Tts)
        self.parent.assertEqual(model.zero_shot_spk_id, config["zero_shot_spk_id"])
        self.parent.assertEqual(model.sample_rate, self.sample_rate)
        return model

    def create_and_check_generate_stream(
        self,
        model: CosyVoice3Tts,
        inputs_dict: dict[str, Any],
    ) -> list[np.ndarray]:
        chunks = list(model.generate(**inputs_dict))
        self.parent.assertGreaterEqual(len(chunks), 1)
        for chunk in chunks:
            self.parent.assertIsInstance(chunk, np.ndarray)
            self.parent.assertEqual(chunk.dtype, np.int16)
            self.parent.assertEqual(chunk.shape[0], self.chunk_samples)

        call_kwargs = model._backend.inference_zero_shot.call_args
        self.parent.assertEqual(call_kwargs.args[0], inputs_dict["input"])
        self.parent.assertEqual(
            call_kwargs.kwargs["zero_shot_spk_id"], model.zero_shot_spk_id
        )
        self.parent.assertTrue(call_kwargs.kwargs["stream"])
        return chunks

    def create_and_check_generate_non_stream(self, model: CosyVoice3Tts, text: str) -> np.ndarray:
        model._backend.inference_zero_shot.return_value = iter(
            [self._fake_chunk(1), self._fake_chunk(2)]
        )
        audio = model.generate(text, stream=False)
        self.parent.assertIsInstance(audio, np.ndarray)
        self.parent.assertEqual(audio.dtype, np.int16)
        self.parent.assertEqual(audio.shape[0], self.chunk_samples * 2)
        return audio

    def create_and_check_generate_return_dict(self, model: CosyVoice3Tts, text: str) -> None:
        model._backend.inference_zero_shot.return_value = iter([self._fake_chunk(1)])
        out = list(model.generate(text, stream=True, return_dict=True))
        self.parent.assertIsInstance(out[0], dict)
        self.parent.assertIn("tts_speech", out[0])

    def create_and_check_add_speaker(self, model: CosyVoice3Tts) -> None:
        model.add_speaker("prompt text", "a.wav", spk_id="spk2")
        args = model._backend.add_zero_shot_spk.call_args.args
        self.parent.assertIn("prompt text", args[0])
        self.parent.assertEqual(args[1], "a.wav")
        self.parent.assertEqual(args[2], "spk2")
        self.parent.assertEqual(model.zero_shot_spk_id, "spk2")


class CosyVoice3TtsModelTest(ModelTesterMixin, unittest.TestCase):
    all_model_classes = (CosyVoice3Tts,)

    def setUp(self) -> None:
        self.model_tester = CosyVoice3TtsModelTester(self)

    def test_config_and_inputs(self):
        config, inputs_dict = self.model_tester.prepare_config_and_inputs()
        self.assertIn("model_dir", config)
        self.assertIsInstance(inputs_dict["input"], str)

    def test_from_pretrained(self):
        config, _ = self.model_tester.prepare_config_and_inputs()
        self.model_tester.create_and_check_from_pretrained(config)

    def test_from_pretrained_without_prompt(self):
        fake_backend = self.model_tester.get_fake_backend()
        ctx, _, fake_cli = _install_fake_vllm_stack(fake_backend)
        import respeak.models.cosyvoice3_tts.modeling_cosyvoice3_tts as mod

        mod._VLLM_REGISTERED = False
        mod._PATHS_READY = False
        with ctx:
            model = CosyVoice3Tts.from_pretrained("dummy/tts", load_vllm=True)
            fake_cli.AutoModel.assert_called_once()
            fake_backend.add_zero_shot_spk.assert_not_called()
        self.assertIsInstance(model, CosyVoice3Tts)

    def test_generate_stream(self):
        _, inputs_dict = self.model_tester.prepare_config_and_inputs()
        model = self.model_tester.get_model()
        self.model_tester.create_and_check_generate_stream(model, inputs_dict)

    def test_generate_non_stream(self):
        model = self.model_tester.get_model()
        self.model_tester.create_and_check_generate_non_stream(model, "测试文本")

    def test_generate_return_dict(self):
        model = self.model_tester.get_model()
        self.model_tester.create_and_check_generate_return_dict(model, "测试文本")

    def test_add_speaker(self):
        model = self.model_tester.get_model()
        self.model_tester.create_and_check_add_speaker(model)

    def test_generate_override_spk_id(self):
        model = self.model_tester.get_model()
        model._backend.inference_zero_shot.return_value = iter(
            [self.model_tester._fake_chunk(1)]
        )
        list(model.generate("hi", stream=True, zero_shot_spk_id="other"))
        self.assertEqual(
            model._backend.inference_zero_shot.call_args.kwargs["zero_shot_spk_id"],
            "other",
        )

    def test_save_spkinfo(self):
        model = self.model_tester.get_model()
        model.save_spkinfo()
        model._backend.save_spkinfo.assert_called_once()

    def test_local_vendor_layout(self):
        root = __import__(
            "respeak.models.cosyvoice3_tts.modeling_cosyvoice3_tts", fromlist=["_PKG_ROOT"]
        )._PKG_ROOT
        self.assertTrue((root / "cosyvoice" / "cli" / "cosyvoice.py").is_file())
        self.assertTrue((root / "matcha" / "utils" / "audio.py").is_file())
        self.assertTrue((root / "matcha" / "models" / "components" / "flow_matching.py").is_file())


if __name__ == "__main__":
    unittest.main()
