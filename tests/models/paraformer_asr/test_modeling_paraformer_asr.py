"""Testing suite for StreamingParaformerAsr (transformers-style)."""

from __future__ import annotations

import sys
import types
import unittest
from typing import Any
from unittest.mock import MagicMock, patch

import numpy as np

from respeak.models.paraformer_asr import StreamingParaformerAsr
from tests.test_modeling_common import ModelTesterMixin


def _patch_funasr_automodel(fake_backend: MagicMock):
    """Install a stub ``funasr`` package so tests do not need torch / real weights."""
    fake_funasr = types.ModuleType("funasr")
    fake_funasr.AutoModel = MagicMock(return_value=fake_backend)
    return patch.dict(sys.modules, {"funasr": fake_funasr}), fake_funasr


class StreamingParaformerAsrModelTester:
    """Builds tiny dummy inputs / a fake FunASR backend for unit tests."""

    def __init__(
        self,
        parent: unittest.TestCase,
        sample_rate: int = 16000,
        chunk_samples: int = 9600,  # 600ms @ 16kHz
        chunk_size: list[int] | None = None,
        encoder_chunk_look_back: int = 4,
        decoder_chunk_look_back: int = 1,
        fake_text: str = "你好世界",
    ) -> None:
        self.parent = parent
        self.sample_rate = sample_rate
        self.chunk_samples = chunk_samples
        self.chunk_size = chunk_size or [0, 10, 5]
        self.encoder_chunk_look_back = encoder_chunk_look_back
        self.decoder_chunk_look_back = decoder_chunk_look_back
        self.fake_text = fake_text

    def prepare_config_and_inputs(self) -> tuple[dict[str, Any], dict[str, Any]]:
        config = {
            "asr": "dummy/asr",
            "punc_model": "dummy/punc",
            "chunk_size": list(self.chunk_size),
            "encoder_chunk_look_back": self.encoder_chunk_look_back,
            "decoder_chunk_look_back": self.decoder_chunk_look_back,
        }
        inputs_dict = {
            "input": np.zeros(self.chunk_samples, dtype=np.float32),
            "is_final": False,
            "cache": {},
        }
        return config, inputs_dict

    def prepare_config_and_inputs_for_common(self) -> tuple[dict[str, Any], dict[str, Any]]:
        return self.prepare_config_and_inputs()

    def get_fake_backend(self) -> MagicMock:
        backend = MagicMock()
        backend.generate.return_value = [{"text": self.fake_text}]
        return backend

    def get_model(self, **overrides: Any) -> StreamingParaformerAsr:
        config, _ = self.prepare_config_and_inputs()
        config.update(overrides)
        return StreamingParaformerAsr(
            self.get_fake_backend(),
            chunk_size=config["chunk_size"],
            encoder_chunk_look_back=config["encoder_chunk_look_back"],
            decoder_chunk_look_back=config["decoder_chunk_look_back"],
        )

    def create_and_check_from_pretrained(self, config: dict[str, Any]) -> StreamingParaformerAsr:
        fake_backend = self.get_fake_backend()
        ctx, fake_funasr = _patch_funasr_automodel(fake_backend)
        with ctx:
            model = StreamingParaformerAsr.from_pretrained(
                asr=config["asr"],
                punc_model=config["punc_model"],
                chunk_size=config["chunk_size"],
                encoder_chunk_look_back=config["encoder_chunk_look_back"],
                decoder_chunk_look_back=config["decoder_chunk_look_back"],
            )
            fake_funasr.AutoModel.assert_called_once()
            call_kwargs = fake_funasr.AutoModel.call_args.kwargs
            self.parent.assertEqual(call_kwargs["model"], config["asr"])
            self.parent.assertEqual(call_kwargs["punc_model"], config["punc_model"])
            self.parent.assertTrue(call_kwargs.get("disable_update", False))

        self.parent.assertIsInstance(model, StreamingParaformerAsr)
        self.parent.assertEqual(model.chunk_size, config["chunk_size"])
        self.parent.assertEqual(model.encoder_chunk_look_back, config["encoder_chunk_look_back"])
        self.parent.assertEqual(model.decoder_chunk_look_back, config["decoder_chunk_look_back"])
        return model

    def create_and_check_generate(
        self,
        model: StreamingParaformerAsr,
        inputs_dict: dict[str, Any],
    ) -> str:
        text = model.generate(**inputs_dict)
        self.parent.assertIsInstance(text, str)
        self.parent.assertEqual(text, self.fake_text)

        backend: MagicMock = model._backend
        backend.generate.assert_called()
        call_kwargs = backend.generate.call_args.kwargs
        self.parent.assertIn("input", call_kwargs)
        self.parent.assertEqual(call_kwargs["is_final"], inputs_dict["is_final"])
        self.parent.assertIs(call_kwargs["cache"], inputs_dict["cache"])
        self.parent.assertEqual(call_kwargs["chunk_size"], model.chunk_size)
        return text

    def create_and_check_generate_override_params(
        self,
        model: StreamingParaformerAsr,
        inputs_dict: dict[str, Any],
    ) -> None:
        override_chunk = [0, 8, 4]
        model.generate(
            **inputs_dict,
            chunk_size=override_chunk,
            encoder_chunk_look_back=2,
            decoder_chunk_look_back=0,
        )
        call_kwargs = model._backend.generate.call_args.kwargs
        self.parent.assertEqual(call_kwargs["chunk_size"], override_chunk)
        self.parent.assertEqual(call_kwargs["encoder_chunk_look_back"], 2)
        self.parent.assertEqual(call_kwargs["decoder_chunk_look_back"], 0)

    def create_and_check_generate_return_dict(
        self,
        model: StreamingParaformerAsr,
        inputs_dict: dict[str, Any],
    ) -> None:
        out = model.generate(**inputs_dict, return_dict=True)
        self.parent.assertIsInstance(out, list)
        self.parent.assertEqual(out[0]["text"], self.fake_text)

    def create_and_check_streaming_cache(
        self,
        model: StreamingParaformerAsr,
        inputs_dict: dict[str, Any],
    ) -> None:
        cache: dict[str, Any] = {}
        audio = inputs_dict["input"]

        # Simulate FunASR mutating cache across chunks.
        def _fake_generate(**kwargs: Any):
            kwargs["cache"]["n"] = kwargs["cache"].get("n", 0) + 1
            return [{"text": self.fake_text if kwargs["is_final"] else "你好"}]

        model._backend.generate.side_effect = _fake_generate

        partial = model.generate(input=audio, is_final=False, cache=cache)
        final = model.generate(input=audio, is_final=True, cache=cache)

        self.parent.assertEqual(partial, "你好")
        self.parent.assertEqual(final, self.fake_text)
        self.parent.assertEqual(cache["n"], 2)


class StreamingParaformerAsrModelTest(ModelTesterMixin, unittest.TestCase):
    all_model_classes = (StreamingParaformerAsr,)

    def setUp(self) -> None:
        self.model_tester = StreamingParaformerAsrModelTester(self)

    def test_config_and_inputs(self):
        config, inputs_dict = self.model_tester.prepare_config_and_inputs()
        self.assertIn("asr", config)
        self.assertIsInstance(inputs_dict["input"], np.ndarray)
        self.assertEqual(inputs_dict["input"].dtype, np.float32)

    def test_from_pretrained(self):
        config, _ = self.model_tester.prepare_config_and_inputs()
        self.model_tester.create_and_check_from_pretrained(config)

    def test_from_pretrained_without_punc(self):
        fake_backend = self.model_tester.get_fake_backend()
        ctx, fake_funasr = _patch_funasr_automodel(fake_backend)
        with ctx:
            model = StreamingParaformerAsr.from_pretrained(asr="dummy/asr")
            self.assertNotIn("punc_model", fake_funasr.AutoModel.call_args.kwargs)
        self.assertIsInstance(model, StreamingParaformerAsr)

    def test_generate(self):
        _, inputs_dict = self.model_tester.prepare_config_and_inputs()
        model = self.model_tester.get_model()
        self.model_tester.create_and_check_generate(model, inputs_dict)

    def test_generate_override_params(self):
        _, inputs_dict = self.model_tester.prepare_config_and_inputs()
        model = self.model_tester.get_model()
        self.model_tester.create_and_check_generate_override_params(model, inputs_dict)

    def test_generate_return_dict(self):
        _, inputs_dict = self.model_tester.prepare_config_and_inputs()
        model = self.model_tester.get_model()
        self.model_tester.create_and_check_generate_return_dict(model, inputs_dict)

    def test_streaming_cache(self):
        _, inputs_dict = self.model_tester.prepare_config_and_inputs()
        model = self.model_tester.get_model()
        self.model_tester.create_and_check_streaming_cache(model, inputs_dict)

    def test_generate_empty_backend_result(self):
        model = self.model_tester.get_model()
        model._backend.generate.return_value = []
        text = model.generate(input=np.zeros(8, dtype=np.float32), is_final=True, cache={})
        self.assertEqual(text, "")


if __name__ == "__main__":
    unittest.main()
