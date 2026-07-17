"""Testing suite for Qwen3LLM (transformers-style)."""

from __future__ import annotations

import queue
import sys
import types
import unittest
from typing import Any
from unittest.mock import MagicMock, patch

from respeak.models.qwen3_llm import Qwen3LLM, build_prompt
from tests.test_modeling_common import ModelTesterMixin


class FakeBatch(dict):
    def __init__(self) -> None:
        super().__init__({"input_ids": [1, 2, 3]})
        self.device = None

    def to(self, device: str) -> "FakeBatch":
        self.device = device
        return self


class FakeTokenizer:
    def __init__(self) -> None:
        self.chat_messages: list[dict[str, str]] | None = None
        self.chat_kwargs: dict[str, Any] | None = None
        self.batch = FakeBatch()

    def apply_chat_template(self, messages: list[dict[str, str]], **kwargs: Any) -> str:
        self.chat_messages = messages
        self.chat_kwargs = kwargs
        return "rendered prompt"

    def __call__(self, texts: list[str], return_tensors: str) -> FakeBatch:
        assert texts == ["rendered prompt"]
        assert return_tensors == "pt"
        return self.batch


class FakeStreamer:
    chunks: list[str] = []

    def __init__(self, tokenizer: Any, skip_prompt: bool, skip_special_tokens: bool) -> None:
        self.tokenizer = tokenizer
        self.skip_prompt = skip_prompt
        self.skip_special_tokens = skip_special_tokens

    def __iter__(self):
        return iter(type(self).chunks)


class FakeBackend:
    device = "cpu"

    def __init__(self) -> None:
        self.generate_kwargs: dict[str, Any] | None = None

    def generate(self, **kwargs: Any) -> None:
        self.generate_kwargs = kwargs


class Qwen3LLMModelTester:
    def __init__(self, parent: unittest.TestCase) -> None:
        self.parent = parent

    def prepare_config_and_inputs(self) -> tuple[dict[str, Any], dict[str, Any]]:
        return {
            "model_path": "dummy/llm",
            "max_new_tokens": 32,
            "sentence_min_chars": 4,
            "sleep_seconds": 0,
        }, {"input": "你好"}

    def prepare_config_and_inputs_for_common(self) -> tuple[dict[str, Any], dict[str, Any]]:
        return self.prepare_config_and_inputs()

    def get_model(self, **overrides: Any) -> Qwen3LLM:
        config, _ = self.prepare_config_and_inputs()
        config.update(overrides)
        return Qwen3LLM(
            FakeBackend(),
            FakeTokenizer(),
            history=[{"role": "system", "content": "你是助手"}],
            max_new_tokens=config["max_new_tokens"],
            sentence_min_chars=config["sentence_min_chars"],
            sleep_seconds=config["sleep_seconds"],
            streamer_cls=FakeStreamer,
        )

    def create_and_check_from_pretrained(self, config: dict[str, Any]) -> Qwen3LLM:
        fake_tokenizer_cls = MagicMock()
        fake_model_cls = MagicMock()
        tokenizer = FakeTokenizer()
        backend = FakeBackend()
        fake_tokenizer_cls.from_pretrained.return_value = tokenizer
        fake_model_cls.from_pretrained.return_value = backend

        model = Qwen3LLM.from_pretrained(
            config["model_path"],
            tokenizer_cls=fake_tokenizer_cls,
            model_cls=fake_model_cls,
            streamer_cls=FakeStreamer,
            persona_setting="人设",
            strategy_prompt="策略",
            max_new_tokens=config["max_new_tokens"],
            sentence_min_chars=config["sentence_min_chars"],
            sleep_seconds=config["sleep_seconds"],
        )

        fake_tokenizer_cls.from_pretrained.assert_called_once_with(config["model_path"])
        fake_model_cls.from_pretrained.assert_called_once()
        self.parent.assertIsInstance(model, Qwen3LLM)
        self.parent.assertEqual(model.history[0]["role"], "system")
        self.parent.assertIn("人设", model.history[0]["content"])
        self.parent.assertIn("策略", model.history[0]["content"])
        return model

    def create_and_check_generate(self, model: Qwen3LLM, inputs_dict: dict[str, Any]) -> str:
        FakeStreamer.chunks = ["这是第一句。", "第二句", "还没完", "！"]
        text = model.generate(**inputs_dict, stream=False)

        self.parent.assertEqual(text, "这是第一句。第二句还没完！")
        self.parent.assertEqual(model.history[-2], {"role": "user", "content": inputs_dict["input"]})
        self.parent.assertEqual(model.history[-1], {"role": "assistant", "content": text})
        self.parent.assertEqual(model._tokenizer.chat_messages[-1]["content"], inputs_dict["input"])
        self.parent.assertFalse(model._tokenizer.chat_kwargs["enable_thinking"])
        self.parent.assertEqual(model._backend.generate_kwargs["max_new_tokens"], model.max_new_tokens)
        return text


class Qwen3LLMTest(ModelTesterMixin, unittest.TestCase):
    all_model_classes = (Qwen3LLM,)

    def setUp(self) -> None:
        self.model_tester = Qwen3LLMModelTester(self)

    def test_from_pretrained(self):
        config, _ = self.model_tester.prepare_config_and_inputs()
        self.model_tester.create_and_check_from_pretrained(config)

    def test_generate(self):
        _, inputs_dict = self.model_tester.prepare_config_and_inputs()
        model = self.model_tester.get_model()
        self.model_tester.create_and_check_generate(model, inputs_dict)

    def test_stream_generate_emits_queues(self):
        model = self.model_tester.get_model()
        FakeStreamer.chunks = ["第一句话。", "第二句话。"]
        output_queue: queue.Queue[str] = queue.Queue()
        event_queue: queue.Queue[dict[str, Any]] = queue.Queue()

        chunks = list(
            model.generate(
                "你好",
                stream=True,
                output_queue=output_queue,
                event_queue=event_queue,
                return_dict=True,
            )
        )

        self.assertEqual(chunks, [
            {"text": "第一句话。", "is_final": False},
            {"text": "第二句话。", "is_final": False},
        ])
        self.assertEqual(output_queue.get_nowait(), "第一句话。")
        self.assertEqual(output_queue.get_nowait(), "第二句话。")
        self.assertEqual(event_queue.get_nowait()["text"], "第一句话。")
        self.assertEqual(event_queue.get_nowait()["text"], "第二句话。")
        self.assertEqual(event_queue.get_nowait(), {"type": "llm", "text": "", "is_final": True})

    def test_build_prompt_from_file(self):
        with patch("pathlib.Path.read_text", return_value="基础提示") as read_text:
            prompt = build_prompt("base.md", persona_setting="人设", strategy_prompt="策略")

        read_text.assert_called_once_with(encoding="utf-8")
        self.assertEqual(prompt, "基础提示\n\n人设\n\n策略")

    def test_from_pretrained_uses_modelscope_when_classes_not_injected(self):
        fake_modelscope = types.ModuleType("modelscope")
        fake_modelscope.AutoTokenizer = MagicMock()
        fake_modelscope.AutoModelForCausalLM = MagicMock()
        fake_modelscope.AutoTokenizer.from_pretrained.return_value = FakeTokenizer()
        fake_modelscope.AutoModelForCausalLM.from_pretrained.return_value = FakeBackend()

        with patch.dict(sys.modules, {"modelscope": fake_modelscope}):
            model = Qwen3LLM.from_pretrained("dummy/llm", streamer_cls=FakeStreamer, sleep_seconds=0)

        self.assertIsInstance(model, Qwen3LLM)
        fake_modelscope.AutoTokenizer.from_pretrained.assert_called_once_with("dummy/llm")
        fake_modelscope.AutoModelForCausalLM.from_pretrained.assert_called_once()


if __name__ == "__main__":
    unittest.main()
