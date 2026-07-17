"""Streaming causal LLM wrapper for respeak generation."""

from __future__ import annotations

import re
import time
from collections.abc import Callable, Iterator
from pathlib import Path
from queue import Queue
from threading import Thread
from typing import Any

from respeak.base import BaseModel

_DEFAULT_MAX_NEW_TOKENS = 2048
_DEFAULT_MAX_HISTORY_MESSAGES = 7
_SENTENCE_PATTERN = re.compile(r"[。！？.!?]")


def build_prompt(
    base_prompt_file: str | Path | None = None,
    *,
    persona_setting: str = "",
    strategy_prompt: str = "",
) -> str:
    """Build a system prompt from a base prompt plus persona / strategy text."""
    prompt = ""
    if base_prompt_file is not None:
        prompt = Path(base_prompt_file).read_text(encoding="utf-8")

    if persona_setting:
        prompt += f"\n\n{persona_setting}" if prompt else persona_setting
    if strategy_prompt:
        prompt += f"\n\n{strategy_prompt}" if prompt else strategy_prompt
    return prompt


class Qwen3LLM(BaseModel):
    """ModelScope / Transformers causal LLM with sentence-level streaming output."""

    def __init__(
        self,
        backend: Any,
        tokenizer: Any,
        *,
        history: list[dict[str, str]] | None = None,
        system_prompt: str | None = None,
        max_new_tokens: int = _DEFAULT_MAX_NEW_TOKENS,
        max_history_messages: int = _DEFAULT_MAX_HISTORY_MESSAGES,
        sentence_min_chars: int = 10,
        sleep_seconds: float = 0.02,
        streamer_cls: type | None = None,
    ) -> None:
        self._backend = backend
        self._tokenizer = tokenizer
        self.max_new_tokens = max_new_tokens
        self.max_history_messages = max_history_messages
        self.sentence_min_chars = sentence_min_chars
        self.sleep_seconds = sleep_seconds
        self._streamer_cls = streamer_cls

        if history is not None:
            self.history = list(history)
        elif system_prompt:
            self.history = [{"role": "system", "content": system_prompt}]
        else:
            self.history = []

    @classmethod
    def from_pretrained(
        cls,
        model_path: str,
        *,
        base_prompt_file: str | Path | None = None,
        persona_setting: str = "",
        strategy_prompt: str = "",
        history: list[dict[str, str]] | None = None,
        torch_dtype: str | Any = "auto",
        device_map: str | dict[str, Any] = "auto",
        max_new_tokens: int = _DEFAULT_MAX_NEW_TOKENS,
        max_history_messages: int = _DEFAULT_MAX_HISTORY_MESSAGES,
        sentence_min_chars: int = 10,
        sleep_seconds: float = 0.02,
        tokenizer_cls: Any = None,
        model_cls: Any = None,
        streamer_cls: type | None = None,
        **kwargs: Any,
    ) -> Qwen3LLM:
        """Load tokenizer and causal LM from a local path or ModelScope id.

        Extra kwargs are forwarded to ``AutoModelForCausalLM.from_pretrained``.
        """
        if tokenizer_cls is None or model_cls is None:
            from modelscope import AutoModelForCausalLM, AutoTokenizer

            tokenizer_cls = tokenizer_cls or AutoTokenizer
            model_cls = model_cls or AutoModelForCausalLM

        tokenizer = tokenizer_cls.from_pretrained(model_path)
        backend = model_cls.from_pretrained(
            model_path,
            torch_dtype=torch_dtype,
            device_map=device_map,
            **kwargs,
        )
        system_prompt = build_prompt(
            base_prompt_file,
            persona_setting=persona_setting,
            strategy_prompt=strategy_prompt,
        )
        return cls(
            backend,
            tokenizer,
            history=history,
            system_prompt=system_prompt or None,
            max_new_tokens=max_new_tokens,
            max_history_messages=max_history_messages,
            sentence_min_chars=sentence_min_chars,
            sleep_seconds=sleep_seconds,
            streamer_cls=streamer_cls,
        )

    def generate(
        self,
        input: str | list[dict[str, str]],
        *,
        stream: bool = True,
        history: list[dict[str, str]] | None = None,
        enable_thinking: bool = False,
        max_new_tokens: int | None = None,
        output_queue: Queue | None = None,
        event_queue: Queue | None = None,
        stop_checker: Callable[[], bool] | None = None,
        update_history: bool = True,
        return_dict: bool = False,
        **generation_kwargs: Any,
    ) -> Iterator[str | dict[str, Any]] | str | dict[str, Any]:
        """Generate a response.

        When ``stream=True``, returns an iterator of sentence chunks. When
        ``stream=False``, returns the full response string. ``input`` may be a
        user utterance or a full chat message list.
        """
        messages, user_text = self._prepare_messages(input, history=history)
        text_inputs = self._tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=enable_thinking,
        )
        model_inputs = self._tokenizer([text_inputs], return_tensors="pt")
        device = getattr(self._backend, "device", None)
        if device is not None and hasattr(model_inputs, "to"):
            model_inputs = model_inputs.to(device)

        streamer = self._make_streamer()
        kwargs = {
            **model_inputs,
            "max_new_tokens": max_new_tokens or self.max_new_tokens,
            "streamer": streamer,
            **generation_kwargs,
        }

        thread = Thread(target=self._backend.generate, kwargs=kwargs)
        thread.start()

        iterator = self._stream_response(
            streamer,
            thread=thread,
            messages=messages,
            user_text=user_text,
            output_queue=output_queue,
            event_queue=event_queue,
            stop_checker=stop_checker,
            update_history=update_history,
            return_dict=return_dict,
        )
        if stream:
            return iterator

        chunks = list(iterator)
        if return_dict:
            text = "".join(chunk["text"] for chunk in chunks)  # type: ignore[index]
            return {"text": text, "chunks": chunks}
        return "".join(chunks)  # type: ignore[arg-type]

    def _prepare_messages(
        self,
        input: str | list[dict[str, str]],
        *,
        history: list[dict[str, str]] | None,
    ) -> tuple[list[dict[str, str]], str | None]:
        if isinstance(input, str):
            base_history = self.history if history is None else history
            return [*base_history, {"role": "user", "content": input}], input
        return list(input), None

    def _make_streamer(self) -> Any:
        streamer_cls = self._streamer_cls
        if streamer_cls is None:
            from transformers import TextIteratorStreamer

            streamer_cls = TextIteratorStreamer
        return streamer_cls(
            self._tokenizer,
            skip_prompt=True,
            skip_special_tokens=True,
        )

    def _stream_response(
        self,
        streamer: Any,
        *,
        thread: Thread,
        messages: list[dict[str, str]],
        user_text: str | None,
        output_queue: Queue | None,
        event_queue: Queue | None,
        stop_checker: Callable[[], bool] | None,
        update_history: bool,
        return_dict: bool,
    ) -> Iterator[str | dict[str, Any]]:
        response = ""
        pending = ""
        stopped = False

        for new_text in streamer:
            if stop_checker is not None and stop_checker():
                stopped = True
                break
            if new_text is None:
                continue

            pending += new_text
            while True:
                sentence, pending = self._pop_sentence(pending, final=False)
                if sentence is None:
                    break
                response += sentence
                self._emit(sentence, output_queue=output_queue, event_queue=event_queue)
                yield {"text": sentence, "is_final": False} if return_dict else sentence

            if self.sleep_seconds:
                time.sleep(self.sleep_seconds)

        if not stopped and pending:
            response += pending
            self._emit(pending, output_queue=output_queue, event_queue=event_queue)
            yield {"text": pending, "is_final": False} if return_dict else pending

        if stopped:
            return

        thread.join()

        if event_queue is not None:
            event_queue.put({"type": "llm", "text": "", "is_final": True})
        if update_history and user_text is not None:
            self._append_history(user_text, response)

    def _pop_sentence(self, text: str, *, final: bool) -> tuple[str | None, str]:
        for match in _SENTENCE_PATTERN.finditer(text):
            if final or match.end() >= self.sentence_min_chars:
                return text[: match.end()], text[match.end() :]
        return None, text

    @staticmethod
    def _emit(
        text: str,
        *,
        output_queue: Queue | None,
        event_queue: Queue | None,
    ) -> None:
        if output_queue is not None:
            output_queue.put(text)
        if event_queue is not None:
            event_queue.put({"type": "llm", "text": text, "is_final": False})

    def _append_history(self, user_text: str, response: str) -> None:
        self.history.append({"role": "user", "content": user_text})
        self.history.append({"role": "assistant", "content": response})
        if len(self.history) <= self.max_history_messages:
            return

        if self.history and self.history[0].get("role") == "system":
            self.history = [self.history[0]] + self.history[-(self.max_history_messages - 1) :]
        else:
            self.history = self.history[-self.max_history_messages :]
