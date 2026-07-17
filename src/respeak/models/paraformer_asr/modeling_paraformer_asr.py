"""Streaming Paraformer ASR (FunASR)."""

from __future__ import annotations

from typing import Any

import numpy as np

from respeak.base import BaseModel

# Default streaming chunk config: [0, 10, 5] ≈ 600ms; [0, 8, 4] ≈ 480ms
_DEFAULT_CHUNK_SIZE = [0, 10, 5]
_DEFAULT_ENCODER_CHUNK_LOOK_BACK = 4
_DEFAULT_DECODER_CHUNK_LOOK_BACK = 1


class StreamingParaformerAsr(BaseModel):
    """Streaming Paraformer ASR with optional punctuation model."""

    def __init__(
        self,
        backend: Any,
        *,
        chunk_size: list[int] | None = None,
        encoder_chunk_look_back: int = _DEFAULT_ENCODER_CHUNK_LOOK_BACK,
        decoder_chunk_look_back: int = _DEFAULT_DECODER_CHUNK_LOOK_BACK,
    ) -> None:
        self._backend = backend
        self.chunk_size = list(chunk_size or _DEFAULT_CHUNK_SIZE)
        self.encoder_chunk_look_back = encoder_chunk_look_back
        self.decoder_chunk_look_back = decoder_chunk_look_back

    @classmethod
    def from_pretrained(
        cls,
        asr: str,
        punc_model: str | None = None,
        *,
        chunk_size: list[int] | None = None,
        encoder_chunk_look_back: int = _DEFAULT_ENCODER_CHUNK_LOOK_BACK,
        decoder_chunk_look_back: int = _DEFAULT_DECODER_CHUNK_LOOK_BACK,
        disable_update: bool = True,
        **kwargs: Any,
    ) -> StreamingParaformerAsr:
        """Load ASR (and optional punctuation) from a model id or local path.

        Extra kwargs are forwarded to ``funasr.AutoModel``.
        """
        from funasr import AutoModel

        backend_kwargs: dict[str, Any] = {
            "model": asr,
            "disable_update": disable_update,
            **kwargs,
        }
        if punc_model is not None:
            backend_kwargs["punc_model"] = punc_model

        backend = AutoModel(**backend_kwargs)
        return cls(
            backend,
            chunk_size=chunk_size,
            encoder_chunk_look_back=encoder_chunk_look_back,
            decoder_chunk_look_back=decoder_chunk_look_back,
        )

    def generate(
        self,
        input: np.ndarray | str,
        *,
        is_final: bool = False,
        cache: dict | None = None,
        chunk_size: list[int] | None = None,
        encoder_chunk_look_back: int | None = None,
        decoder_chunk_look_back: int | None = None,
        return_dict: bool = False,
        **kwargs: Any,
    ) -> str | list[dict[str, Any]]:
        """Run streaming ASR on an audio chunk (or path / array).

        Parameters passed here override the instance defaults for this call only.
        """
        if cache is None:
            cache = {}

        res = self._backend.generate(
            input=input,
            cache=cache,
            is_final=is_final,
            chunk_size=chunk_size or self.chunk_size,
            encoder_chunk_look_back=(
                self.encoder_chunk_look_back
                if encoder_chunk_look_back is None
                else encoder_chunk_look_back
            ),
            decoder_chunk_look_back=(
                self.decoder_chunk_look_back
                if decoder_chunk_look_back is None
                else decoder_chunk_look_back
            ),
            **kwargs,
        )
        if return_dict:
            return res
        return res[0]["text"] if res else ""
