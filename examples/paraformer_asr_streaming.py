#!/usr/bin/env python3
"""Streaming Paraformer ASR example.

Usage:
    python examples/paraformer_asr_streaming.py \\
        --asr /home/ubuntu/workspace/codebases/openFaceBot/models_hub/paraformer-zh-streaming \\
        --punc /home/ubuntu/workspace/codebases/openFaceBot/models_hub/punc_ct-transformer_cn-en-common-vocab471067-large \\
        --audio src/respeak/models/cosyvoice3_tts/data/man.wav

Requires: pip install -e .
"""

from __future__ import annotations

import argparse
import wave
from pathlib import Path

import numpy as np

from respeak import StreamingParaformerAsr


def load_wav_mono(path: str, target_sr: int = 16000) -> np.ndarray:
    """Load WAV as float32 mono @ target_sr (simple linear resample if needed)."""
    with wave.open(path, "rb") as wf:
        channels = wf.getnchannels()
        sample_rate = wf.getframerate()
        sample_width = wf.getsampwidth()
        n_frames = wf.getnframes()
        raw = wf.readframes(n_frames)

    if sample_width == 2:
        audio = np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32768.0
    elif sample_width == 4:
        audio = np.frombuffer(raw, dtype=np.int32).astype(np.float32) / 2147483648.0
    else:
        raise ValueError(f"unsupported sample width: {sample_width}")

    if channels > 1:
        audio = audio.reshape(-1, channels).mean(axis=1)

    if sample_rate != target_sr:
        duration = audio.shape[0] / sample_rate
        target_len = int(duration * target_sr)
        audio = np.interp(
            np.linspace(0.0, duration, target_len, endpoint=False),
            np.linspace(0.0, duration, audio.shape[0], endpoint=False),
            audio,
        ).astype(np.float32)

    return audio


def stream_asr(
    model: StreamingParaformerAsr,
    audio: np.ndarray,
    *,
    chunk_ms: int = 600,
    sample_rate: int = 16000,
) -> str:
    """Simulate streaming ASR by feeding fixed-size chunks."""
    chunk_samples = int(sample_rate * chunk_ms / 1000)
    cache: dict = {}
    text = ""
    final_text = ""
    for start in range(0, len(audio), chunk_samples):
        chunk = audio[start : start + chunk_samples]
        is_final = start + chunk_samples >= len(audio)
        text = model.generate(input=chunk, is_final=is_final, cache=cache)
        final_text += text
        print(f"[partial] is_final={is_final}: {final_text}")

    return final_text


def main() -> None:
    parser = argparse.ArgumentParser(description="Streaming Paraformer ASR demo")
    parser.add_argument("--asr", required=True, help="ASR model path or model id")
    parser.add_argument("--punc", default=None, help="Punctuation model path or model id")
    parser.add_argument("--audio", required=True, help="Input wav file")
    parser.add_argument("--chunk-ms", type=int, default=600, help="Streaming chunk size in ms")
    args = parser.parse_args()

    audio_path = Path(args.audio)
    if not audio_path.is_file():
        raise FileNotFoundError(audio_path)

    print("Loading ASR model...")
    model = StreamingParaformerAsr.from_pretrained(
        asr=args.asr,
        punc_model=args.punc,
    )

    print(f"Reading audio: {audio_path}")
    audio = load_wav_mono(str(audio_path))

    print("Running streaming ASR...")
    final_text = stream_asr(model, audio, chunk_ms=args.chunk_ms)
    print(f"\nFinal transcript:\n{final_text}")


if __name__ == "__main__":
    main()
