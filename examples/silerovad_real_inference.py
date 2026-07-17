#!/usr/bin/env python3
"""Silero VAD real inference example.

Usage:
    python examples/silerovad_real_inference.py \
        --audio input.wav \
        --threshold 0.5

Requires: pip install -e ".[vad]"
"""

from __future__ import annotations

import argparse
import wave
from pathlib import Path

import numpy as np

from respeak import SileroVad


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

    return audio.astype(np.float32, copy=False)


def run_vad(
    model: SileroVad,
    audio: np.ndarray,
    *,
    sample_rate: int = 16000,
    chunk_samples: int = 512,
    threshold: float = 0.5,
) -> list[tuple[float, float, float]]:
    """Run chunked VAD and return speech chunks as (start_sec, end_sec, prob)."""
    speech_chunks: list[tuple[float, float, float]] = []

    for start in range(0, len(audio), chunk_samples):
        chunk = audio[start : start + chunk_samples]
        if len(chunk) < chunk_samples:
            chunk = np.pad(chunk, (0, chunk_samples - len(chunk)))

        prob = model.generate(chunk, sampling_rate=sample_rate)
        start_sec = start / sample_rate
        end_sec = min(start + chunk_samples, len(audio)) / sample_rate
        is_speech = prob >= threshold

        print(
            f"[{start_sec:7.3f}s - {end_sec:7.3f}s] "
            f"speech_prob={prob:.4f} speech={is_speech}"
        )

        if is_speech:
            speech_chunks.append((start_sec, end_sec, prob))

    return speech_chunks


def main() -> None:
    parser = argparse.ArgumentParser(description="Silero VAD real inference demo")
    parser.add_argument("--audio", required=True, help="Input wav file")
    parser.add_argument("--threshold", type=float, default=0.8, help="Speech threshold")
    parser.add_argument("--sample-rate", type=int, default=16000, help="Target sample rate")
    parser.add_argument("--chunk-samples", type=int, default=512, help="VAD chunk size")
    parser.add_argument("--onnx", action="store_true", help="Load ONNX Silero VAD")
    parser.add_argument("--opset-version", type=int, default=16, help="ONNX opset version")
    args = parser.parse_args()

    audio_path = Path(args.audio)
    if not audio_path.is_file():
        raise FileNotFoundError(audio_path)

    print("Loading Silero VAD...")
    vad = SileroVad.from_pretrained(
        onnx=args.onnx,
        opset_version=args.opset_version,
        sampling_rate=args.sample_rate,
    )

    print(f"Reading audio: {audio_path}")
    audio = load_wav_mono(str(audio_path), target_sr=args.sample_rate)

    print("Running VAD...")
    speech_chunks = run_vad(
        vad,
        audio,
        sample_rate=args.sample_rate,
        chunk_samples=args.chunk_samples,
        threshold=args.threshold,
    )

    speech_seconds = sum(end - start for start, end, _ in speech_chunks)
    total_seconds = len(audio) / args.sample_rate
    print(
        f"\nSummary: {len(speech_chunks)} speech chunks, "
        f"{speech_seconds:.2f}s speech / {total_seconds:.2f}s total"
    )


if __name__ == "__main__":
    main()
