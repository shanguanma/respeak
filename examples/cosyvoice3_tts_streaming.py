#!/usr/bin/env python3
"""CosyVoice3 TTS example (vLLM-accelerated streaming).

Usage:
    python examples/cosyvoice3_tts_streaming.py \\
        --model-dir path/to/Fun-CosyVoice3 \\
        --prompt-wav src/respeak/models/cosyvoice3_tts/data/man.wav \\
        --prompt-text "近日，除了葛洲坝股价下跌外，其余三家均有不同程度的上涨。" \\
        --text "你好，欢迎使用本系统。" \\
        --output output.wav

Requires: pip install -e ".[tts]"
"""

from __future__ import annotations

import argparse
import wave
from pathlib import Path
from typing import Union
import numpy as np

from respeak import CosyVoice3Tts


def save_wav_int16(path: str, audio: np.ndarray, sample_rate: int) -> None:
    """Save int16 PCM wav."""
    audio = np.asarray(audio, dtype=np.int16)
    with wave.open(path, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)
        wf.writeframes(audio.tobytes())


def synthesize_streaming(
    model: CosyVoice3Tts,
    text: str,
    *,
    target_sr: int = 16000,
) -> tuple[np.ndarray, int]:
    """Stream TTS chunks and concatenate to one int16 waveform."""
    chunks: list[np.ndarray] = []

    for chunk in model.generate(text, stream=True, target_sr=target_sr):
        chunks.append(np.asarray(chunk, dtype=np.int16))
        print(f"[chunk] samples={len(chunk)}")

    if not chunks:
        return np.zeros(0, dtype=np.int16), target_sr

    return np.concatenate(chunks), target_sr


def main() -> None:
    parser = argparse.ArgumentParser(description="CosyVoice3 TTS streaming demo")
    parser.add_argument("--model-dir", required=True, help="CosyVoice3 model directory")
    parser.add_argument("--prompt-wav", required=True, help="Zero-shot prompt wav")
    parser.add_argument("--prompt-text", required=True, help="Zero-shot prompt text")
    parser.add_argument("--text", required=True, help="Text to synthesize")
    parser.add_argument("--output", default="output.wav", help="Output wav path")
    parser.add_argument("--spk-id", default="demo_spk", help="Zero-shot speaker id")
    parser.add_argument("--target-sr", type=int, default=16000, help="Output sample rate")
    parser.add_argument("--no-vllm", action="store_true", help="Disable vLLM acceleration")
    parser.add_argument("--warmup", default="你好。", help="Warmup text (empty to skip)")
    args = parser.parse_args()

    prompt_wav = Path(args.prompt_wav)
    if not prompt_wav.is_file():
        raise FileNotFoundError(prompt_wav)

    print("Loading CosyVoice3 TTS...")
    tts = CosyVoice3Tts.from_pretrained(
        model_dir=args.model_dir,
        prompt_text=args.prompt_text,
        prompt_wav=str(prompt_wav),
        zero_shot_spk_id=args.spk_id,
        load_vllm=not args.no_vllm,
        warmup_text=args.warmup or None,
        save_spkinfo=True,
    )

    print(f"Synthesizing: {args.text}")
    audio, sample_rate = synthesize_streaming(
        tts,
        args.text,
        target_sr=args.target_sr,
    )

    output_path = Path(args.output)
    save_wav_int16(str(output_path), audio, sample_rate)
    print(f"Saved: {output_path} ({len(audio)} samples @ {sample_rate} Hz)")


if __name__ == "__main__":
    main()
