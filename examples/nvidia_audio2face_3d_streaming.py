#!/usr/bin/env python3
"""NVIDIA Audio2Face 3D streaming example.

Usage:
    python examples/nvidia_audio2face_3d_streaming.py \
        --model-dir path/to/audio2face-3d-v2.3.1-claire \
        --audio path/to/test.wav \
        --output arkit_weights.npy

Requires: pip install -e ".[a2f]"
"""

from __future__ import annotations

import argparse
import wave
from pathlib import Path

import numpy as np

from respeak import NvidiaAudio2Face3D


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


def main() -> None:
    parser = argparse.ArgumentParser(description="NVIDIA Audio2Face 3D streaming demo")
    parser.add_argument(
        "--model-dir",
        required=True,
        help="Audio2Face 3D model folder (contains network.onnx, *.npz, etc.)",
    )
    parser.add_argument("--audio", required=True, help="Input wav file")
    parser.add_argument(
        "--output",
        default=None,
        help="Optional path to save [T, N] ARKit weights as .npy",
    )
    parser.add_argument("--output-fps", type=int, default=25, help="Blendshape FPS")
    parser.add_argument("--cpu", action="store_true", help="Force CPUExecutionProvider")
    parser.add_argument(
        "--pad-to",
        type=int,
        default=None,
        help="Pad each frame to this length (e.g. 61 for Live Link)",
    )
    parser.add_argument(
        "--max-frames",
        type=int,
        default=None,
        help="Print / keep at most this many frames (debug)",
    )
    args = parser.parse_args()

    model_dir = Path(args.model_dir)
    audio_path = Path(args.audio)
    if not model_dir.is_dir():
        raise FileNotFoundError(model_dir)
    if not audio_path.is_file():
        raise FileNotFoundError(audio_path)

    print(f"Loading Audio2Face 3D from {model_dir} ...")
    model = NvidiaAudio2Face3D.from_pretrained(
        str(model_dir),
        use_cuda=not args.cpu,
        output_fps=args.output_fps,
    )
    pose_names = model.get_arkit_pose_names()
    print(
        f"Loaded: sample_rate={model.sample_rate}, "
        f"window={model.window_samples} samples, poses={len(pose_names)}"
    )

    print(f"Reading audio: {audio_path}")
    audio = load_wav_mono(str(audio_path), target_sr=model.sample_rate)
    print(f"Audio: {len(audio) / model.sample_rate:.2f}s ({len(audio)} samples)")

    print("Running streaming inference ...")
    frames = model.generate(
        audio,
        stream=True,
        is_final=True,
        pad_to=args.pad_to,
    )
    if args.max_frames is not None:
        frames = frames[: args.max_frames]

    weights = np.stack(frames, axis=0)
    print(f"Frames: {weights.shape[0]} @ {args.output_fps} FPS, dim={weights.shape[1]}")

    for i, frame in enumerate(frames[: min(5, len(frames))]):
        active = int(np.sum(frame > 0.01))
        max_idx = int(np.argmax(frame[: len(pose_names)]))
        max_name = pose_names[max_idx] if max_idx < len(pose_names) else str(max_idx)
        print(
            f"  frame {i:3d}: active={active}, "
            f"max={max_name}={float(frame[max_idx]):.3f}"
        )

    if args.output:
        out_path = Path(args.output)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        np.save(out_path, weights)
        print(f"Saved: {out_path}")


if __name__ == "__main__":
    main()
