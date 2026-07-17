#!/usr/bin/env python3
"""NVIDIA Audio2Face 3D streaming example with optional UE5 TCP rendering.

Usage:
    python examples/nvidia_audio2face_3d_streaming.py \
        --model-dir path/to/audio2face-3d-v2.3.1-claire \
        --audio path/to/test.wav \
        --output arkit_weights.npy

    # Stream blendshapes to Unreal Engine Audio2Face plugin
    python examples/nvidia_audio2face_3d_streaming.py \
        --model-dir path/to/audio2face-3d-v2.3.1-claire \
        --audio path/to/test.wav \
        --ue5-tcp-ip 127.0.0.1 \
        --ue5-tcp-port 12030

Requires: pip install -e ".[a2f]" or pip install "respeak-ai[a2f-trt]"
"""

from __future__ import annotations

import argparse
import wave
from pathlib import Path

import numpy as np

from respeak.models.nvidia_audio2face_3d import NvidiaAudio2Face3D, Ue5BlendshapeRenderer


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


def save_wav_mono(path: Path, audio: np.ndarray, sample_rate: int) -> None:
    audio = np.clip(audio, -1.0, 1.0)
    pcm = (audio * 32767.0).astype(np.int16)
    path.parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(path), "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)
        wf.writeframes(pcm.tobytes())


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
    parser.add_argument(
        "--output-audio",
        default=None,
        help="Optional path to save synced output audio as .wav",
    )
    parser.add_argument("--output-fps", type=int, default=25, help="Blendshape FPS")
    parser.add_argument("--cpu", action="store_true", help="Force CPUExecutionProvider")
    parser.add_argument(
        "--tensorrt",
        action="store_true",
        help=(
            "Enable TensorRT EP (requires system TensorRT / libnvinfer). "
            "If missing, automatically falls back to CUDA."
        ),
    )
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
    parser.add_argument("--ue5-tcp-ip", default=None, help="UE5 Audio2Face plugin TCP IP")
    parser.add_argument(
        "--ue5-tcp-port",
        type=int,
        default=12030,
        help="UE5 Audio2Face plugin TCP port",
    )
    parser.add_argument(
        "--ue5-subject",
        default="Audio2Face",
        help="UE5 subject name in Live Link payload",
    )
    parser.add_argument(
        "--ue5-realtime",
        action="store_true",
        help="Pace UE5 frame sending at output FPS",
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
        enable_tensorrt=args.tensorrt,
        output_fps=args.output_fps,
    )
    pose_names = model.get_arkit_pose_names()
    print(
        f"Loaded: provider={model.active_provider}, "
        f"sample_rate={model.sample_rate}, "
        f"window={model.window_samples} samples, poses={len(pose_names)}"
    )

    print(f"Reading audio: {audio_path}")
    audio = load_wav_mono(str(audio_path), target_sr=model.sample_rate)
    print(f"Audio: {len(audio) / model.sample_rate:.2f}s ({len(audio)} samples)")

    print("Running streaming inference ...")
    result = model.generate(
        audio,
        stream=True,
        is_final=True,
        return_dict=True,
        pad_to=args.pad_to,
    )
    frames = result["frames"]
    if args.max_frames is not None:
        frames = frames[: args.max_frames]

    synced_audio = np.concatenate([frame["audio"] for frame in frames], axis=0)
    weights = np.stack([frame["arkit_weights"] for frame in frames], axis=0)
    print(
        f"Frames: {weights.shape[0]} @ {args.output_fps} FPS, "
        f"weights_dim={weights.shape[1]}, synced_audio={synced_audio.shape[0]} samples"
    )

    for i, frame in enumerate(frames[: min(5, len(frames))]):
        w = frame["arkit_weights"]
        active = int(np.sum(w > 0.01))
        max_idx = int(np.argmax(w[: len(pose_names)]))
        max_name = pose_names[max_idx] if max_idx < len(pose_names) else str(max_idx)
        print(
            f"  frame {i:3d}: audio={frame['audio'].shape[0]} samples, "
            f"active={active}, max={max_name}={float(w[max_idx]):.3f}"
        )

    if args.output:
        out_path = Path(args.output)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        np.save(out_path, weights)
        print(f"Saved weights: {out_path}")

    if args.output_audio:
        out_audio = Path(args.output_audio)
        save_wav_mono(out_audio, synced_audio, model.sample_rate)
        print(f"Saved synced audio: {out_audio}")

    if args.ue5_tcp_ip:
        print(
            f"Streaming to UE5 at {args.ue5_tcp_ip}:{args.ue5_tcp_port} "
            f"(realtime={args.ue5_realtime}) ..."
        )
        renderer = Ue5BlendshapeRenderer(
            tcp_ip=args.ue5_tcp_ip,
            tcp_port=args.ue5_tcp_port,
            output_fps=args.output_fps,
            subject_name=args.ue5_subject,
        )
        with renderer:
            sent = renderer.stream_frames(
                frames,
                realtime=args.ue5_realtime,
            )
        print(f"UE5 frames sent: {sent}")


if __name__ == "__main__":
    main()
