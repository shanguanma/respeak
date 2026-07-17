"""Stream ARKit blendshapes to Unreal Engine (NVIDIA Audio2Face Live Link TCP)."""

from __future__ import annotations

import json
import math
import random
import socket
import struct
import time
from typing import Any, Iterable

import numpy as np

BLENDSHAPE_NAMES_55 = [
    "EyeBlinkLeft",
    "EyeLookDownLeft",
    "EyeLookInLeft",
    "EyeLookOutLeft",
    "EyeLookUpLeft",
    "EyeSquintLeft",
    "EyeWideLeft",
    "EyeBlinkRight",
    "EyeLookDownRight",
    "EyeLookInRight",
    "EyeLookOutRight",
    "EyeLookUpRight",
    "EyeSquintRight",
    "EyeWideRight",
    "JawForward",
    "JawLeft",
    "JawRight",
    "JawOpen",
    "MouthClose",
    "MouthFunnel",
    "MouthPucker",
    "MouthLeft",
    "MouthRight",
    "MouthSmileLeft",
    "MouthSmileRight",
    "MouthFrownLeft",
    "MouthFrownRight",
    "MouthDimpleLeft",
    "MouthDimpleRight",
    "MouthStretchLeft",
    "MouthStretchRight",
    "MouthRollLower",
    "MouthRollUpper",
    "MouthShrugLower",
    "MouthShrugUpper",
    "MouthPressLeft",
    "MouthPressRight",
    "MouthLowerDownLeft",
    "MouthLowerDownRight",
    "MouthUpperUpLeft",
    "MouthUpperUpRight",
    "BrowDownLeft",
    "BrowDownRight",
    "BrowInnerUp",
    "BrowOuterUpLeft",
    "BrowOuterUpRight",
    "CheekPuff",
    "CheekSquintLeft",
    "CheekSquintRight",
    "NoseSneerLeft",
    "NoseSneerRight",
    "TongueOut",
    "HeadRoll",
    "HeadPitch",
    "HeadYaw",
]

DEFAULT_SCALING_FACTOR = 1.2 * np.array(
    [
        1.0,
        0.0,
        0.0,
        0.0,
        0.0,
        1.0,
        1.0,
        1.0,
        0.0,
        0.0,
        0.0,
        0.0,
        1.0,
        1.0,
        0.7,
        0.2,
        0.2,
        1.0,
        0.0,
        1.5,
        1.5,
        0.2,
        0.2,
        0.7,
        0.7,
        0.5,
        0.5,
        0.5,
        0.5,
        0.2,
        0.2,
        0.6,
        0.4,
        0.7,
        0.4,
        1.0,
        1.0,
        0.8,
        0.8,
        0.8,
        0.8,
        1.0,
        1.0,
        1.0,
        1.0,
        1.0,
        0.2,
        1.0,
        1.0,
        0.8,
        0.8,
        0.0,
        1.0,
        1.0,
        1.0,
        1.0,
        1.0,
        1.0,
        1.0,
        1.0,
        1.0,
    ],
    dtype=np.float32,
)


class IdleMotionGenerator:
    """Procedural blink / head motion for idle frames (ported from openfacebot)."""

    def __init__(
        self,
        fps: float,
        *,
        enable_idle_motion: bool = True,
        enable_head_motion: bool = True,
        enable_blink: bool = True,
        head_center: float = 0.0,
        head_motion_strength: float = 0.12,
        head_motion_hz: float = 0.04,
        head_random_interval_min_sec: float = 1.5,
        head_random_interval_max_sec: float = 4.0,
        head_random_speed: float = 5.0,
        speech_energy_smooth: float = 6.0,
        speech_emphasis_threshold: float = 0.04,
        speech_emphasis_strength: float = 0.7,
        speech_emphasis_decay: float = 6.0,
        speech_emphasis_pitch: float = -0.06,
        speech_emphasis_yaw: float = 0.0,
        blink_rate_min_sec: float = 2.0,
        blink_rate_max_sec: float = 8.0,
        blink_duration_min_ms: float = 100.0,
        blink_duration_max_ms: float = 180.0,
        blink_strength: float = 1.0,
    ) -> None:
        self.fps = max(1.0, float(fps))
        self.enable_idle_motion = bool(enable_idle_motion)
        self.enable_head_motion = bool(enable_head_motion)
        self.enable_blink = bool(enable_blink)
        self.head_center = float(head_center)
        self.head_motion_strength = float(head_motion_strength)
        self.head_motion_hz = float(head_motion_hz)
        self.head_random_interval_min_sec = float(head_random_interval_min_sec)
        self.head_random_interval_max_sec = float(head_random_interval_max_sec)
        self.head_random_speed = float(head_random_speed)
        self.speech_energy_smooth = float(speech_energy_smooth)
        self.speech_emphasis_threshold = float(speech_emphasis_threshold)
        self.speech_emphasis_strength = float(speech_emphasis_strength)
        self.speech_emphasis_decay = float(speech_emphasis_decay)
        self.speech_emphasis_pitch = float(speech_emphasis_pitch)
        self.speech_emphasis_yaw = float(speech_emphasis_yaw)
        self.blink_rate_min_sec = float(blink_rate_min_sec)
        self.blink_rate_max_sec = float(blink_rate_max_sec)
        self.blink_duration_min_ms = float(blink_duration_min_ms)
        self.blink_duration_max_ms = float(blink_duration_max_ms)
        self.blink_strength = float(blink_strength)

        self.start_time = time.time()
        self.in_blink = False
        self.blink_start_time = 0.0
        self.blink_end_time = 0.0
        self.next_blink_time = 0.0
        self._schedule_next_blink(self.start_time)

        self.head_target = np.array(
            [self.head_center, self.head_center, self.head_center], dtype=np.float32
        )
        self.head_current = self.head_target.copy()
        self.head_next_time = self.start_time
        self.last_head_update_time = self.start_time

        self.speech_smoothed = 0.0
        self.speech_last = 0.0
        self.speech_pulse = 0.0
        self.emphasis_yaw_dir = 1.0

    def _schedule_next_blink(self, now: float) -> None:
        if self.blink_rate_min_sec <= 0.0 or self.blink_rate_max_sec <= 0.0:
            self.next_blink_time = now + 10.0
            return
        low = min(self.blink_rate_min_sec, self.blink_rate_max_sec)
        high = max(self.blink_rate_min_sec, self.blink_rate_max_sec)
        self.next_blink_time = now + random.uniform(low, high)

    def _blink_value(self, now: float) -> float:
        if not self.enable_blink:
            return 0.0
        if not self.in_blink and now >= self.next_blink_time:
            duration_ms = random.uniform(self.blink_duration_min_ms, self.blink_duration_max_ms)
            duration = max(0.03, duration_ms / 1000.0)
            self.blink_start_time = now
            self.blink_end_time = now + duration
            self.in_blink = True
        if not self.in_blink:
            return 0.0
        if now >= self.blink_end_time:
            self.in_blink = False
            self._schedule_next_blink(now)
            return 0.0
        t = (now - self.blink_start_time) / (self.blink_end_time - self.blink_start_time)
        value = t / 0.5 if t < 0.5 else (1.0 - t) / 0.5
        return max(0.0, min(1.0, value * self.blink_strength))

    def _head_values(
        self, now: float, speech_energy: float, is_speaking: bool
    ) -> tuple[float, float, float]:
        if not self.enable_head_motion:
            return 0.0, 0.0, 0.0

        elapsed = now - self.start_time
        base = self.head_center
        amp = self.head_motion_strength

        if is_speaking:
            self.head_target = np.array([base, base, base], dtype=np.float32)
            self.head_next_time = now
        elif now >= self.head_next_time:
            low = min(self.head_random_interval_min_sec, self.head_random_interval_max_sec)
            high = max(self.head_random_interval_min_sec, self.head_random_interval_max_sec)
            interval = random.uniform(max(0.4, low * 0.6), max(0.6, high * 0.6))
            self.head_next_time = now + interval
            yaw_mag = random.uniform(max(0.05, amp * 0.8), max(0.08, amp * 1.2))
            yaw_dir = 1.0 if random.random() > 0.5 else -1.0
            self.head_target = np.array(
                [
                    base + random.uniform(-amp * 0.08, amp * 0.08),
                    base + random.uniform(-amp * 0.12, amp * 0.12),
                    base + yaw_mag * yaw_dir,
                ],
                dtype=np.float32,
            )

        dt = max(0.0, now - self.last_head_update_time)
        self.last_head_update_time = now
        speed = self.head_random_speed * (1.8 if is_speaking else 1.0)
        alpha = 1.0 - math.exp(-speed * dt) if speed > 0.0 else 1.0
        self.head_current = self.head_current + alpha * (self.head_target - self.head_current)

        wobble_amp = amp * (0.08 if is_speaking else 0.22)
        if wobble_amp > 0.0 and self.head_motion_hz > 0.0:
            phase = 2.0 * math.pi * self.head_motion_hz * elapsed
            if is_speaking:
                self.head_current[2] += wobble_amp * math.sin(phase)
                self.head_current[1] += (wobble_amp * 0.6) * math.sin(phase + 1.1)
                self.head_current[0] += (wobble_amp * 0.4) * math.sin(phase + 2.2)
            else:
                self.head_current[2] += (wobble_amp * 1.1) * math.sin(phase)
                self.head_current[1] += (wobble_amp * 0.25) * math.sin(phase + 1.1)
                self.head_current[0] += (wobble_amp * 0.2) * math.sin(phase + 2.2)

        if is_speaking:
            smooth_alpha = min(1.0, dt * max(0.1, self.speech_energy_smooth))
            self.speech_smoothed = (
                (1.0 - smooth_alpha) * self.speech_smoothed + smooth_alpha * speech_energy
            )
            delta = self.speech_smoothed - self.speech_last
            self.speech_last = self.speech_smoothed
            if delta > self.speech_emphasis_threshold:
                self.speech_pulse = 1.0
                self.emphasis_yaw_dir = 1.0 if random.random() > 0.5 else -1.0

        decay = math.exp(-max(0.0, self.speech_emphasis_decay) * dt)
        self.speech_pulse *= decay
        pulse = self.speech_pulse * max(0.0, self.speech_emphasis_strength)
        if pulse > 0.0:
            self.head_current[1] += pulse * self.speech_emphasis_pitch
            self.head_current[2] += pulse * self.speech_emphasis_yaw * self.emphasis_yaw_dir

        return (
            max(-1.0, min(1.0, float(self.head_current[0]))),
            max(-1.0, min(1.0, float(self.head_current[1]))),
            max(-1.0, min(1.0, float(self.head_current[2]))),
        )

    def step(
        self,
        now: float | None = None,
        *,
        speech_energy: float = 0.0,
        is_speaking: bool = False,
    ) -> np.ndarray:
        now = time.time() if now is None else now
        weights = np.zeros(61, dtype=np.float32)
        if self.enable_idle_motion:
            roll, pitch, yaw = self._head_values(now, speech_energy, is_speaking)
            if self.enable_head_motion:
                weights[52] = roll
                weights[53] = pitch
                weights[54] = yaw
        if self.enable_blink:
            blink = self._blink_value(now)
            weights[0] = blink
            weights[7] = blink
        return weights


class Ue5BlendshapeRenderer:
    """TCP client for NVIDIA UE Audio2Face Live Link plugin."""

    def __init__(
        self,
        *,
        tcp_ip: str = "127.0.0.1",
        tcp_port: int = 12030,
        output_fps: int = 25,
        subject_name: str = "Audio2Face",
        reconnect_interval_sec: float = 2.0,
        enable_idle_motion: bool = True,
        enable_head_motion: bool = True,
        enable_blink: bool = True,
        idle_mix_when_speaking: float = 0.5,
        speech_active_threshold: float = 0.15,
        speech_hold_sec: float = 1.0,
        scaling_factor: np.ndarray | None = None,
        idle_motion: IdleMotionGenerator | None = None,
    ) -> None:
        self.tcp_ip = tcp_ip
        self.tcp_port = int(tcp_port)
        self.output_fps = int(output_fps)
        self.subject_name = subject_name
        self.reconnect_interval_sec = float(reconnect_interval_sec)
        self.enable_idle_motion = enable_idle_motion
        self.idle_mix_when_speaking = float(idle_mix_when_speaking)
        self.speech_active_threshold = float(speech_active_threshold)
        self.speech_hold_sec = float(speech_hold_sec)
        self.scaling_factor = (
            scaling_factor.copy()
            if scaling_factor is not None
            else DEFAULT_SCALING_FACTOR.copy()
        )
        self.idle_gen = idle_motion or IdleMotionGenerator(
            fps=self.output_fps,
            enable_idle_motion=enable_idle_motion,
            enable_head_motion=enable_head_motion,
            enable_blink=enable_blink,
        )

        self._socket: socket.socket | None = None
        self._header_sent = False
        self._last_connect_attempt = 0.0
        self._speech_active = False
        self._last_speech_time = 0.0
        self.sent_frames = 0

    @staticmethod
    def pack_message(payload_bytes: bytes) -> bytes:
        return struct.pack("!Q", len(payload_bytes)) + payload_bytes

    @staticmethod
    def to_61_weights(raw_weights: np.ndarray) -> np.ndarray:
        raw = np.asarray(raw_weights, dtype=np.float32).reshape(-1)
        raw_len = raw.shape[0]
        if raw_len == 61:
            return raw
        weights = np.zeros(61, dtype=np.float32)
        if raw_len >= 55:
            weights[: min(61, raw_len)] = raw[: min(61, raw_len)]
        elif raw_len >= 52:
            weights[:52] = raw[:52]
        elif raw_len >= 51:
            weights[:51] = raw[:51]
        return weights

    @staticmethod
    def to_55_weights(weights_61: np.ndarray) -> np.ndarray:
        out = np.zeros(55, dtype=np.float32)
        copy_count = min(52, weights_61.shape[0])
        out[:copy_count] = weights_61[:copy_count]
        if weights_61.shape[0] >= 55:
            out[52] = weights_61[52]
            out[53] = weights_61[53]
            out[54] = weights_61[54]
        out[:52] = np.clip(out[:52], 0.0, 1.0)
        out[52:55] = np.clip(out[52:55], -1.0, 1.0)
        return out

    @staticmethod
    def blend_idle_into_base(
        base: np.ndarray, idle: np.ndarray, mix: float
    ) -> np.ndarray:
        mix = max(0.0, min(1.0, mix))
        if mix <= 0.0:
            return base
        result = base.copy()
        for idx in (0, 7, 52, 53, 54):
            result[idx] = (1.0 - mix) * result[idx] + mix * idle[idx]
        return result

    @staticmethod
    def speech_energy_from_weights(weights_61: np.ndarray) -> float:
        return max(
            0.0,
            min(
                1.0,
                float(
                    max(
                        weights_61[17],
                        weights_61[18] * 0.5,
                        weights_61[19],
                        weights_61[20],
                        weights_61[23],
                        weights_61[24],
                    )
                ),
            ),
        )

    def connect(self, *, timeout_sec: float = 2.0) -> bool:
        if self._socket is not None:
            return True
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.settimeout(timeout_sec)
        sock.connect((self.tcp_ip, self.tcp_port))
        sock.settimeout(None)
        self._socket = sock
        self._header_sent = False
        return True

    def close(self, *, send_eos: bool = True) -> None:
        if self._socket is None:
            return
        try:
            if send_eos:
                self._send_ascii("EOS")
        except OSError:
            pass
        finally:
            try:
                self._socket.close()
            except OSError:
                pass
            self._socket = None
            self._header_sent = False

    def __enter__(self) -> Ue5BlendshapeRenderer:
        self.connect()
        return self

    def __exit__(self, *_exc: Any) -> None:
        self.close()

    def _send_ascii(self, payload: str) -> None:
        if self._socket is None:
            raise RuntimeError("UE5 renderer is not connected")
        self._socket.sendall(self.pack_message(payload.encode("ascii")))

    def _prepare_frame_weights(self, raw_weights: np.ndarray | None) -> np.ndarray:
        now = time.time()
        if raw_weights is None:
            if not self.enable_idle_motion:
                return np.zeros(61, dtype=np.float32)
            return self.idle_gen.step(now)

        weights_61 = self.to_61_weights(raw_weights) * self.scaling_factor
        speech_energy = self.speech_energy_from_weights(weights_61)
        if speech_energy > self.speech_active_threshold:
            self._speech_active = True
            self._last_speech_time = now
        elif self._speech_active and (now - self._last_speech_time) > self.speech_hold_sec:
            self._speech_active = False

        if self.enable_idle_motion and self.idle_mix_when_speaking > 0.0:
            idle = self.idle_gen.step(
                now,
                speech_energy=speech_energy,
                is_speaking=self._speech_active,
            )
            mix = self.idle_mix_when_speaking + (1.0 - self.idle_mix_when_speaking) * (
                1.0 - speech_energy
            )
            weights_61 = self.blend_idle_into_base(weights_61, idle, mix)
        return weights_61

    def build_frame_payload(self, raw_weights: np.ndarray | None) -> str:
        weights_61 = self._prepare_frame_weights(raw_weights)
        weights_55 = self.to_55_weights(weights_61)
        frame_dict = {
            self.subject_name: {
                "Body": {},
                "Facial": {
                    "Names": BLENDSHAPE_NAMES_55,
                    "Weights": [float(v) for v in weights_55],
                },
            }
        }
        return json.dumps(frame_dict, separators=(",", ":"), ensure_ascii=True)

    def send_frame(self, raw_weights: np.ndarray | None) -> None:
        if self._socket is None:
            raise RuntimeError("UE5 renderer is not connected")
        if not self._header_sent:
            self._send_ascii(f"A2F:{self.output_fps}")
            self._header_sent = True
        self._send_ascii(self.build_frame_payload(raw_weights))
        self.sent_frames += 1

    def stream_frames(
        self,
        frames: Iterable[np.ndarray | dict[str, Any] | None],
        *,
        realtime: bool = True,
    ) -> int:
        """Send frames to UE5. Each item may be weights or ``{"arkit_weights": ...}``."""
        if self._socket is None:
            self.connect()
        sent = 0
        frame_interval = 1.0 / max(1, self.output_fps)
        next_time = time.perf_counter()
        for item in frames:
            if isinstance(item, dict):
                weights = item.get("arkit_weights")
            else:
                weights = item
            self.send_frame(weights)
            sent += 1
            if realtime:
                next_time += frame_interval
                sleep_for = next_time - time.perf_counter()
                if sleep_for > 0:
                    time.sleep(sleep_for)
        return sent
