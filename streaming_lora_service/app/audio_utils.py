from __future__ import annotations

import io
from typing import Iterable

import numpy as np
import soundfile as sf


def float_audio_to_pcm16le_bytes(audio: np.ndarray | Iterable[float]) -> bytes:
    waveform = np.asarray(audio, dtype=np.float32).reshape(-1)
    waveform = np.clip(waveform, -1.0, 1.0)
    pcm = (waveform * 32767.0).astype(np.int16)
    return pcm.tobytes()


def chunk_audio_bytes(audio_bytes: bytes, chunk_size_bytes: int) -> list[bytes]:
    if chunk_size_bytes <= 0:
        raise ValueError("chunk_size_bytes must be > 0")
    if not audio_bytes:
        return []
    return [audio_bytes[idx : idx + chunk_size_bytes] for idx in range(0, len(audio_bytes), chunk_size_bytes)]


def pcm16le_bytes_to_float_audio(audio_bytes: bytes, *, channels: int = 1) -> np.ndarray:
    waveform = np.frombuffer(audio_bytes, dtype=np.int16).astype(np.float32) / 32767.0
    if channels > 1:
        waveform = waveform.reshape(-1, channels)
    return waveform


def pcm16le_bytes_to_wav_bytes(audio_bytes: bytes, *, sample_rate: int, channels: int = 1) -> bytes:
    waveform = pcm16le_bytes_to_float_audio(audio_bytes, channels=channels)
    buffer = io.BytesIO()
    sf.write(buffer, waveform, samplerate=sample_rate, format="WAV")
    return buffer.getvalue()