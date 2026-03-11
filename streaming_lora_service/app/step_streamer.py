from __future__ import annotations

from dataclasses import dataclass
import math

from .incremental_decoder import IncrementalAudioDecoder, IncrementalDecoderConfig
from .models import SynthesizedAudio


@dataclass(frozen=True)
class AudioStepStreamerConfig:
    samples_per_step: int = 1920
    chunk_steps: int = 4
    left_context_steps: int = 25

    def __post_init__(self) -> None:
        if self.samples_per_step <= 0:
            raise ValueError("samples_per_step must be > 0")
        if self.chunk_steps <= 0:
            raise ValueError("chunk_steps must be > 0")
        if self.left_context_steps < 0:
            raise ValueError("left_context_steps must be >= 0")


class AudioStepStreamer:
    def __init__(self, config: AudioStepStreamerConfig | None = None) -> None:
        self.config = config or AudioStepStreamerConfig()

    def iter_audio_chunks(self, synthesized: SynthesizedAudio) -> list[bytes]:
        if not synthesized.audio_bytes:
            return []

        if synthesized.codec_steps is not None and synthesized.codec_steps > 0:
            total_steps = synthesized.codec_steps
            bytes_per_step = max(1, math.ceil(len(synthesized.audio_bytes) / total_steps))
        else:
            bytes_per_step = self.config.samples_per_step * max(1, synthesized.channels) * 2
            total_steps = max(1, math.ceil(len(synthesized.audio_bytes) / bytes_per_step))

        decoder = IncrementalAudioDecoder(
            IncrementalDecoderConfig(
                chunk_steps=self.config.chunk_steps,
                left_context_steps=self.config.left_context_steps,
            )
        )

        def decode_fn(start_step: int, end_step: int) -> bytes:
            if synthesized.decode_step_range is not None:
                return synthesized.decode_step_range(start_step, end_step)
            start = start_step * bytes_per_step
            end = min(end_step * bytes_per_step, len(synthesized.audio_bytes))
            return synthesized.audio_bytes[start:end]

        chunks: list[bytes] = []
        next_step = self.config.chunk_steps
        while next_step < total_steps:
            chunk = decoder.decode(
                next_step,
                decode_fn=decode_fn,
                bytes_per_step=bytes_per_step,
                force=False,
                finished=False,
            )
            if chunk:
                chunks.append(chunk)
            next_step += self.config.chunk_steps

        final_chunk = decoder.decode(
            total_steps,
            decode_fn=decode_fn,
            bytes_per_step=bytes_per_step,
            force=True,
            finished=True,
        )
        if final_chunk:
            chunks.append(final_chunk)
        return chunks