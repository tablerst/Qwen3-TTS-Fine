from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from itertools import islice
from typing import Any, Callable, Iterable, Sequence

from .models import DecodePlan


class IncrementalDecoderError(RuntimeError):
    """Raised when decoder state becomes inconsistent."""


@dataclass(frozen=True)
class IncrementalDecoderConfig:
    chunk_steps: int = 4
    left_context_steps: int = 25
    first_chunk_steps: int | None = None
    crossfade_samples: int = 0

    def __post_init__(self) -> None:
        if self.chunk_steps <= 0:
            raise ValueError("chunk_steps must be > 0")
        if self.left_context_steps < 0:
            raise ValueError("left_context_steps must be >= 0")
        if self.first_chunk_steps is not None and self.first_chunk_steps <= 0:
            raise ValueError("first_chunk_steps must be > 0 when provided")
        if self.crossfade_samples < 0:
            raise ValueError("crossfade_samples must be >= 0")


class IncrementalAudioDecoder:
    def __init__(self, config: IncrementalDecoderConfig | None = None) -> None:
        self.config = config or IncrementalDecoderConfig()
        self.emitted_until_step = 0
        self.buffer_start_step = 0
        self._codec_buffer: deque[Any] = deque()
        self._pending_crossfade_tail = b""

    @property
    def buffered_until_step(self) -> int:
        return self.buffer_start_step + len(self._codec_buffer)

    @property
    def buffered_step_count(self) -> int:
        return len(self._codec_buffer)

    def reset(self) -> None:
        self.emitted_until_step = 0
        self.buffer_start_step = 0
        self._codec_buffer.clear()
        self._pending_crossfade_tail = b""

    def push_codec_step(self, codec_ids: Any) -> int:
        self._codec_buffer.append(codec_ids)
        return self.buffered_until_step

    def extend_codec_steps(self, codec_steps: Iterable[Any]) -> int:
        for codec_ids in codec_steps:
            self._codec_buffer.append(codec_ids)
        return self.buffered_until_step

    def plan(self, total_steps: int, *, force: bool = False, finished: bool = False) -> DecodePlan | None:
        if total_steps < self.emitted_until_step:
            raise IncrementalDecoderError(
                f"total_steps {total_steps} cannot be smaller than emitted_until_step {self.emitted_until_step}"
            )
        new_steps = total_steps - self.emitted_until_step
        if new_steps == 0:
            return None
        required_steps = (
            self.config.first_chunk_steps
            if self.emitted_until_step == 0 and self.config.first_chunk_steps is not None
            else self.config.chunk_steps
        )
        if not force and not finished and new_steps < required_steps:
            return None

        start_step = max(0, self.emitted_until_step - self.config.left_context_steps)
        return DecodePlan(
            start_step=start_step,
            end_step=total_steps,
            emit_from_step=self.emitted_until_step,
        )

    def decode(
        self,
        total_steps: int,
        *,
        decode_fn: Callable[[int, int], bytes],
        bytes_per_step: int,
        channels: int = 1,
        force: bool = False,
        finished: bool = False,
    ) -> bytes | None:
        if bytes_per_step <= 0:
            raise ValueError("bytes_per_step must be > 0")
        if channels <= 0:
            raise ValueError("channels must be > 0")

        plan = self.plan(total_steps, force=force, finished=finished)
        if plan is None:
            return None

        decoded = decode_fn(plan.start_step, plan.end_step)
        overlap_steps = plan.emit_from_step - plan.start_step
        overlap_bytes = overlap_steps * bytes_per_step
        if len(decoded) < overlap_bytes:
            raise IncrementalDecoderError(
                f"Decoded audio is shorter than the overlap window: {len(decoded)} < {overlap_bytes}"
            )

        self.emitted_until_step = plan.end_step
        return self._apply_crossfade(decoded[overlap_bytes:], channels=channels, finished=finished)

    def decode_buffered(
        self,
        *,
        decode_fn: Callable[[Sequence[Any]], bytes],
        bytes_per_step: int,
        channels: int = 1,
        force: bool = False,
        finished: bool = False,
    ) -> bytes | None:
        if bytes_per_step <= 0:
            raise ValueError("bytes_per_step must be > 0")
        if channels <= 0:
            raise ValueError("channels must be > 0")

        total_steps = self.buffered_until_step
        plan = self.plan(total_steps, force=force, finished=finished)
        if plan is None:
            return None

        start_offset = plan.start_step - self.buffer_start_step
        end_offset = plan.end_step - self.buffer_start_step
        if start_offset < 0 or end_offset > len(self._codec_buffer):
            raise IncrementalDecoderError(
                "Buffered codec steps do not cover the requested decode window: "
                f"buffer={self.buffer_start_step}..{self.buffered_until_step}, plan={plan.start_step}..{plan.end_step}"
            )

        decoded = decode_fn(tuple(islice(self._codec_buffer, start_offset, end_offset)))
        overlap_steps = plan.emit_from_step - plan.start_step
        overlap_bytes = overlap_steps * bytes_per_step
        if len(decoded) < overlap_bytes:
            raise IncrementalDecoderError(
                f"Decoded audio is shorter than the overlap window: {len(decoded)} < {overlap_bytes}"
            )

        self.emitted_until_step = plan.end_step
        self._trim_codec_buffer()
        return self._apply_crossfade(decoded[overlap_bytes:], channels=channels, finished=finished)

    def snapshot(self) -> dict[str, int]:
        return {
            "emitted_until_step": self.emitted_until_step,
            "buffer_start_step": self.buffer_start_step,
            "buffered_step_count": len(self._codec_buffer),
        }

    def _trim_codec_buffer(self) -> None:
        keep_from_step = max(0, self.emitted_until_step - self.config.left_context_steps)
        drop_count = keep_from_step - self.buffer_start_step
        for _ in range(max(0, drop_count)):
            self._codec_buffer.popleft()
        self.buffer_start_step = keep_from_step

    def _apply_crossfade(self, audio_bytes: bytes, *, channels: int, finished: bool) -> bytes:
        if self.config.crossfade_samples <= 0:
            return audio_bytes

        frame_bytes = channels * 2
        if not audio_bytes:
            if finished and self._pending_crossfade_tail:
                pending = self._pending_crossfade_tail
                self._pending_crossfade_tail = b""
                return pending
            return b""

        output = bytearray()
        remaining = audio_bytes
        if self._pending_crossfade_tail:
            blend_bytes = min(len(self._pending_crossfade_tail), len(remaining))
            blend_bytes -= blend_bytes % frame_bytes
            if blend_bytes > 0:
                output.extend(
                    self._crossfade_pcm16(
                        self._pending_crossfade_tail[:blend_bytes],
                        remaining[:blend_bytes],
                        channels=channels,
                    )
                )
                self._pending_crossfade_tail = self._pending_crossfade_tail[blend_bytes:]
                remaining = remaining[blend_bytes:]
            if self._pending_crossfade_tail:
                if finished:
                    output.extend(self._pending_crossfade_tail)
                    self._pending_crossfade_tail = b""
                return bytes(output)

        if finished:
            output.extend(remaining)
            self._pending_crossfade_tail = b""
            return bytes(output)

        holdback_bytes = min(len(remaining), self.config.crossfade_samples * frame_bytes)
        holdback_bytes -= holdback_bytes % frame_bytes
        if holdback_bytes <= 0:
            output.extend(remaining)
            self._pending_crossfade_tail = b""
            return bytes(output)

        output.extend(remaining[:-holdback_bytes])
        self._pending_crossfade_tail = remaining[-holdback_bytes:]
        return bytes(output)

    @staticmethod
    def _crossfade_pcm16(previous: bytes, current: bytes, *, channels: int) -> bytes:
        if len(previous) != len(current):
            raise IncrementalDecoderError("Crossfade inputs must have the same byte length")
        if not previous:
            return b""

        frame_bytes = channels * 2
        if len(previous) % frame_bytes != 0:
            raise IncrementalDecoderError("Crossfade inputs must align to PCM16 frame boundaries")

        frame_count = len(previous) // frame_bytes
        previous_view = memoryview(previous).cast("h")
        current_view = memoryview(current).cast("h")
        blended = bytearray(len(previous))
        blended_view = memoryview(blended).cast("h")

        for frame_index in range(frame_count):
            if frame_count == 1:
                current_weight = 0.5
            else:
                current_weight = frame_index / float(frame_count - 1)
            previous_weight = 1.0 - current_weight
            base_index = frame_index * channels
            for channel_index in range(channels):
                sample_index = base_index + channel_index
                sample = int(round(
                    previous_view[sample_index] * previous_weight
                    + current_view[sample_index] * current_weight
                ))
                blended_view[sample_index] = max(-32768, min(32767, sample))
        return bytes(blended)
