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

    def __post_init__(self) -> None:
        if self.chunk_steps <= 0:
            raise ValueError("chunk_steps must be > 0")
        if self.left_context_steps < 0:
            raise ValueError("left_context_steps must be >= 0")


class IncrementalAudioDecoder:
    def __init__(self, config: IncrementalDecoderConfig | None = None) -> None:
        self.config = config or IncrementalDecoderConfig()
        self.emitted_until_step = 0
        self.buffer_start_step = 0
        self._codec_buffer: deque[Any] = deque()

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
        if not force and not finished and new_steps < self.config.chunk_steps:
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
        force: bool = False,
        finished: bool = False,
    ) -> bytes | None:
        if bytes_per_step <= 0:
            raise ValueError("bytes_per_step must be > 0")

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
        return decoded[overlap_bytes:]

    def decode_buffered(
        self,
        *,
        decode_fn: Callable[[Sequence[Any]], bytes],
        bytes_per_step: int,
        force: bool = False,
        finished: bool = False,
    ) -> bytes | None:
        if bytes_per_step <= 0:
            raise ValueError("bytes_per_step must be > 0")

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
        return decoded[overlap_bytes:]

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
