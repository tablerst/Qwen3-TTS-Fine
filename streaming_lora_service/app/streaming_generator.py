from __future__ import annotations

from dataclasses import dataclass, field
import time
from typing import Any, Iterator

import torch

from .audio_utils import float_audio_to_pcm16le_bytes
from .incremental_decoder import IncrementalAudioDecoder, IncrementalDecoderConfig
from .prompt_builder import build_custom_voice_talker_prompt
from .runtime_session import RuntimeSession


class StreamingGenerationError(RuntimeError):
    """Raised when the true-streaming generator cannot make progress."""


@dataclass(frozen=True)
class StreamingAudioChunk:
    audio_bytes: bytes
    start_step: int
    end_step: int
    sample_rate: int
    channels: int = 1


@dataclass
class StreamingGenerationMetrics:
    generated_steps: int = 0
    emitted_chunks: int = 0
    first_emitted_step: int | None = None
    finish_reason: str | None = None
    state_sync_calls: int = 0
    prompt_build_ms: float | None = None
    prefill_ms: float | None = None
    state_restore_ms: float | None = None
    init_total_ms: float | None = None
    first_step_ms: float | None = None
    first_forward_ms: float | None = None
    first_decode_ms: float | None = None
    first_chunk_ready_ms: float | None = None
    total_step_ms: float = 0.0
    total_forward_ms: float = 0.0
    total_decode_ms: float = 0.0
    total_state_sync_ms: float = 0.0


@dataclass
class StreamingGenerationState:
    attention_mask: torch.Tensor
    attention_mask_buffer: torch.Tensor
    attention_length: int
    past_key_values: Any
    past_hidden: torch.Tensor
    generation_step: int
    trailing_text_hidden: torch.Tensor
    tts_pad_embed: torch.Tensor
    next_logits: torch.Tensor
    generated_code_buffer: torch.Tensor | None = None
    generated_code_count: int = 0
    generated_codes: list[torch.Tensor] = field(default_factory=list)
    sampled_tokens: list[int] = field(default_factory=list)
    finished: bool = False


class StreamingCustomVoiceGenerator:
    def __init__(
        self,
        qwen3tts: Any,
        *,
        text: str,
        speaker: str,
        language: str = "Auto",
        instruct: str | None = None,
        runtime_session: RuntimeSession | None = None,
        chunk_steps: int = 4,
        left_context_steps: int = 25,
        first_chunk_steps: int | None = None,
        crossfade_samples: int = 0,
        runtime_session_sync_mode: str = "chunk",
        **generate_kwargs: Any,
    ) -> None:
        self._created_at = time.perf_counter()
        self.qwen3tts = qwen3tts
        self.text = text
        self.speaker = speaker
        self.language = language
        self.instruct = instruct
        self.runtime_session = runtime_session if isinstance(runtime_session, RuntimeSession) else None
        self.runtime_session_sync_mode = self._normalize_runtime_session_sync_mode(runtime_session_sync_mode)
        prompt_started_at = time.perf_counter()
        self.prompt = build_custom_voice_talker_prompt(
            qwen3tts,
            text=text,
            speaker=speaker,
            language=language,
            instruct=instruct,
            non_streaming_mode=False,
            **generate_kwargs,
        )
        self.metrics = StreamingGenerationMetrics()
        self.metrics.prompt_build_ms = (time.perf_counter() - prompt_started_at) * 1000.0
        decoder_config = IncrementalDecoderConfig(
            chunk_steps=chunk_steps,
            left_context_steps=left_context_steps,
            first_chunk_steps=first_chunk_steps,
            crossfade_samples=crossfade_samples,
        )
        resumable_session = self.runtime_session if self.runtime_session is not None and self.runtime_session.can_resume_generation(
            text=text,
            speaker_name=speaker,
            language_type=language,
            instructions=instruct,
        ) else None
        if resumable_session is not None:
            state_restore_started_at = time.perf_counter()
            existing_decoder = resumable_session.state.incremental_decoder
            if not isinstance(existing_decoder, IncrementalAudioDecoder):
                raise StreamingGenerationError("Runtime session stored an invalid incremental decoder")
            self.decoder = existing_decoder
            self._state = self._state_from_runtime_session(resumable_session)
            self.metrics.state_restore_ms = (time.perf_counter() - state_restore_started_at) * 1000.0
        else:
            self.decoder = IncrementalAudioDecoder(decoder_config)
            if self.runtime_session is not None:
                self.runtime_session.reset_generation_state()
            prefill_started_at = time.perf_counter()
            self._state = self._prefill()
            self.metrics.prefill_ms = (time.perf_counter() - prefill_started_at) * 1000.0
        self.metrics.init_total_ms = (time.perf_counter() - self._created_at) * 1000.0
        self._bytes_per_step = self.prompt.decode_upsample_rate * 2
        self._sync_runtime_session()

    @property
    def state(self) -> StreamingGenerationState:
        return self._state

    @property
    def sample_rate(self) -> int:
        return self.prompt.sample_rate

    @staticmethod
    def _normalize_runtime_session_sync_mode(mode: str) -> str:
        normalized = (mode or "chunk").strip().lower()
        if normalized not in {"step", "chunk", "final"}:
            raise ValueError(
                "runtime_session_sync_mode must be one of ('step', 'chunk', 'final'); "
                f"got {mode!r}"
            )
        return normalized

    @staticmethod
    def _prepare_next_token_logits(logits: torch.Tensor) -> torch.Tensor:
        return logits[:, -1, :].to(dtype=torch.float32, device=logits.device, copy=True)

    def _build_attention_mask_buffer(
        self,
        attention_mask: torch.Tensor,
        *,
        min_capacity: int | None = None,
    ) -> torch.Tensor:
        current_length = int(attention_mask.shape[1])
        requested_capacity = current_length + self.prompt.sampling.max_new_tokens + 1
        capacity = max(current_length, requested_capacity, int(min_capacity or 0))
        buffer = torch.ones(
            (attention_mask.shape[0], capacity),
            device=attention_mask.device,
            dtype=attention_mask.dtype,
        )
        buffer[:, :current_length] = attention_mask
        return buffer

    def _ensure_attention_mask_capacity(self, min_capacity: int) -> None:
        if self.state.attention_mask_buffer.shape[1] >= min_capacity:
            return
        grown_capacity = max(min_capacity, self.state.attention_mask_buffer.shape[1] * 2)
        buffer = self._build_attention_mask_buffer(self.state.attention_mask, min_capacity=grown_capacity)
        self.state.attention_mask_buffer = buffer
        self.state.attention_mask = buffer[:, : self.state.attention_length]

    def _build_generated_code_buffer(
        self,
        codes: torch.Tensor,
        *,
        min_capacity: int | None = None,
    ) -> torch.Tensor:
        if codes.ndim != 2:
            raise StreamingGenerationError(f"Expected generated code buffer source with shape (steps, width), got {tuple(codes.shape)}")
        current_steps = int(codes.shape[0])
        width = int(codes.shape[1])
        requested_capacity = max(1, self.prompt.sampling.max_new_tokens)
        capacity = max(current_steps, requested_capacity, int(min_capacity or 0))
        buffer = torch.empty(
            (capacity, width),
            device=codes.device,
            dtype=codes.dtype,
        )
        if current_steps > 0:
            buffer[:current_steps].copy_(codes)
        return buffer

    def _ensure_generated_code_capacity(self, min_capacity: int, *, width: int, device: torch.device, dtype: torch.dtype) -> None:
        buffer = self.state.generated_code_buffer
        if buffer is not None and buffer.shape[0] >= min_capacity and buffer.shape[1] == width:
            return

        if buffer is None:
            source = torch.empty((0, width), device=device, dtype=dtype)
            grown_capacity = max(min_capacity, max(1, self.prompt.sampling.max_new_tokens))
        else:
            source = buffer[: self.state.generated_code_count]
            grown_capacity = max(min_capacity, buffer.shape[0] * 2)
        self.state.generated_code_buffer = self._build_generated_code_buffer(source, min_capacity=grown_capacity)

    def _append_generated_code(self, codec_ids: torch.Tensor) -> torch.Tensor:
        if codec_ids.ndim != 1:
            raise StreamingGenerationError(f"Expected codec_ids with shape (width,), got {tuple(codec_ids.shape)}")

        normalized = codec_ids.to(dtype=torch.long, copy=False)
        next_index = self.state.generated_code_count
        self._ensure_generated_code_capacity(
            next_index + 1,
            width=int(normalized.shape[0]),
            device=normalized.device,
            dtype=normalized.dtype,
        )
        assert self.state.generated_code_buffer is not None
        self.state.generated_code_buffer[next_index].copy_(normalized)
        stored = self.state.generated_code_buffer[next_index]
        self.state.generated_codes.append(stored)
        self.state.generated_code_count = next_index + 1
        return stored

    def iter_audio_chunks(self) -> Iterator[bytes]:
        while not self.state.finished:
            chunk = self.step()
            if chunk is not None and chunk.audio_bytes:
                yield chunk.audio_bytes

    def step(self) -> StreamingAudioChunk | None:
        if self.state.finished:
            return None

        step_started_at = time.perf_counter()
        next_token = self._sample_next_codec_token(self.state.next_logits)
        token_id = int(next_token.item())
        if token_id == self.prompt.sampling.eos_token_id:
            self.metrics.finish_reason = "eos"
            self.state.finished = True
            decode_started_at = time.perf_counter()
            chunk = self._emit_ready_audio(force=True, finished=True)
            decode_elapsed_ms = (time.perf_counter() - decode_started_at) * 1000.0
            self.metrics.total_decode_ms += decode_elapsed_ms
            if chunk is not None and self.metrics.first_decode_ms is None:
                self.metrics.first_decode_ms = decode_elapsed_ms
            step_elapsed_ms = (time.perf_counter() - step_started_at) * 1000.0
            self.metrics.total_step_ms += step_elapsed_ms
            if self.metrics.first_step_ms is None:
                self.metrics.first_step_ms = step_elapsed_ms
            if chunk is not None and chunk.audio_bytes and self.metrics.first_chunk_ready_ms is None:
                self.metrics.first_chunk_ready_ms = (time.perf_counter() - self._created_at) * 1000.0
            self._maybe_sync_runtime_session(chunk_emitted=bool(chunk and chunk.audio_bytes), force=True)
            return chunk

        forward_started_at = time.perf_counter()
        step_outputs = self._run_single_step(next_token)
        forward_elapsed_ms = (time.perf_counter() - forward_started_at) * 1000.0
        self.metrics.total_forward_ms += forward_elapsed_ms
        if self.metrics.first_forward_ms is None:
            self.metrics.first_forward_ms = forward_elapsed_ms
        codec_ids = self._extract_codec_ids(step_outputs)
        self._append_generated_code(codec_ids)
        self.state.sampled_tokens.append(token_id)
        self.metrics.generated_steps += 1
        self.state.attention_mask = getattr(step_outputs, "streaming_attention_mask")
        self.state.attention_length = int(getattr(step_outputs, "streaming_attention_length", self.state.attention_mask.shape[1]))
        self.state.past_key_values = step_outputs.past_key_values
        self.state.past_hidden = step_outputs.past_hidden
        self.state.generation_step = int(step_outputs.generation_step)
        self.state.next_logits = self._prepare_next_token_logits(step_outputs.logits)
        self.state.trailing_text_hidden = step_outputs.trailing_text_hidden
        self.state.tts_pad_embed = step_outputs.tts_pad_embed

        if self.metrics.generated_steps >= self.prompt.sampling.max_new_tokens:
            self.metrics.finish_reason = "length"
            self.state.finished = True
            decode_started_at = time.perf_counter()
            chunk = self._emit_ready_audio(force=True, finished=True)
            decode_elapsed_ms = (time.perf_counter() - decode_started_at) * 1000.0
            self.metrics.total_decode_ms += decode_elapsed_ms
            if chunk is not None and self.metrics.first_decode_ms is None:
                self.metrics.first_decode_ms = decode_elapsed_ms
            step_elapsed_ms = (time.perf_counter() - step_started_at) * 1000.0
            self.metrics.total_step_ms += step_elapsed_ms
            if self.metrics.first_step_ms is None:
                self.metrics.first_step_ms = step_elapsed_ms
            if chunk is not None and chunk.audio_bytes and self.metrics.first_chunk_ready_ms is None:
                self.metrics.first_chunk_ready_ms = (time.perf_counter() - self._created_at) * 1000.0
            self._maybe_sync_runtime_session(chunk_emitted=bool(chunk and chunk.audio_bytes), force=True)
            return chunk

        decode_started_at = time.perf_counter()
        chunk = self._emit_ready_audio(force=False, finished=False)
        decode_elapsed_ms = (time.perf_counter() - decode_started_at) * 1000.0
        self.metrics.total_decode_ms += decode_elapsed_ms
        if chunk is not None and self.metrics.first_decode_ms is None:
            self.metrics.first_decode_ms = decode_elapsed_ms
        step_elapsed_ms = (time.perf_counter() - step_started_at) * 1000.0
        self.metrics.total_step_ms += step_elapsed_ms
        if self.metrics.first_step_ms is None:
            self.metrics.first_step_ms = step_elapsed_ms
        if chunk is not None and chunk.audio_bytes and self.metrics.first_chunk_ready_ms is None:
            self.metrics.first_chunk_ready_ms = (time.perf_counter() - self._created_at) * 1000.0
        self._maybe_sync_runtime_session(chunk_emitted=bool(chunk and chunk.audio_bytes), force=False)
        return chunk

    def _prefill(self) -> StreamingGenerationState:
        talker = self.qwen3tts.model.talker
        if hasattr(talker, "rope_deltas"):
            talker.rope_deltas = None
        with torch.inference_mode():
            outputs = talker(
                inputs_embeds=self.prompt.talker_input_embeds,
                attention_mask=self.prompt.attention_mask,
                trailing_text_hidden=self.prompt.trailing_text_hidden,
                tts_pad_embed=self.prompt.tts_pad_embed,
                use_cache=True,
                output_hidden_states=True,
                return_dict=True,
                subtalker_dosample=self.prompt.sampling.subtalker_dosample,
                subtalker_top_k=self.prompt.sampling.subtalker_top_k,
                subtalker_top_p=self.prompt.sampling.subtalker_top_p,
                subtalker_temperature=self.prompt.sampling.subtalker_temperature,
            )
        attention_mask_buffer = self._build_attention_mask_buffer(self.prompt.attention_mask)
        return StreamingGenerationState(
            attention_mask=attention_mask_buffer[:, : self.prompt.attention_mask.shape[1]],
            attention_mask_buffer=attention_mask_buffer,
            attention_length=int(self.prompt.attention_mask.shape[1]),
            past_key_values=outputs.past_key_values,
            past_hidden=outputs.past_hidden,
            generation_step=int(outputs.generation_step),
            trailing_text_hidden=outputs.trailing_text_hidden,
            tts_pad_embed=outputs.tts_pad_embed,
            next_logits=self._prepare_next_token_logits(outputs.logits),
            generated_code_buffer=None,
            generated_code_count=0,
        )

    def _state_from_runtime_session(self, runtime_session: RuntimeSession) -> StreamingGenerationState:
        state = runtime_session.state
        if state.attention_mask is None:
            raise StreamingGenerationError("Runtime session is missing attention_mask for generation restore")
        attention_length = int(state.attention_mask.shape[1])
        attention_mask_buffer = self._build_attention_mask_buffer(state.attention_mask)
        generated_codes = list(state.generated_codes)
        generated_code_buffer: torch.Tensor | None = None
        generated_code_count = len(generated_codes)
        if generated_codes:
            restored_codes = torch.stack(generated_codes, dim=0)
            generated_code_buffer = self._build_generated_code_buffer(restored_codes)
            generated_codes = [generated_code_buffer[index] for index in range(generated_code_count)]
        return StreamingGenerationState(
            attention_mask=attention_mask_buffer[:, :attention_length],
            attention_mask_buffer=attention_mask_buffer,
            attention_length=attention_length,
            past_key_values=state.past_key_values,
            past_hidden=state.past_hidden,
            generation_step=state.generation_step,
            trailing_text_hidden=state.trailing_text_hidden,
            tts_pad_embed=state.tts_pad_embed,
            next_logits=state.next_logits.to(dtype=torch.float32, copy=True),
            generated_code_buffer=generated_code_buffer,
            generated_code_count=generated_code_count,
            generated_codes=generated_codes,
            sampled_tokens=list(state.sampled_tokens),
            finished=state.generation_finished,
        )

    def _run_single_step(self, next_token: torch.Tensor):
        next_length = self.state.attention_length + int(next_token.shape[1])
        self._ensure_attention_mask_capacity(next_length)
        next_attention_mask = self.state.attention_mask_buffer[:, :next_length]
        cache_position = torch.arange(
            self.state.attention_length,
            next_length,
            device=next_token.device,
        )
        with torch.inference_mode():
            outputs = self.qwen3tts.model.talker(
                input_ids=next_token,
                attention_mask=next_attention_mask,
                cache_position=cache_position,
                past_key_values=self.state.past_key_values,
                past_hidden=self.state.past_hidden,
                generation_step=self.state.generation_step,
                trailing_text_hidden=self.state.trailing_text_hidden,
                tts_pad_embed=self.state.tts_pad_embed,
                use_cache=True,
                output_hidden_states=True,
                return_dict=True,
                subtalker_dosample=self.prompt.sampling.subtalker_dosample,
                subtalker_top_k=self.prompt.sampling.subtalker_top_k,
                subtalker_top_p=self.prompt.sampling.subtalker_top_p,
                subtalker_temperature=self.prompt.sampling.subtalker_temperature,
            )
            setattr(outputs, "streaming_attention_mask", next_attention_mask)
            setattr(outputs, "streaming_attention_length", next_length)
            return outputs

    def _extract_codec_ids(self, step_outputs: Any) -> torch.Tensor:
        hidden_states = getattr(step_outputs, "hidden_states", None)
        if not hidden_states or len(hidden_states) < 2 or hidden_states[1] is None:
            raise StreamingGenerationError("talker forward did not return codec ids in hidden_states")
        codec_ids = hidden_states[1]
        if codec_ids.ndim == 1:
            return codec_ids.to(dtype=torch.long)
        if codec_ids.ndim != 2 or codec_ids.shape[0] != 1:
            raise StreamingGenerationError(f"Unexpected codec_ids shape: {tuple(codec_ids.shape)}")
        return codec_ids[0].to(dtype=torch.long)

    def _emit_ready_audio(self, *, force: bool, finished: bool) -> StreamingAudioChunk | None:
        audio_bytes = self.decoder.decode(
            self.state.generated_code_count,
            decode_fn=self._decode_codec_window,
            bytes_per_step=self._bytes_per_step,
            channels=1,
            force=force,
            finished=finished,
        )
        if not audio_bytes:
            return None

        end_step = self.decoder.emitted_until_step
        start_step = end_step - max(1, len(audio_bytes) // self._bytes_per_step)
        self.metrics.emitted_chunks += 1
        if self.metrics.first_emitted_step is None:
            self.metrics.first_emitted_step = end_step
        return StreamingAudioChunk(
            audio_bytes=audio_bytes,
            start_step=max(0, start_step),
            end_step=end_step,
            sample_rate=self.prompt.sample_rate,
            channels=1,
        )

    def _decode_codec_window(self, start_step: int, end_step: int) -> bytes:
        if start_step >= end_step:
            return b""
        if self.state.generated_code_buffer is None:
            raise StreamingGenerationError("Generated code buffer is unavailable for decode")
        codes = self.state.generated_code_buffer[start_step:end_step]
        wavs, _ = self.qwen3tts.model.speech_tokenizer.decode([{"audio_codes": codes}])
        return float_audio_to_pcm16le_bytes(wavs[0])

    def _sync_runtime_session(self) -> None:
        if self.runtime_session is None:
            return
        sync_started_at = time.perf_counter()
        self.runtime_session.bind_generation_state(
            text=self.text,
            speaker_name=self.speaker,
            language_type=self.language,
            instructions=self.instruct,
            attention_mask=self.state.attention_mask,
            past_key_values=self.state.past_key_values,
            past_hidden=self.state.past_hidden,
            generation_step=self.state.generation_step,
            trailing_text_hidden=self.state.trailing_text_hidden,
            tts_pad_embed=self.state.tts_pad_embed,
            next_logits=self.state.next_logits,
            generated_codes=self.state.generated_codes,
            sampled_tokens=self.state.sampled_tokens,
            decoded_until_step=self.decoder.emitted_until_step,
            generation_finished=self.state.finished,
            incremental_decoder=self.decoder,
            metrics=self._build_metrics_snapshot(),
        )
        sync_elapsed_ms = (time.perf_counter() - sync_started_at) * 1000.0
        self.metrics.state_sync_calls += 1
        self.metrics.total_state_sync_ms += sync_elapsed_ms
        self.runtime_session.state.last_generation_metrics["state_sync_calls"] = self.metrics.state_sync_calls
        self.runtime_session.state.last_generation_metrics["total_state_sync_ms"] = round(self.metrics.total_state_sync_ms, 2)

    def _maybe_sync_runtime_session(self, *, chunk_emitted: bool, force: bool) -> None:
        if self.runtime_session is None:
            return
        if force or self.runtime_session_sync_mode == "step":
            self._sync_runtime_session()
            return
        if self.runtime_session_sync_mode == "chunk" and chunk_emitted:
            self._sync_runtime_session()

    def _build_metrics_snapshot(self) -> dict[str, Any]:
        generated_steps = max(0, self.metrics.generated_steps)
        emitted_chunks = max(0, self.metrics.emitted_chunks)
        state_sync_calls = max(0, self.metrics.state_sync_calls)
        return {
            "generated_steps": self.metrics.generated_steps,
            "emitted_chunks": self.metrics.emitted_chunks,
            "first_emitted_step": self.metrics.first_emitted_step,
            "finish_reason": self.metrics.finish_reason,
            "state_sync_calls": self.metrics.state_sync_calls,
            "prompt_build_ms": round(self.metrics.prompt_build_ms, 2) if self.metrics.prompt_build_ms is not None else None,
            "prefill_ms": round(self.metrics.prefill_ms, 2) if self.metrics.prefill_ms is not None else None,
            "state_restore_ms": round(self.metrics.state_restore_ms, 2) if self.metrics.state_restore_ms is not None else None,
            "init_total_ms": round(self.metrics.init_total_ms, 2) if self.metrics.init_total_ms is not None else None,
            "first_step_ms": round(self.metrics.first_step_ms, 2) if self.metrics.first_step_ms is not None else None,
            "first_forward_ms": round(self.metrics.first_forward_ms, 2) if self.metrics.first_forward_ms is not None else None,
            "first_decode_ms": round(self.metrics.first_decode_ms, 2) if self.metrics.first_decode_ms is not None else None,
            "first_chunk_ready_ms": round(self.metrics.first_chunk_ready_ms, 2) if self.metrics.first_chunk_ready_ms is not None else None,
            "total_step_ms": round(self.metrics.total_step_ms, 2),
            "total_forward_ms": round(self.metrics.total_forward_ms, 2),
            "total_decode_ms": round(self.metrics.total_decode_ms, 2),
            "total_state_sync_ms": round(self.metrics.total_state_sync_ms, 2),
            "avg_step_ms": round(self.metrics.total_step_ms / generated_steps, 2) if generated_steps else None,
            "avg_forward_ms": round(self.metrics.total_forward_ms / generated_steps, 2) if generated_steps else None,
            "avg_decode_ms": round(self.metrics.total_decode_ms / emitted_chunks, 2) if emitted_chunks else None,
            "avg_state_sync_ms": round(self.metrics.total_state_sync_ms / state_sync_calls, 2) if state_sync_calls else None,
            "decoder": self.decoder.snapshot(),
            "runtime_session_sync_mode": self.runtime_session_sync_mode,
        }

    def _sample_next_codec_token(self, logits: torch.Tensor) -> torch.Tensor:
        if logits.ndim != 2 or logits.shape[0] != 1:
            raise StreamingGenerationError(f"Expected logits with shape (1, vocab), got {tuple(logits.shape)}")

        processed = logits.to(dtype=torch.float32, copy=True)
        if self.prompt.sampling.suppress_tokens:
            processed[:, list(self.prompt.sampling.suppress_tokens)] = -torch.inf
        self._apply_repetition_penalty(processed)

        if not self.prompt.sampling.do_sample or self.prompt.sampling.temperature <= 0:
            return torch.argmax(processed, dim=-1, keepdim=True)

        processed = processed / self.prompt.sampling.temperature
        processed = self._apply_top_k_top_p(processed)
        probs = torch.softmax(processed, dim=-1)
        return torch.multinomial(probs, num_samples=1)

    def _apply_repetition_penalty(self, logits: torch.Tensor) -> None:
        penalty = self.prompt.sampling.repetition_penalty
        if penalty <= 1.0:
            return
        for token_id in sorted(set(self.state.sampled_tokens)):
            current = logits[0, token_id]
            logits[0, token_id] = current / penalty if current >= 0 else current * penalty

    def _apply_top_k_top_p(self, logits: torch.Tensor) -> torch.Tensor:
        top_k = self.prompt.sampling.top_k
        if top_k > 0 and top_k < logits.shape[-1]:
            threshold = torch.topk(logits, top_k, dim=-1).values[..., -1, None]
            logits = logits.masked_fill(logits < threshold, -torch.inf)

        top_p = self.prompt.sampling.top_p
        if 0 < top_p < 1.0:
            sorted_logits, sorted_indices = torch.sort(logits, descending=True, dim=-1)
            sorted_probs = torch.softmax(sorted_logits, dim=-1)
            cumulative_probs = torch.cumsum(sorted_probs, dim=-1)
            sorted_indices_to_remove = cumulative_probs > top_p
            sorted_indices_to_remove[..., 0] = False
            indices_to_remove = torch.zeros_like(sorted_indices_to_remove, dtype=torch.bool)
            indices_to_remove.scatter_(1, sorted_indices, sorted_indices_to_remove)
            logits = logits.masked_fill(indices_to_remove, -torch.inf)
        return logits


__all__ = [
    "StreamingAudioChunk",
    "StreamingCustomVoiceGenerator",
    "StreamingGenerationError",
    "StreamingGenerationMetrics",
    "StreamingGenerationState",
]
