from __future__ import annotations

from dataclasses import dataclass, field
import time
from typing import Any, Iterator, Sequence

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
    prompt_build_ms: float | None = None
    prefill_ms: float | None = None
    state_restore_ms: float | None = None
    init_total_ms: float | None = None
    first_step_ms: float | None = None
    first_decode_ms: float | None = None
    first_chunk_ready_ms: float | None = None
    total_step_ms: float = 0.0
    total_decode_ms: float = 0.0


@dataclass
class StreamingGenerationState:
    attention_mask: torch.Tensor
    past_key_values: Any
    past_hidden: torch.Tensor
    generation_step: int
    trailing_text_hidden: torch.Tensor
    tts_pad_embed: torch.Tensor
    next_logits: torch.Tensor
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
        **generate_kwargs: Any,
    ) -> None:
        self._created_at = time.perf_counter()
        self.qwen3tts = qwen3tts
        self.text = text
        self.speaker = speaker
        self.language = language
        self.instruct = instruct
        self.runtime_session = runtime_session if isinstance(runtime_session, RuntimeSession) else None
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
    def _prepare_next_token_logits(logits: torch.Tensor) -> torch.Tensor:
        return logits[:, -1, :].to(dtype=torch.float32, device=logits.device, copy=True)

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
            self._sync_runtime_session()
            return chunk

        step_outputs = self._run_single_step(next_token)
        codec_ids = self._extract_codec_ids(step_outputs)
        self.state.generated_codes.append(codec_ids)
        self.state.sampled_tokens.append(token_id)
        self.decoder.push_codec_step(codec_ids)
        self.metrics.generated_steps += 1
        self.state.attention_mask = getattr(step_outputs, "streaming_attention_mask")
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
            self._sync_runtime_session()
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
        self._sync_runtime_session()
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
        return StreamingGenerationState(
            attention_mask=self.prompt.attention_mask.clone(),
            past_key_values=outputs.past_key_values,
            past_hidden=outputs.past_hidden,
            generation_step=int(outputs.generation_step),
            trailing_text_hidden=outputs.trailing_text_hidden,
            tts_pad_embed=outputs.tts_pad_embed,
            next_logits=self._prepare_next_token_logits(outputs.logits),
        )

    def _state_from_runtime_session(self, runtime_session: RuntimeSession) -> StreamingGenerationState:
        state = runtime_session.state
        return StreamingGenerationState(
            attention_mask=state.attention_mask,
            past_key_values=state.past_key_values,
            past_hidden=state.past_hidden,
            generation_step=state.generation_step,
            trailing_text_hidden=state.trailing_text_hidden,
            tts_pad_embed=state.tts_pad_embed,
            next_logits=state.next_logits.to(dtype=torch.float32, copy=True),
            generated_codes=list(state.generated_codes),
            sampled_tokens=list(state.sampled_tokens),
            finished=state.generation_finished,
        )

    def _run_single_step(self, next_token: torch.Tensor):
        next_attention_mask = torch.cat(
            [self.state.attention_mask, torch.ones_like(self.state.attention_mask[:, :1])], dim=1
        )
        cache_position = torch.arange(
            self.state.attention_mask.shape[1],
            self.state.attention_mask.shape[1] + next_token.shape[1],
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
        audio_bytes = self.decoder.decode_buffered(
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

    def _decode_codec_window(self, codec_window: Sequence[torch.Tensor]) -> bytes:
        if not codec_window:
            return b""
        codes = torch.stack(list(codec_window), dim=0)
        wavs, _ = self.qwen3tts.model.speech_tokenizer.decode([{"audio_codes": codes}])
        return float_audio_to_pcm16le_bytes(wavs[0])

    def _sync_runtime_session(self) -> None:
        if self.runtime_session is None:
            return
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
            metrics={
                "generated_steps": self.metrics.generated_steps,
                "emitted_chunks": self.metrics.emitted_chunks,
                "first_emitted_step": self.metrics.first_emitted_step,
                "finish_reason": self.metrics.finish_reason,
                "prompt_build_ms": round(self.metrics.prompt_build_ms, 2) if self.metrics.prompt_build_ms is not None else None,
                "prefill_ms": round(self.metrics.prefill_ms, 2) if self.metrics.prefill_ms is not None else None,
                "state_restore_ms": round(self.metrics.state_restore_ms, 2) if self.metrics.state_restore_ms is not None else None,
                "init_total_ms": round(self.metrics.init_total_ms, 2) if self.metrics.init_total_ms is not None else None,
                "first_step_ms": round(self.metrics.first_step_ms, 2) if self.metrics.first_step_ms is not None else None,
                "first_decode_ms": round(self.metrics.first_decode_ms, 2) if self.metrics.first_decode_ms is not None else None,
                "first_chunk_ready_ms": round(self.metrics.first_chunk_ready_ms, 2) if self.metrics.first_chunk_ready_ms is not None else None,
                "total_step_ms": round(self.metrics.total_step_ms, 2),
                "total_decode_ms": round(self.metrics.total_decode_ms, 2),
                "decoder": self.decoder.snapshot(),
            },
        )

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
