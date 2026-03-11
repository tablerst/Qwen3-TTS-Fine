from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from .models import SessionOptions


class RuntimeSessionError(RuntimeError):
    """Raised when a session operation is invalid for the current state."""


@dataclass
class RuntimeSessionState:
    session_id: str
    options: SessionOptions
    raw_text_buffer: str = ""
    committed_text: str = ""
    pending_text_tail: str = ""
    committed_chunks: list[str] = field(default_factory=list)
    active_generation_text: str = ""
    active_speaker_name: str | None = None
    active_language_type: str = "Auto"
    active_instructions: str = ""
    attention_mask: Any = None
    past_key_values: Any = None
    past_hidden: Any = None
    generation_step: int = 0
    trailing_text_hidden: Any = None
    tts_pad_embed: Any = None
    next_logits: Any = None
    generated_codes: list[Any] = field(default_factory=list)
    sampled_tokens: list[int] = field(default_factory=list)
    decoded_until_step: int = 0
    generation_finished: bool = False
    incremental_decoder: Any = None
    last_generation_metrics: dict[str, Any] = field(default_factory=dict)
    finished: bool = False


class RuntimeSession:
    def __init__(self, session_id: str, options: SessionOptions) -> None:
        self.state = RuntimeSessionState(session_id=session_id, options=options)
        self.reset_generation_state()

    @property
    def session_id(self) -> str:
        return self.state.session_id

    @property
    def options(self) -> SessionOptions:
        return self.state.options

    def append_text(self, text: str) -> None:
        if self.state.finished:
            raise RuntimeSessionError("Cannot append text after session.finish()")
        if not text:
            return
        self.state.raw_text_buffer += text
        self.state.pending_text_tail = self.state.raw_text_buffer[len(self.state.committed_text) :]

    def commit(self) -> str:
        if self.state.finished:
            raise RuntimeSessionError("Cannot commit text after session.finish()")
        if not self.state.pending_text_tail:
            raise RuntimeSessionError("No pending text to commit")
        committed_chunk = self.state.pending_text_tail
        self.state.committed_text = self.state.raw_text_buffer
        self.state.pending_text_tail = ""
        self.state.committed_chunks.append(committed_chunk)
        return committed_chunk

    def clear_pending_text(self) -> None:
        if self.state.finished:
            raise RuntimeSessionError("Cannot clear text after session.finish()")
        self.state.raw_text_buffer = self.state.committed_text
        self.state.pending_text_tail = ""

    def finish(self) -> str | None:
        if self.state.finished:
            return None
        final_chunk = self.commit() if self.state.pending_text_tail else None
        self.state.finished = True
        return final_chunk

    def reset_generation_state(self) -> None:
        self.state.active_generation_text = ""
        self.state.active_speaker_name = None
        self.state.active_language_type = self.state.options.language_type
        self.state.active_instructions = self.state.options.instructions
        self.state.attention_mask = None
        self.state.past_key_values = None
        self.state.past_hidden = None
        self.state.generation_step = 0
        self.state.trailing_text_hidden = None
        self.state.tts_pad_embed = None
        self.state.next_logits = None
        self.state.generated_codes = []
        self.state.sampled_tokens = []
        self.state.decoded_until_step = 0
        self.state.generation_finished = False
        self.state.incremental_decoder = None
        self.state.last_generation_metrics = {}

    def can_resume_generation(
        self,
        *,
        text: str,
        speaker_name: str,
        language_type: str,
        instructions: str | None,
    ) -> bool:
        return (
            not self.state.finished
            and not self.state.generation_finished
            and self.state.active_generation_text == text
            and self.state.active_speaker_name == speaker_name
            and self.state.active_language_type == language_type
            and self.state.active_instructions == (instructions or "")
            and self.state.attention_mask is not None
            and self.state.past_key_values is not None
            and self.state.past_hidden is not None
            and self.state.trailing_text_hidden is not None
            and self.state.tts_pad_embed is not None
            and self.state.next_logits is not None
            and self.state.incremental_decoder is not None
        )

    def bind_generation_state(
        self,
        *,
        text: str,
        speaker_name: str,
        language_type: str,
        instructions: str | None,
        attention_mask: Any,
        past_key_values: Any,
        past_hidden: Any,
        generation_step: int,
        trailing_text_hidden: Any,
        tts_pad_embed: Any,
        next_logits: Any,
        generated_codes: list[Any],
        sampled_tokens: list[int],
        decoded_until_step: int,
        generation_finished: bool,
        incremental_decoder: Any,
        metrics: dict[str, Any] | None = None,
    ) -> None:
        self.state.active_generation_text = text
        self.state.active_speaker_name = speaker_name
        self.state.active_language_type = language_type
        self.state.active_instructions = instructions or ""
        self.state.attention_mask = attention_mask
        self.state.past_key_values = past_key_values
        self.state.past_hidden = past_hidden
        self.state.generation_step = generation_step
        self.state.trailing_text_hidden = trailing_text_hidden
        self.state.tts_pad_embed = tts_pad_embed
        self.state.next_logits = next_logits
        self.state.generated_codes = list(generated_codes)
        self.state.sampled_tokens = list(sampled_tokens)
        self.state.decoded_until_step = decoded_until_step
        self.state.generation_finished = generation_finished
        self.state.incremental_decoder = incremental_decoder
        self.state.last_generation_metrics = dict(metrics or {})

    def snapshot(self) -> dict[str, object]:
        return {
            "session_id": self.state.session_id,
            "model": self.state.options.model,
            "voice": self.state.options.voice,
            "mode": self.state.options.mode,
            "raw_text_buffer": self.state.raw_text_buffer,
            "committed_text": self.state.committed_text,
            "pending_text_tail": self.state.pending_text_tail,
            "finished": self.state.finished,
            "committed_chunks": list(self.state.committed_chunks),
            "generation_state": {
                "active_generation_text": self.state.active_generation_text,
                "active_speaker_name": self.state.active_speaker_name,
                "active_language_type": self.state.active_language_type,
                "generated_steps": len(self.state.generated_codes),
                "decoded_until_step": self.state.decoded_until_step,
                "generation_finished": self.state.generation_finished,
                "has_past_key_values": self.state.past_key_values is not None,
                "has_past_hidden": self.state.past_hidden is not None,
                "metrics": dict(self.state.last_generation_metrics),
            },
        }
