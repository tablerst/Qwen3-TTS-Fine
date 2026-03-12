from __future__ import annotations

import base64
from itertools import count
import re
from typing import Any, Callable, Iterable, Iterator

from .models import PublicVoiceProfile, SessionOptions, SynthesizedAudio
from .runtime_session import RuntimeSession, RuntimeSessionError
from .voice_registry import VoiceRegistry, VoiceRegistryError


class QwenRealtimeProtocolAdapter:
    def __init__(
        self,
        voice_registry: VoiceRegistry,
        *,
        session_factory: Callable[[str, SessionOptions], RuntimeSession] | None = None,
        synthesize_callback: Callable[[RuntimeSession, str], SynthesizedAudio] | None = None,
        stream_synthesize_callback: Callable[[RuntimeSession, str], Iterable[bytes]] | None = None,
        audio_chunker: Callable[[SynthesizedAudio], list[bytes]] | None = None,
        audio_chunk_duration_ms: int = 320,
        auto_commit_chars: int = 80,
        supported_response_formats: tuple[str, ...] = ("pcm",),
        supported_sample_rates: tuple[int, ...] = (24000,),
    ) -> None:
        self.voice_registry = voice_registry
        self._session_factory = session_factory or RuntimeSession
        self._synthesize_callback = synthesize_callback
        self._stream_synthesize_callback = stream_synthesize_callback
        self._audio_chunker = audio_chunker
        self._audio_chunk_duration_ms = audio_chunk_duration_ms
        self._auto_commit_chars = auto_commit_chars
        self._supported_response_formats = tuple(item.lower() for item in supported_response_formats)
        self._supported_sample_rates = tuple(int(item) for item in supported_sample_rates)
        self._session: RuntimeSession | None = None
        self._session_counter = count(1)
        self._response_counter = count(1)
        self._item_counter = count(1)
        self._punctuation_pattern = re.compile(r"[。！？!?；;\n]$")

    @property
    def current_session(self) -> RuntimeSession | None:
        return self._session

    def open_connection(self, default_options: SessionOptions) -> list[dict[str, Any]]:
        profile = self.voice_registry.resolve(default_options.voice, model=default_options.model)
        if self._session is None:
            session_id = self._next_id("sess", self._session_counter)
            self._session = self._session_factory(session_id, default_options)
        payload = self._session_payload(default_options, profile, self._session.session_id)
        return [
            {
                "event_id": self._next_id("event", self._response_counter),
                "type": "session.created",
                "session": payload,
            }
        ]

    def handle_event(self, event: dict[str, Any]) -> list[dict[str, Any]]:
        return list(self.iter_events(event))

    def iter_events(self, event: dict[str, Any]) -> Iterator[dict[str, Any]]:
        try:
            event_type = event.get("type")
            if not event_type:
                yield self._error("invalid_value", "Missing field: type")
                return

            if event_type == "session.update":
                yield from self._handle_session_update(event)
                return
            if self._session is None:
                yield self._error("invalid_state", "Session has not been initialized")
                return
            if event_type == "input_text_buffer.append":
                text = str(event.get("text", ""))
                self._session.append_text(text)
                if self._session.options.mode == "server_commit" and self._should_auto_commit():
                    yield from self._iter_commit_events()
                return
            if event_type == "input_text_buffer.commit":
                yield from self._iter_commit_events()
                return
            if event_type == "input_text_buffer.clear":
                self._session.clear_pending_text()
                yield {
                    "event_id": self._next_id("event", self._response_counter),
                    "type": "input_text_buffer.cleared",
                }
                return
            if event_type == "session.finish":
                yield from self._iter_finish_events()
                return
            yield self._error("invalid_value", f"Unsupported event type: {event_type}")
        except (RuntimeSessionError, VoiceRegistryError, ValueError) as exc:
            yield self._error("invalid_value", str(exc))

    def _handle_session_update(self, event: dict[str, Any]) -> list[dict[str, Any]]:
        try:
            options = SessionOptions.from_session_update(event.get("session", {}))
            profile = self.voice_registry.resolve(options.voice, model=options.model)
            self._validate_realtime_audio_contract(options)
        except Exception as exc:  # pragma: no cover - converted to protocol error response
            return [self._error("invalid_value", str(exc))]

        had_session = self._session is not None
        if self._session is not None:
            session_id = self._session.session_id
        else:
            session_id = self._next_id("sess", self._session_counter)
        self._session = self._session_factory(session_id, options)

        payload = self._session_payload(options, profile, session_id)
        events: list[dict[str, Any]] = []
        if not had_session and event.get("_emit_created", True):
            events.append({"event_id": self._next_id("event", self._response_counter), "type": "session.created", "session": payload})
        events.append({"event_id": self._next_id("event", self._response_counter), "type": "session.updated", "session": payload})
        return events

    def _iter_commit_events(self) -> Iterator[dict[str, Any]]:
        assert self._session is not None
        committed_text = self._session.commit()
        yield from self._iter_generation_events(committed_text)

    def _iter_generation_events(self, committed_text: str) -> Iterator[dict[str, Any]]:
        assert self._session is not None
        response_id = self._next_id("resp", self._response_counter)
        item_id = self._next_id("item", self._item_counter)
        yield {
            "event_id": self._next_id("event", self._response_counter),
            "type": "input_text_buffer.committed",
            "item_id": item_id,
            "text": committed_text,
        }
        yield {
            "event_id": self._next_id("event", self._response_counter),
            "type": "response.created",
            "response": {
                "id": response_id,
                "object": "realtime.response",
                "conversation_id": "",
                "status": "in_progress",
                "modalities": ["audio"],
                "voice": self._session.options.voice,
                "output": [],
            },
        }

        yield {
            "event_id": self._next_id("event", self._response_counter),
            "type": "response.output_item.added",
            "response_id": response_id,
            "output_index": 0,
            "item": {
                "id": item_id,
                "object": "realtime.item",
                "type": "message",
                "status": "in_progress",
                "role": "assistant",
                "content": [],
            },
        }
        yield {
            "event_id": self._next_id("event", self._response_counter),
            "type": "response.content_part.added",
            "response_id": response_id,
            "item_id": item_id,
            "output_index": 0,
            "content_index": 0,
            "part": {"type": "audio", "text": ""},
        }

        if self._stream_synthesize_callback is not None:
            for chunk in self._stream_synthesize_callback(self._session, committed_text):
                if not chunk:
                    continue
                yield {
                    "event_id": self._next_id("event", self._response_counter),
                    "type": "response.audio.delta",
                    "response_id": response_id,
                    "item_id": item_id,
                    "output_index": 0,
                    "content_index": 0,
                    "delta": base64.b64encode(chunk).decode("ascii"),
                }
        elif self._synthesize_callback is not None:
            synthesized = self._synthesize_callback(self._session, committed_text)
            chunk_size_bytes = self._chunk_size_bytes(synthesized.sample_rate, synthesized.channels)
            if synthesized.audio_bytes:
                chunks = self._audio_chunker(synthesized) if self._audio_chunker is not None else self._chunk_audio(synthesized.audio_bytes, chunk_size_bytes)
                for chunk in chunks:
                    yield {
                        "event_id": self._next_id("event", self._response_counter),
                        "type": "response.audio.delta",
                        "response_id": response_id,
                        "item_id": item_id,
                        "output_index": 0,
                        "content_index": 0,
                        "delta": base64.b64encode(chunk).decode("ascii"),
                    }

        yield {
            "event_id": self._next_id("event", self._response_counter),
            "type": "response.audio.done",
            "response_id": response_id,
            "item_id": item_id,
            "output_index": 0,
            "content_index": 0,
        }
        yield {
            "event_id": self._next_id("event", self._response_counter),
            "type": "response.content_part.done",
            "response_id": response_id,
            "item_id": item_id,
            "output_index": 0,
            "content_index": 0,
            "part": {"type": "audio", "text": ""},
        }
        yield {
            "event_id": self._next_id("event", self._response_counter),
            "type": "response.output_item.done",
            "response_id": response_id,
            "output_index": 0,
            "item": {
                "id": item_id,
                "object": "realtime.item",
                "type": "message",
                "status": "completed",
                "role": "assistant",
                "content": [{"type": "audio", "text": ""}],
            },
        }
        yield {
            "event_id": self._next_id("event", self._response_counter),
            "type": "response.done",
            "response": {
                "id": response_id,
                "object": "realtime.response",
                "conversation_id": "",
                "status": "completed",
                "modalities": ["audio"],
                "voice": self._session.options.voice,
                "output": [
                    {
                        "id": item_id,
                        "object": "realtime.item",
                        "type": "message",
                        "status": "completed",
                        "role": "assistant",
                        "content": [{"type": "audio", "text": ""}],
                    }
                ],
                "usage": {"characters": len(committed_text)},
            },
        }

    def _iter_finish_events(self) -> Iterator[dict[str, Any]]:
        assert self._session is not None
        final_chunk = self._session.finish()
        if final_chunk:
            yield from self._iter_generation_events(final_chunk)
        yield {
            "event_id": self._next_id("event", self._response_counter),
            "type": "session.finished",
        }

    def _session_payload(
        self,
        options: SessionOptions,
        profile: PublicVoiceProfile,
        session_id: str,
    ) -> dict[str, Any]:
        return {
            "object": "realtime.session",
            "id": session_id,
            "model": options.model,
            "voice": profile.voice_alias,
            "language_type": options.language_type,
            "mode": options.mode,
            "response_format": options.response_format,
            "sample_rate": options.sample_rate,
            "instructions": options.instructions,
            "optimize_instructions": options.optimize_instructions,
        }

    def _should_auto_commit(self) -> bool:
        assert self._session is not None
        pending = self._session.state.pending_text_tail
        if not pending:
            return False
        return bool(self._punctuation_pattern.search(pending)) or len(pending) >= self._auto_commit_chars

    def _chunk_size_bytes(self, sample_rate: int, channels: int) -> int:
        samples_per_chunk = max(1, int(sample_rate * self._audio_chunk_duration_ms / 1000.0))
        return samples_per_chunk * max(1, channels) * 2

    def _validate_realtime_audio_contract(self, options: SessionOptions) -> None:
        if options.response_format not in self._supported_response_formats:
            supported = ", ".join(self._supported_response_formats)
            raise ValueError(
                "Realtime websocket currently only supports "
                f"response_format in ({supported}); got {options.response_format!r}"
            )
        if options.sample_rate not in self._supported_sample_rates:
            supported = ", ".join(str(item) for item in self._supported_sample_rates)
            raise ValueError(
                "Realtime websocket currently only supports "
                f"sample_rate in ({supported}); got {options.sample_rate}"
            )

    @staticmethod
    def _chunk_audio(audio_bytes: bytes, chunk_size_bytes: int) -> list[bytes]:
        if not audio_bytes:
            return []
        return [audio_bytes[idx : idx + chunk_size_bytes] for idx in range(0, len(audio_bytes), chunk_size_bytes)]

    @staticmethod
    def _next_id(prefix: str, counter: count) -> str:
        return f"{prefix}_{next(counter):04d}"

    @staticmethod
    def _error(code: str, message: str) -> dict[str, Any]:
        return {
            "event_id": "event_error",
            "type": "error",
            "error": {
                "code": code,
                "message": message,
            },
        }
