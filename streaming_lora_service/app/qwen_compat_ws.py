from __future__ import annotations

import base64
from itertools import count
import re
from typing import Any, Callable

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
        audio_chunk_duration_ms: int = 320,
        auto_commit_chars: int = 80,
    ) -> None:
        self.voice_registry = voice_registry
        self._session_factory = session_factory or RuntimeSession
        self._synthesize_callback = synthesize_callback
        self._audio_chunk_duration_ms = audio_chunk_duration_ms
        self._auto_commit_chars = auto_commit_chars
        self._session: RuntimeSession | None = None
        self._session_counter = count(1)
        self._response_counter = count(1)
        self._item_counter = count(1)
        self._punctuation_pattern = re.compile(r"[。！？!?；;\n]$")

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
        try:
            event_type = event.get("type")
            if not event_type:
                return [self._error("invalid_value", "Missing field: type")]

            if event_type == "session.update":
                return self._handle_session_update(event)
            if self._session is None:
                return [self._error("invalid_state", "Session has not been initialized")]
            if event_type == "input_text_buffer.append":
                text = str(event.get("text", ""))
                self._session.append_text(text)
                if self._session.options.mode == "server_commit" and self._should_auto_commit():
                    return self._handle_commit()
                return []
            if event_type == "input_text_buffer.commit":
                return self._handle_commit()
            if event_type == "input_text_buffer.clear":
                self._session.clear_pending_text()
                return []
            if event_type == "session.finish":
                return self._handle_finish()
            return [self._error("invalid_value", f"Unsupported event type: {event_type}")]
        except (RuntimeSessionError, VoiceRegistryError, ValueError) as exc:
            return [self._error("invalid_value", str(exc))]

    def _handle_session_update(self, event: dict[str, Any]) -> list[dict[str, Any]]:
        try:
            options = SessionOptions.from_session_update(event.get("session", {}))
            profile = self.voice_registry.resolve(options.voice, model=options.model)
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

    def _handle_commit(self) -> list[dict[str, Any]]:
        assert self._session is not None
        committed_text = self._session.commit()
        return self._build_generation_events(committed_text)

    def _build_generation_events(self, committed_text: str) -> list[dict[str, Any]]:
        assert self._session is not None
        response_id = self._next_id("resp", self._response_counter)
        item_id = self._next_id("item", self._item_counter)
        events: list[dict[str, Any]] = [
            {
                "event_id": self._next_id("event", self._response_counter),
                "type": "input_text_buffer.committed",
                "item_id": item_id,
                "text": committed_text,
            },
            {
                "event_id": self._next_id("event", self._response_counter),
                "type": "response.created",
                "response": {
                    "id": response_id,
                    "object": "realtime.response",
                    "status": "in_progress",
                    "voice": self._session.options.voice,
                    "output": [],
                },
            },
        ]

        if self._synthesize_callback is None:
            return events

        synthesized = self._synthesize_callback(self._session, committed_text)
        chunk_size_bytes = self._chunk_size_bytes(synthesized.sample_rate, synthesized.channels)
        events.append(
            {
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
        )
        events.append(
            {
                "event_id": self._next_id("event", self._response_counter),
                "type": "response.content_part.added",
                "response_id": response_id,
                "item_id": item_id,
                "output_index": 0,
                "content_index": 0,
                "part": {"type": "audio", "text": ""},
            }
        )

        if synthesized.audio_bytes:
            for chunk in self._chunk_audio(synthesized.audio_bytes, chunk_size_bytes):
                events.append(
                    {
                        "event_id": self._next_id("event", self._response_counter),
                        "type": "response.audio.delta",
                        "response_id": response_id,
                        "item_id": item_id,
                        "output_index": 0,
                        "content_index": 0,
                        "delta": base64.b64encode(chunk).decode("ascii"),
                    }
                )

        events.extend(
            [
                {
                    "event_id": self._next_id("event", self._response_counter),
                    "type": "response.audio.done",
                    "response_id": response_id,
                    "item_id": item_id,
                    "output_index": 0,
                    "content_index": 0,
                },
                {
                    "event_id": self._next_id("event", self._response_counter),
                    "type": "response.content_part.done",
                    "response_id": response_id,
                    "item_id": item_id,
                    "output_index": 0,
                    "content_index": 0,
                    "part": {"type": "audio", "text": ""},
                },
                {
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
                },
                {
                    "event_id": self._next_id("event", self._response_counter),
                    "type": "response.done",
                    "response": {
                        "id": response_id,
                        "object": "realtime.response",
                        "status": "completed",
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
                },
            ]
        )
        return events

    def _handle_finish(self) -> list[dict[str, Any]]:
        assert self._session is not None
        final_chunk = self._session.finish()
        events: list[dict[str, Any]] = []
        if final_chunk:
            events.extend(self._build_generation_events(final_chunk))
        events.append(
            {
                "event_id": self._next_id("event", self._response_counter),
                "type": "session.finished",
            }
        )
        return events

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
