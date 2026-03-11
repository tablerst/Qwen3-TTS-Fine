from __future__ import annotations

from dataclasses import dataclass, field

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
    finished: bool = False


class RuntimeSession:
    def __init__(self, session_id: str, options: SessionOptions) -> None:
        self.state = RuntimeSessionState(session_id=session_id, options=options)

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
        }
