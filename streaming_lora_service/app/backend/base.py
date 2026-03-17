from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Iterable, Protocol, runtime_checkable

if TYPE_CHECKING:
    from ..models import BundleArtifacts, LoadedBundle, SynthesizedAudio
    from ..runtime_session import RuntimeSession


@dataclass(frozen=True)
class BackendCapabilities:
    supports_state_resume: bool = False
    supports_runtime_session_sync_mode: bool = False
    supports_compile_talker: bool = False
    supports_trace_attention_backend: bool = False


@runtime_checkable
class SpeechBackend(Protocol):
    kind: str
    sample_rate: int
    capabilities: BackendCapabilities

    def list_supported_speakers(self) -> tuple[str, ...]:
        ...

    def synthesize(
        self,
        *,
        text: str,
        speaker: str,
        language: str = "Auto",
        instruct: str | None = None,
    ) -> "SynthesizedAudio":
        ...

    def stream_synthesize(
        self,
        *,
        text: str,
        speaker: str,
        language: str = "Auto",
        instruct: str | None = None,
        runtime_session: "RuntimeSession | None" = None,
        chunk_steps: int = 4,
        left_context_steps: int = 25,
        first_chunk_steps: int | None = 1,
        crossfade_samples: int = 0,
        runtime_session_sync_mode: str = "chunk",
    ) -> Iterable[bytes]:
        ...


@dataclass(frozen=True)
class BackendLoadResult:
    backend: SpeechBackend
    speaker_name: str
    supported_speakers: tuple[str, ...]
    tts_model_type: str
    loaded_bundle: "LoadedBundle | None" = None
    bundle_artifacts: "BundleArtifacts | None" = None
    metadata: dict[str, Any] = field(default_factory=dict)
