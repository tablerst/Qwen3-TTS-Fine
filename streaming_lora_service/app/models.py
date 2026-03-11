from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal, Mapping

SessionMode = Literal["server_commit", "commit"]


@dataclass(frozen=True)
class BundleArtifacts:
    bundle_dir: Path
    base_model: str
    adapter_dir: Path
    config_patch_file: Path
    speaker_patch_file: Path
    speaker_name: str | None = None
    manifest: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class LoadedBundle:
    artifacts: BundleArtifacts
    qwen3tts: Any
    speaker_name: str
    speaker_id: int
    config_patch: dict[str, Any]
    tts_model_type: str


@dataclass(frozen=True)
class PublicVoiceProfile:
    voice_alias: str
    speaker_name: str
    bundle_key: str = "default"
    supported_models: tuple[str, ...] = ()
    language_type: str = "Auto"
    description: str = ""


@dataclass(frozen=True)
class SessionOptions:
    model: str
    voice: str
    language_type: str = "Auto"
    mode: SessionMode = "server_commit"
    response_format: str = "pcm"
    sample_rate: int = 24000
    instructions: str = ""
    optimize_instructions: bool = False

    @classmethod
    def from_session_update(cls, session: Mapping[str, Any]) -> "SessionOptions":
        mode = str(session.get("mode", "server_commit"))
        if mode not in ("server_commit", "commit"):
            raise ValueError(f"Unsupported session mode: {mode}")

        model = str(session.get("model") or "").strip()
        voice = str(session.get("voice") or "").strip()
        if not model:
            raise ValueError("session.model is required")
        if not voice:
            raise ValueError("session.voice is required")

        return cls(
            model=model,
            voice=voice,
            language_type=str(session.get("language_type", "Auto")),
            mode=mode,
            response_format=str(session.get("response_format", "pcm")),
            sample_rate=int(session.get("sample_rate", 24000)),
            instructions=str(session.get("instructions", "")),
            optimize_instructions=bool(session.get("optimize_instructions", False)),
        )


@dataclass(frozen=True)
class DecodePlan:
    start_step: int
    end_step: int
    emit_from_step: int


@dataclass(frozen=True)
class SynthesizedAudio:
    audio_bytes: bytes
    sample_rate: int
    channels: int = 1
