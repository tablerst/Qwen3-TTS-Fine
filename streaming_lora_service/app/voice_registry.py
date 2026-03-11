from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterable, Mapping

from .models import PublicVoiceProfile


class VoiceRegistryError(KeyError):
    """Raised when a public voice alias cannot be resolved."""


class VoiceRegistry:
    def __init__(
        self,
        profiles: Iterable[PublicVoiceProfile] | None = None,
        *,
        default_voice: str | None = None,
    ) -> None:
        self._profiles: dict[str, PublicVoiceProfile] = {}
        self._default_voice = default_voice
        for profile in profiles or []:
            self.register(profile)

    @property
    def default_voice(self) -> str | None:
        return self._default_voice

    def register(self, profile: PublicVoiceProfile) -> None:
        alias = profile.voice_alias.strip()
        if not alias:
            raise ValueError("voice_alias must be non-empty")
        if alias in self._profiles:
            raise ValueError(f"Duplicate voice alias: {alias}")
        self._profiles[alias] = profile
        if self._default_voice is None:
            self._default_voice = alias

    def resolve(self, voice_alias: str | None, *, model: str | None = None) -> PublicVoiceProfile:
        alias = (voice_alias or self._default_voice or "").strip()
        if not alias:
            raise VoiceRegistryError("No voice alias provided and no default voice configured")
        if alias not in self._profiles:
            raise VoiceRegistryError(f"Unknown voice alias: {alias}")

        profile = self._profiles[alias]
        if model and profile.supported_models and model not in profile.supported_models:
            raise VoiceRegistryError(
                f"Voice '{alias}' does not support model '{model}'. Supported models: {profile.supported_models}"
            )
        return profile

    def list_aliases(self) -> list[str]:
        return sorted(self._profiles)

    def list_profiles(self) -> list[PublicVoiceProfile]:
        return [self._profiles[alias] for alias in self.list_aliases()]

    @classmethod
    def from_mapping(
        cls,
        mapping: Mapping[str, Mapping[str, Any]],
        *,
        default_voice: str | None = None,
    ) -> "VoiceRegistry":
        profiles = []
        for alias, data in mapping.items():
            profiles.append(
                PublicVoiceProfile(
                    voice_alias=alias,
                    speaker_name=str(data["speaker_name"]),
                    bundle_key=str(data.get("bundle_key", "default")),
                    supported_models=tuple(str(item) for item in data.get("supported_models", ())),
                    language_type=str(data.get("language_type", "Auto")),
                    description=str(data.get("description", "")),
                )
            )
        return cls(profiles, default_voice=default_voice)

    @classmethod
    def from_file(cls, file_path: str | Path) -> "VoiceRegistry":
        path = Path(file_path)
        suffix = path.suffix.lower()
        if suffix == ".json":
            payload = json.loads(path.read_text(encoding="utf-8"))
        elif suffix in {".yml", ".yaml"}:
            try:
                import yaml
            except ImportError as exc:  # pragma: no cover - optional dependency branch
                raise RuntimeError("PyYAML is required to load YAML voice registry files") from exc
            payload = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
        else:
            raise ValueError(f"Unsupported voice registry file type: {path.suffix}")

        if not isinstance(payload, Mapping):
            raise TypeError("Voice registry file must contain a mapping at the top level")

        default_voice = str(payload.get("default_voice", "")).strip() or None
        voices = payload.get("voices", payload)
        if not isinstance(voices, Mapping):
            raise TypeError("Voice registry 'voices' section must be a mapping")
        return cls.from_mapping(voices, default_voice=default_voice)

    @classmethod
    def single_voice(
        cls,
        *,
        voice_alias: str,
        speaker_name: str,
        model_alias: str,
        description: str = "",
        language_type: str = "Auto",
    ) -> "VoiceRegistry":
        return cls(
            [
                PublicVoiceProfile(
                    voice_alias=voice_alias,
                    speaker_name=speaker_name,
                    supported_models=(model_alias,),
                    language_type=language_type,
                    description=description,
                )
            ],
            default_voice=voice_alias,
        )
