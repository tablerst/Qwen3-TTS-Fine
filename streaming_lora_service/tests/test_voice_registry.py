from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from streaming_lora_service.app.models import PublicVoiceProfile
from streaming_lora_service.app.voice_registry import VoiceRegistry, VoiceRegistryError


class VoiceRegistryTests(unittest.TestCase):
    def make_temp_dir(self) -> Path:
        temp_dir = tempfile.TemporaryDirectory()
        self.addCleanup(temp_dir.cleanup)
        return Path(temp_dir.name)

    def test_resolve_default_voice_alias(self) -> None:
        registry = VoiceRegistry(
            [
                PublicVoiceProfile(
                    voice_alias="yachiyo_formal",
                    speaker_name="inference_speaker",
                    supported_models=("qwen3-tts-flash-realtime",),
                )
            ]
        )

        profile = registry.resolve(None, model="qwen3-tts-flash-realtime")

        self.assertEqual(profile.voice_alias, "yachiyo_formal")
        self.assertEqual(profile.speaker_name, "inference_speaker")

    def test_resolve_rejects_unknown_alias(self) -> None:
        registry = VoiceRegistry()

        with self.assertRaisesRegex(VoiceRegistryError, "Unknown voice alias"):
            registry.resolve("missing_voice")

    def test_resolve_rejects_unsupported_model(self) -> None:
        registry = VoiceRegistry(
            [
                PublicVoiceProfile(
                    voice_alias="yachiyo_formal",
                    speaker_name="inference_speaker",
                    supported_models=("qwen3-tts-flash-realtime",),
                )
            ]
        )

        with self.assertRaisesRegex(VoiceRegistryError, "does not support model"):
            registry.resolve("yachiyo_formal", model="qwen3-tts-instruct-flash-realtime")

    def test_from_file_loads_default_voice_and_profiles(self) -> None:
        work_dir = self.make_temp_dir()
        registry_path = work_dir / "voice_registry.json"
        registry_path.write_text(
            json.dumps(
                {
                    "default_voice": "yachiyo_formal",
                    "voices": {
                        "yachiyo_formal": {
                            "speaker_name": "inference_speaker",
                            "supported_models": ["qwen3-tts-flash-realtime"],
                            "description": "Default bundle voice"
                        }
                    },
                },
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
        )

        registry = VoiceRegistry.from_file(registry_path)

        self.assertEqual(registry.default_voice, "yachiyo_formal")
        self.assertEqual(registry.resolve(None).speaker_name, "inference_speaker")


if __name__ == "__main__":
    unittest.main()
