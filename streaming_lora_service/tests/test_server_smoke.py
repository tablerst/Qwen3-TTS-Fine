from __future__ import annotations

import unittest

from fastapi.testclient import TestClient

from streaming_lora_service.app.models import PublicVoiceProfile, SessionOptions, SynthesizedAudio
from streaming_lora_service.app.qwen_compat_ws import QwenRealtimeProtocolAdapter
from streaming_lora_service.app.server import create_app
from streaming_lora_service.app.voice_registry import VoiceRegistry


class FakeService:
    public_model_alias = "qwen3-tts-flash-realtime"
    default_voice_alias = "yachiyo_formal"

    def __init__(self) -> None:
        self.voice_registry = VoiceRegistry(
            [
                PublicVoiceProfile(
                    voice_alias="yachiyo_formal",
                    speaker_name="inference_speaker",
                    supported_models=("qwen3-tts-flash-realtime",),
                )
            ]
        )

    def build_initial_session_options(self) -> SessionOptions:
        return SessionOptions(model=self.public_model_alias, voice=self.default_voice_alias)

    def create_protocol_adapter(self) -> QwenRealtimeProtocolAdapter:
        return QwenRealtimeProtocolAdapter(
            voice_registry=self.voice_registry,
            synthesize_callback=lambda _session, text: SynthesizedAudio(
                audio_bytes=(text.encode("utf-8") or b"x") * 8,
                sample_rate=24000,
            ),
            audio_chunk_duration_ms=20,
        )

    def list_voices(self):
        return [
            {
                "voice": "yachiyo_formal",
                "speaker_name": "inference_speaker",
                "supported_models": [self.public_model_alias],
                "language_type": "Auto",
                "description": "fake voice",
            }
        ]


class ServerSmokeTests(unittest.TestCase):
    def test_health_and_websocket_flow(self) -> None:
        app = create_app(service=FakeService())
        client = TestClient(app)

        response = client.get("/healthz")
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()["status"], "ok")

        with client.websocket_connect("/api-ws/v1/realtime") as websocket:
            created = websocket.receive_json()
            self.assertEqual(created["type"], "session.created")

            websocket.send_json(
                {
                    "type": "session.update",
                    "session": {
                        "model": "qwen3-tts-flash-realtime",
                        "voice": "yachiyo_formal",
                        "mode": "commit",
                    },
                }
            )
            updated = websocket.receive_json()
            self.assertEqual(updated["type"], "session.updated")

            websocket.send_json({"type": "input_text_buffer.append", "text": "你好，MVP。"})
            websocket.send_json({"type": "input_text_buffer.commit"})

            event_types = [websocket.receive_json()["type"] for _ in range(9)]
            self.assertEqual(
                event_types,
                [
                    "input_text_buffer.committed",
                    "response.created",
                    "response.output_item.added",
                    "response.content_part.added",
                    "response.audio.delta",
                    "response.audio.done",
                    "response.content_part.done",
                    "response.output_item.done",
                    "response.done",
                ],
            )


if __name__ == "__main__":
    unittest.main()