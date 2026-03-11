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
        from streaming_lora_service.app.server import InMemoryAudioStore

        self.audio_store = InMemoryAudioStore()

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
        return []

    def synthesize_http(self, *, text: str, model: str, voice: str, language_type: str, instructions: str = "") -> SynthesizedAudio:
        return SynthesizedAudio(audio_bytes=(text.encode("utf-8") or b"x") * 16, sample_rate=24000)

    def store_wav_asset(self, synthesized: SynthesizedAudio):
        from streaming_lora_service.app.audio_utils import pcm16le_bytes_to_wav_bytes

        asset = self.audio_store.save(
            pcm16le_bytes_to_wav_bytes(synthesized.audio_bytes, sample_rate=synthesized.sample_rate),
            media_type="audio/wav",
        )
        return asset


class HttpTTSSmokeTests(unittest.TestCase):
    def test_http_tts_returns_official_style_payload_and_audio_url(self) -> None:
        client = TestClient(create_app(service=FakeService()))

        response = client.post(
            "/v1/tts",
            json={
                "model": "qwen3-tts-flash-realtime",
                "text": "你好，HTTP。",
                "voice": "yachiyo_formal",
                "language_type": "Chinese",
                "stream": False,
            },
        )

        self.assertEqual(response.status_code, 200)
        payload = response.json()
        self.assertEqual(payload["status_code"], 200)
        self.assertEqual(payload["output"]["finish_reason"], "stop")
        self.assertTrue(payload["output"]["audio"]["url"])
        self.assertEqual(payload["usage"]["characters"], len("你好，HTTP。"))

        audio_response = client.get(payload["output"]["audio"]["url"])
        self.assertEqual(audio_response.status_code, 200)
        self.assertEqual(audio_response.headers["content-type"], "audio/wav")
        self.assertTrue(audio_response.content)

    def test_http_tts_stream_returns_ndjson_chunks(self) -> None:
        client = TestClient(create_app(service=FakeService()))

        with client.stream(
            "POST",
            "/api/v1/services/aigc/multimodal-generation/generation",
            json={
                "model": "qwen3-tts-flash-realtime",
                "text": "你好，流式HTTP。",
                "voice": "yachiyo_formal",
                "stream": True,
            },
        ) as response:
            self.assertEqual(response.status_code, 200)
            chunks = [line for line in response.iter_lines() if line]

        self.assertGreaterEqual(len(chunks), 2)
        self.assertIn('"status_code": 200', chunks[0])
        self.assertIn('"finish_reason": "stop"', chunks[-1])


if __name__ == "__main__":
    unittest.main()