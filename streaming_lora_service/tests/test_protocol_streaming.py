from __future__ import annotations

import unittest

from streaming_lora_service.app.models import PublicVoiceProfile, SessionOptions
from streaming_lora_service.app.qwen_compat_ws import QwenRealtimeProtocolAdapter
from streaming_lora_service.app.voice_registry import VoiceRegistry


class ProtocolStreamingTests(unittest.TestCase):
    def test_iter_events_supports_streaming_audio_source(self) -> None:
        registry = VoiceRegistry(
            [
                PublicVoiceProfile(
                    voice_alias="yachiyo_formal",
                    speaker_name="inference_speaker",
                    supported_models=("qwen3-tts-flash-realtime",),
                )
            ]
        )
        adapter = QwenRealtimeProtocolAdapter(
            voice_registry=registry,
            stream_synthesize_callback=lambda _session, _text: [b"chunk-1", b"chunk-2"],
        )
        adapter.open_connection(SessionOptions(model="qwen3-tts-flash-realtime", voice="yachiyo_formal"))
        adapter.handle_event(
            {
                "type": "session.update",
                "session": {
                    "model": "qwen3-tts-flash-realtime",
                    "voice": "yachiyo_formal",
                    "mode": "commit",
                },
            }
        )
        adapter.handle_event({"type": "input_text_buffer.append", "text": "你好，流式协议。"})

        events = list(adapter.iter_events({"type": "input_text_buffer.commit"}))

        event_types = [event["type"] for event in events]
        self.assertEqual(event_types.count("response.audio.delta"), 2)
        self.assertEqual(event_types[0], "input_text_buffer.committed")
        self.assertEqual(event_types[-1], "response.done")


if __name__ == "__main__":
    unittest.main()
