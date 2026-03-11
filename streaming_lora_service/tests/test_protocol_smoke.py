from __future__ import annotations

import unittest

from streaming_lora_service.app.models import PublicVoiceProfile, SessionOptions, SynthesizedAudio
from streaming_lora_service.app.qwen_compat_ws import QwenRealtimeProtocolAdapter
from streaming_lora_service.app.voice_registry import VoiceRegistry


class ProtocolSmokeTests(unittest.TestCase):
    def make_adapter(self) -> QwenRealtimeProtocolAdapter:
        registry = VoiceRegistry(
            [
                PublicVoiceProfile(
                    voice_alias="yachiyo_formal",
                    speaker_name="inference_speaker",
                    supported_models=("qwen3-tts-flash-realtime",),
                )
            ]
        )
        return QwenRealtimeProtocolAdapter(
            voice_registry=registry,
            synthesize_callback=lambda _session, text: SynthesizedAudio(
                audio_bytes=(text.encode("utf-8") or b"x") * 4,
                sample_rate=24000,
            ),
            audio_chunk_duration_ms=20,
        )

    def test_session_update_returns_created_and_updated(self) -> None:
        adapter = self.make_adapter()

        events = adapter.handle_event(
            {
                "type": "session.update",
                "session": {
                    "model": "qwen3-tts-flash-realtime",
                    "voice": "yachiyo_formal",
                    "mode": "server_commit",
                },
            }
        )

        self.assertEqual([event["type"] for event in events], ["session.created", "session.updated"])
        self.assertEqual(events[1]["session"]["voice"], "yachiyo_formal")

    def test_commit_flow_returns_qwen_style_events(self) -> None:
        adapter = self.make_adapter()
        adapter.open_connection(
            SessionOptions(
                model="qwen3-tts-flash-realtime",
                voice="yachiyo_formal",
            )
        )
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
        adapter.handle_event({"type": "input_text_buffer.append", "text": "你好，世界。"})

        events = adapter.handle_event({"type": "input_text_buffer.commit"})

        self.assertEqual(
            [event["type"] for event in events],
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
        self.assertEqual(events[0]["text"], "你好，世界。")
        self.assertEqual(events[1]["response"]["status"], "in_progress")
        self.assertTrue(events[4]["delta"])

    def test_finish_returns_response_done_and_session_finished(self) -> None:
        adapter = self.make_adapter()
        adapter.open_connection(
            SessionOptions(
                model="qwen3-tts-flash-realtime",
                voice="yachiyo_formal",
            )
        )
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
        adapter.handle_event({"type": "input_text_buffer.append", "text": "收尾。"})

        events = adapter.handle_event({"type": "session.finish"})

        self.assertEqual(events[-2]["type"], "response.done")
        self.assertEqual(events[-1]["type"], "session.finished")
        self.assertEqual(events[-2]["response"]["status"], "completed")


if __name__ == "__main__":
    unittest.main()
