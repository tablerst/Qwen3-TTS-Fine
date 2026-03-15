from __future__ import annotations

import unittest

from streaming_lora_service.ws_benchmark import (
    audio_duration_seconds,
    choose_voice_alias,
    compute_rtf,
    derive_http_base_url,
    percentile,
    summarize_numeric,
)


class WsBenchmarkHelpersTests(unittest.TestCase):
    def test_audio_duration_seconds_24khz_pcm16(self) -> None:
        self.assertAlmostEqual(audio_duration_seconds(48000, sample_rate=24000), 1.0)

    def test_compute_rtf_returns_none_without_audio(self) -> None:
        self.assertIsNone(compute_rtf(1234.0, 0.0))

    def test_percentile_interpolates(self) -> None:
        self.assertAlmostEqual(percentile([10.0, 20.0, 30.0, 40.0], 0.95) or 0.0, 38.5)

    def test_summarize_numeric_empty(self) -> None:
        summary = summarize_numeric([])
        self.assertEqual(summary["count"], 0)
        self.assertIsNone(summary["p50"])

    def test_choose_voice_prefers_default_compatible_voice(self) -> None:
        payload = {
            "default_voice": "voice_b",
            "data": [
                {"voice": "voice_a", "supported_models": ["other-model"]},
                {"voice": "voice_b", "supported_models": ["qwen3-tts-flash-realtime"]},
            ],
        }
        selection = choose_voice_alias(payload, requested_voice=None, model="qwen3-tts-flash-realtime")
        self.assertEqual(selection.voice, "voice_b")
        self.assertEqual(selection.source, "default_voice")

    def test_choose_voice_raises_for_missing_requested_voice(self) -> None:
        with self.assertRaisesRegex(ValueError, "Requested voice"):
            choose_voice_alias({"data": [{"voice": "voice_a"}]}, requested_voice="missing", model="demo")

    def test_derive_http_base_url_from_ws_url(self) -> None:
        self.assertEqual(
            derive_http_base_url("ws://127.0.0.1:9010/api-ws/v1/realtime"),
            "http://127.0.0.1:9010",
        )


if __name__ == "__main__":
    unittest.main()