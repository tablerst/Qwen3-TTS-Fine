from __future__ import annotations

import unittest

from streaming_lora_service.app.models import SynthesizedAudio
from streaming_lora_service.app.step_streamer import AudioStepStreamer, AudioStepStreamerConfig


class AudioStepStreamerTests(unittest.TestCase):
    def test_iter_audio_chunks_reconstructs_original_audio(self) -> None:
        original = (bytes(range(256)) * 100)
        synthesized = SynthesizedAudio(audio_bytes=original, sample_rate=24000, channels=1)
        streamer = AudioStepStreamer(
            AudioStepStreamerConfig(samples_per_step=1920, chunk_steps=4, left_context_steps=2)
        )

        chunks = streamer.iter_audio_chunks(synthesized)

        self.assertGreaterEqual(len(chunks), 1)
        self.assertEqual(b"".join(chunks), original)

    def test_iter_audio_chunks_handles_short_audio(self) -> None:
        synthesized = SynthesizedAudio(audio_bytes=b"\x01\x02\x03\x04", sample_rate=24000, channels=1)
        streamer = AudioStepStreamer(
            AudioStepStreamerConfig(samples_per_step=1920, chunk_steps=4, left_context_steps=2)
        )

        chunks = streamer.iter_audio_chunks(synthesized)

        self.assertEqual(chunks, [b"\x01\x02\x03\x04"])


if __name__ == "__main__":
    unittest.main()