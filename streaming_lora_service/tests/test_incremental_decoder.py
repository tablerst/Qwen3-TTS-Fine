from __future__ import annotations

import unittest

from streaming_lora_service.app.incremental_decoder import IncrementalAudioDecoder, IncrementalDecoderConfig


class IncrementalDecoderTests(unittest.TestCase):
    @staticmethod
    def fake_decode(start_step: int, end_step: int) -> bytes:
        return b"".join(step.to_bytes(2, byteorder="little", signed=False) for step in range(start_step, end_step))

    def test_decode_waits_until_chunk_threshold(self) -> None:
        decoder = IncrementalAudioDecoder(IncrementalDecoderConfig(chunk_steps=4, left_context_steps=2))

        self.assertIsNone(decoder.decode(3, decode_fn=self.fake_decode, bytes_per_step=2))
        audio = decoder.decode(4, decode_fn=self.fake_decode, bytes_per_step=2)

        self.assertEqual(audio, self.fake_decode(0, 4))
        self.assertEqual(decoder.emitted_until_step, 4)

    def test_decode_uses_left_context_but_only_emits_new_audio(self) -> None:
        decoder = IncrementalAudioDecoder(IncrementalDecoderConfig(chunk_steps=4, left_context_steps=2))
        decoder.decode(4, decode_fn=self.fake_decode, bytes_per_step=2)

        audio = decoder.decode(6, decode_fn=self.fake_decode, bytes_per_step=2, finished=True)

        self.assertEqual(audio, self.fake_decode(4, 6))
        self.assertEqual(decoder.emitted_until_step, 6)


if __name__ == "__main__":
    unittest.main()
