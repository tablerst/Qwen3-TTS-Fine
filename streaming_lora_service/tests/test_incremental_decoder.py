from __future__ import annotations

import unittest

from streaming_lora_service.app.incremental_decoder import IncrementalAudioDecoder, IncrementalDecoderConfig


class IncrementalDecoderTests(unittest.TestCase):
    @staticmethod
    def fake_decode(start_step: int, end_step: int) -> bytes:
        return b"".join(step.to_bytes(2, byteorder="little", signed=False) for step in range(start_step, end_step))

    @staticmethod
    def fake_pcm16_decode(start_step: int, end_step: int) -> bytes:
        return b"".join(step.to_bytes(2, byteorder="little", signed=True) for step in range(start_step, end_step))

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

    def test_decode_buffered_uses_codec_ring_buffer_and_trims_history(self) -> None:
        decoder = IncrementalAudioDecoder(IncrementalDecoderConfig(chunk_steps=2, left_context_steps=1))

        decoder.push_codec_step(0)
        decoder.push_codec_step(1)
        first_audio = decoder.decode_buffered(decode_fn=lambda steps: bytes(steps), bytes_per_step=1)

        self.assertEqual(first_audio, b"\x00\x01")
        self.assertEqual(decoder.buffer_start_step, 1)
        self.assertEqual(decoder.buffered_step_count, 1)

        decoder.push_codec_step(2)
        second_audio = decoder.decode_buffered(
            decode_fn=lambda steps: bytes(steps),
            bytes_per_step=1,
            finished=True,
        )

        self.assertEqual(second_audio, b"\x02")
        self.assertEqual(decoder.emitted_until_step, 3)
        self.assertEqual(decoder.buffer_start_step, 2)
        self.assertEqual(decoder.buffered_step_count, 1)

    def test_first_chunk_steps_delays_only_the_initial_emit(self) -> None:
        decoder = IncrementalAudioDecoder(
            IncrementalDecoderConfig(chunk_steps=2, left_context_steps=1, first_chunk_steps=3)
        )

        self.assertIsNone(decoder.decode(2, decode_fn=self.fake_decode, bytes_per_step=2))
        first_audio = decoder.decode(3, decode_fn=self.fake_decode, bytes_per_step=2)
        second_wait = decoder.decode(4, decode_fn=self.fake_decode, bytes_per_step=2)
        second_audio = decoder.decode(5, decode_fn=self.fake_decode, bytes_per_step=2, finished=True)

        self.assertEqual(first_audio, self.fake_decode(0, 3))
        self.assertIsNone(second_wait)
        self.assertEqual(second_audio, self.fake_decode(3, 5))
        self.assertEqual(decoder.emitted_until_step, 5)

    def test_crossfade_blends_previous_tail_with_next_chunk_head(self) -> None:
        decoder = IncrementalAudioDecoder(
            IncrementalDecoderConfig(chunk_steps=2, left_context_steps=0, crossfade_samples=1)
        )

        first_audio = decoder.decode(2, decode_fn=self.fake_pcm16_decode, bytes_per_step=2)
        second_audio = decoder.decode(4, decode_fn=self.fake_pcm16_decode, bytes_per_step=2, finished=True)

        self.assertEqual(first_audio, (0).to_bytes(2, byteorder="little", signed=True))
        self.assertEqual(
            second_audio,
            (2).to_bytes(2, byteorder="little", signed=True) + (3).to_bytes(2, byteorder="little", signed=True),
        )

    def test_crossfade_can_delay_initial_emit_until_enough_audio_arrives(self) -> None:
        decoder = IncrementalAudioDecoder(
            IncrementalDecoderConfig(chunk_steps=1, left_context_steps=0, crossfade_samples=2)
        )

        first_audio = decoder.decode(1, decode_fn=self.fake_pcm16_decode, bytes_per_step=2)
        second_audio = decoder.decode(3, decode_fn=self.fake_pcm16_decode, bytes_per_step=2, finished=True)

        self.assertEqual(first_audio, b"")
        self.assertEqual(
            second_audio,
            (0).to_bytes(2, byteorder="little", signed=True)
            + (2).to_bytes(2, byteorder="little", signed=True),
        )


if __name__ == "__main__":
    unittest.main()
