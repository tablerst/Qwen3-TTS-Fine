from __future__ import annotations

import unittest

import numpy as np
import torch

from streaming_lora_service.app.step_generator import generate_custom_voice_step_aware


class FakeSpeechTokenizer:
    def decode(self, items):
        codes = items[0]["audio_codes"]
        step_count = int(codes.shape[0])
        wav = np.linspace(-0.5, 0.5, num=max(step_count, 1) * 4, dtype=np.float32)
        return [wav], 24000


class FakeModel:
    def __init__(self) -> None:
        self.tts_model_type = "custom_voice"
        self.tts_model_size = "1p7b"
        self.speech_tokenizer = FakeSpeechTokenizer()

    def generate(self, **kwargs):
        codes = torch.tensor(
            [
                [1, 10, 11],
                [2, 12, 13],
                [3, 14, 15],
            ],
            dtype=torch.long,
        )
        return [codes], [torch.zeros((3, 8), dtype=torch.float32)]


class FakeQwen3TTS:
    def __init__(self) -> None:
        self.model = FakeModel()
        self._validated_languages = []
        self._validated_speakers = []

    def _validate_languages(self, languages):
        self._validated_languages.extend(languages)

    def _validate_speakers(self, speakers):
        self._validated_speakers.extend(speakers)

    def _build_assistant_text(self, text: str) -> str:
        return text

    def _build_instruct_text(self, instruct: str) -> str:
        return instruct

    def _tokenize_texts(self, texts):
        return [torch.tensor([[101, 102, 103]], dtype=torch.long) for _ in texts]

    def _merge_generate_kwargs(self, **kwargs):
        return kwargs


class StepGeneratorTests(unittest.TestCase):
    def test_generate_custom_voice_step_aware_returns_codec_metadata(self) -> None:
        qwen3tts = FakeQwen3TTS()

        synthesized = generate_custom_voice_step_aware(
            qwen3tts,
            text="你好，step aware。",
            speaker="inference_speaker",
            language="Chinese",
            instruct="自然一些",
        )

        self.assertEqual(qwen3tts._validated_languages, ["Chinese"])
        self.assertEqual(qwen3tts._validated_speakers, ["inference_speaker"])
        self.assertEqual(synthesized.sample_rate, 24000)
        self.assertEqual(synthesized.codec_steps, 3)
        self.assertEqual(synthesized.codec_tokens, ((1, 10, 11), (2, 12, 13), (3, 14, 15)))
        self.assertTrue(synthesized.audio_bytes)
        self.assertIsNotNone(synthesized.decode_step_range)
        self.assertTrue(synthesized.decode_step_range(0, 2))


if __name__ == "__main__":
    unittest.main()