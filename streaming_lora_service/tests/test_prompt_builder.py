from __future__ import annotations

from types import SimpleNamespace
import unittest

import torch

from streaming_lora_service.app.prompt_builder import build_custom_voice_talker_prompt


class FakeEmbedding:
    def __init__(self, hidden_size: int = 4) -> None:
        self.hidden_size = hidden_size
        self.weight = torch.zeros((4096, hidden_size), dtype=torch.float32)

    def __call__(self, ids):
        tensor = torch.as_tensor(ids, dtype=torch.long)
        if tensor.ndim == 0:
            tensor = tensor.view(1)
        if tensor.ndim == 1:
            tensor = tensor.unsqueeze(0)
        return tensor.unsqueeze(-1).expand(*tensor.shape, self.hidden_size).to(torch.float32) / 100.0


class FakeTextProjection:
    def __call__(self, tensor: torch.Tensor) -> torch.Tensor:
        return tensor.to(torch.float32)


class FakeSpeechTokenizer:
    def get_output_sample_rate(self) -> int:
        return 24000

    def get_decode_upsample_rate(self) -> int:
        return 1920


class FakeTalker:
    def __init__(self, config) -> None:
        self.config = config
        self._codec_embedding = FakeEmbedding()
        self._text_embedding = FakeEmbedding()
        self.text_projection = FakeTextProjection()

    def get_input_embeddings(self):
        return self._codec_embedding

    def get_text_embeddings(self):
        return self._text_embedding


class FakeModel:
    def __init__(self) -> None:
        talker_config = SimpleNamespace(
            spk_id={"inference_speaker": 3000},
            spk_is_dialect={"inference_speaker": False},
            codec_language_id={"chinese": 42, "auto": 0},
            codec_nothink_id=1,
            codec_think_bos_id=2,
            codec_think_eos_id=3,
            codec_think_id=4,
            codec_pad_id=5,
            codec_bos_id=6,
            codec_eos_token_id=7,
            vocab_size=4096,
        )
        self.config = SimpleNamespace(
            talker_config=talker_config,
            tts_bos_token_id=11,
            tts_eos_token_id=12,
            tts_pad_token_id=13,
        )
        self.talker = FakeTalker(talker_config)
        self.speech_tokenizer = FakeSpeechTokenizer()
        self.tts_model_type = "custom_voice"
        self.tts_model_size = "1p7b"


class FakeQwen3TTS:
    def __init__(self) -> None:
        self.model = FakeModel()
        self.validated_languages = []
        self.validated_speakers = []

    def _validate_languages(self, languages):
        self.validated_languages.extend(languages)

    def _validate_speakers(self, speakers):
        self.validated_speakers.extend(speakers)

    def _build_assistant_text(self, text: str) -> str:
        return text

    def _build_instruct_text(self, instruct: str) -> str:
        return instruct

    def _tokenize_texts(self, texts):
        return [torch.tensor([[101, 102, 103, 104, 105, 106, 107, 108, 109, 110]], dtype=torch.long) for _ in texts]

    def _merge_generate_kwargs(self, **kwargs):
        merged = {
            "max_new_tokens": 32,
            "do_sample": False,
            "top_k": 0,
            "top_p": 1.0,
            "temperature": 1.0,
            "subtalker_dosample": False,
            "subtalker_top_k": 0,
            "subtalker_top_p": 1.0,
            "subtalker_temperature": 1.0,
            "repetition_penalty": 1.0,
            "suppress_tokens": [],
        }
        merged.update(kwargs)
        return merged


class PromptBuilderTests(unittest.TestCase):
    def test_build_custom_voice_prompt_returns_prefill_tensors(self) -> None:
        qwen3tts = FakeQwen3TTS()

        prompt = build_custom_voice_talker_prompt(
            qwen3tts,
            text="你好，prompt。",
            speaker="inference_speaker",
            language="Chinese",
            instruct="自然一些",
        )

        self.assertEqual(qwen3tts.validated_languages, ["Chinese"])
        self.assertEqual(qwen3tts.validated_speakers, ["inference_speaker"])
        self.assertEqual(prompt.sample_rate, 24000)
        self.assertEqual(prompt.decode_upsample_rate, 1920)
        self.assertEqual(prompt.attention_mask.shape[1], prompt.talker_input_embeds.shape[1])
        self.assertTrue(torch.all(prompt.attention_mask == 1))
        self.assertGreater(prompt.trailing_text_hidden.shape[1], 0)
        self.assertEqual(prompt.sampling.max_new_tokens, 32)

    def test_instruction_prefix_increases_prefill_length(self) -> None:
        qwen3tts = FakeQwen3TTS()

        without_instruct = build_custom_voice_talker_prompt(
            qwen3tts,
            text="你好。",
            speaker="inference_speaker",
            language="Chinese",
        )
        with_instruct = build_custom_voice_talker_prompt(
            qwen3tts,
            text="你好。",
            speaker="inference_speaker",
            language="Chinese",
            instruct="更正式一些",
        )

        self.assertGreater(with_instruct.talker_input_embeds.shape[1], without_instruct.talker_input_embeds.shape[1])


if __name__ == "__main__":
    unittest.main()
