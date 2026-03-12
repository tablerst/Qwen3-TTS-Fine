from __future__ import annotations

from types import SimpleNamespace
from typing import cast
import unittest

import numpy as np
import torch

from streaming_lora_service.app.audio_utils import float_audio_to_pcm16le_bytes
from streaming_lora_service.app.models import SessionOptions
from streaming_lora_service.app.runtime_session import RuntimeSession
from streaming_lora_service.app.streaming_generator import StreamingCustomVoiceGenerator


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
        return 4

    def decode(self, items):
        wavs = []
        for item in items:
            codes = item["audio_codes"]
            samples = []
            for row in codes.tolist():
                samples.extend([row[0] / 100.0] * 4)
            wavs.append(np.asarray(samples, dtype=np.float32))
        return wavs, 24000


class FakeTalker:
    def __init__(self, config) -> None:
        self.config = config
        self._codec_embedding = FakeEmbedding()
        self._text_embedding = FakeEmbedding()
        self.text_projection = FakeTextProjection()
        self.rope_deltas = None
        self._planned_tokens = [11, 12, 13, config.codec_eos_token_id]
        self._cursor = 0
        self._hidden_size = 4

    def get_input_embeddings(self):
        return self._codec_embedding

    def get_text_embeddings(self):
        return self._text_embedding

    def __call__(self, **kwargs):
        if kwargs.get("inputs_embeds") is not None and kwargs["inputs_embeds"].shape[1] > 1:
            prompt_length = kwargs["inputs_embeds"].shape[1]
            return SimpleNamespace(
                logits=self._build_logits(self._planned_tokens[0], length=prompt_length),
                past_key_values={"cursor": 0},
                past_hidden=torch.zeros((1, 1, self._hidden_size), dtype=torch.float32),
                generation_step=0,
                trailing_text_hidden=kwargs["trailing_text_hidden"],
                tts_pad_embed=kwargs["tts_pad_embed"],
                hidden_states=((torch.zeros((1, prompt_length, self._hidden_size), dtype=torch.float32),), None),
            )

        attention_mask = kwargs["attention_mask"]
        cache_position = kwargs["cache_position"]
        expected_attention_length = int(cache_position[-1].item()) + 1
        if attention_mask.shape[1] != expected_attention_length:
            raise AssertionError(
                f"generation attention_mask length {attention_mask.shape[1]} must match cache_position end {expected_attention_length}"
            )

        token = int(kwargs["input_ids"][0, 0].item())
        self._cursor += 1
        next_token = self._planned_tokens[self._cursor]
        codec_ids = torch.tensor([[token, token + 20, token + 40]], dtype=torch.long)
        return SimpleNamespace(
            logits=self._build_logits(next_token, length=1),
            past_key_values={"cursor": self._cursor},
            past_hidden=torch.full((1, 1, self._hidden_size), float(token), dtype=torch.float32),
            generation_step=kwargs["generation_step"] + 1,
            trailing_text_hidden=kwargs["trailing_text_hidden"],
            tts_pad_embed=kwargs["tts_pad_embed"],
            hidden_states=((torch.zeros((1, 1, self._hidden_size), dtype=torch.float32),), codec_ids),
        )

    def _build_logits(self, next_token: int, *, length: int) -> torch.Tensor:
        logits = torch.full((1, length, self.config.vocab_size), -1000.0, dtype=torch.float32)
        logits[:, -1, next_token] = 1000.0
        return logits


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

    def _validate_languages(self, languages):
        return None

    def _validate_speakers(self, speakers):
        return None

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


class StreamingGeneratorTests(unittest.TestCase):
    @staticmethod
    def make_session() -> RuntimeSession:
        return RuntimeSession(
            session_id="sess_streaming",
            options=SessionOptions(
                model="qwen3-tts-flash-realtime",
                voice="yachiyo_formal",
                language_type="Chinese",
            ),
        )

    def test_iter_audio_chunks_generates_incremental_audio_and_metrics(self) -> None:
        qwen3tts = FakeQwen3TTS()
        generator = StreamingCustomVoiceGenerator(
            qwen3tts,
            text="你好，真流式。",
            speaker="inference_speaker",
            language="Chinese",
            chunk_steps=2,
            left_context_steps=1,
        )

        chunks = list(generator.iter_audio_chunks())

        expected_audio = np.asarray([0.11] * 4 + [0.12] * 4 + [0.13] * 4, dtype=np.float32)
        self.assertEqual(b"".join(chunks), float_audio_to_pcm16le_bytes(expected_audio))
        self.assertEqual(generator.metrics.generated_steps, 3)
        self.assertEqual(generator.metrics.emitted_chunks, 2)
        self.assertEqual(generator.metrics.first_emitted_step, 2)
        self.assertEqual(generator.metrics.finish_reason, "eos")
        self.assertTrue(generator.state.finished)

    def test_step_returns_none_until_chunk_threshold_then_flushes_on_finish(self) -> None:
        qwen3tts = FakeQwen3TTS()
        generator = StreamingCustomVoiceGenerator(
            qwen3tts,
            text="你好，逐步。",
            speaker="inference_speaker",
            language="Chinese",
            chunk_steps=3,
            left_context_steps=1,
        )

        first = generator.step()
        second = generator.step()
        third = generator.step()
        final = generator.step()

        self.assertIsNone(first)
        self.assertIsNone(second)
        self.assertIsNotNone(third)
        self.assertIsNone(final)
        self.assertTrue(generator.state.finished)

    def test_runtime_session_binds_and_resumes_true_generation_state(self) -> None:
        qwen3tts = FakeQwen3TTS()
        session = self.make_session()

        first_generator = StreamingCustomVoiceGenerator(
            qwen3tts,
            text="你好，恢复状态。",
            speaker="inference_speaker",
            language="Chinese",
            runtime_session=session,
            chunk_steps=2,
            left_context_steps=1,
        )

        self.assertEqual(session.state.active_generation_text, "你好，恢复状态。")
        self.assertEqual(session.state.decoded_until_step, 0)
        self.assertIsNotNone(session.state.incremental_decoder)

        first_chunk = first_generator.step()

        self.assertIsNone(first_chunk)
        self.assertEqual(len(session.state.generated_codes), 1)
        self.assertEqual(session.state.sampled_tokens, [11])
        self.assertEqual(session.state.past_key_values, {"cursor": 1})
        self.assertFalse(session.state.generation_finished)

        resumed_generator = StreamingCustomVoiceGenerator(
            qwen3tts,
            text="你好，恢复状态。",
            speaker="inference_speaker",
            language="Chinese",
            runtime_session=session,
            chunk_steps=2,
            left_context_steps=1,
        )

        chunks = list(resumed_generator.iter_audio_chunks())
        expected_audio = np.asarray([0.11] * 4 + [0.12] * 4 + [0.13] * 4, dtype=np.float32)
        generation_snapshot = cast(dict[str, object], session.snapshot()["generation_state"])

        self.assertEqual(b"".join(chunks), float_audio_to_pcm16le_bytes(expected_audio))
        self.assertEqual(session.state.decoded_until_step, 3)
        self.assertTrue(session.state.generation_finished)
        self.assertEqual(generation_snapshot["generated_steps"], 3)
        self.assertEqual(session.state.last_generation_metrics["finish_reason"], "eos")
        self.assertEqual(qwen3tts.model.talker._cursor, 3)

    def test_iter_audio_chunks_marks_length_finish_reason_when_hitting_max_new_tokens(self) -> None:
        qwen3tts = FakeQwen3TTS()
        generator = StreamingCustomVoiceGenerator(
            qwen3tts,
            text="你好，长度上限。",
            speaker="inference_speaker",
            language="Chinese",
            chunk_steps=4,
            left_context_steps=1,
            max_new_tokens=2,
        )

        chunks = list(generator.iter_audio_chunks())

        expected_audio = np.asarray([0.11] * 4 + [0.12] * 4, dtype=np.float32)
        self.assertEqual(b"".join(chunks), float_audio_to_pcm16le_bytes(expected_audio))
        self.assertEqual(generator.metrics.generated_steps, 2)
        self.assertEqual(generator.metrics.finish_reason, "length")
        self.assertTrue(generator.state.finished)


if __name__ == "__main__":
    unittest.main()
