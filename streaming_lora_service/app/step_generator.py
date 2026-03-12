from __future__ import annotations

from typing import Any, Optional

import torch

from .audio_utils import float_audio_to_pcm16le_bytes
from .models import SynthesizedAudio


def generate_custom_voice_step_aware(
    qwen3tts: Any,
    *,
    text: str,
    speaker: str,
    language: str = "Auto",
    instruct: str | None = None,
    non_streaming_mode: bool = False,
    **kwargs,
) -> SynthesizedAudio:
    if qwen3tts.model.tts_model_type != "custom_voice":
        raise ValueError("step-aware custom voice generation only supports custom_voice models")

    if qwen3tts.model.tts_model_size in "0b6":
        instruct = None

    qwen3tts._validate_languages([language])
    qwen3tts._validate_speakers([speaker])

    input_ids = qwen3tts._tokenize_texts([qwen3tts._build_assistant_text(text)])

    instruct_ids: list[Optional[torch.Tensor]] = []
    if instruct is None or instruct == "":
        instruct_ids.append(None)
    else:
        instruct_ids.append(qwen3tts._tokenize_texts([qwen3tts._build_instruct_text(instruct)])[0])

    gen_kwargs = qwen3tts._merge_generate_kwargs(**kwargs)

    talker_codes_list, _ = qwen3tts.model.generate(
        input_ids=input_ids,
        instruct_ids=instruct_ids,
        languages=[language],
        speakers=[speaker],
        non_streaming_mode=non_streaming_mode,
        **gen_kwargs,
    )

    talker_codes = talker_codes_list[0]
    wavs, sample_rate = qwen3tts.model.speech_tokenizer.decode([{"audio_codes": talker_codes}])
    audio_bytes = float_audio_to_pcm16le_bytes(wavs[0])
    codec_tokens = tuple(tuple(int(item) for item in row) for row in talker_codes.tolist())

    def decode_step_range(start_step: int, end_step: int) -> bytes:
        if start_step < 0 or end_step < start_step:
            raise ValueError(f"Invalid step range: {start_step}..{end_step}")
        sliced_codes = talker_codes[start_step:end_step]
        if sliced_codes.numel() == 0:
            return b""
        sliced_wavs, _ = qwen3tts.model.speech_tokenizer.decode([{"audio_codes": sliced_codes}])
        return float_audio_to_pcm16le_bytes(sliced_wavs[0])

    return SynthesizedAudio(
        audio_bytes=audio_bytes,
        sample_rate=int(sample_rate),
        channels=1,
        codec_steps=int(talker_codes.shape[0]),
        decode_step_range=decode_step_range,
        codec_tokens=codec_tokens,
    )