from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch


@dataclass(frozen=True)
class TalkerSamplingConfig:
    max_new_tokens: int
    do_sample: bool
    top_k: int
    top_p: float
    temperature: float
    subtalker_dosample: bool
    subtalker_top_k: int
    subtalker_top_p: float
    subtalker_temperature: float
    eos_token_id: int
    repetition_penalty: float
    suppress_tokens: tuple[int, ...]


@dataclass(frozen=True)
class CustomVoiceTalkerPrompt:
    talker_input_embeds: torch.Tensor
    attention_mask: torch.Tensor
    trailing_text_hidden: torch.Tensor
    tts_pad_embed: torch.Tensor
    sample_rate: int
    decode_upsample_rate: int
    sampling: TalkerSamplingConfig


class PromptBuilderError(RuntimeError):
    """Raised when a custom-voice talker prompt cannot be prepared."""


def build_talker_sampling_config(model: Any, merged_generate_kwargs: dict[str, Any]) -> TalkerSamplingConfig:
    talker_config = model.config.talker_config
    eos_token_id = int(merged_generate_kwargs.get("eos_token_id") or talker_config.codec_eos_token_id)
    suppress_tokens = tuple(
        int(item)
        for item in merged_generate_kwargs.get(
            "suppress_tokens",
            [
                i
                for i in range(max(0, talker_config.vocab_size - 1024), talker_config.vocab_size)
                if i not in (talker_config.codec_eos_token_id,)
            ],
        )
    )
    return TalkerSamplingConfig(
        max_new_tokens=int(merged_generate_kwargs.get("max_new_tokens", 4096)),
        do_sample=bool(merged_generate_kwargs.get("do_sample", True)),
        top_k=int(merged_generate_kwargs.get("top_k", 50)),
        top_p=float(merged_generate_kwargs.get("top_p", 1.0)),
        temperature=float(merged_generate_kwargs.get("temperature", 0.9)),
        subtalker_dosample=bool(merged_generate_kwargs.get("subtalker_dosample", True)),
        subtalker_top_k=int(merged_generate_kwargs.get("subtalker_top_k", 50)),
        subtalker_top_p=float(merged_generate_kwargs.get("subtalker_top_p", 1.0)),
        subtalker_temperature=float(merged_generate_kwargs.get("subtalker_temperature", 0.9)),
        eos_token_id=eos_token_id,
        repetition_penalty=float(merged_generate_kwargs.get("repetition_penalty", 1.05)),
        suppress_tokens=suppress_tokens,
    )


def build_custom_voice_talker_prompt(
    qwen3tts: Any,
    *,
    text: str,
    speaker: str,
    language: str = "Auto",
    instruct: str | None = None,
    non_streaming_mode: bool = False,
    **generate_kwargs: Any,
) -> CustomVoiceTalkerPrompt:
    if qwen3tts.model.tts_model_type != "custom_voice":
        raise PromptBuilderError("custom voice talker prompt builder only supports custom_voice models")

    if qwen3tts.model.tts_model_size in "0b6":
        instruct = None

    qwen3tts._validate_languages([language])
    qwen3tts._validate_speakers([speaker])

    input_id = qwen3tts._tokenize_texts([qwen3tts._build_assistant_text(text)])[0]
    instruct_id = None
    if instruct not in (None, ""):
        instruct_id = qwen3tts._tokenize_texts([qwen3tts._build_instruct_text(instruct)])[0]

    merged_generate_kwargs = qwen3tts._merge_generate_kwargs(**generate_kwargs)
    sampling = build_talker_sampling_config(qwen3tts.model, merged_generate_kwargs)

    talker = qwen3tts.model.talker
    talker_config = qwen3tts.model.config.talker_config

    talker_input_parts: list[torch.Tensor] = []
    if instruct_id is not None:
        talker_input_parts.append(talker.text_projection(talker.get_text_embeddings()(instruct_id)))

    speaker_embed = _resolve_speaker_embed(qwen3tts, input_id, speaker)
    language_id = _resolve_language_id(qwen3tts, language, speaker)

    tts_prompt_ids = torch.tensor(
        [[qwen3tts.model.config.tts_bos_token_id, qwen3tts.model.config.tts_eos_token_id, qwen3tts.model.config.tts_pad_token_id]],
        device=talker.get_input_embeddings().weight.device,
        dtype=input_id.dtype,
    )
    tts_bos_embed, tts_eos_embed, tts_pad_embed = talker.text_projection(
        talker.get_text_embeddings()(tts_prompt_ids)
    ).chunk(3, dim=1)

    if language_id is None:
        codec_prefill_list = [[
            talker_config.codec_nothink_id,
            talker_config.codec_think_bos_id,
            talker_config.codec_think_eos_id,
        ]]
    else:
        codec_prefill_list = [[
            talker_config.codec_think_id,
            talker_config.codec_think_bos_id,
            language_id,
            talker_config.codec_think_eos_id,
        ]]

    codec_input_embedding_0 = talker.get_input_embeddings()(
        torch.tensor(codec_prefill_list, device=input_id.device, dtype=input_id.dtype)
    )
    codec_input_embedding_1 = talker.get_input_embeddings()(
        torch.tensor(
            [[talker_config.codec_pad_id, talker_config.codec_bos_id]],
            device=input_id.device,
            dtype=input_id.dtype,
        )
    )
    if speaker_embed is None:
        codec_input_embedding = torch.cat([codec_input_embedding_0, codec_input_embedding_1], dim=1)
    else:
        codec_input_embedding = torch.cat(
            [codec_input_embedding_0, speaker_embed.view(1, 1, -1), codec_input_embedding_1], dim=1
        )

    talker_input_embed_role = talker.text_projection(talker.get_text_embeddings()(input_id[:, :3]))
    codec_prefill_embed = torch.cat(
        (
            tts_pad_embed.expand(-1, codec_input_embedding.shape[1] - 2, -1),
            tts_bos_embed,
        ),
        dim=1,
    ) + codec_input_embedding[:, :-1]
    talker_input_embed = torch.cat((talker_input_embed_role, codec_prefill_embed), dim=1)
    first_text_with_codec = talker.text_projection(talker.get_text_embeddings()(input_id[:, 3:4])) + codec_input_embedding[:, -1:]
    talker_input_embed = torch.cat([talker_input_embed, first_text_with_codec], dim=1)

    if non_streaming_mode:
        talker_input_embed = talker_input_embed[:, :-1]
        text_embed = torch.cat(
            (
                talker.text_projection(talker.get_text_embeddings()(input_id[:, 3:-5])),
                tts_eos_embed,
            ),
            dim=1,
        ) + talker.get_input_embeddings()(
            torch.tensor(
                [[talker_config.codec_pad_id] * (input_id[:, 3:-5].shape[1] + 1)],
                device=input_id.device,
                dtype=input_id.dtype,
            )
        )
        bos_embed = tts_pad_embed + talker.get_input_embeddings()(
            torch.tensor([[talker_config.codec_bos_id]], device=input_id.device, dtype=input_id.dtype)
        )
        trailing_text_hidden = tts_pad_embed
        talker_input_embed = torch.cat([talker_input_embed, text_embed, bos_embed], dim=1)
    else:
        trailing_text_hidden = torch.cat(
            (
                talker.text_projection(talker.get_text_embeddings()(input_id[:, 4:-5])),
                tts_eos_embed,
            ),
            dim=1,
        )

    talker_input_parts.append(talker_input_embed)
    talker_input_embeds = torch.cat(talker_input_parts, dim=1)
    attention_mask = torch.ones(
        (talker_input_embeds.shape[0], talker_input_embeds.shape[1]),
        device=talker_input_embeds.device,
        dtype=torch.long,
    )

    speech_tokenizer = qwen3tts.model.speech_tokenizer
    return CustomVoiceTalkerPrompt(
        talker_input_embeds=talker_input_embeds,
        attention_mask=attention_mask,
        trailing_text_hidden=trailing_text_hidden,
        tts_pad_embed=tts_pad_embed,
        sample_rate=int(speech_tokenizer.get_output_sample_rate()),
        decode_upsample_rate=int(speech_tokenizer.get_decode_upsample_rate()),
        sampling=sampling,
    )


def _resolve_speaker_embed(qwen3tts: Any, input_id: torch.Tensor, speaker: str) -> torch.Tensor | None:
    if not speaker:
        return None
    talker_config = qwen3tts.model.config.talker_config
    speaker_key = speaker.lower()
    if speaker_key not in talker_config.spk_id:
        raise PromptBuilderError(f"Speaker {speaker} not implemented")
    speaker_id = talker_config.spk_id[speaker_key]
    return qwen3tts.model.talker.get_input_embeddings()(
        torch.tensor(speaker_id, device=input_id.device, dtype=input_id.dtype)
    )


def _resolve_language_id(qwen3tts: Any, language: str, speaker: str) -> int | None:
    talker_config = qwen3tts.model.config.talker_config
    normalized_language = (language or "Auto").lower()
    if normalized_language == "auto":
        language_id = None
    else:
        if normalized_language not in talker_config.codec_language_id:
            raise PromptBuilderError(f"Language {language} not implemented")
        language_id = talker_config.codec_language_id[normalized_language]

    speaker_key = (speaker or "").lower()
    if normalized_language in {"chinese", "auto"} and speaker_key and talker_config.spk_is_dialect[speaker_key] is not False:
        dialect = talker_config.spk_is_dialect[speaker_key]
        language_id = talker_config.codec_language_id[dialect]
    return language_id


__all__ = [
    "CustomVoiceTalkerPrompt",
    "PromptBuilderError",
    "TalkerSamplingConfig",
    "build_custom_voice_talker_prompt",
    "build_talker_sampling_config",
]
