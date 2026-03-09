from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any, Iterable, Sequence

import librosa
import torch
import yaml
from peft import (LoraConfig, TaskType, get_peft_model_state_dict,
                  inject_adapter_in_model, set_peft_model_state_dict)
from safetensors.torch import load_file, save_file

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

DEFAULT_TARGET_MODULES = [
    "q_proj",
    "k_proj",
    "v_proj",
    "o_proj",
    "gate_proj",
    "up_proj",
    "down_proj",
]

PHASE2_OPTIONAL_MODULES = [
    "linear_fc1",
    "linear_fc2",
]


def ensure_dir(path: str | Path) -> Path:
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


def load_yaml_config(path: str | Path | None) -> dict[str, Any]:
    if path is None:
        return {}
    with open(path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    if not isinstance(data, dict):
        raise TypeError(f"Config file must contain a mapping, got {type(data)!r}")
    return data


def save_json(data: dict[str, Any] | list[Any], path: str | Path) -> None:
    path = Path(path)
    ensure_dir(path.parent)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


def load_json(path: str | Path) -> dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def resolve_setting(
    cli_value: Any,
    config: dict[str, Any],
    section: str | None,
    key: str,
    default: Any = None,
) -> Any:
    if cli_value is not None:
        return cli_value
    if section and isinstance(config.get(section), dict) and key in config[section]:
        return config[section][key]
    if key in config:
        return config[key]
    return default


def normalize_string_list(value: Any, default: Sequence[str] | None = None) -> list[str]:
    if value is None:
        return list(default or [])
    if isinstance(value, str):
        return [item.strip() for item in value.split(",") if item.strip()]
    if isinstance(value, (list, tuple, set)):
        return [str(item).strip() for item in value if str(item).strip()]
    raise TypeError(f"Expected string or sequence for module list, got {type(value)!r}")


def parse_torch_dtype(dtype_name: str | None) -> torch.dtype:
    normalized = (dtype_name or "bfloat16").lower()
    mapping = {
        "float32": torch.float32,
        "fp32": torch.float32,
        "float16": torch.float16,
        "fp16": torch.float16,
        "half": torch.float16,
        "bfloat16": torch.bfloat16,
        "bf16": torch.bfloat16,
    }
    if normalized not in mapping:
        raise ValueError(f"Unsupported dtype: {dtype_name}")
    return mapping[normalized]


def freeze_all_parameters(model: torch.nn.Module) -> None:
    for param in model.parameters():
        param.requires_grad = False


def mark_trainable_by_name(model: torch.nn.Module, name_fragments: Iterable[str]) -> list[str]:
    enabled: list[str] = []
    fragments = [frag for frag in name_fragments if frag]
    if not fragments:
        return enabled
    for name, param in model.named_parameters():
        if any(fragment in name for fragment in fragments):
            param.requires_grad = True
            enabled.append(name)
    return enabled


def count_parameters(model: torch.nn.Module) -> tuple[int, int]:
    trainable = sum(param.numel() for param in model.parameters() if param.requires_grad)
    total = sum(param.numel() for param in model.parameters())
    return trainable, total


def build_lora_config(
    r: int,
    alpha: int,
    dropout: float,
    target_modules: Sequence[str],
    bias: str = "none",
) -> LoraConfig:
    return LoraConfig(
        r=r,
        lora_alpha=alpha,
        lora_dropout=dropout,
        bias=bias,
        target_modules=list(target_modules),
        task_type=TaskType.CAUSAL_LM,
    )


def inject_lora(
    model: torch.nn.Module,
    lora_config: LoraConfig,
    extra_trainable_modules: Sequence[str] | None = None,
) -> tuple[int, int, list[str]]:
    freeze_all_parameters(model)
    inject_adapter_in_model(lora_config, model)
    enabled = mark_trainable_by_name(model, extra_trainable_modules or [])
    return (*count_parameters(model), enabled)


def extract_target_speaker_embedding(qwen3tts: Any, ref_audio: str | Path) -> torch.Tensor:
    normalized = qwen3tts._normalize_audio_inputs(str(ref_audio))
    wav, sr = normalized[0]
    target_sr = qwen3tts.model.speaker_encoder_sample_rate
    if sr != target_sr:
        wav = librosa.resample(wav, orig_sr=sr, target_sr=target_sr)
        sr = target_sr
    speaker_embedding = qwen3tts.model.extract_speaker_embedding(wav, sr).detach().cpu()
    return speaker_embedding


def make_config_patch(speaker_name: str, speaker_id: int) -> dict[str, Any]:
    return {
        "tts_model_type": "custom_voice",
        "talker_config": {
            "spk_id": {
                speaker_name: speaker_id,
            },
            "spk_is_dialect": {
                speaker_name: False,
            },
        },
    }


def save_speaker_patch(path: str | Path, speaker_id: int, speaker_embedding: torch.Tensor) -> None:
    path = Path(path)
    ensure_dir(path.parent)
    tensor_dict = {
        "speaker_id": torch.tensor([speaker_id], dtype=torch.int64),
        "embedding": speaker_embedding.detach().cpu(),
    }
    save_file(tensor_dict, str(path))


def load_speaker_patch(path: str | Path) -> tuple[int, torch.Tensor]:
    state = load_file(str(path))
    if "speaker_id" not in state or "embedding" not in state:
        raise KeyError(f"Speaker patch {path} must contain 'speaker_id' and 'embedding'")
    speaker_id = int(state["speaker_id"].view(-1)[0].item())
    embedding = state["embedding"]
    return speaker_id, embedding


def save_lora_adapter(model: torch.nn.Module, adapter_dir: str | Path, lora_config: LoraConfig) -> Path:
    adapter_dir = ensure_dir(adapter_dir)
    adapter_state = get_peft_model_state_dict(model)
    adapter_state = {key: value.detach().cpu() for key, value in adapter_state.items()}
    save_file(adapter_state, str(adapter_dir / "adapter_model.safetensors"))
    if hasattr(lora_config, "save_pretrained"):
        lora_config.save_pretrained(str(adapter_dir))
    else:
        save_json(lora_config.to_dict(), adapter_dir / "adapter_config.json")
    return adapter_dir


def load_lora_adapter(model: torch.nn.Module, adapter_dir: str | Path) -> LoraConfig:
    adapter_dir = Path(adapter_dir)
    lora_config = LoraConfig.from_pretrained(str(adapter_dir))
    inject_adapter_in_model(lora_config, model)
    adapter_state = load_file(str(adapter_dir / "adapter_model.safetensors"))
    outcome = set_peft_model_state_dict(model, adapter_state)
    if getattr(outcome, "unexpected_keys", None):
        raise ValueError(f"Unexpected adapter keys: {outcome.unexpected_keys}")
    return lora_config


def apply_config_patch(model: Any, config_patch: dict[str, Any]) -> None:
    tts_model_type = config_patch.get("tts_model_type")
    talker_patch = config_patch.get("talker_config", {})
    spk_id = talker_patch.get("spk_id", {})
    spk_is_dialect = talker_patch.get("spk_is_dialect", {})

    if tts_model_type is not None:
        model.config.tts_model_type = tts_model_type
        model.tts_model_type = tts_model_type

    if spk_id:
        model.config.talker_config.spk_id.update(spk_id)
    if spk_is_dialect:
        model.config.talker_config.spk_is_dialect.update(spk_is_dialect)

    model.supported_speakers = list(model.config.talker_config.spk_id.keys())


def apply_speaker_patch(model: Any, speaker_patch_file: str | Path) -> int:
    speaker_id, embedding = load_speaker_patch(speaker_patch_file)
    target_weight = model.talker.model.codec_embedding.weight
    if speaker_id >= target_weight.shape[0]:
        raise IndexError(
            f"speaker_id {speaker_id} is out of range for codec embedding size {target_weight.shape[0]}"
        )
    embedding = embedding.to(device=target_weight.device, dtype=target_weight.dtype).view(-1)
    if embedding.shape[0] != target_weight.shape[1]:
        raise ValueError(
            f"Speaker embedding dim mismatch: expected {target_weight.shape[1]}, got {embedding.shape[0]}"
        )
    with torch.no_grad():
        target_weight[speaker_id].copy_(embedding)
    return speaker_id


def build_bundle_manifest(
    base_model_path: str,
    speaker_name: str,
    speaker_id: int,
    adapter_subdir: str = "adapter",
) -> dict[str, Any]:
    return {
        "format_version": 1,
        "base_model_path": base_model_path,
        "adapter_dir": adapter_subdir,
        "speaker_embedding_file": "speaker_embedding.safetensors",
        "config_patch_file": "config_patch.json",
        "speaker_name": speaker_name,
        "speaker_id": speaker_id,
    }
