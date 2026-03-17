from __future__ import annotations

import argparse
import shutil
import time
from pathlib import Path
from typing import Any

import torch
from safetensors.torch import save_model

from ..contracts import validate_exported_model_dir
from lora_finetuning.common import (
    apply_config_patch,
    apply_speaker_patch,
    ensure_dir,
    load_json,
    load_lora_adapter,
    parse_torch_dtype,
    save_json,
)
from qwen_tts import Qwen3TTSModel
from streaming_lora_service.app.bundle_loader import infer_speaker_name, resolve_bundle_artifacts


SUMMARY_FILENAME = "merged_export_summary.json"
SOURCE_MANIFEST_FILENAME = "source_bundle_manifest.json"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Export a LoRA custom-voice bundle as a fully merged local model directory"
    )
    parser.add_argument("--bundle_dir", required=True, help="Bundle directory created by export_custom_voice.py")
    parser.add_argument("--output_dir", required=True, help="Output directory for the merged local model")
    parser.add_argument("--base_model", default=None, help="Optional override for the base model path/repo")
    parser.add_argument("--speaker_name", default=None, help="Optional override for the exported speaker name")
    parser.add_argument("--device_map", default="cuda:0", help="Device map passed into Qwen3TTSModel.from_pretrained")
    parser.add_argument("--torch_dtype", default="bfloat16", help="Torch dtype: bfloat16/fp16/fp32")
    parser.add_argument("--attn_implementation", default="sdpa", help="Attention backend for loading the base model")
    parser.add_argument("--local_files_only", action="store_true", help="Only load files from local paths")
    return parser


def _load_base_qwen_model(
    base_model: str,
    *,
    model_factory: Any,
    device_map: str,
    dtype: Any,
    attn_implementation: str,
    local_files_only: bool,
):
    common_kwargs = {
        "device_map": device_map,
        "attn_implementation": attn_implementation,
        "local_files_only": local_files_only,
    }
    try:
        return model_factory.from_pretrained(base_model, dtype=dtype, **common_kwargs)
    except TypeError as exc:
        message = str(exc)
        if "dtype" not in message or "unexpected keyword" not in message:
            raise
        return model_factory.from_pretrained(base_model, torch_dtype=dtype, **common_kwargs)


def _merge_injected_lora_layers(model: Any) -> int:
    merged_count = 0
    for module in model.modules():
        merge_fn = getattr(module, "merge", None)
        already_merged = bool(getattr(module, "merged", False))
        if callable(merge_fn) and not already_merged:
            merge_fn()
            merged_count += 1
    return merged_count


def _strip_lora_wrappers(module: Any) -> int:
    stripped_count = 0
    child_modules = getattr(module, "_modules", None)
    if not isinstance(child_modules, dict):
        return stripped_count

    for name, child in list(child_modules.items()):
        if child is None:
            continue
        get_base_layer = getattr(child, "get_base_layer", None)
        if callable(get_base_layer):
            child_modules[name] = get_base_layer()
            stripped_count += 1
            continue
        stripped_count += _strip_lora_wrappers(child)
    return stripped_count


def _json_safe(value: Any) -> Any:
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, torch.dtype):
        return str(value).replace("torch.", "")
    if isinstance(value, dict):
        normalized: dict[str, Any] = {}
        for key, item in value.items():
            key_str = str(key)
            if key_str == "dtype" or callable(item):
                continue
            normalized[key_str] = _json_safe(item)
        return normalized
    if isinstance(value, (list, tuple, set)):
        return [_json_safe(item) for item in value]
    if hasattr(value, "to_dict") and callable(getattr(value, "to_dict")):
        return _json_safe(value.to_dict())
    if hasattr(value, "__dict__") and not isinstance(value, type):
        normalized_obj: dict[str, Any] = {}
        for key, item in vars(value).items():
            key_str = str(key)
            if key_str == "dtype" or callable(item):
                continue
            normalized_obj[key_str] = _json_safe(item)
        return normalized_obj
    if callable(value):
        return None
    return str(value)


def _save_merged_model(qwen3tts: Any, output_path: Path) -> None:
    save_model(qwen3tts.model, str(output_path / "model.safetensors"))


def _build_export_config(base_model: str, qwen3tts: Any, config_patch: dict[str, Any]) -> dict[str, Any]:
    base_path = Path(base_model)
    base_config_path = base_path / "config.json"
    if base_config_path.exists():
        config = load_json(base_config_path)
    else:
        config_obj = qwen3tts.model.config
        config_dict_getter = getattr(config_obj, "to_dict", None)
        if callable(config_dict_getter):
            config = _json_safe(config_dict_getter())
        else:
            config = _json_safe(vars(config_obj))

    config["tts_model_type"] = config_patch.get("tts_model_type", config.get("tts_model_type"))

    talker_config = config.setdefault("talker_config", {})
    patch_talker_config = config_patch.get("talker_config", {})
    spk_id = patch_talker_config.get("spk_id", {})
    spk_is_dialect = patch_talker_config.get("spk_is_dialect", {})

    if spk_id:
        talker_config.setdefault("spk_id", {}).update(spk_id)
    if spk_is_dialect:
        talker_config.setdefault("spk_is_dialect", {}).update(spk_is_dialect)

    return config


def _save_generation_config(base_model: str, qwen3tts: Any, output_path: Path) -> None:
    base_generation_config_path = Path(base_model) / "generation_config.json"
    if base_generation_config_path.exists():
        shutil.copy2(base_generation_config_path, output_path / "generation_config.json")
        return

    generation_config = getattr(qwen3tts.model, "generation_config", None)
    if generation_config is not None:
        to_dict = getattr(generation_config, "to_dict", None)
        if callable(to_dict):
            save_json(_json_safe(to_dict()), output_path / "generation_config.json")


def _copy_speech_tokenizer_assets(base_model: str, output_path: Path) -> None:
    source_dir = Path(base_model) / "speech_tokenizer"
    target_dir = output_path / "speech_tokenizer"
    if source_dir.exists() and source_dir.is_dir():
        shutil.copytree(source_dir, target_dir, dirs_exist_ok=True)


def merge_bundle_to_local_model(
    bundle_dir: str | Path,
    output_dir: str | Path,
    *,
    base_model: str | None = None,
    speaker_name: str | None = None,
    device_map: str = "cuda:0",
    torch_dtype: str = "bfloat16",
    attn_implementation: str = "sdpa",
    local_files_only: bool = False,
    model_factory: Any | None = None,
) -> dict[str, Any]:
    model_factory = model_factory or Qwen3TTSModel
    started_at = time.perf_counter()

    artifacts = resolve_bundle_artifacts(
        bundle_dir,
        base_model=base_model,
        speaker_name=speaker_name,
    )
    config_patch = load_json(artifacts.config_patch_file)
    resolved_speaker_name = infer_speaker_name(config_patch, artifacts.speaker_name)
    dtype = parse_torch_dtype(torch_dtype)

    qwen3tts = _load_base_qwen_model(
        artifacts.base_model,
        model_factory=model_factory,
        device_map=device_map,
        dtype=dtype,
        attn_implementation=attn_implementation,
        local_files_only=local_files_only,
    )

    load_lora_adapter(qwen3tts.model, artifacts.adapter_dir)
    merged_layer_count = _merge_injected_lora_layers(qwen3tts.model)
    if merged_layer_count <= 0:
        raise RuntimeError("No injected LoRA layers were merged; export would be incomplete")
    stripped_wrapper_count = _strip_lora_wrappers(qwen3tts.model)

    apply_config_patch(qwen3tts.model, config_patch)
    applied_speaker_id = apply_speaker_patch(qwen3tts.model, artifacts.speaker_patch_file)
    qwen3tts.model.eval()

    output_path = ensure_dir(output_dir)
    _save_merged_model(qwen3tts, output_path)
    save_json(_build_export_config(artifacts.base_model, qwen3tts, config_patch), output_path / "config.json")
    _save_generation_config(artifacts.base_model, qwen3tts, output_path)
    _copy_speech_tokenizer_assets(artifacts.base_model, output_path)
    if not hasattr(qwen3tts.processor, "save_pretrained"):
        raise TypeError("Loaded Qwen3TTS processor does not support save_pretrained()")
    qwen3tts.processor.save_pretrained(str(output_path))

    save_json(artifacts.manifest, output_path / SOURCE_MANIFEST_FILENAME)
    contract = validate_exported_model_dir(output_path, expected_speaker=resolved_speaker_name)

    summary = {
        "source_bundle_dir": str(Path(bundle_dir).resolve()),
        "output_dir": str(output_path.resolve()),
        "base_model": artifacts.base_model,
        "adapter_dir": str(artifacts.adapter_dir),
        "resolved_speaker_name": resolved_speaker_name,
        "speaker_id": applied_speaker_id,
        "merged_layer_count": merged_layer_count,
        "stripped_wrapper_count": stripped_wrapper_count,
        "tts_model_type": str(getattr(qwen3tts.model, "tts_model_type", "")),
        "device_map": device_map,
        "torch_dtype": torch_dtype,
        "attn_implementation": attn_implementation,
        "local_files_only": local_files_only,
        "elapsed_seconds": round(time.perf_counter() - started_at, 4),
        "validated_contract": contract,
    }
    save_json(summary, output_path / SUMMARY_FILENAME)
    return summary


def parse_args() -> argparse.Namespace:
    return build_parser().parse_args()


def main() -> None:
    args = parse_args()
    summary = merge_bundle_to_local_model(
        bundle_dir=args.bundle_dir,
        output_dir=args.output_dir,
        base_model=args.base_model,
        speaker_name=args.speaker_name,
        device_map=args.device_map,
        torch_dtype=args.torch_dtype,
        attn_implementation=args.attn_implementation,
        local_files_only=args.local_files_only,
    )

    print(f"Merged export completed: {summary['output_dir']}")
    print(f"Speaker: {summary['resolved_speaker_name']} (id={summary['speaker_id']})")
    print(f"Model type: {summary['tts_model_type']}")
    print(f"Summary: {Path(summary['output_dir']) / SUMMARY_FILENAME}")


if __name__ == "__main__":
    main()
