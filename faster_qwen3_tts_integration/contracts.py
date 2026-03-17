from __future__ import annotations

from pathlib import Path
from typing import Any

from lora_finetuning.common import load_json


WEIGHT_FILE_CANDIDATES = (
    "model.safetensors",
    "model.safetensors.index.json",
    "pytorch_model.bin",
    "pytorch_model.bin.index.json",
)

PROCESSOR_FILE_CANDIDATES = (
    "processor_config.json",
    "preprocessor_config.json",
    "tokenizer_config.json",
    "tokenizer.json",
)


class ContractError(RuntimeError):
    """Raised when an exported merged model directory is incomplete or incompatible."""


def load_model_config(model_dir: str | Path) -> dict[str, Any]:
    model_path = Path(model_dir)
    config_path = model_path / "config.json"
    if not config_path.exists():
        raise ContractError(f"Missing config.json in exported model directory: {model_path}")
    config = load_json(config_path)
    if not isinstance(config, dict):
        raise ContractError(f"config.json must contain a JSON object, got {type(config)!r}")
    return config


def get_speaker_map(config: dict[str, Any]) -> dict[str, int]:
    talker_config = config.get("talker_config") or {}
    speaker_map = talker_config.get("spk_id") or {}
    if not isinstance(speaker_map, dict):
        raise ContractError("config.talker_config.spk_id must be a mapping")
    normalized: dict[str, int] = {}
    for name, value in speaker_map.items():
        normalized[str(name)] = int(value)
    return normalized


def infer_speaker_name_from_config(
    config: dict[str, Any],
    explicit_speaker: str | None = None,
    *,
    require_single: bool = True,
) -> str:
    speaker_map = get_speaker_map(config)
    if not speaker_map:
        raise ContractError("Exported model config does not define any custom_voice speakers")

    speakers_by_lower = {speaker.lower(): speaker for speaker in speaker_map}
    if explicit_speaker:
        matched = speakers_by_lower.get(explicit_speaker.lower())
        if matched is None:
            raise ContractError(
                f"Requested speaker {explicit_speaker!r} not found in exported model. "
                f"Available speakers: {sorted(speaker_map)}"
            )
        return matched

    if require_single and len(speaker_map) != 1:
        raise ContractError(
            "Exported model defines multiple speakers; pass an explicit speaker name to disambiguate"
        )
    return next(iter(speaker_map.keys()))


def infer_speaker_name(
    model_dir: str | Path,
    explicit_speaker: str | None = None,
    *,
    require_single: bool = True,
) -> str:
    return infer_speaker_name_from_config(
        load_model_config(model_dir),
        explicit_speaker=explicit_speaker,
        require_single=require_single,
    )


def _has_any_file(model_dir: Path, candidates: tuple[str, ...]) -> bool:
    return any((model_dir / name).exists() for name in candidates)


def validate_exported_model_dir(
    model_dir: str | Path,
    *,
    expected_speaker: str | None = None,
) -> dict[str, Any]:
    model_path = Path(model_dir)
    if not model_path.exists():
        raise ContractError(f"Exported model directory does not exist: {model_path}")
    if not model_path.is_dir():
        raise ContractError(f"Exported model path is not a directory: {model_path}")

    config = load_model_config(model_path)
    tts_model_type = str(config.get("tts_model_type") or "")
    if tts_model_type != "custom_voice":
        raise ContractError(
            f"Expected exported model to be custom_voice, got tts_model_type={tts_model_type!r}"
        )

    if not _has_any_file(model_path, WEIGHT_FILE_CANDIDATES):
        raise ContractError(
            "Exported model directory is missing saved weights; expected one of "
            f"{WEIGHT_FILE_CANDIDATES}"
        )

    if not _has_any_file(model_path, PROCESSOR_FILE_CANDIDATES):
        raise ContractError(
            "Exported model directory is missing processor/tokenizer files; expected one of "
            f"{PROCESSOR_FILE_CANDIDATES}"
        )

    speaker_map = get_speaker_map(config)
    resolved_speaker = infer_speaker_name_from_config(
        config,
        explicit_speaker=expected_speaker,
        require_single=expected_speaker is None,
    )

    return {
        "model_dir": str(model_path.resolve()),
        "tts_model_type": tts_model_type,
        "speaker_names": list(speaker_map.keys()),
        "speaker_count": len(speaker_map),
        "resolved_speaker": resolved_speaker,
        "has_weight_files": True,
        "has_processor_files": True,
    }
