from __future__ import annotations

from pathlib import Path, PureWindowsPath
from typing import Any

from lora_finetuning.common import (
    apply_config_patch,
    apply_speaker_patch,
    load_json,
    load_lora_adapter,
    parse_torch_dtype,
)

from .models import BundleArtifacts, LoadedBundle


class BundleLoaderError(RuntimeError):
    """Raised when a LoRA bundle cannot be resolved or loaded."""


def _candidate_workspace_roots(bundle_path: Path) -> list[Path]:
    return [bundle_path, *bundle_path.parents]


def _resolve_local_model_path(base_model_ref: str, bundle_path: Path) -> str:
    base_model_path = Path(base_model_ref)
    if base_model_path.exists():
        return str(base_model_path)

    windows_path = PureWindowsPath(base_model_ref)
    suffix_candidates: list[Path] = []
    parts = list(windows_path.parts)
    if "models" in parts:
        models_index = parts.index("models")
        suffix_candidates.append(Path(*parts[models_index:]))
    if windows_path.name:
        suffix_candidates.append(Path("models") / windows_path.name)

    if not base_model_path.is_absolute():
        suffix_candidates.append(base_model_path)

    for workspace_root in _candidate_workspace_roots(bundle_path.resolve()):
        for suffix in suffix_candidates:
            candidate = workspace_root / suffix
            if candidate.exists():
                return str(candidate)

    return base_model_ref


def resolve_bundle_artifacts(
    bundle_dir: str | Path,
    *,
    base_model: str | None = None,
    adapter_dir: str | Path | None = None,
    config_patch_file: str | Path | None = None,
    speaker_patch_file: str | Path | None = None,
    speaker_name: str | None = None,
) -> BundleArtifacts:
    bundle_path = Path(bundle_dir)
    manifest_path = bundle_path / "manifest.json"
    manifest = load_json(manifest_path)

    resolved_base_model = _resolve_local_model_path(
        base_model or str(manifest["base_model_path"]),
        bundle_path,
    )
    resolved_adapter_dir = Path(adapter_dir) if adapter_dir else bundle_path / manifest.get("adapter_dir", "adapter")
    resolved_config_patch = (
        Path(config_patch_file)
        if config_patch_file
        else bundle_path / manifest.get("config_patch_file", "config_patch.json")
    )
    resolved_speaker_patch = (
        Path(speaker_patch_file)
        if speaker_patch_file
        else bundle_path / manifest.get("speaker_embedding_file", "speaker_embedding.safetensors")
    )
    resolved_speaker_name = speaker_name or manifest.get("speaker_name")

    return BundleArtifacts(
        bundle_dir=bundle_path,
        base_model=resolved_base_model,
        adapter_dir=resolved_adapter_dir,
        config_patch_file=resolved_config_patch,
        speaker_patch_file=resolved_speaker_patch,
        speaker_name=resolved_speaker_name,
        manifest=manifest,
    )


def infer_speaker_name(config_patch: dict[str, Any], explicit_speaker_name: str | None = None) -> str:
    if explicit_speaker_name:
        return explicit_speaker_name
    speaker_map = config_patch.get("talker_config", {}).get("spk_id", {})
    if not speaker_map:
        raise BundleLoaderError("Config patch does not define any speakers")
    if len(speaker_map) > 1:
        raise BundleLoaderError(
            "Config patch defines multiple speakers; an explicit speaker_name is required for V1"
        )
    return str(next(iter(speaker_map.keys())))


class BundleLoader:
    def __init__(self, model_factory: Any | None = None) -> None:
        self._model_factory = model_factory

    @property
    def model_factory(self) -> Any:
        if self._model_factory is None:
            from qwen_tts import Qwen3TTSModel  # lazy import to keep tests lightweight

            self._model_factory = Qwen3TTSModel
        return self._model_factory

    def load(
        self,
        bundle_dir: str | Path,
        *,
        base_model: str | None = None,
        adapter_dir: str | Path | None = None,
        config_patch_file: str | Path | None = None,
        speaker_patch_file: str | Path | None = None,
        speaker_name: str | None = None,
        device_map: str = "cuda:0",
        torch_dtype: str = "bfloat16",
        attn_implementation: str = "sdpa",
        local_files_only: bool = False,
    ) -> LoadedBundle:
        artifacts = resolve_bundle_artifacts(
            bundle_dir,
            base_model=base_model,
            adapter_dir=adapter_dir,
            config_patch_file=config_patch_file,
            speaker_patch_file=speaker_patch_file,
            speaker_name=speaker_name,
        )
        dtype = parse_torch_dtype(torch_dtype)
        qwen3tts = self.model_factory.from_pretrained(
            artifacts.base_model,
            device_map=device_map,
            dtype=dtype,
            attn_implementation=attn_implementation,
            local_files_only=local_files_only,
        )

        load_lora_adapter(qwen3tts.model, artifacts.adapter_dir)
        config_patch = load_json(artifacts.config_patch_file)
        apply_config_patch(qwen3tts.model, config_patch)
        applied_speaker_id = apply_speaker_patch(qwen3tts.model, artifacts.speaker_patch_file)
        qwen3tts.model.eval()

        resolved_speaker_name = infer_speaker_name(config_patch, artifacts.speaker_name)
        tts_model_type = str(getattr(qwen3tts.model, "tts_model_type", ""))
        if tts_model_type != "custom_voice":
            raise BundleLoaderError(
                f"Loaded bundle must resolve to a custom_voice model, got {tts_model_type!r}"
            )

        supported_speakers = getattr(qwen3tts.model, "supported_speakers", None)
        if supported_speakers is not None and resolved_speaker_name not in supported_speakers:
            raise BundleLoaderError(
                f"Resolved speaker {resolved_speaker_name!r} was not injected into supported_speakers"
            )

        return LoadedBundle(
            artifacts=artifacts,
            qwen3tts=qwen3tts,
            speaker_name=resolved_speaker_name,
            speaker_id=applied_speaker_id,
            config_patch=config_patch,
            tts_model_type=tts_model_type,
        )
