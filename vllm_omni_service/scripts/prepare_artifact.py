from __future__ import annotations

import argparse
import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

CURRENT_DIR = Path(__file__).resolve().parent
SERVICE_ROOT = CURRENT_DIR.parent
REPO_ROOT = SERVICE_ROOT.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Normalize an existing Qwen3-TTS LoRA bundle into a vllm_omni_service experiment artifact."
    )
    parser.add_argument("--bundle-dir", required=True, help="Path to the exported custom voice bundle.")
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Output directory. Defaults to outputs/vllm_omni/<bundle-name>.",
    )
    parser.add_argument(
        "--base-model",
        default=None,
        help="Optional explicit base model ref/path override.",
    )
    parser.add_argument(
        "--bundle-copy-mode",
        choices=["symlink", "copy"],
        default="symlink",
        help="How to materialize the bundle under the output artifact directory.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Overwrite the output directory if it already exists.",
    )
    return parser.parse_args()


def prepare_bundle_materialization(source_bundle_dir: Path, target_bundle_dir: Path, mode: str) -> None:
    if target_bundle_dir.exists() or target_bundle_dir.is_symlink():
        if target_bundle_dir.is_symlink() or target_bundle_dir.is_file():
            target_bundle_dir.unlink()
        else:
            shutil.rmtree(target_bundle_dir)

    if mode == "copy":
        shutil.copytree(source_bundle_dir, target_bundle_dir)
        return

    target_bundle_dir.symlink_to(source_bundle_dir.resolve(), target_is_directory=True)


def infer_speaker_info(config_patch: dict[str, Any], explicit_speaker_name: str | None) -> tuple[str | None, int | None]:
    speaker_map = config_patch.get("talker_config", {}).get("spk_id", {})
    if not speaker_map:
        return explicit_speaker_name, None

    speaker_name = explicit_speaker_name or next(iter(speaker_map.keys()))
    speaker_id = speaker_map.get(speaker_name)
    return speaker_name, int(speaker_id) if speaker_id is not None else None


def main() -> None:
    from lora_finetuning.common import ensure_dir, load_json, save_json
    from streaming_lora_service.app.bundle_loader import resolve_bundle_artifacts

    args = parse_args()
    bundle_dir = Path(args.bundle_dir).expanduser().resolve()
    if not bundle_dir.exists():
        raise FileNotFoundError(f"Bundle directory does not exist: {bundle_dir}")

    output_dir = (
        Path(args.output_dir).expanduser().resolve()
        if args.output_dir
        else (REPO_ROOT / "outputs" / "vllm_omni" / bundle_dir.name).resolve()
    )

    if output_dir.exists() and not args.force:
        raise FileExistsError(f"Output directory already exists: {output_dir}. Use --force to overwrite it.")
    if output_dir.exists() and args.force:
        shutil.rmtree(output_dir)

    ensure_dir(output_dir)
    artifacts = resolve_bundle_artifacts(bundle_dir, base_model=args.base_model)
    config_patch = load_json(artifacts.config_patch_file)
    speaker_name, speaker_id = infer_speaker_info(config_patch, artifacts.speaker_name)
    runtime_tts_model_type = str(config_patch.get("tts_model_type", "unknown"))

    prepared_bundle_dir = output_dir / "bundle"
    prepare_bundle_materialization(bundle_dir, prepared_bundle_dir, args.bundle_copy_mode)

    service_artifact = {
        "format_version": 1,
        "prepared_at": datetime.now(timezone.utc).isoformat(),
        "source_bundle_dir": str(bundle_dir),
        "prepared_bundle_dir": str(prepared_bundle_dir),
        "bundle_copy_mode": args.bundle_copy_mode,
        "base_model_ref": artifacts.manifest.get("base_model_path", artifacts.base_model),
        "resolved_base_model_ref": artifacts.base_model,
        "runtime_tts_model_type": runtime_tts_model_type,
        "speaker_name": speaker_name,
        "speaker_id": speaker_id,
        "adapter_dir": str(artifacts.adapter_dir),
        "config_patch_file": str(artifacts.config_patch_file),
        "speaker_patch_file": str(artifacts.speaker_patch_file),
        "recommended_stage_config": str(SERVICE_ROOT / "configs" / "qwen3_tts.stage.yaml"),
        "recommended_api": {
            "task_type": "CustomVoice" if runtime_tts_model_type == "custom_voice" else None,
            "response_format": "pcm",
            "stream": True,
            "sample_rate": 24000,
            "voice_hint": speaker_name,
        },
        "compatibility_status": {
            "direct_peft_runtime_loading": "unknown",
            "merged_local_model_dir": "not_prepared",
        },
        "next_actions": [
            "Install vllm-omni and sync qwen3_tts stage config into vllm_omni_service/configs/.",
            "Validate whether vllm-omni can load this bundle's PEFT LoRA structure directly for Qwen3-TTS online serving.",
            "If direct loading fails, prepare a merged local model directory and update serve.sh to point --model to it.",
        ],
    }

    save_json(service_artifact, output_dir / "service_artifact.json")

    print(f"Prepared artifact directory: {output_dir}")
    print(f"Source bundle: {bundle_dir}")
    print(f"Resolved base model: {artifacts.base_model}")
    print(f"Runtime TTS model type: {runtime_tts_model_type}")
    print(f"Speaker: {speaker_name!r} (id={speaker_id})")
    print(f"Artifact manifest: {output_dir / 'service_artifact.json'}")


if __name__ == "__main__":
    main()
