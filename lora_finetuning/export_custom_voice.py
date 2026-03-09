from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

CURRENT_DIR = Path(__file__).resolve().parent
REPO_ROOT = CURRENT_DIR.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from lora_finetuning.common import build_bundle_manifest, ensure_dir, load_json, save_json


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Bundle Qwen3-TTS LoRA artifacts into a reusable custom voice package")
    parser.add_argument("--base_model", required=True, help="Base model repo id or local path used for training")
    parser.add_argument("--source_dir", required=True, help="Training artifact dir, e.g. outputs/lora_single_speaker")
    parser.add_argument("--output_dir", required=True, help="Export bundle directory")
    parser.add_argument("--speaker_name", default=None, help="Override speaker name stored in config patch")
    return parser.parse_args()


def copy_if_exists(src: Path, dst: Path) -> None:
    if src.is_dir():
        shutil.copytree(src, dst, dirs_exist_ok=True)
    elif src.is_file():
        ensure_dir(dst.parent)
        shutil.copy2(src, dst)


def main() -> None:
    args = parse_args()
    source_dir = Path(args.source_dir)
    output_dir = ensure_dir(args.output_dir)

    adapter_dir = source_dir / "adapter"
    config_patch_file = source_dir / "config_patch.json"
    speaker_patch_file = source_dir / "speaker_embedding.safetensors"

    if not adapter_dir.exists():
        raise FileNotFoundError(f"Adapter dir not found: {adapter_dir}")
    if not config_patch_file.exists():
        raise FileNotFoundError(f"Config patch not found: {config_patch_file}")
    if not speaker_patch_file.exists():
        raise FileNotFoundError(f"Speaker patch not found: {speaker_patch_file}")

    config_patch = load_json(config_patch_file)
    speaker_name = args.speaker_name or next(iter(config_patch["talker_config"]["spk_id"].keys()))
    speaker_id = int(config_patch["talker_config"]["spk_id"][speaker_name])

    copy_if_exists(adapter_dir, output_dir / "adapter")
    copy_if_exists(config_patch_file, output_dir / "config_patch.json")
    copy_if_exists(speaker_patch_file, output_dir / "speaker_embedding.safetensors")
    copy_if_exists(source_dir / "train_args.json", output_dir / "train_args.json")
    copy_if_exists(source_dir / "metrics.json", output_dir / "metrics.json")

    manifest = build_bundle_manifest(
        base_model_path=args.base_model,
        speaker_name=speaker_name,
        speaker_id=speaker_id,
        adapter_subdir="adapter",
    )
    save_json(manifest, output_dir / "manifest.json")

    print(f"Export completed: {output_dir}")
    print(f"Speaker: {speaker_name} (id={speaker_id})")
    print(f"Base model: {args.base_model}")


if __name__ == "__main__":
    main()
