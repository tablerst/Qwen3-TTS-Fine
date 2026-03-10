from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

CURRENT_DIR = Path(__file__).resolve().parent
REPO_ROOT = CURRENT_DIR.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import soundfile as sf
import torch

from lora_finetuning.common import (apply_config_patch, apply_speaker_patch,
                                    ensure_dir, load_json,
                                    load_lora_adapter, parse_torch_dtype,
                                    save_json)


LANGUAGE_TAGS = {
    "Japanese": "ja",
    "Chinese": "zh",
    "English": "en",
}


def chunked(items: list[dict[str, Any]], batch_size: int) -> list[list[dict[str, Any]]]:
    if batch_size <= 0:
        raise ValueError(f"batch_size must be > 0, got {batch_size}")
    return [items[index:index + batch_size] for index in range(0, len(items), batch_size)]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate multilingual Yachiyo preset samples with a LoRA bundle")
    parser.add_argument("--bundle_dir", required=True, help="Bundle dir created by export_custom_voice.py")
    parser.add_argument(
        "--preset_file",
        default=str(REPO_ROOT / "resources" / "yachiyo_presets" / "presets.json"),
        help="Preset definition JSON file",
    )
    parser.add_argument(
        "--output_dir",
        default=str(REPO_ROOT / "outputs" / "yachiyo_multilingual_samples"),
        help="Directory to store generated wav files and reports",
    )
    parser.add_argument("--base_model", type=str, default=None, help="Optional override for base model path")
    parser.add_argument("--speaker_name", type=str, default=None, help="Optional override speaker name")
    parser.add_argument("--device_map", default="cuda:0")
    parser.add_argument("--torch_dtype", default="bfloat16")
    parser.add_argument("--attn_implementation", default="sdpa")
    parser.add_argument(
        "--batch_size",
        type=int,
        default=3,
        help="Number of samples to generate per model call; on a single GPU this is usually faster than naive parallelism.",
    )
    parser.add_argument("--local_files_only", action="store_true")
    return parser.parse_args()


def resolve_bundle(bundle_dir: Path, base_model_override: str | None) -> tuple[str, Path, Path, Path, str | None]:
    manifest = load_json(bundle_dir / "manifest.json")
    base_model = base_model_override or manifest["base_model_path"]
    adapter_dir = bundle_dir / manifest.get("adapter_dir", "adapter")
    config_patch_file = bundle_dir / manifest.get("config_patch_file", "config_patch.json")
    speaker_patch_file = bundle_dir / manifest.get("speaker_embedding_file", "speaker_embedding.safetensors")
    speaker_name = manifest.get("speaker_name")
    return base_model, adapter_dir, config_patch_file, speaker_patch_file, speaker_name


def load_qwen_model(
    base_model: str,
    adapter_dir: Path,
    config_patch_file: Path,
    speaker_patch_file: Path,
    device_map: str,
    torch_dtype: str,
    attn_implementation: str,
    local_files_only: bool,
):
    from qwen_tts import Qwen3TTSModel

    dtype = parse_torch_dtype(torch_dtype)
    qwen3tts = Qwen3TTSModel.from_pretrained(
        base_model,
        device_map=device_map,
        dtype=dtype,
        attn_implementation=attn_implementation,
        local_files_only=local_files_only,
    )
    load_lora_adapter(qwen3tts.model, adapter_dir)
    apply_config_patch(qwen3tts.model, load_json(config_patch_file))
    apply_speaker_patch(qwen3tts.model, speaker_patch_file)
    qwen3tts.model.eval()
    return qwen3tts


def load_presets(preset_file: Path) -> list[dict[str, Any]]:
    presets = load_json(preset_file)
    if not isinstance(presets, list) or not presets:
        raise ValueError(f"Preset file must contain a non-empty list: {preset_file}")
    return presets


def build_markdown(report: dict[str, Any]) -> str:
    lines = [
        "# Yachiyo multilingual sample report",
        "",
        f"- speaker: `{report['speaker_name']}`",
        f"- bundle_dir: `{report['bundle_dir']}`",
        f"- base_model: `{report['base_model']}`",
        f"- total_samples: `{report['total_samples']}`",
        "",
    ]

    for preset in report["presets"]:
        lines.append(f"## {preset['speaker_id']} · {preset['style']}")
        lines.append("")
        lines.append(f"- description: {preset['description']}")
        lines.append(f"- instruct: {preset['style_instruct']}")
        lines.append("")
        for sample in preset["samples"]:
            lines.append(
                f"- {sample['language']} ({sample['language_tag']}): `{sample['output_wav']}` · "
                f"duration `{sample['duration_seconds']:.2f}s`"
            )
        lines.append("")

    return "\n".join(lines)


def build_generation_jobs(
    presets: list[dict[str, Any]],
    output_dir: Path,
    speaker_name: str,
) -> tuple[list[dict[str, Any]], dict[str, dict[str, Any]]]:
    jobs: list[dict[str, Any]] = []
    preset_reports: dict[str, dict[str, Any]] = {}

    for preset in presets:
        preset_id = preset["speaker_id"]
        preset_output_dir = ensure_dir(output_dir / preset_id)
        prompt_texts = preset.get("prompt_texts") or {"Japanese": preset.get("prompt_text", "")}
        style_instruct = preset.get("style_instruct") or preset.get("description") or ""

        preset_reports[preset_id] = {
            "speaker_id": preset_id,
            "style": preset.get("style", preset_id),
            "description": preset.get("description", ""),
            "style_instruct": style_instruct,
            "samples": [],
        }

        for language, text in prompt_texts.items():
            if not text:
                continue
            language_tag = LANGUAGE_TAGS.get(language, language.lower())
            jobs.append(
                {
                    "preset_id": preset_id,
                    "speaker_name": speaker_name,
                    "language": language,
                    "language_tag": language_tag,
                    "text": text,
                    "instruct": style_instruct or None,
                    "output_path": preset_output_dir / f"{language_tag}.wav",
                }
            )

    return jobs, preset_reports


def main() -> None:
    args = parse_args()
    bundle_dir = Path(args.bundle_dir)
    preset_file = Path(args.preset_file)
    output_dir = ensure_dir(args.output_dir)

    base_model, adapter_dir, config_patch_file, speaker_patch_file, bundle_speaker_name = resolve_bundle(
        bundle_dir,
        args.base_model,
    )
    speaker_name = args.speaker_name or bundle_speaker_name
    if speaker_name is None:
        config_patch = load_json(config_patch_file)
        speaker_name = next(iter(config_patch["talker_config"]["spk_id"].keys()))

    presets = load_presets(preset_file)
    qwen3tts = load_qwen_model(
        base_model=base_model,
        adapter_dir=adapter_dir,
        config_patch_file=config_patch_file,
        speaker_patch_file=speaker_patch_file,
        device_map=args.device_map,
        torch_dtype=args.torch_dtype,
        attn_implementation=args.attn_implementation,
        local_files_only=args.local_files_only,
    )

    report: dict[str, Any] = {
        "bundle_dir": str(bundle_dir),
        "base_model": base_model,
        "speaker_name": speaker_name,
        "preset_file": str(preset_file),
        "batch_size": args.batch_size,
        "total_samples": 0,
        "presets": [],
    }

    jobs, preset_reports = build_generation_jobs(presets, output_dir, speaker_name)

    for batch_jobs in chunked(jobs, args.batch_size):
        wavs, sample_rate = qwen3tts.generate_custom_voice(
            text=[job["text"] for job in batch_jobs],
            language=[job["language"] for job in batch_jobs],
            speaker=[job["speaker_name"] for job in batch_jobs],
            instruct=[job["instruct"] for job in batch_jobs],
        )

        for job, wav in zip(batch_jobs, wavs):
            output_path = job["output_path"]
            sf.write(str(output_path), wav, sample_rate)

            duration_seconds = float(len(wav) / sample_rate)
            sample_info = {
                "language": job["language"],
                "language_tag": job["language_tag"],
                "text": job["text"],
                "output_wav": str(output_path),
                "sample_rate": sample_rate,
                "duration_seconds": duration_seconds,
            }
            preset_reports[job["preset_id"]]["samples"].append(sample_info)
            report["total_samples"] += 1

            print(
                f"Generated {job['preset_id']} [{job['language']}] -> {output_path} "
                f"({duration_seconds:.2f}s, sr={sample_rate})"
            )

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    report["presets"] = [preset_reports[preset["speaker_id"]] for preset in presets]

    save_json(report, output_dir / "report.json")
    (output_dir / "report.md").write_text(build_markdown(report), encoding="utf-8")
    print(f"Done. Generated {report['total_samples']} samples in {output_dir}")


if __name__ == "__main__":
    main()