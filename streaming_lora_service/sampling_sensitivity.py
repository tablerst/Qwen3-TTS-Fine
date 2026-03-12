from __future__ import annotations

import argparse
from dataclasses import asdict
import json
from pathlib import Path

import torch

from .app.server import RealtimeServerConfig, BundleSpeechService, build_dependencies
from .app.step_generator import generate_custom_voice_step_aware
from .app.streaming_generator import StreamingCustomVoiceGenerator
from .quality_regression import DEFAULT_VALIDATION_CASES, compare_codec_sequences, normalize_codec_tokens


DEFAULT_VARIANTS: tuple[tuple[str, dict[str, object]], ...] = (
    ("baseline_sample", {}),
    ("sample_no_rep_penalty", {"repetition_penalty": 1.0}),
    (
        "greedy_no_rep_penalty",
        {
            "do_sample": False,
            "repetition_penalty": 1.0,
            "top_k": 0,
            "top_p": 1.0,
            "temperature": 1.0,
            "subtalker_dosample": False,
            "subtalker_top_k": 0,
            "subtalker_top_p": 1.0,
            "subtalker_temperature": 1.0,
        },
    ),
    (
        "greedy_with_rep_penalty",
        {
            "do_sample": False,
            "repetition_penalty": 1.05,
            "top_k": 0,
            "top_p": 1.0,
            "temperature": 1.0,
            "subtalker_dosample": False,
            "subtalker_top_k": 0,
            "subtalker_top_p": 1.0,
            "subtalker_temperature": 1.0,
        },
    ),
)


def reseed(seed: int) -> None:
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run sampling sensitivity diagnostics for streaming_lora_service")
    parser.add_argument("--bundle_dir", required=True)
    parser.add_argument("--voice_registry_file", required=True)
    parser.add_argument("--default_voice_alias", default="default")
    parser.add_argument("--public_model_alias", default="qwen3-tts-flash-realtime")
    parser.add_argument("--case_id", default="zh_formal")
    parser.add_argument("--output_path", required=True)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--variant_names", nargs="*", default=None)
    parser.add_argument("--local_files_only", action="store_true")
    parser.add_argument("--device_map", default="cuda:0")
    parser.add_argument("--torch_dtype", default="bfloat16")
    parser.add_argument("--attn_implementation", default="sdpa")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    config = RealtimeServerConfig(
        bundle_dir=Path(args.bundle_dir),
        public_model_alias=args.public_model_alias,
        default_voice_alias=args.default_voice_alias,
        voice_registry_file=Path(args.voice_registry_file),
        local_files_only=args.local_files_only,
        device_map=args.device_map,
        torch_dtype=args.torch_dtype,
        attn_implementation=args.attn_implementation,
    )
    service = BundleSpeechService(build_dependencies(config))
    case = next(item for item in DEFAULT_VALIDATION_CASES if item.id == args.case_id)
    profile = service.deps.voice_registry.resolve(service.default_voice_alias, model=service.public_model_alias)

    report: dict[str, object] = {
        "bundle_dir": str(config.bundle_dir),
        "case": {
            "id": case.id,
            "text": case.text,
            "language_type": case.language_type,
            "instructions": case.instructions,
            "voice": service.default_voice_alias,
            "speaker_name": profile.speaker_name,
            "seed": args.seed,
        },
        "variants": [],
    }

    selected_variant_names = set(args.variant_names) if args.variant_names else None
    variants = [
        (name, variant_kwargs)
        for name, variant_kwargs in DEFAULT_VARIANTS
        if selected_variant_names is None or name in selected_variant_names
    ]
    if not variants:
        raise ValueError(f"No variants selected. Available variants: {[name for name, _ in DEFAULT_VARIANTS]}")

    print(
        f"running sampling sensitivity for case={case.id}, voice={service.default_voice_alias}, "
        f"variants={[name for name, _ in variants]}",
        flush=True,
    )

    for index, (name, variant_kwargs) in enumerate(variants, start=1):
        kwargs = dict(variant_kwargs)
        print(f"[{index}/{len(variants)}] starting variant={name} kwargs={kwargs}", flush=True)

        reseed(args.seed)
        synthesized = generate_custom_voice_step_aware(
            service.deps.loaded_bundle.qwen3tts,
            text=case.text,
            speaker=profile.speaker_name,
            language=case.language_type,
            instruct=case.instructions or None,
            **kwargs,
        )

        reseed(args.seed)
        generator = StreamingCustomVoiceGenerator(
            service.deps.loaded_bundle.qwen3tts,
            text=case.text,
            speaker=profile.speaker_name,
            language=case.language_type,
            instruct=case.instructions or None,
            chunk_steps=service.deps.chunk_steps,
            left_context_steps=service.deps.left_context_steps,
            **kwargs,
        )
        for _ in generator.iter_audio_chunks():
            pass

        comparison = compare_codec_sequences(
            "http_non_streaming",
            synthesized.codec_tokens,
            "streaming_sampler_full_decode",
            normalize_codec_tokens(generator.state.generated_codes),
        )
        report["variants"].append(
            {
                "name": name,
                "kwargs": kwargs,
                "http_non_streaming_codec_steps": synthesized.codec_steps,
                "streaming_sampler_codec_steps": len(generator.state.generated_codes),
                "finish_reason": generator.metrics.finish_reason,
                "comparison": asdict(comparison) if comparison is not None else None,
            }
        )
        comparison_summary = report["variants"][-1]["comparison"]
        print(f"[{index}/{len(variants)}] finished variant={name} comparison={comparison_summary}", flush=True)

    output_path = Path(args.output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    print(f"saved: {output_path}")


if __name__ == "__main__":
    main()
