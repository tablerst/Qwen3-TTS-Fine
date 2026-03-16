from __future__ import annotations

import argparse
import asyncio
import json
import math
from pathlib import Path
from statistics import mean
from types import SimpleNamespace
from typing import Any

from http_stream_probe import probe as http_probe
from ws_stream_probe import stream_probe as ws_probe


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run serial benchmark loops for vLLM-Omni HTTP / WebSocket speech endpoints."
    )
    parser.add_argument("--transport", choices=["http", "ws", "both"], default="both")
    parser.add_argument("--iterations", type=int, default=3, help="Measured serial runs per transport.")
    parser.add_argument("--warmups", type=int, default=1, help="Warmup runs per transport before measurement.")
    parser.add_argument(
        "--output",
        default=None,
        help="Summary JSON path. Defaults to outputs/vllm_omni/serial_benchmark/summary.json.",
    )

    parser.add_argument("--api-base", default="http://localhost:8091")
    parser.add_argument("--ws-url", default="ws://localhost:8091/v1/audio/speech/stream")
    parser.add_argument("--model", default=None)
    parser.add_argument("--text", required=True)
    parser.add_argument("--voice", default="Vivian")
    parser.add_argument("--task-type", default="CustomVoice", choices=["CustomVoice", "VoiceDesign", "Base"])
    parser.add_argument("--language", default="Auto")
    parser.add_argument("--instructions", default=None)
    parser.add_argument("--ref-audio", default=None)
    parser.add_argument("--ref-text", default=None)
    parser.add_argument("--x-vector-only-mode", action="store_true", default=False)
    parser.add_argument(
        "--initial-codec-chunk-frames",
        type=int,
        default=None,
        help="Optional per-request/session initial chunk override for lower TTFA.",
    )

    parser.add_argument("--response-format", default="pcm", choices=["wav", "pcm", "flac", "mp3", "aac", "opus"])
    parser.add_argument(
        "--http-stream",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Whether HTTP benchmark requests should use stream=true.",
    )
    parser.add_argument("--ws-stream-audio", action="store_true", default=False)
    parser.add_argument("--simulate-stt", action="store_true", default=False)
    parser.add_argument("--stt-delay", type=float, default=0.1)
    parser.add_argument("--timeout", type=float, default=300.0)
    parser.add_argument("--trust-env", action="store_true", default=False)
    parser.add_argument("--use-env-proxy", action="store_true", default=False)
    return parser.parse_args()


def _percentile(values: list[float], percentile: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    if len(ordered) == 1:
        return ordered[0]
    position = (len(ordered) - 1) * percentile
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    weight = position - lower
    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight


def _summarize_runs(runs: list[dict[str, Any]], numeric_keys: list[str]) -> dict[str, Any]:
    summary: dict[str, Any] = {"runs": runs}
    aggregates: dict[str, Any] = {}
    for key in numeric_keys:
        values = [float(run[key]) for run in runs if run.get(key) is not None]
        if not values:
            continue
        aggregates[key] = {
            "min": round(min(values), 4),
            "p50": round(_percentile(values, 0.5) or 0.0, 4),
            "p95": round(_percentile(values, 0.95) or 0.0, 4),
            "mean": round(mean(values), 4),
            "max": round(max(values), 4),
        }
    summary["aggregates"] = aggregates
    return summary


def _http_args(base_args: argparse.Namespace, output_root: Path, run_label: str) -> SimpleNamespace:
    return SimpleNamespace(
        api_base=base_args.api_base,
        model=base_args.model,
        text=base_args.text,
        voice=base_args.voice,
        task_type=base_args.task_type,
        language=base_args.language,
        instructions=base_args.instructions,
        ref_audio=base_args.ref_audio,
        ref_text=base_args.ref_text,
        x_vector_only_mode=base_args.x_vector_only_mode,
        initial_codec_chunk_frames=base_args.initial_codec_chunk_frames,
        response_format=base_args.response_format,
        stream=base_args.http_stream,
        output=str(output_root / "http" / f"{run_label}.{base_args.response_format}"),
        timeout=base_args.timeout,
        trust_env=base_args.trust_env,
    )


def _ws_args(base_args: argparse.Namespace, output_root: Path, run_label: str) -> SimpleNamespace:
    return SimpleNamespace(
        url=base_args.ws_url,
        model=base_args.model,
        text=base_args.text,
        voice=base_args.voice,
        task_type=base_args.task_type,
        language=base_args.language,
        instructions=base_args.instructions,
        ref_audio=base_args.ref_audio,
        ref_text=base_args.ref_text,
        x_vector_only_mode=base_args.x_vector_only_mode,
        initial_codec_chunk_frames=base_args.initial_codec_chunk_frames,
        response_format=base_args.response_format,
        stream_audio=base_args.ws_stream_audio,
        simulate_stt=base_args.simulate_stt,
        stt_delay=base_args.stt_delay,
        output_dir=str(output_root / "ws" / run_label),
        use_env_proxy=base_args.use_env_proxy,
    )


def run_http_serial(base_args: argparse.Namespace, output_root: Path) -> dict[str, Any]:
    for warmup_index in range(base_args.warmups):
        http_probe(_http_args(base_args, output_root, f"warmup_{warmup_index + 1:02d}"))

    runs: list[dict[str, Any]] = []
    for run_index in range(base_args.iterations):
        metrics = http_probe(_http_args(base_args, output_root, f"run_{run_index + 1:02d}"))
        metrics["run_index"] = run_index + 1
        runs.append(metrics)

    return _summarize_runs(runs, ["first_chunk_ms", "elapsed_ms", "audio_duration_s", "rtf"])


def run_ws_serial(base_args: argparse.Namespace, output_root: Path) -> dict[str, Any]:
    for warmup_index in range(base_args.warmups):
        asyncio.run(ws_probe(_ws_args(base_args, output_root, f"warmup_{warmup_index + 1:02d}")))

    runs: list[dict[str, Any]] = []
    for run_index in range(base_args.iterations):
        metrics = asyncio.run(ws_probe(_ws_args(base_args, output_root, f"run_{run_index + 1:02d}")))
        metrics["run_index"] = run_index + 1
        runs.append(metrics)

    return _summarize_runs(runs, ["first_audio_ms", "elapsed_ms", "audio_duration_s", "rtf"])


def main() -> None:
    args = parse_args()
    output_path = (
        Path(args.output).expanduser().resolve()
        if args.output
        else Path(__file__).resolve().parents[2] / "outputs" / "vllm_omni" / "serial_benchmark" / "summary.json"
    )
    output_root = output_path.parent
    output_root.mkdir(parents=True, exist_ok=True)

    summary: dict[str, Any] = {
        "transport": args.transport,
        "iterations": args.iterations,
        "warmups": args.warmups,
        "response_format": args.response_format,
        "task_type": args.task_type,
        "text": args.text,
        "model": args.model,
    }

    if args.transport in {"http", "both"}:
        summary["http"] = run_http_serial(args, output_root)
    if args.transport in {"ws", "both"}:
        summary["ws"] = run_ws_serial(args, output_root)

    output_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()