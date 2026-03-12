from __future__ import annotations

import argparse
import base64
from dataclasses import asdict, dataclass
import json
from pathlib import Path
import time
from typing import Any, Iterable

import torch

from .app.audio_utils import float_audio_to_pcm16le_bytes, pcm16le_bytes_to_wav_bytes
from .app.qwen_compat_ws import QwenRealtimeProtocolAdapter
from .app.server import BundleSpeechService, RealtimeServerConfig, build_dependencies


@dataclass(frozen=True)
class ValidationCase:
    id: str
    text: str
    language_type: str = "Auto"
    instructions: str = ""
    voice: str | None = None


@dataclass
class PathMetrics:
    path: str
    sample_rate: int
    channels: int
    total_audio_bytes: int
    duration_s: float
    elapsed_ms: float | None = None
    ttfb_ms: float | None = None
    delta_chunks: int | None = None
    generated_steps: int | None = None
    emitted_chunks: int | None = None
    first_emitted_step: int | None = None
    finish_reason: str | None = None
    codec_steps: int | None = None
    output_wav_path: str | None = None


@dataclass
class CollectedPathResult:
    metrics: PathMetrics
    audio_bytes: bytes


@dataclass
class ValidationCaseResult:
    case: ValidationCase
    offline_non_streaming: PathMetrics
    http_non_streaming: PathMetrics
    http_streaming_runtime: PathMetrics
    websocket_realtime: PathMetrics
    warnings: list[str]


DEFAULT_VALIDATION_CASES: tuple[ValidationCase, ...] = (
    ValidationCase(
        id="zh_formal",
        language_type="Chinese",
        instructions="正式，平静，清晰。",
        text="你好，欢迎使用 Qwen3-TTS 实时语音服务。这是一段用于端到端验证的正式女声测试。",
    ),
    ValidationCase(
        id="ja_formal",
        language_type="Japanese",
        instructions="落ち着いて、丁寧に。",
        text="こんにちは。本日はリアルタイム音声サービスのエンドツーエンド検証を行っています。",
    ),
)


def audio_duration_seconds(audio_bytes: bytes, *, sample_rate: int, channels: int = 1) -> float:
    if not audio_bytes or sample_rate <= 0 or channels <= 0:
        return 0.0
    return len(audio_bytes) / float(sample_rate * channels * 2)


def build_case_warnings(
    result: ValidationCaseResult,
    *,
    max_stream_to_offline_ratio: float = 1.5,
) -> list[str]:
    warnings: list[str] = []
    offline_duration = result.offline_non_streaming.duration_s

    for path_metrics in (result.http_streaming_runtime, result.websocket_realtime):
        if path_metrics.total_audio_bytes <= 0:
            warnings.append(f"{path_metrics.path}: no audio bytes were produced")
        if path_metrics.finish_reason not in (None, "eos"):
            warnings.append(
                f"{path_metrics.path}: finish_reason={path_metrics.finish_reason!r}, expected 'eos' for short text"
            )
        if offline_duration > 0:
            ratio = path_metrics.duration_s / offline_duration
            if ratio > max_stream_to_offline_ratio:
                warnings.append(
                    f"{path_metrics.path}: duration ratio vs offline is {ratio:.2f} (> {max_stream_to_offline_ratio:.2f})"
                )
        if result.http_non_streaming.codec_steps and path_metrics.generated_steps:
            step_ratio = path_metrics.generated_steps / result.http_non_streaming.codec_steps
            if step_ratio > max_stream_to_offline_ratio:
                warnings.append(
                    f"{path_metrics.path}: generated_steps ratio vs http_non_streaming.codec_steps is "
                    f"{step_ratio:.2f} (> {max_stream_to_offline_ratio:.2f})"
                )

    if result.http_streaming_runtime.total_audio_bytes != result.websocket_realtime.total_audio_bytes:
        warnings.append(
            "http_streaming_runtime and websocket_realtime produced different audio sizes: "
            f"{result.http_streaming_runtime.total_audio_bytes} vs {result.websocket_realtime.total_audio_bytes}"
        )

    return warnings


def build_summary(results: Iterable[ValidationCaseResult]) -> dict[str, Any]:
    result_list = list(results)
    warning_count = sum(len(item.warnings) for item in result_list)
    warning_case_count = sum(1 for item in result_list if item.warnings)
    return {
        "case_count": len(result_list),
        "warning_count": warning_count,
        "warning_case_count": warning_case_count,
        "case_ids_with_warnings": [item.case.id for item in result_list if item.warnings],
    }


def load_cases(cases_json: Path | None) -> list[ValidationCase]:
    if cases_json is None:
        return list(DEFAULT_VALIDATION_CASES)
    payload = json.loads(cases_json.read_text(encoding="utf-8"))
    if not isinstance(payload, list):
        raise ValueError("cases_json must contain a JSON array")
    return [ValidationCase(**item) for item in payload]


def collect_case_result(
    service: BundleSpeechService,
    case: ValidationCase,
    *,
    max_stream_to_offline_ratio: float,
    seed: int | None,
) -> ValidationCaseResult:
    _set_torch_seed(seed)
    offline = collect_offline_non_streaming(service, case)
    _set_torch_seed(seed)
    http_non_stream = collect_http_non_streaming(service, case)
    _set_torch_seed(seed)
    http_stream = collect_http_streaming_runtime(service, case)
    _set_torch_seed(seed)
    websocket = collect_websocket_realtime(service, case)
    result = ValidationCaseResult(
        case=case,
        offline_non_streaming=offline.metrics,
        http_non_streaming=http_non_stream.metrics,
        http_streaming_runtime=http_stream.metrics,
        websocket_realtime=websocket.metrics,
        warnings=[],
    )
    result.warnings = build_case_warnings(result, max_stream_to_offline_ratio=max_stream_to_offline_ratio)
    result._artifacts = {
        "offline_non_streaming": offline,
        "http_non_streaming": http_non_stream,
        "http_streaming_runtime": http_stream,
        "websocket_realtime": websocket,
    }
    return result


def run_validation_suite(
    *,
    config: RealtimeServerConfig,
    cases: Iterable[ValidationCase],
    output_dir: Path | None = None,
    max_stream_to_offline_ratio: float = 1.5,
    seed: int | None = 1234,
) -> dict[str, Any]:
    service = BundleSpeechService(build_dependencies(config))
    results: list[ValidationCaseResult] = []
    target_dir = output_dir
    if target_dir is not None:
        target_dir.mkdir(parents=True, exist_ok=True)

    for case in cases:
        result = collect_case_result(
            service,
            case,
            max_stream_to_offline_ratio=max_stream_to_offline_ratio,
            seed=seed,
        )
        if target_dir is not None:
            _write_case_artifacts(target_dir, result)
        results.append(result)

    summary = build_summary(results)
    payload = {
        "config": {
            "bundle_dir": str(config.bundle_dir),
            "public_model_alias": config.public_model_alias,
            "default_voice_alias": config.default_voice_alias,
            "voice_registry_file": str(config.voice_registry_file) if config.voice_registry_file else None,
            "chunk_steps": config.chunk_steps,
            "left_context_steps": config.left_context_steps,
            "samples_per_step": config.samples_per_step,
            "max_stream_to_offline_ratio": max_stream_to_offline_ratio,
            "seed": seed,
        },
        "summary": summary,
        "cases": [
            {
                **asdict(_strip_case_result(result)),
            }
            for result in results
        ],
    }
    if target_dir is not None:
        metrics_path = target_dir / "metrics.json"
        metrics_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        payload["metrics_path"] = str(metrics_path)
    return payload


def collect_offline_non_streaming(service: BundleSpeechService, case: ValidationCase) -> CollectedPathResult:
    voice_alias = case.voice or service.default_voice_alias
    profile = service.deps.voice_registry.resolve(voice_alias, model=service.public_model_alias)
    started = time.perf_counter()
    wavs, sample_rate = service.deps.loaded_bundle.qwen3tts.generate_custom_voice(
        text=case.text,
        speaker=profile.speaker_name,
        language=case.language_type,
        instruct=case.instructions or None,
        non_streaming_mode=True,
    )
    elapsed_ms = (time.perf_counter() - started) * 1000.0
    audio_bytes = float_audio_to_pcm16le_bytes(wavs[0])
    return CollectedPathResult(
        metrics=PathMetrics(
            path="offline_non_streaming",
            sample_rate=int(sample_rate),
            channels=1,
            total_audio_bytes=len(audio_bytes),
            duration_s=audio_duration_seconds(audio_bytes, sample_rate=int(sample_rate), channels=1),
            elapsed_ms=round(elapsed_ms, 2),
        ),
        audio_bytes=audio_bytes,
    )


def collect_http_non_streaming(service: BundleSpeechService, case: ValidationCase) -> CollectedPathResult:
    voice_alias = case.voice or service.default_voice_alias
    started = time.perf_counter()
    synthesized = service.synthesize_http(
        text=case.text,
        model=service.public_model_alias,
        voice=voice_alias,
        language_type=case.language_type,
        instructions=case.instructions,
    )
    elapsed_ms = (time.perf_counter() - started) * 1000.0
    return CollectedPathResult(
        metrics=PathMetrics(
            path="http_non_streaming",
            sample_rate=int(synthesized.sample_rate),
            channels=int(synthesized.channels),
            total_audio_bytes=len(synthesized.audio_bytes),
            duration_s=audio_duration_seconds(
                synthesized.audio_bytes,
                sample_rate=int(synthesized.sample_rate),
                channels=int(synthesized.channels),
            ),
            elapsed_ms=round(elapsed_ms, 2),
            codec_steps=synthesized.codec_steps,
        ),
        audio_bytes=synthesized.audio_bytes,
    )


def collect_http_streaming_runtime(service: BundleSpeechService, case: ValidationCase) -> CollectedPathResult:
    voice_alias = case.voice or service.default_voice_alias
    session = service.build_runtime_session(
        model=service.public_model_alias,
        voice=voice_alias,
        language_type=case.language_type,
        instructions=case.instructions,
        session_id=f"validation_http_{case.id}",
    )
    started = time.perf_counter()
    first_chunk_at: float | None = None
    chunks: list[bytes] = []
    for chunk in service.stream_synthesize(session, case.text):
        if first_chunk_at is None:
            first_chunk_at = time.perf_counter()
        chunks.append(chunk)
    elapsed_ms = (time.perf_counter() - started) * 1000.0
    audio_bytes = b"".join(chunks)
    sample_rate = int(service.deps.loaded_bundle.qwen3tts.model.speech_tokenizer.get_output_sample_rate())
    metrics = dict(session.state.last_generation_metrics)
    return CollectedPathResult(
        metrics=PathMetrics(
            path="http_streaming_runtime",
            sample_rate=sample_rate,
            channels=1,
            total_audio_bytes=len(audio_bytes),
            duration_s=audio_duration_seconds(audio_bytes, sample_rate=sample_rate, channels=1),
            elapsed_ms=round(elapsed_ms, 2),
            ttfb_ms=round((first_chunk_at - started) * 1000.0, 2) if first_chunk_at is not None else None,
            delta_chunks=len(chunks),
            generated_steps=metrics.get("generated_steps"),
            emitted_chunks=metrics.get("emitted_chunks"),
            first_emitted_step=metrics.get("first_emitted_step"),
            finish_reason=metrics.get("finish_reason"),
        ),
        audio_bytes=audio_bytes,
    )


def collect_websocket_realtime(service: BundleSpeechService, case: ValidationCase) -> CollectedPathResult:
    voice_alias = case.voice or service.default_voice_alias
    adapter: QwenRealtimeProtocolAdapter = service.create_protocol_adapter()
    adapter.open_connection(service.build_initial_session_options())
    update_events = adapter.handle_event(
        {
            "type": "session.update",
            "session": {
                "model": service.public_model_alias,
                "voice": voice_alias,
                "language_type": case.language_type,
                "mode": "commit",
                "response_format": "pcm",
                "sample_rate": 24000,
                "instructions": case.instructions,
                "optimize_instructions": False,
            },
        }
    )
    if any(event.get("type") == "error" for event in update_events):
        raise RuntimeError(f"session.update failed for case {case.id}: {update_events}")

    adapter.handle_event({"type": "input_text_buffer.append", "text": case.text})

    started = time.perf_counter()
    first_delta_at: float | None = None
    chunks: list[bytes] = []
    for event in adapter.iter_events({"type": "input_text_buffer.commit"}):
        if event.get("type") != "response.audio.delta":
            continue
        if first_delta_at is None:
            first_delta_at = time.perf_counter()
        chunks.append(base64.b64decode(event["delta"]))

    elapsed_ms = (time.perf_counter() - started) * 1000.0
    audio_bytes = b"".join(chunks)
    metrics = dict(adapter.current_session.state.last_generation_metrics) if adapter.current_session is not None else {}
    return CollectedPathResult(
        metrics=PathMetrics(
            path="websocket_realtime",
            sample_rate=24000,
            channels=1,
            total_audio_bytes=len(audio_bytes),
            duration_s=audio_duration_seconds(audio_bytes, sample_rate=24000, channels=1),
            elapsed_ms=round(elapsed_ms, 2),
            ttfb_ms=round((first_delta_at - started) * 1000.0, 2) if first_delta_at is not None else None,
            delta_chunks=len(chunks),
            generated_steps=metrics.get("generated_steps"),
            emitted_chunks=metrics.get("emitted_chunks"),
            first_emitted_step=metrics.get("first_emitted_step"),
            finish_reason=metrics.get("finish_reason"),
        ),
        audio_bytes=audio_bytes,
    )


def _strip_case_result(result: ValidationCaseResult) -> ValidationCaseResult:
    if hasattr(result, "_artifacts"):
        delattr(result, "_artifacts")
    return result


def _write_case_artifacts(output_dir: Path, result: ValidationCaseResult) -> None:
    artifacts = getattr(result, "_artifacts", {})
    case_dir = output_dir / result.case.id
    case_dir.mkdir(parents=True, exist_ok=True)
    for key, collected in artifacts.items():
        wav_path = case_dir / f"{key}.wav"
        wav_path.write_bytes(
            pcm16le_bytes_to_wav_bytes(
                collected.audio_bytes,
                sample_rate=collected.metrics.sample_rate,
                channels=collected.metrics.channels,
            )
        )
        getattr(result, key).output_wav_path = str(wav_path)


def _set_torch_seed(seed: int | None) -> None:
    if seed is None:
        return
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Compare offline, HTTP, and WebSocket streaming paths for a LoRA bundle")
    parser.add_argument("--bundle_dir", required=True)
    parser.add_argument("--output_dir", default=None)
    parser.add_argument("--voice_registry_file", default=None)
    parser.add_argument("--public_model_alias", default="qwen3-tts-flash-realtime")
    parser.add_argument("--default_voice_alias", default="default")
    parser.add_argument("--chunk_steps", type=int, default=4)
    parser.add_argument("--left_context_steps", type=int, default=25)
    parser.add_argument("--samples_per_step", type=int, default=1920)
    parser.add_argument("--device_map", default="cuda:0")
    parser.add_argument("--torch_dtype", default="bfloat16")
    parser.add_argument("--attn_implementation", default="sdpa")
    parser.add_argument("--local_files_only", action="store_true")
    parser.add_argument("--cases_json", default=None)
    parser.add_argument("--max_stream_to_offline_ratio", type=float, default=1.5)
    parser.add_argument("--seed", type=int, default=1234)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    config = RealtimeServerConfig(
        bundle_dir=Path(args.bundle_dir),
        public_model_alias=args.public_model_alias,
        default_voice_alias=args.default_voice_alias,
        voice_registry_file=Path(args.voice_registry_file) if args.voice_registry_file else None,
        chunk_steps=args.chunk_steps,
        left_context_steps=args.left_context_steps,
        samples_per_step=args.samples_per_step,
        device_map=args.device_map,
        torch_dtype=args.torch_dtype,
        attn_implementation=args.attn_implementation,
        local_files_only=args.local_files_only,
    )
    output_dir = Path(args.output_dir) if args.output_dir else None
    cases = load_cases(Path(args.cases_json) if args.cases_json else None)
    payload = run_validation_suite(
        config=config,
        cases=cases,
        output_dir=output_dir,
        max_stream_to_offline_ratio=args.max_stream_to_offline_ratio,
        seed=args.seed,
    )
    print(json.dumps(payload["summary"], ensure_ascii=False, indent=2))
    if payload.get("metrics_path"):
        print(f"metrics_path: {payload['metrics_path']}")


if __name__ == "__main__":
    main()