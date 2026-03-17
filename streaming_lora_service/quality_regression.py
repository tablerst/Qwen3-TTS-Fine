from __future__ import annotations

import argparse
import base64
from dataclasses import asdict, dataclass, field
import json
from pathlib import Path
import time
from typing import Any, Iterable, Sequence

import numpy as np
import torch

from .app.audio_utils import float_audio_to_pcm16le_bytes, pcm16le_bytes_to_float_audio, pcm16le_bytes_to_wav_bytes
from .app.qwen_compat_ws import QwenRealtimeProtocolAdapter
from .app.server import BundleSpeechService, RealtimeServerConfig, build_dependencies
from .app.streaming_generator import StreamingCustomVoiceGenerator


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
    runtime_metrics: dict[str, Any] | None = None


@dataclass
class CollectedPathResult:
    metrics: PathMetrics
    audio_bytes: bytes
    codec_tokens: tuple[tuple[int, ...], ...] | None = None
    chunk_boundaries_samples: tuple[int, ...] = ()


@dataclass(frozen=True)
class CodecComparison:
    reference_path: str
    candidate_path: str
    reference_codec_steps: int
    candidate_codec_steps: int
    shared_prefix_steps: int
    first_divergence_step: int | None
    reference_step_tokens: list[int] | None
    candidate_step_tokens: list[int] | None
    identical: bool


@dataclass(frozen=True)
class BoundaryWindowComparison:
    boundary_sample: int
    start_sample: int
    end_sample: int
    mean_abs_diff: float
    rms_diff: float
    max_abs_diff: float


@dataclass(frozen=True)
class WaveformComparison:
    reference_path: str
    candidate_path: str
    compared_samples: int
    reference_total_samples: int
    candidate_total_samples: int
    length_delta_samples: int
    mean_abs_diff: float
    rms_diff: float
    max_abs_diff: float
    pearson_corr: float
    boundary_windows: list[BoundaryWindowComparison] = field(default_factory=list)


@dataclass
class ValidationCaseResult:
    case: ValidationCase
    offline_non_streaming: PathMetrics
    http_non_streaming: PathMetrics
    http_streaming_runtime: PathMetrics
    websocket_realtime: PathMetrics
    warnings: list[str]
    streaming_sampler_full_decode: PathMetrics | None = None
    codec_diagnostics: dict[str, CodecComparison] = field(default_factory=dict)
    waveform_diagnostics: dict[str, WaveformComparison] = field(default_factory=dict)
    _artifacts: dict[str, CollectedPathResult] = field(default_factory=dict, repr=False)


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
    sampler_full_decode = collect_streaming_sampler_full_decode(service, case)
    _set_torch_seed(seed)
    http_stream = collect_http_streaming_runtime(service, case)
    _set_torch_seed(seed)
    websocket = collect_websocket_realtime(service, case)
    result = ValidationCaseResult(
        case=case,
        offline_non_streaming=offline.metrics,
        http_non_streaming=http_non_stream.metrics,
        streaming_sampler_full_decode=sampler_full_decode.metrics,
        http_streaming_runtime=http_stream.metrics,
        websocket_realtime=websocket.metrics,
        warnings=[],
    )
    result.warnings = build_case_warnings(result, max_stream_to_offline_ratio=max_stream_to_offline_ratio)
    result._artifacts = {
        "offline_non_streaming": offline,
        "http_non_streaming": http_non_stream,
        "streaming_sampler_full_decode": sampler_full_decode,
        "http_streaming_runtime": http_stream,
        "websocket_realtime": websocket,
    }
    result.codec_diagnostics = build_codec_diagnostics(result._artifacts)
    result.waveform_diagnostics = build_waveform_diagnostics(
        result._artifacts,
        boundary_window_samples=max(1, service.deps.samples_per_step),
    )
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
            "first_chunk_steps": config.first_chunk_steps,
            "crossfade_samples": config.crossfade_samples,
            "samples_per_step": config.samples_per_step,
            "max_stream_to_offline_ratio": max_stream_to_offline_ratio,
            "seed": seed,
        },
        "summary": summary,
        "cases": [
            _serialize_case_result(result)
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
        codec_tokens=synthesized.codec_tokens,
    )


def collect_streaming_sampler_full_decode(service: BundleSpeechService, case: ValidationCase) -> CollectedPathResult:
    voice_alias = case.voice or service.default_voice_alias
    profile = service.deps.voice_registry.resolve(voice_alias, model=service.public_model_alias)
    session = service.build_runtime_session(
        model=service.public_model_alias,
        voice=voice_alias,
        language_type=case.language_type,
        instructions=case.instructions,
        session_id=f"validation_sampler_full_{case.id}",
    )
    generator = StreamingCustomVoiceGenerator(
        service.deps.loaded_bundle.qwen3tts,
        text=case.text,
        language=case.language_type,
        speaker=profile.speaker_name,
        instruct=case.instructions or None,
        runtime_session=session,
        chunk_steps=service.deps.chunk_steps,
        left_context_steps=service.deps.left_context_steps,
        first_chunk_steps=service.deps.first_chunk_steps,
    )

    started = time.perf_counter()
    first_chunk_at: float | None = None
    streamed_chunks: list[bytes] = []
    for chunk in generator.iter_audio_chunks():
        if first_chunk_at is None and chunk:
            first_chunk_at = time.perf_counter()
        streamed_chunks.append(chunk)
    elapsed_ms = (time.perf_counter() - started) * 1000.0

    sample_rate = int(service.deps.loaded_bundle.qwen3tts.model.speech_tokenizer.get_output_sample_rate())
    generated_codes = [code.detach() for code in generator.state.generated_codes]
    codec_tokens = normalize_codec_tokens(generated_codes)
    if generated_codes:
        wavs, decoded_sample_rate = service.deps.loaded_bundle.qwen3tts.model.speech_tokenizer.decode(
            [{"audio_codes": torch.stack(generated_codes, dim=0)}]
        )
        audio_bytes = float_audio_to_pcm16le_bytes(wavs[0])
        sample_rate = int(decoded_sample_rate)
    else:
        audio_bytes = b""

    return CollectedPathResult(
        metrics=PathMetrics(
            path="streaming_sampler_full_decode",
            sample_rate=sample_rate,
            channels=1,
            total_audio_bytes=len(audio_bytes),
            duration_s=audio_duration_seconds(audio_bytes, sample_rate=sample_rate, channels=1),
            elapsed_ms=round(elapsed_ms, 2),
            ttfb_ms=round((first_chunk_at - started) * 1000.0, 2) if first_chunk_at is not None else None,
            delta_chunks=generator.metrics.emitted_chunks,
            generated_steps=generator.metrics.generated_steps,
            emitted_chunks=generator.metrics.emitted_chunks,
            first_emitted_step=generator.metrics.first_emitted_step,
            finish_reason=generator.metrics.finish_reason,
            codec_steps=len(generated_codes),
            runtime_metrics=dict(session.state.last_generation_metrics),
        ),
        audio_bytes=audio_bytes,
        codec_tokens=codec_tokens,
        chunk_boundaries_samples=build_chunk_boundaries(streamed_chunks, channels=1),
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
    codec_tokens = normalize_codec_tokens(session.state.generated_codes)
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
            runtime_metrics=metrics,
        ),
        audio_bytes=audio_bytes,
        codec_tokens=codec_tokens,
        chunk_boundaries_samples=build_chunk_boundaries(chunks, channels=1),
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
    codec_tokens = (
        normalize_codec_tokens(adapter.current_session.state.generated_codes)
        if adapter.current_session is not None
        else None
    )
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
            runtime_metrics=metrics,
        ),
        audio_bytes=audio_bytes,
        codec_tokens=codec_tokens,
        chunk_boundaries_samples=build_chunk_boundaries(chunks, channels=1),
    )


def _serialize_case_result(result: ValidationCaseResult) -> dict[str, Any]:
    return {
        "case": asdict(result.case),
        "offline_non_streaming": asdict(result.offline_non_streaming),
        "http_non_streaming": asdict(result.http_non_streaming),
        "streaming_sampler_full_decode": (
            asdict(result.streaming_sampler_full_decode)
            if result.streaming_sampler_full_decode is not None
            else None
        ),
        "http_streaming_runtime": asdict(result.http_streaming_runtime),
        "websocket_realtime": asdict(result.websocket_realtime),
        "warnings": list(result.warnings),
        "codec_diagnostics": {
            key: asdict(value)
            for key, value in result.codec_diagnostics.items()
        },
        "waveform_diagnostics": {
            key: asdict(value)
            for key, value in result.waveform_diagnostics.items()
        },
    }


def build_codec_diagnostics(artifacts: dict[str, CollectedPathResult]) -> dict[str, CodecComparison]:
    diagnostics: dict[str, CodecComparison] = {}
    pair_specs = (
        ("http_non_vs_streaming_sampler_full_decode", "http_non_streaming", "streaming_sampler_full_decode"),
        ("streaming_sampler_full_decode_vs_http_streaming_runtime", "streaming_sampler_full_decode", "http_streaming_runtime"),
        ("http_streaming_runtime_vs_websocket_realtime", "http_streaming_runtime", "websocket_realtime"),
    )
    for key, reference_name, candidate_name in pair_specs:
        reference = artifacts.get(reference_name)
        candidate = artifacts.get(candidate_name)
        if reference is None or candidate is None:
            continue
        comparison = compare_codec_sequences(
            reference_name,
            reference.codec_tokens,
            candidate_name,
            candidate.codec_tokens,
        )
        if comparison is not None:
            diagnostics[key] = comparison
    return diagnostics


def build_waveform_diagnostics(
    artifacts: dict[str, CollectedPathResult],
    *,
    boundary_window_samples: int,
) -> dict[str, WaveformComparison]:
    diagnostics: dict[str, WaveformComparison] = {}
    pair_specs = (
        ("http_non_vs_streaming_sampler_full_decode", "http_non_streaming", "streaming_sampler_full_decode"),
        ("streaming_sampler_full_decode_vs_http_streaming_runtime", "streaming_sampler_full_decode", "http_streaming_runtime"),
        ("http_streaming_runtime_vs_websocket_realtime", "http_streaming_runtime", "websocket_realtime"),
    )
    for key, reference_name, candidate_name in pair_specs:
        reference = artifacts.get(reference_name)
        candidate = artifacts.get(candidate_name)
        if reference is None or candidate is None:
            continue
        diagnostics[key] = compare_waveforms(
            reference_name,
            reference.audio_bytes,
            candidate_name,
            candidate.audio_bytes,
            channels=max(reference.metrics.channels, candidate.metrics.channels),
            boundary_samples=candidate.chunk_boundaries_samples,
            boundary_window_samples=boundary_window_samples,
        )
    return diagnostics


def compare_codec_sequences(
    reference_path: str,
    reference_tokens: Sequence[Sequence[int]] | None,
    candidate_path: str,
    candidate_tokens: Sequence[Sequence[int]] | None,
) -> CodecComparison | None:
    if reference_tokens is None or candidate_tokens is None:
        return None

    shared_prefix_steps = 0
    for reference_step, candidate_step in zip(reference_tokens, candidate_tokens):
        if tuple(reference_step) != tuple(candidate_step):
            break
        shared_prefix_steps += 1

    identical = (
        len(reference_tokens) == len(candidate_tokens)
        and shared_prefix_steps == len(reference_tokens)
    )
    first_divergence_step = None if identical else shared_prefix_steps

    reference_step_tokens = None
    candidate_step_tokens = None
    if first_divergence_step is not None:
        if first_divergence_step < len(reference_tokens):
            reference_step_tokens = [int(item) for item in reference_tokens[first_divergence_step]]
        if first_divergence_step < len(candidate_tokens):
            candidate_step_tokens = [int(item) for item in candidate_tokens[first_divergence_step]]

    return CodecComparison(
        reference_path=reference_path,
        candidate_path=candidate_path,
        reference_codec_steps=len(reference_tokens),
        candidate_codec_steps=len(candidate_tokens),
        shared_prefix_steps=shared_prefix_steps,
        first_divergence_step=first_divergence_step,
        reference_step_tokens=reference_step_tokens,
        candidate_step_tokens=candidate_step_tokens,
        identical=identical,
    )


def compare_waveforms(
    reference_path: str,
    reference_audio_bytes: bytes,
    candidate_path: str,
    candidate_audio_bytes: bytes,
    *,
    channels: int = 1,
    boundary_samples: Sequence[int] = (),
    boundary_window_samples: int = 0,
) -> WaveformComparison:
    reference_wave = pcm16le_bytes_to_float_audio(reference_audio_bytes, channels=channels).reshape(-1)
    candidate_wave = pcm16le_bytes_to_float_audio(candidate_audio_bytes, channels=channels).reshape(-1)
    compared_samples = min(len(reference_wave), len(candidate_wave))

    if compared_samples == 0:
        return WaveformComparison(
            reference_path=reference_path,
            candidate_path=candidate_path,
            compared_samples=0,
            reference_total_samples=len(reference_wave),
            candidate_total_samples=len(candidate_wave),
            length_delta_samples=len(candidate_wave) - len(reference_wave),
            mean_abs_diff=0.0,
            rms_diff=0.0,
            max_abs_diff=0.0,
            pearson_corr=1.0,
            boundary_windows=[],
        )

    reference_aligned = reference_wave[:compared_samples]
    candidate_aligned = candidate_wave[:compared_samples]
    diff = candidate_aligned - reference_aligned
    abs_diff = np.abs(diff)

    if np.allclose(reference_aligned, candidate_aligned):
        pearson_corr = 1.0
    else:
        ref_std = float(np.std(reference_aligned))
        cand_std = float(np.std(candidate_aligned))
        if ref_std == 0.0 or cand_std == 0.0:
            pearson_corr = 0.0
        else:
            pearson_corr = float(np.corrcoef(reference_aligned, candidate_aligned)[0, 1])

    boundary_windows: list[BoundaryWindowComparison] = []
    if boundary_window_samples > 0:
        for boundary_sample in boundary_samples:
            if boundary_sample <= 0 or boundary_sample >= compared_samples:
                continue
            start = max(0, boundary_sample - boundary_window_samples)
            end = min(compared_samples, boundary_sample + boundary_window_samples)
            if end <= start:
                continue
            window = diff[start:end]
            window_abs = np.abs(window)
            boundary_windows.append(
                BoundaryWindowComparison(
                    boundary_sample=int(boundary_sample),
                    start_sample=int(start),
                    end_sample=int(end),
                    mean_abs_diff=float(window_abs.mean()),
                    rms_diff=float(np.sqrt(np.mean(np.square(window)))),
                    max_abs_diff=float(window_abs.max()),
                )
            )

    return WaveformComparison(
        reference_path=reference_path,
        candidate_path=candidate_path,
        compared_samples=compared_samples,
        reference_total_samples=len(reference_wave),
        candidate_total_samples=len(candidate_wave),
        length_delta_samples=len(candidate_wave) - len(reference_wave),
        mean_abs_diff=float(abs_diff.mean()),
        rms_diff=float(np.sqrt(np.mean(np.square(diff)))),
        max_abs_diff=float(abs_diff.max()),
        pearson_corr=pearson_corr,
        boundary_windows=boundary_windows,
    )


def build_chunk_boundaries(chunks: Sequence[bytes], *, channels: int = 1) -> tuple[int, ...]:
    if channels <= 0:
        raise ValueError("channels must be > 0")
    bytes_per_sample = channels * 2
    total_samples = 0
    boundaries: list[int] = []
    for chunk in chunks[:-1]:
        if not chunk:
            continue
        total_samples += len(chunk) // bytes_per_sample
        boundaries.append(total_samples)
    return tuple(boundaries)


def normalize_codec_tokens(codec_steps: Sequence[torch.Tensor]) -> tuple[tuple[int, ...], ...]:
    return tuple(
        tuple(int(item) for item in step.detach().cpu().tolist())
        for step in codec_steps
    )


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
    parser.add_argument("--first_chunk_steps", type=int, default=None)
    parser.add_argument("--crossfade_samples", type=int, default=0)
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
        backend="native",
        public_model_alias=args.public_model_alias,
        default_voice_alias=args.default_voice_alias,
        voice_registry_file=Path(args.voice_registry_file) if args.voice_registry_file else None,
        chunk_steps=args.chunk_steps,
        left_context_steps=args.left_context_steps,
        first_chunk_steps=args.first_chunk_steps,
        crossfade_samples=args.crossfade_samples,
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