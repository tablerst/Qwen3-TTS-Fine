from __future__ import annotations

import argparse
import base64
from dataclasses import asdict, dataclass
import json
from pathlib import Path
import time
from typing import Any, Sequence

import httpx

from .app.audio_utils import pcm16le_bytes_to_wav_bytes
from .ws_benchmark import (
    DEFAULT_CHANNELS,
    DEFAULT_SAMPLE_RATE,
    BenchmarkCase,
    BenchmarkError,
    BenchmarkFailure,
    audio_duration_seconds,
    compute_rtf,
    parse_cases,
    preflight_service,
    summarize_numeric,
)


@dataclass(frozen=True)
class HttpBenchmarkRunMetrics:
    case_id: str
    iteration: int
    model: str
    voice: str
    endpoint: str
    sample_rate: int
    response_headers_ms: float
    ttft_ms: float | None
    final_chunk_ms: float | None
    total_elapsed_ms: float
    total_audio_bytes: int
    audio_duration_s: float
    rtf: float | None
    audio_chunks: int
    request_id: str | None
    final_finish_reason: str | None
    audio_id: str | None
    audio_url: str | None
    usage_characters: int | None


def extract_stream_payload_fields(payload: dict[str, Any]) -> dict[str, Any]:
    output = payload.get("output") if isinstance(payload, dict) else None
    if not isinstance(output, dict):
        raise BenchmarkError(f"Invalid streaming payload without output object: {payload!r}")
    audio = output.get("audio")
    if not isinstance(audio, dict):
        raise BenchmarkError(f"Invalid streaming payload without audio object: {payload!r}")
    usage = payload.get("usage")
    usage_characters = None
    if isinstance(usage, dict) and usage.get("characters") is not None:
        usage_characters = int(usage["characters"])
    return {
        "request_id": payload.get("request_id"),
        "finish_reason": output.get("finish_reason"),
        "audio_data": str(audio.get("data") or ""),
        "audio_id": audio.get("id"),
        "audio_url": audio.get("url"),
        "usage_characters": usage_characters,
        "status_code": payload.get("status_code"),
    }


async def run_single_http_iteration(
    *,
    client: httpx.AsyncClient,
    base_url: str,
    endpoint: str,
    case: BenchmarkCase,
    iteration: int,
    model: str,
    voice: str,
    sample_rate: int,
    optimize_instructions: bool,
    timeout_s: float,
    save_audio_dir: Path | None,
) -> HttpBenchmarkRunMetrics:
    request_payload = {
        "model": model,
        "text": case.text,
        "voice": voice,
        "language_type": case.language_type,
        "instructions": case.instructions,
        "optimize_instructions": optimize_instructions,
        "stream": True,
    }
    request_url = f"{base_url.rstrip('/')}{endpoint}"
    request_started = time.perf_counter()
    timeout = httpx.Timeout(timeout_s)

    first_audio_at: float | None = None
    final_chunk_at: float | None = None
    request_id: str | None = None
    final_finish_reason: str | None = None
    audio_id: str | None = None
    audio_url: str | None = None
    usage_characters: int | None = None
    audio_chunks = 0
    audio_bytes = bytearray()

    async with client.stream("POST", request_url, json=request_payload, timeout=timeout) as response:
        response.raise_for_status()
        headers_received_at = time.perf_counter()

        async for line in response.aiter_lines():
            if not line:
                continue
            event_at = time.perf_counter()
            payload = json.loads(line)
            fields = extract_stream_payload_fields(payload)
            request_id = str(fields["request_id"]) if fields["request_id"] is not None else request_id
            if fields["audio_data"]:
                if first_audio_at is None:
                    first_audio_at = event_at
                audio_bytes.extend(base64.b64decode(fields["audio_data"]))
                audio_chunks += 1
                continue
            final_chunk_at = event_at
            final_finish_reason = (
                str(fields["finish_reason"]) if fields["finish_reason"] is not None else final_finish_reason
            )
            audio_id = str(fields["audio_id"]) if fields["audio_id"] else audio_id
            audio_url = str(fields["audio_url"]) if fields["audio_url"] else audio_url
            usage_characters = fields["usage_characters"] if fields["usage_characters"] is not None else usage_characters

    finished_at = final_chunk_at or time.perf_counter()
    total_audio_bytes = len(audio_bytes)
    audio_duration_s = audio_duration_seconds(total_audio_bytes, sample_rate=sample_rate, channels=DEFAULT_CHANNELS)
    total_elapsed_ms = round((finished_at - request_started) * 1000.0, 4)

    if save_audio_dir is not None and total_audio_bytes > 0:
        save_audio_dir.mkdir(parents=True, exist_ok=True)
        wav_path = save_audio_dir / f"{case.id}_iter{iteration:02d}.wav"
        wav_path.write_bytes(
            pcm16le_bytes_to_wav_bytes(bytes(audio_bytes), sample_rate=sample_rate, channels=DEFAULT_CHANNELS)
        )

    return HttpBenchmarkRunMetrics(
        case_id=case.id,
        iteration=iteration,
        model=model,
        voice=voice,
        endpoint=endpoint,
        sample_rate=sample_rate,
        response_headers_ms=round((headers_received_at - request_started) * 1000.0, 4),
        ttft_ms=round((first_audio_at - request_started) * 1000.0, 4) if first_audio_at is not None else None,
        final_chunk_ms=round((final_chunk_at - request_started) * 1000.0, 4) if final_chunk_at is not None else None,
        total_elapsed_ms=total_elapsed_ms,
        total_audio_bytes=total_audio_bytes,
        audio_duration_s=round(audio_duration_s, 4),
        rtf=round(compute_rtf(total_elapsed_ms, audio_duration_s) or 0.0, 6) if audio_duration_s > 0 else None,
        audio_chunks=audio_chunks,
        request_id=request_id,
        final_finish_reason=final_finish_reason,
        audio_id=audio_id,
        audio_url=audio_url,
        usage_characters=usage_characters,
    )


def summarize_http_case_runs(runs: Sequence[HttpBenchmarkRunMetrics]) -> dict[str, Any]:
    headers_values = [item.response_headers_ms for item in runs]
    ttft_values = [item.ttft_ms for item in runs if item.ttft_ms is not None]
    final_chunk_values = [item.final_chunk_ms for item in runs if item.final_chunk_ms is not None]
    elapsed_values = [item.total_elapsed_ms for item in runs]
    rtf_values = [item.rtf for item in runs if item.rtf is not None]
    audio_duration_values = [item.audio_duration_s for item in runs]
    chunk_values = [float(item.audio_chunks) for item in runs]
    bytes_values = [float(item.total_audio_bytes) for item in runs]
    return {
        "run_count": len(runs),
        "response_headers_ms": summarize_numeric(headers_values),
        "ttft_ms": summarize_numeric(ttft_values),
        "final_chunk_ms": summarize_numeric(final_chunk_values),
        "total_elapsed_ms": summarize_numeric(elapsed_values),
        "audio_duration_s": summarize_numeric(audio_duration_values),
        "rtf": summarize_numeric(rtf_values),
        "audio_chunks": summarize_numeric(chunk_values),
        "total_audio_bytes": summarize_numeric(bytes_values),
    }


async def run_http_benchmark(args: argparse.Namespace) -> dict[str, Any]:
    cases = parse_cases(args)
    preflight = None
    selected_voice = args.voice
    if not args.skip_preflight:
        try:
            preflight = await preflight_service(
                http_base_url=args.http_base_url,
                model=args.model,
                requested_voice=args.voice,
                timeout_s=args.timeout,
            )
            selected_voice = preflight["voice_selection"]["voice"]
        except Exception as exc:
            if not args.voice:
                raise
            preflight = {
                "error": str(exc),
                "voice_selection": {"voice": args.voice, "source": "cli_after_preflight_failure"},
            }
            selected_voice = args.voice
            print(f"[preflight] failed, falling back to --voice={args.voice}: {exc}", flush=True)
    if not selected_voice:
        raise ValueError("voice is required when --skip-preflight is used")

    measured_iterations = max(1, args.iterations)
    warmup_runs = max(0, args.warmup)
    save_audio_dir = Path(args.save_audio_dir) if args.save_audio_dir else None
    successes: list[HttpBenchmarkRunMetrics] = []
    failures: list[BenchmarkFailure] = []

    async with httpx.AsyncClient(trust_env=False) as client:
        for case in cases:
            for warmup_index in range(1, warmup_runs + 1):
                try:
                    await run_single_http_iteration(
                        client=client,
                        base_url=args.http_base_url,
                        endpoint=args.endpoint,
                        case=case,
                        iteration=warmup_index,
                        model=args.model,
                        voice=selected_voice,
                        sample_rate=args.sample_rate,
                        optimize_instructions=args.optimize_instructions,
                        timeout_s=args.timeout,
                        save_audio_dir=None,
                    )
                except Exception as exc:  # pragma: no cover - warmup best effort
                    print(f"[warmup][{case.id}][{warmup_index}] failed: {exc}", flush=True)

            for iteration in range(1, measured_iterations + 1):
                print(
                    f"[http-benchmark][{case.id}][{iteration}/{measured_iterations}] posting to {args.http_base_url}{args.endpoint}",
                    flush=True,
                )
                try:
                    metrics = await run_single_http_iteration(
                        client=client,
                        base_url=args.http_base_url,
                        endpoint=args.endpoint,
                        case=case,
                        iteration=iteration,
                        model=args.model,
                        voice=selected_voice,
                        sample_rate=args.sample_rate,
                        optimize_instructions=args.optimize_instructions,
                        timeout_s=args.timeout,
                        save_audio_dir=save_audio_dir,
                    )
                except Exception as exc:
                    failure = BenchmarkFailure(case_id=case.id, iteration=iteration, error=str(exc))
                    failures.append(failure)
                    print(f"[http-benchmark][{case.id}][{iteration}] failed: {failure.error}", flush=True)
                    continue
                successes.append(metrics)
                print(
                    f"[http-benchmark][{case.id}][{iteration}] ttft_ms={metrics.ttft_ms} total_elapsed_ms={metrics.total_elapsed_ms} "
                    f"audio_duration_s={metrics.audio_duration_s} rtf={metrics.rtf}",
                    flush=True,
                )

    case_summaries: dict[str, Any] = {}
    for case in cases:
        case_runs = [item for item in successes if item.case_id == case.id]
        case_summaries[case.id] = {
            "case": asdict(case),
            "summary": summarize_http_case_runs(case_runs),
            "runs": [asdict(item) for item in case_runs],
            "failure_count": sum(1 for item in failures if item.case_id == case.id),
        }

    payload = {
        "config": {
            "http_base_url": args.http_base_url,
            "endpoint": args.endpoint,
            "model": args.model,
            "voice": selected_voice,
            "voice_resolution": preflight["voice_selection"] if preflight is not None else {"voice": selected_voice, "source": "cli"},
            "sample_rate": args.sample_rate,
            "iterations": measured_iterations,
            "warmup": warmup_runs,
            "timeout": args.timeout,
            "optimize_instructions": args.optimize_instructions,
        },
        "preflight": preflight,
        "summary": {
            "case_count": len(cases),
            "success_count": len(successes),
            "failure_count": len(failures),
            "overall": summarize_http_case_runs(successes),
        },
        "cases": case_summaries,
        "failures": [asdict(item) for item in failures],
    }
    if args.output_path:
        output_path = Path(args.output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        payload["output_path"] = str(output_path)
    return payload


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Benchmark the streaming_lora_service HTTP streaming endpoint and report TTFT/RTF metrics")
    parser.add_argument("--http_base_url", default="http://127.0.0.1:9010")
    parser.add_argument("--endpoint", default="/v1/tts")
    parser.add_argument("--model", default="qwen3-tts-flash-realtime")
    parser.add_argument("--voice", default=None)
    parser.add_argument("--sample_rate", type=int, default=DEFAULT_SAMPLE_RATE)
    parser.add_argument("--iterations", type=int, default=3)
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--timeout", type=float, default=600.0)
    parser.add_argument("--skip_preflight", action="store_true")
    parser.add_argument("--optimize_instructions", action="store_true")
    parser.add_argument("--text", default=None)
    parser.add_argument("--case_id", default="custom")
    parser.add_argument("--language_type", default="Chinese")
    parser.add_argument("--instructions", default="正式，平静，清晰。")
    parser.add_argument("--cases_json", default=None)
    parser.add_argument("--output_path", default=None)
    parser.add_argument("--save_audio_dir", default=None)
    return parser


def print_human_summary(payload: dict[str, Any]) -> None:
    config = payload["config"]
    overall = payload["summary"]["overall"]
    ttft = overall["ttft_ms"]
    rtf = overall["rtf"]
    elapsed = overall["total_elapsed_ms"]
    print("=" * 72)
    print(f"HTTP streaming benchmark finished for {config['http_base_url']}{config['endpoint']}")
    print(f"model={config['model']} voice={config['voice']} iterations={config['iterations']} warmup={config['warmup']}")
    print(
        "overall: "
        f"ttft p50={ttft['p50']} ms, p95={ttft['p95']} ms | "
        f"rtf mean={rtf['mean']} | total_elapsed p50={elapsed['p50']} ms"
    )
    print(f"success={payload['summary']['success_count']} failure={payload['summary']['failure_count']}")
    if payload.get("output_path"):
        print(f"output_path: {payload['output_path']}")
    print("=" * 72)


def main() -> None:
    import asyncio

    args = build_parser().parse_args()
    payload = asyncio.run(run_http_benchmark(args))
    print_human_summary(payload)


if __name__ == "__main__":
    main()