from __future__ import annotations

import argparse
import asyncio
from dataclasses import asdict, dataclass
import json
from pathlib import Path
import time
from typing import Any, Sequence

import httpx

from .http_streaming_benchmark import run_single_http_iteration
from .ws_benchmark import (
    BenchmarkCase,
    derive_http_base_url,
    parse_cases,
    preflight_service,
    run_single_iteration as run_single_ws_iteration,
    summarize_numeric,
)


@dataclass(frozen=True)
class LoadTestRequestMetrics:
    request_index: int
    case_id: str
    transport: str
    model: str
    voice: str
    started_offset_ms: float
    ended_offset_ms: float
    ttft_ms: float | None
    total_elapsed_ms: float
    audio_duration_s: float
    rtf: float | None
    audio_chunks: int
    total_audio_bytes: int
    success: bool
    error: str | None = None


def summarize_load_results(results: Sequence[LoadTestRequestMetrics]) -> dict[str, Any]:
    successes = [item for item in results if item.success]
    failures = [item for item in results if not item.success]
    ttft_values = [item.ttft_ms for item in successes if item.ttft_ms is not None]
    elapsed_values = [item.total_elapsed_ms for item in successes]
    rtf_values = [item.rtf for item in successes if item.rtf is not None]
    duration_values = [item.audio_duration_s for item in successes]
    chunk_values = [float(item.audio_chunks) for item in successes]
    bytes_values = [float(item.total_audio_bytes) for item in successes]
    return {
        "request_count": len(results),
        "success_count": len(successes),
        "failure_count": len(failures),
        "ttft_ms": summarize_numeric(ttft_values),
        "total_elapsed_ms": summarize_numeric(elapsed_values),
        "audio_duration_s": summarize_numeric(duration_values),
        "rtf": summarize_numeric(rtf_values),
        "audio_chunks": summarize_numeric(chunk_values),
        "total_audio_bytes": summarize_numeric(bytes_values),
    }


def select_case(cases: Sequence[BenchmarkCase], request_index: int) -> BenchmarkCase:
    if not cases:
        raise ValueError("cases must not be empty")
    return cases[(request_index - 1) % len(cases)]


async def execute_single_request(
    *,
    transport: str,
    request_index: int,
    case: BenchmarkCase,
    wall_start: float,
    model: str,
    voice: str,
    timeout_s: float,
    http_client: httpx.AsyncClient | None,
    http_base_url: str,
    endpoint: str,
    ws_url: str,
    mode: str,
    sample_rate: int,
    response_format: str,
    optimize_instructions: bool,
) -> LoadTestRequestMetrics:
    started_at = time.perf_counter()
    started_offset_ms = round((started_at - wall_start) * 1000.0, 4)
    try:
        if transport == "http-streaming":
            if http_client is None:
                raise ValueError("http_client is required for http-streaming transport")
            metrics = await run_single_http_iteration(
                client=http_client,
                base_url=http_base_url,
                endpoint=endpoint,
                case=case,
                iteration=request_index,
                model=model,
                voice=voice,
                sample_rate=sample_rate,
                optimize_instructions=optimize_instructions,
                timeout_s=timeout_s,
                save_audio_dir=None,
            )
            ended_at = time.perf_counter()
            return LoadTestRequestMetrics(
                request_index=request_index,
                case_id=case.id,
                transport=transport,
                model=model,
                voice=voice,
                started_offset_ms=started_offset_ms,
                ended_offset_ms=round((ended_at - wall_start) * 1000.0, 4),
                ttft_ms=metrics.ttft_ms,
                total_elapsed_ms=metrics.total_elapsed_ms,
                audio_duration_s=metrics.audio_duration_s,
                rtf=metrics.rtf,
                audio_chunks=metrics.audio_chunks,
                total_audio_bytes=metrics.total_audio_bytes,
                success=True,
            )

        metrics = await run_single_ws_iteration(
            ws_url=ws_url,
            case=case,
            iteration=request_index,
            model=model,
            voice=voice,
            mode=mode,
            sample_rate=sample_rate,
            response_format=response_format,
            optimize_instructions=optimize_instructions,
            timeout_s=timeout_s,
            save_audio_dir=None,
        )
        ended_at = time.perf_counter()
        return LoadTestRequestMetrics(
            request_index=request_index,
            case_id=case.id,
            transport=transport,
            model=model,
            voice=voice,
            started_offset_ms=started_offset_ms,
            ended_offset_ms=round((ended_at - wall_start) * 1000.0, 4),
            ttft_ms=metrics.ttft_ms,
            total_elapsed_ms=metrics.total_elapsed_ms,
            audio_duration_s=metrics.audio_duration_s,
            rtf=metrics.rtf,
            audio_chunks=metrics.audio_chunks,
            total_audio_bytes=metrics.total_audio_bytes,
            success=True,
        )
    except Exception as exc:
        ended_at = time.perf_counter()
        return LoadTestRequestMetrics(
            request_index=request_index,
            case_id=case.id,
            transport=transport,
            model=model,
            voice=voice,
            started_offset_ms=started_offset_ms,
            ended_offset_ms=round((ended_at - wall_start) * 1000.0, 4),
            ttft_ms=None,
            total_elapsed_ms=round((ended_at - started_at) * 1000.0, 4),
            audio_duration_s=0.0,
            rtf=None,
            audio_chunks=0,
            total_audio_bytes=0,
            success=False,
            error=str(exc),
        )


async def run_concurrent_benchmark(args: argparse.Namespace) -> dict[str, Any]:
    cases = parse_cases(args)
    http_base_url = args.http_base_url or derive_http_base_url(args.ws_url)
    preflight = None
    selected_voice = args.voice
    if not args.skip_preflight:
        try:
            preflight = await preflight_service(
                http_base_url=http_base_url,
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

    warmup_runs = max(0, args.warmup)
    total_requests = max(1, args.requests)
    concurrency = max(1, args.concurrency)

    async with httpx.AsyncClient(trust_env=False) as client:
        for warmup_index in range(1, warmup_runs + 1):
            case = select_case(cases, warmup_index)
            try:
                await execute_single_request(
                    transport=args.transport,
                    request_index=warmup_index,
                    case=case,
                    wall_start=time.perf_counter(),
                    model=args.model,
                    voice=selected_voice,
                    timeout_s=args.timeout,
                    http_client=client if args.transport == "http-streaming" else None,
                    http_base_url=http_base_url,
                    endpoint=args.endpoint,
                    ws_url=args.ws_url,
                    mode=args.mode,
                    sample_rate=args.sample_rate,
                    response_format=args.response_format,
                    optimize_instructions=args.optimize_instructions,
                )
            except Exception as exc:  # pragma: no cover - warmup best effort
                print(f"[warmup][{args.transport}][{warmup_index}] failed: {exc}", flush=True)

        semaphore = asyncio.Semaphore(concurrency)
        wall_start = time.perf_counter()

        async def worker(request_index: int) -> LoadTestRequestMetrics:
            case = select_case(cases, request_index)
            async with semaphore:
                print(
                    f"[load-test][{args.transport}][{request_index}/{total_requests}] start case={case.id}",
                    flush=True,
                )
                result = await execute_single_request(
                    transport=args.transport,
                    request_index=request_index,
                    case=case,
                    wall_start=wall_start,
                    model=args.model,
                    voice=selected_voice,
                    timeout_s=args.timeout,
                    http_client=client if args.transport == "http-streaming" else None,
                    http_base_url=http_base_url,
                    endpoint=args.endpoint,
                    ws_url=args.ws_url,
                    mode=args.mode,
                    sample_rate=args.sample_rate,
                    response_format=args.response_format,
                    optimize_instructions=args.optimize_instructions,
                )
                print(
                    f"[load-test][{args.transport}][{request_index}] success={result.success} ttft_ms={result.ttft_ms} elapsed_ms={result.total_elapsed_ms}",
                    flush=True,
                )
                return result

        results = await asyncio.gather(*(worker(index) for index in range(1, total_requests + 1)))
        wall_end = time.perf_counter()

    wall_time_s = max(0.0, wall_end - wall_start)
    summary = summarize_load_results(results)
    successes = [item for item in results if item.success]
    total_audio_duration_s = round(sum(item.audio_duration_s for item in successes), 4)
    payload = {
        "config": {
            "transport": args.transport,
            "http_base_url": http_base_url,
            "endpoint": args.endpoint,
            "ws_url": args.ws_url,
            "model": args.model,
            "voice": selected_voice,
            "voice_resolution": preflight["voice_selection"] if preflight is not None else {"voice": selected_voice, "source": "cli"},
            "concurrency": concurrency,
            "requests": total_requests,
            "warmup": warmup_runs,
            "timeout": args.timeout,
            "mode": args.mode,
            "sample_rate": args.sample_rate,
            "response_format": args.response_format,
            "optimize_instructions": args.optimize_instructions,
        },
        "preflight": preflight,
        "summary": {
            **summary,
            "wall_time_s": round(wall_time_s, 4),
            "throughput_rps": round(summary["success_count"] / wall_time_s, 4) if wall_time_s > 0 else None,
            "audio_seconds_per_wall_second": round(total_audio_duration_s / wall_time_s, 4) if wall_time_s > 0 else None,
            "total_audio_duration_s": total_audio_duration_s,
        },
        "requests": [asdict(item) for item in results],
    }
    case_summaries: dict[str, Any] = {}
    for case in cases:
        case_results = [item for item in results if item.case_id == case.id]
        case_summaries[case.id] = {
            "case": asdict(case),
            "summary": summarize_load_results(case_results),
        }
    payload["cases"] = case_summaries
    if args.output_path:
        output_path = Path(args.output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        payload["output_path"] = str(output_path)
    return payload


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run concurrent load tests against streaming_lora_service HTTP streaming or WebSocket endpoints")
    parser.add_argument("--transport", default="http-streaming", choices=("http-streaming", "ws"))
    parser.add_argument("--http_base_url", default=None)
    parser.add_argument("--endpoint", default="/v1/tts")
    parser.add_argument("--ws_url", default="ws://127.0.0.1:9010/api-ws/v1/realtime")
    parser.add_argument("--model", default="qwen3-tts-flash-realtime")
    parser.add_argument("--voice", default=None)
    parser.add_argument("--concurrency", type=int, default=2)
    parser.add_argument("--requests", type=int, default=4)
    parser.add_argument("--warmup", type=int, default=0)
    parser.add_argument("--timeout", type=float, default=600.0)
    parser.add_argument("--skip_preflight", action="store_true")
    parser.add_argument("--mode", default="commit", choices=("commit", "server_commit"))
    parser.add_argument("--sample_rate", type=int, default=24000)
    parser.add_argument("--response_format", default="pcm")
    parser.add_argument("--optimize_instructions", action="store_true")
    parser.add_argument("--text", default=None)
    parser.add_argument("--case_id", default="custom")
    parser.add_argument("--language_type", default="Chinese")
    parser.add_argument("--instructions", default="正式，平静，清晰。")
    parser.add_argument("--cases_json", default=None)
    parser.add_argument("--output_path", default=None)
    return parser


def print_human_summary(payload: dict[str, Any]) -> None:
    config = payload["config"]
    summary = payload["summary"]
    print("=" * 72)
    print(
        f"Concurrent load test finished for transport={config['transport']} "
        f"concurrency={config['concurrency']} requests={config['requests']}"
    )
    print(
        f"success={summary['success_count']} failure={summary['failure_count']} | "
        f"throughput={summary['throughput_rps']} req/s | wall_time={summary['wall_time_s']} s"
    )
    print(
        f"ttft p50={summary['ttft_ms']['p50']} ms, p95={summary['ttft_ms']['p95']} ms | "
        f"rtf mean={summary['rtf']['mean']}"
    )
    if payload.get("output_path"):
        print(f"output_path: {payload['output_path']}")
    print("=" * 72)


def main() -> None:
    import asyncio

    args = build_parser().parse_args()
    payload = asyncio.run(run_concurrent_benchmark(args))
    print_human_summary(payload)


if __name__ == "__main__":
    main()