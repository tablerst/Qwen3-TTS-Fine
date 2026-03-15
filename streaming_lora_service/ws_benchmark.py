from __future__ import annotations

import argparse
import asyncio
import base64
from dataclasses import asdict, dataclass
import json
from pathlib import Path
import statistics
import time
from typing import Any, Iterable, Sequence
from urllib.parse import urlparse, urlunparse

import httpx
import websockets

from .app.audio_utils import pcm16le_bytes_to_wav_bytes


DEFAULT_SAMPLE_RATE = 24000
DEFAULT_CHANNELS = 1


@dataclass(frozen=True)
class BenchmarkCase:
    id: str
    text: str
    language_type: str = "Auto"
    instructions: str = ""


@dataclass(frozen=True)
class BenchmarkRunMetrics:
    case_id: str
    iteration: int
    model: str
    voice: str
    mode: str
    sample_rate: int
    connect_ms: float
    session_created_ms: float
    session_update_ms: float
    commit_ack_ms: float | None
    response_created_ms: float | None
    ttft_ms: float | None
    audio_done_ms: float | None
    total_elapsed_ms: float
    total_audio_bytes: int
    audio_duration_s: float
    rtf: float | None
    audio_chunks: int
    usage_characters: int | None


@dataclass(frozen=True)
class BenchmarkFailure:
    case_id: str
    iteration: int
    error: str


@dataclass(frozen=True)
class VoiceSelection:
    voice: str
    source: str


DEFAULT_CASES: tuple[BenchmarkCase, ...] = (
    BenchmarkCase(
        id="zh_formal",
        language_type="Chinese",
        instructions="正式，平静，清晰。",
        text="你好，欢迎使用 Qwen3-TTS WebSocket benchmark。这是一段用于测量首包时延与实时因子的中文测试文本。",
    ),
    BenchmarkCase(
        id="ja_formal",
        language_type="Japanese",
        instructions="落ち着いて、丁寧に。",
        text="こんにちは。こちらは WebSocket ベンチマーク用の日本語テキストで、TTFT と RTF を測定します。",
    ),
)


class BenchmarkError(RuntimeError):
    """Raised when the benchmark protocol flow fails."""


def audio_duration_seconds(total_audio_bytes: int, *, sample_rate: int, channels: int = 1) -> float:
    if total_audio_bytes <= 0 or sample_rate <= 0 or channels <= 0:
        return 0.0
    return total_audio_bytes / float(sample_rate * channels * 2)


def compute_rtf(total_elapsed_ms: float, audio_duration_s: float) -> float | None:
    if audio_duration_s <= 0:
        return None
    return total_elapsed_ms / 1000.0 / audio_duration_s


def percentile(values: Sequence[float], q: float) -> float | None:
    if not values:
        return None
    if len(values) == 1:
        return float(values[0])
    ordered = sorted(float(item) for item in values)
    position = max(0.0, min(1.0, q)) * (len(ordered) - 1)
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    if lower == upper:
        return ordered[lower]
    weight = position - lower
    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight


def summarize_numeric(values: Sequence[float]) -> dict[str, float | int | None]:
    if not values:
        return {
            "count": 0,
            "min": None,
            "max": None,
            "mean": None,
            "median": None,
            "stdev": None,
            "p50": None,
            "p95": None,
        }

    numeric = [float(item) for item in values]
    return {
        "count": len(numeric),
        "min": round(min(numeric), 4),
        "max": round(max(numeric), 4),
        "mean": round(statistics.fmean(numeric), 4),
        "median": round(statistics.median(numeric), 4),
        "stdev": round(statistics.stdev(numeric), 4) if len(numeric) >= 2 else 0.0,
        "p50": round(percentile(numeric, 0.50) or 0.0, 4),
        "p95": round(percentile(numeric, 0.95) or 0.0, 4),
    }


def derive_http_base_url(ws_url: str) -> str:
    parsed = urlparse(ws_url)
    if parsed.scheme not in {"ws", "wss"}:
        raise ValueError(f"Unsupported websocket URL scheme: {parsed.scheme!r}")
    http_scheme = "https" if parsed.scheme == "wss" else "http"
    return urlunparse((http_scheme, parsed.netloc, "", "", "", ""))


def parse_cases(args: argparse.Namespace) -> list[BenchmarkCase]:
    if args.text:
        return [
            BenchmarkCase(
                id=args.case_id,
                text=args.text,
                language_type=args.language_type,
                instructions=args.instructions,
            )
        ]
    if args.cases_json:
        payload = json.loads(Path(args.cases_json).read_text(encoding="utf-8"))
        if not isinstance(payload, list):
            raise ValueError("cases_json must contain a JSON array")
        return [BenchmarkCase(**item) for item in payload]
    return list(DEFAULT_CASES)


def _extract_voice_entries(payload: Any) -> list[dict[str, Any]]:
    if isinstance(payload, dict):
        data = payload.get("data")
        if isinstance(data, list):
            return [item for item in data if isinstance(item, dict)]
    if isinstance(payload, list):
        return [item for item in payload if isinstance(item, dict)]
    return []


def choose_voice_alias(payload: Any, *, requested_voice: str | None, model: str) -> VoiceSelection:
    voices = _extract_voice_entries(payload)
    default_voice = payload.get("default_voice") if isinstance(payload, dict) else None
    if requested_voice:
        for voice_item in voices:
            if str(voice_item.get("voice")) == requested_voice:
                return VoiceSelection(voice=requested_voice, source="requested")
        raise ValueError(f"Requested voice {requested_voice!r} was not found in /v1/voices")

    compatible = [
        str(item.get("voice"))
        for item in voices
        if item.get("voice")
        and (
            not item.get("supported_models")
            or model in {str(entry) for entry in item.get("supported_models", [])}
        )
    ]
    if default_voice and default_voice in compatible:
        return VoiceSelection(voice=str(default_voice), source="default_voice")
    if compatible:
        return VoiceSelection(voice=compatible[0], source="first_compatible_voice")
    if voices:
        fallback = str(voices[0].get("voice") or "")
        if fallback:
            return VoiceSelection(voice=fallback, source="first_voice")
    raise ValueError("Could not resolve any voice alias from /v1/voices; pass --voice explicitly")


async def preflight_service(
    *,
    http_base_url: str,
    model: str,
    requested_voice: str | None,
    timeout_s: float,
) -> dict[str, Any]:
    timeout = httpx.Timeout(timeout_s)
    async with httpx.AsyncClient(timeout=timeout, trust_env=False) as client:
        health_response = await client.get(f"{http_base_url}/healthz")
        health_response.raise_for_status()
        voices_response = await client.get(f"{http_base_url}/v1/voices")
        voices_response.raise_for_status()

    health = health_response.json()
    voices_payload = voices_response.json()
    selection = choose_voice_alias(voices_payload, requested_voice=requested_voice, model=model)
    return {
        "health": health,
        "voices": voices_payload,
        "voice_selection": asdict(selection),
    }


async def recv_json(ws: Any, *, timeout_s: float) -> dict[str, Any]:
    raw = await asyncio.wait_for(ws.recv(), timeout=timeout_s)
    if isinstance(raw, bytes):
        raw = raw.decode("utf-8")
    payload = json.loads(raw)
    if not isinstance(payload, dict):
        raise BenchmarkError(f"Expected JSON object event, got: {payload!r}")
    return payload


async def run_single_iteration(
    *,
    ws_url: str,
    case: BenchmarkCase,
    iteration: int,
    model: str,
    voice: str,
    mode: str,
    sample_rate: int,
    response_format: str,
    optimize_instructions: bool,
    timeout_s: float,
    save_audio_dir: Path | None,
) -> BenchmarkRunMetrics:
    connect_started = time.perf_counter()
    async with websockets.connect(ws_url, max_size=None, open_timeout=timeout_s) as ws:
        connected_at = time.perf_counter()
        created_event = await recv_json(ws, timeout_s=timeout_s)
        if created_event.get("type") != "session.created":
            raise BenchmarkError(f"Expected session.created, got {created_event}")
        session_created_at = time.perf_counter()

        update_payload = {
            "type": "session.update",
            "session": {
                "model": model,
                "voice": voice,
                "language_type": case.language_type,
                "mode": mode,
                "response_format": response_format,
                "sample_rate": sample_rate,
                "instructions": case.instructions,
                "optimize_instructions": optimize_instructions,
            },
        }

        update_sent_at = time.perf_counter()
        await ws.send(json.dumps(update_payload, ensure_ascii=False))
        update_event = await recv_json(ws, timeout_s=timeout_s)
        update_received_at = time.perf_counter()
        if update_event.get("type") == "error":
            raise BenchmarkError(update_event.get("error", {}).get("message", "session.update failed"))
        if update_event.get("type") != "session.updated":
            raise BenchmarkError(f"Expected session.updated, got {update_event}")

        await ws.send(json.dumps({"type": "input_text_buffer.append", "text": case.text}, ensure_ascii=False))

        commit_sent_at = time.perf_counter()
        await ws.send(json.dumps({"type": "input_text_buffer.commit"}))

        first_response_created_at: float | None = None
        first_audio_delta_at: float | None = None
        audio_done_at: float | None = None
        commit_ack_at: float | None = None
        usage_characters: int | None = None
        audio_chunks = 0
        audio_bytes = bytearray()

        while True:
            event = await recv_json(ws, timeout_s=timeout_s)
            event_type = str(event.get("type"))
            event_at = time.perf_counter()

            if event_type == "error":
                raise BenchmarkError(event.get("error", {}).get("message", "generation failed"))
            if event_type == "input_text_buffer.committed" and commit_ack_at is None:
                commit_ack_at = event_at
                continue
            if event_type == "response.created" and first_response_created_at is None:
                first_response_created_at = event_at
                continue
            if event_type == "response.audio.delta":
                if first_audio_delta_at is None:
                    first_audio_delta_at = event_at
                delta = event.get("delta")
                if isinstance(delta, str) and delta:
                    audio_bytes.extend(base64.b64decode(delta))
                    audio_chunks += 1
                continue
            if event_type == "response.audio.done" and audio_done_at is None:
                audio_done_at = event_at
                continue
            if event_type == "response.done":
                usage = event.get("response", {}).get("usage", {})
                usage_characters = usage.get("characters")
                total_elapsed_at = event_at
                break

        total_audio_bytes = len(audio_bytes)
        audio_duration_s = audio_duration_seconds(total_audio_bytes, sample_rate=sample_rate, channels=DEFAULT_CHANNELS)
        total_elapsed_ms = round((total_elapsed_at - commit_sent_at) * 1000.0, 4)

        if save_audio_dir is not None and total_audio_bytes > 0:
            save_audio_dir.mkdir(parents=True, exist_ok=True)
            wav_path = save_audio_dir / f"{case.id}_iter{iteration:02d}.wav"
            wav_path.write_bytes(
                pcm16le_bytes_to_wav_bytes(bytes(audio_bytes), sample_rate=sample_rate, channels=DEFAULT_CHANNELS)
            )

        return BenchmarkRunMetrics(
            case_id=case.id,
            iteration=iteration,
            model=model,
            voice=voice,
            mode=mode,
            sample_rate=sample_rate,
            connect_ms=round((connected_at - connect_started) * 1000.0, 4),
            session_created_ms=round((session_created_at - connected_at) * 1000.0, 4),
            session_update_ms=round((update_received_at - update_sent_at) * 1000.0, 4),
            commit_ack_ms=round((commit_ack_at - commit_sent_at) * 1000.0, 4) if commit_ack_at is not None else None,
            response_created_ms=(
                round((first_response_created_at - commit_sent_at) * 1000.0, 4)
                if first_response_created_at is not None
                else None
            ),
            ttft_ms=round((first_audio_delta_at - commit_sent_at) * 1000.0, 4) if first_audio_delta_at is not None else None,
            audio_done_ms=round((audio_done_at - commit_sent_at) * 1000.0, 4) if audio_done_at is not None else None,
            total_elapsed_ms=total_elapsed_ms,
            total_audio_bytes=total_audio_bytes,
            audio_duration_s=round(audio_duration_s, 4),
            rtf=round(compute_rtf(total_elapsed_ms, audio_duration_s) or 0.0, 6) if audio_duration_s > 0 else None,
            audio_chunks=audio_chunks,
            usage_characters=int(usage_characters) if usage_characters is not None else None,
        )


def summarize_case_runs(runs: Sequence[BenchmarkRunMetrics]) -> dict[str, Any]:
    ttft_values = [item.ttft_ms for item in runs if item.ttft_ms is not None]
    response_created_values = [item.response_created_ms for item in runs if item.response_created_ms is not None]
    elapsed_values = [item.total_elapsed_ms for item in runs]
    rtf_values = [item.rtf for item in runs if item.rtf is not None]
    audio_duration_values = [item.audio_duration_s for item in runs]
    chunk_values = [float(item.audio_chunks) for item in runs]
    audio_bytes_values = [float(item.total_audio_bytes) for item in runs]
    return {
        "run_count": len(runs),
        "ttft_ms": summarize_numeric(ttft_values),
        "response_created_ms": summarize_numeric(response_created_values),
        "total_elapsed_ms": summarize_numeric(elapsed_values),
        "audio_duration_s": summarize_numeric(audio_duration_values),
        "rtf": summarize_numeric(rtf_values),
        "audio_chunks": summarize_numeric(chunk_values),
        "total_audio_bytes": summarize_numeric(audio_bytes_values),
    }


async def run_benchmark(args: argparse.Namespace) -> dict[str, Any]:
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

    successes: list[BenchmarkRunMetrics] = []
    failures: list[BenchmarkFailure] = []
    warmup_runs = max(0, args.warmup)
    measured_iterations = max(1, args.iterations)
    save_audio_dir = Path(args.save_audio_dir) if args.save_audio_dir else None

    for case in cases:
        for warmup_index in range(1, warmup_runs + 1):
            try:
                await run_single_iteration(
                    ws_url=args.ws_url,
                    case=case,
                    iteration=warmup_index,
                    model=args.model,
                    voice=selected_voice,
                    mode=args.mode,
                    sample_rate=args.sample_rate,
                    response_format=args.response_format,
                    optimize_instructions=args.optimize_instructions,
                    timeout_s=args.timeout,
                    save_audio_dir=None,
                )
            except Exception as exc:  # pragma: no cover - warmup is best effort
                print(f"[warmup][{case.id}][{warmup_index}] failed: {exc}", flush=True)

        for iteration in range(1, measured_iterations + 1):
            print(
                f"[benchmark][{case.id}][{iteration}/{measured_iterations}] connecting to {args.ws_url}",
                flush=True,
            )
            try:
                metrics = await run_single_iteration(
                    ws_url=args.ws_url,
                    case=case,
                    iteration=iteration,
                    model=args.model,
                    voice=selected_voice,
                    mode=args.mode,
                    sample_rate=args.sample_rate,
                    response_format=args.response_format,
                    optimize_instructions=args.optimize_instructions,
                    timeout_s=args.timeout,
                    save_audio_dir=save_audio_dir,
                )
            except Exception as exc:
                failure = BenchmarkFailure(case_id=case.id, iteration=iteration, error=str(exc))
                failures.append(failure)
                print(f"[benchmark][{case.id}][{iteration}] failed: {failure.error}", flush=True)
                continue
            successes.append(metrics)
            print(
                f"[benchmark][{case.id}][{iteration}] ttft_ms={metrics.ttft_ms} total_elapsed_ms={metrics.total_elapsed_ms} "
                f"audio_duration_s={metrics.audio_duration_s} rtf={metrics.rtf}",
                flush=True,
            )

    case_summaries: dict[str, Any] = {}
    for case in cases:
        case_runs = [item for item in successes if item.case_id == case.id]
        case_summaries[case.id] = {
            "case": asdict(case),
            "summary": summarize_case_runs(case_runs),
            "runs": [asdict(item) for item in case_runs],
            "failure_count": sum(1 for item in failures if item.case_id == case.id),
        }

    payload = {
        "config": {
            "ws_url": args.ws_url,
            "http_base_url": http_base_url,
            "model": args.model,
            "voice": selected_voice,
            "voice_resolution": preflight["voice_selection"] if preflight is not None else {"voice": selected_voice, "source": "cli"},
            "mode": args.mode,
            "sample_rate": args.sample_rate,
            "response_format": args.response_format,
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
            "overall": summarize_case_runs(successes),
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
    parser = argparse.ArgumentParser(description="Benchmark the streaming_lora_service WebSocket endpoint and report TTFT/RTF metrics")
    parser.add_argument("--ws_url", default="ws://127.0.0.1:9010/api-ws/v1/realtime")
    parser.add_argument("--http_base_url", default=None)
    parser.add_argument("--model", default="qwen3-tts-flash-realtime")
    parser.add_argument("--voice", default=None)
    parser.add_argument("--mode", default="commit", choices=("commit", "server_commit"))
    parser.add_argument("--sample_rate", type=int, default=DEFAULT_SAMPLE_RATE)
    parser.add_argument("--response_format", default="pcm")
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
    print(f"WS benchmark finished for {config['ws_url']}")
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
    args = build_parser().parse_args()
    payload = asyncio.run(run_benchmark(args))
    print_human_summary(payload)


if __name__ == "__main__":
    main()