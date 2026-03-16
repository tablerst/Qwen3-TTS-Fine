from __future__ import annotations

import argparse
import base64
import json
import mimetypes
import os
import time
from pathlib import Path
from typing import Any

import httpx

RESPONSE_FORMATS = {"wav", "mp3", "flac", "pcm", "aac", "opus"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Probe the vLLM-Omni /v1/audio/speech endpoint.")
    parser.add_argument("--api-base", default="http://localhost:8091", help="Base URL without trailing /v1.")
    parser.add_argument("--model", default=None, help="Optional model name/path.")
    parser.add_argument("--text", required=True, help="Input text.")
    parser.add_argument("--voice", default="vivian", help="Voice name.")
    parser.add_argument("--task-type", default=None, choices=["CustomVoice", "VoiceDesign", "Base"])
    parser.add_argument("--language", default=None, help="Language value passed to the API.")
    parser.add_argument("--instructions", default=None, help="Style/voice instructions.")
    parser.add_argument("--ref-audio", default=None, help="Reference audio URL or base64 data URL for Base task.")
    parser.add_argument("--ref-text", default=None, help="Reference text for Base task.")
    parser.add_argument("--x-vector-only-mode", action="store_true", help="Use speaker embedding only mode for Base task.")
    parser.add_argument("--response-format", default="pcm", choices=sorted(RESPONSE_FORMATS))
    parser.add_argument("--stream", action="store_true", help="Request raw HTTP streaming from the API.")
    parser.add_argument(
        "--output",
        default=None,
        help="Output file path. Defaults to outputs/vllm_omni/http_probe.<ext>.",
    )
    parser.add_argument("--timeout", type=float, default=300.0)
    parser.add_argument("--trust-env", action="store_true", help="Allow httpx to inherit HTTP(S)_PROXY and related environment variables.")
    return parser.parse_args()


def build_payload(args: argparse.Namespace) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "input": args.text,
        "voice": args.voice,
        "response_format": args.response_format,
    }
    if args.model:
        payload["model"] = args.model
    if args.task_type:
        payload["task_type"] = args.task_type
    if args.language:
        payload["language"] = args.language
    if args.instructions:
        payload["instructions"] = args.instructions
    if args.ref_audio:
        payload["ref_audio"] = normalize_ref_audio(args.ref_audio)
    if args.ref_text:
        payload["ref_text"] = args.ref_text
    if args.x_vector_only_mode:
        payload["x_vector_only_mode"] = True
    if args.stream:
        payload["stream"] = True
    return payload


def normalize_ref_audio(ref_audio: str) -> str:
    if ref_audio.startswith(("http://", "https://", "data:", "file://")):
        return ref_audio
    if not os.path.exists(ref_audio):
        return ref_audio

    mime_type, _ = mimetypes.guess_type(ref_audio)
    if mime_type is None:
        mime_type = "audio/wav"
    with open(ref_audio, "rb") as handle:
        payload = base64.b64encode(handle.read()).decode("utf-8")
    return f"data:{mime_type};base64,{payload}"


def default_output_path(response_format: str) -> Path:
    return Path(__file__).resolve().parents[2] / "outputs" / "vllm_omni" / f"http_probe.{response_format}"


def main() -> None:
    args = parse_args()
    if args.stream and args.response_format != "pcm":
        raise ValueError("HTTP streaming probe should use --response-format pcm to match the official API contract.")

    output_path = Path(args.output).expanduser().resolve() if args.output else default_output_path(args.response_format)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    payload = build_payload(args)
    url = f"{args.api_base.rstrip('/')}/v1/audio/speech"

    started_at = time.perf_counter()
    first_chunk_ms: float | None = None
    total_bytes = 0
    chunk_count = 0

    with httpx.Client(timeout=args.timeout, trust_env=args.trust_env) as client:
        with client.stream("POST", url, json=payload) as response:
            response.raise_for_status()
            with output_path.open("wb") as handle:
                for chunk in response.iter_bytes():
                    if not chunk:
                        continue
                    if first_chunk_ms is None:
                        first_chunk_ms = (time.perf_counter() - started_at) * 1000.0
                    handle.write(chunk)
                    total_bytes += len(chunk)
                    chunk_count += 1

    elapsed_ms = (time.perf_counter() - started_at) * 1000.0
    metrics = {
        "endpoint": url,
        "stream": args.stream,
        "response_format": args.response_format,
        "output_path": str(output_path),
        "chunk_count": chunk_count,
        "total_bytes": total_bytes,
        "first_chunk_ms": round(first_chunk_ms, 2) if first_chunk_ms is not None else None,
        "elapsed_ms": round(elapsed_ms, 2),
    }
    print(json.dumps(metrics, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
