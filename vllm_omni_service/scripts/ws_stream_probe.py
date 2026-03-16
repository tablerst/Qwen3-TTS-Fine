from __future__ import annotations

import argparse
import asyncio
import base64
import json
import mimetypes
import os
import time
from pathlib import Path
from typing import Any

import websockets

PCM24K_MONO_S16_BYTES_PER_SECOND = 24_000 * 2


async def stream_probe(args: argparse.Namespace) -> dict[str, Any]:
    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    session_config: dict[str, Any] = {
        "voice": args.voice,
        "task_type": args.task_type,
        "language": args.language,
        "response_format": args.response_format,
    }
    if args.model:
        session_config["model"] = args.model
    if args.instructions:
        session_config["instructions"] = args.instructions
    if args.ref_audio:
        session_config["ref_audio"] = normalize_ref_audio(args.ref_audio)
    if args.ref_text:
        session_config["ref_text"] = args.ref_text
    if args.x_vector_only_mode:
        session_config["x_vector_only_mode"] = True
    if args.initial_codec_chunk_frames is not None:
        session_config["initial_codec_chunk_frames"] = args.initial_codec_chunk_frames
    if args.stream_audio:
        session_config["stream_audio"] = True

    first_audio_ms: float | None = None
    total_audio_bytes = 0
    total_audio_chunks = 0
    total_sentences = 0
    current_sentence_index = 0
    current_chunks: list[bytes] = []

    async with websockets.connect(
        args.url,
        max_size=None,
        proxy=True if args.use_env_proxy else None,
    ) as ws:
        await ws.send(json.dumps({"type": "session.config", **session_config}))
        started_at = time.perf_counter()

        if args.simulate_stt:
            words = args.text.split(" ")
            for index, word in enumerate(words):
                chunk = word + (" " if index < len(words) - 1 else "")
                await ws.send(json.dumps({"type": "input.text", "text": chunk}))
                await asyncio.sleep(args.stt_delay)
        else:
            await ws.send(json.dumps({"type": "input.text", "text": args.text}))

        await ws.send(json.dumps({"type": "input.done"}))

        while True:
            message = await ws.recv()
            if isinstance(message, bytes):
                if first_audio_ms is None:
                    first_audio_ms = (time.perf_counter() - started_at) * 1000.0
                current_chunks.append(message)
                total_audio_bytes += len(message)
                total_audio_chunks += 1
                continue

            payload = json.loads(message)
            msg_type = payload.get("type")

            if msg_type == "audio.start":
                current_sentence_index = int(payload.get("sentence_index", 0))
                current_chunks = []
            elif msg_type == "audio.done":
                extension = args.response_format.lower()
                target_file = output_dir / f"sentence_{current_sentence_index:03d}.{extension}"
                target_file.write_bytes(b"".join(current_chunks))
                total_sentences += 1
                current_chunks = []
            elif msg_type == "session.done":
                break
            elif msg_type == "error":
                raise RuntimeError(payload.get("message", "Unknown websocket error"))

    elapsed_ms = (time.perf_counter() - started_at) * 1000.0
    metrics = {
        "url": args.url,
        "stream_audio": args.stream_audio,
        "response_format": args.response_format,
        "output_dir": str(output_dir),
        "sentences": total_sentences,
        "audio_chunks": total_audio_chunks,
        "audio_bytes": total_audio_bytes,
        "first_audio_ms": round(first_audio_ms, 2) if first_audio_ms is not None else None,
        "elapsed_ms": round(elapsed_ms, 2),
    }
    if args.response_format == "pcm" and total_audio_bytes > 0:
        audio_duration_s = total_audio_bytes / PCM24K_MONO_S16_BYTES_PER_SECOND
        metrics["audio_duration_s"] = round(audio_duration_s, 4)
        metrics["rtf"] = round(elapsed_ms / 1000.0 / audio_duration_s, 4)
    return metrics


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Probe the vLLM-Omni websocket speech streaming endpoint.")
    parser.add_argument("--url", default="ws://localhost:8091/v1/audio/speech/stream")
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
        help="Optional per-session override for initial_codec_chunk_frames to reduce TTFA.",
    )
    parser.add_argument("--response-format", default="pcm", choices=["wav", "pcm", "flac", "mp3", "aac", "opus"])
    parser.add_argument("--stream-audio", action="store_true", help="Ask for chunked audio within each sentence.")
    parser.add_argument("--simulate-stt", action="store_true")
    parser.add_argument("--stt-delay", type=float, default=0.1)
    parser.add_argument(
        "--output-dir",
        default=str(Path(__file__).resolve().parents[2] / "outputs" / "vllm_omni" / "ws_probe"),
    )
    parser.add_argument("--use-env-proxy", action="store_true", help="Allow websockets.connect to use proxy settings from the environment.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.stream_audio and args.response_format != "pcm":
        raise ValueError("--stream-audio should be paired with --response-format pcm for the official websocket API.")
    metrics = asyncio.run(stream_probe(args))
    print(json.dumps(metrics, indent=2, ensure_ascii=False))


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


if __name__ == "__main__":
    main()
