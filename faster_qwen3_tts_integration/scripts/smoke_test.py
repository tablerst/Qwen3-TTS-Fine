from __future__ import annotations

import argparse
import importlib
import time
from pathlib import Path
from typing import Any

import numpy as np
import soundfile as sf
import torch

from ..contracts import infer_speaker_name, validate_exported_model_dir
from lora_finetuning.common import ensure_dir, save_json


DEFAULT_SUMMARY_SUFFIX = ".summary.json"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run a minimal faster-qwen3-tts custom-voice smoke test")
    parser.add_argument("--model_dir", required=True, help="Merged local model directory produced by export_merged_model.py")
    parser.add_argument("--text", required=True, help="Text to synthesize")
    parser.add_argument("--output_wav", required=True, help="Output WAV path")
    parser.add_argument("--speaker", default=None, help="Speaker name to use; required for multi-speaker models")
    parser.add_argument("--language", default="Auto", help="Language to synthesize in")
    parser.add_argument("--instruct", default=None, help="Optional custom voice instruction text")
    parser.add_argument("--device", default="cuda", help="Device passed to FasterQwen3TTS.from_pretrained")
    parser.add_argument("--dtype", default="bf16", choices=["bf16", "fp16", "fp32"], help="Inference dtype")
    parser.add_argument("--attn_implementation", default="sdpa", help="Attention backend for FasterQwen3TTS")
    parser.add_argument("--max_seq_len", type=int, default=2048, help="Static cache sequence length for faster-qwen3-tts")
    parser.add_argument("--max_new_tokens", type=int, default=2048)
    parser.add_argument("--temperature", type=float, default=0.9)
    parser.add_argument("--top_k", type=int, default=50)
    parser.add_argument("--top_p", type=float, default=1.0)
    parser.add_argument("--repetition_penalty", type=float, default=1.05)
    parser.add_argument("--greedy", action="store_true", help="Disable sampling and use greedy decoding")
    parser.add_argument("--summary_json", default=None, help="Optional explicit summary JSON path")
    parser.add_argument("--streaming", action="store_true", help="Use generate_custom_voice_streaming and record TTFT/RTF")
    parser.add_argument("--chunk_size", type=int, default=8, help="Streaming chunk size in codec steps")
    parser.add_argument(
        "--warmup_max_new_tokens",
        type=int,
        default=20,
        help="Warmup generation length used before streaming metrics to capture CUDA graphs",
    )
    return parser


def parse_dtype(dtype_name: str) -> torch.dtype:
    mapping = {
        "bf16": torch.bfloat16,
        "fp16": torch.float16,
        "fp32": torch.float32,
    }
    return mapping[dtype_name]


def load_faster_model_class() -> Any:
    candidates = (
        ("faster_qwen3_tts", "FasterQwen3TTS"),
        ("faster_qwen3_tts.model", "FasterQwen3TTS"),
    )
    errors: list[str] = []
    for module_name, attr_name in candidates:
        try:
            module = importlib.import_module(module_name)
            return getattr(module, attr_name)
        except Exception as exc:  # pragma: no cover - best-effort error surfacing
            errors.append(f"{module_name}: {exc}")
    joined = " | ".join(errors)
    raise ImportError(
        "Could not import external faster-qwen3-tts package. Install it in the active environment first. "
        f"Tried: {joined}"
    )


def _to_numpy_audio(audio: Any) -> np.ndarray:
    if hasattr(audio, "detach"):
        audio = audio.detach().cpu().numpy()
    elif hasattr(audio, "cpu") and hasattr(audio, "numpy"):
        audio = audio.cpu().numpy()
    audio_array = np.asarray(audio, dtype=np.float32).reshape(-1)
    if audio_array.size == 0:
        raise RuntimeError("Smoke test returned an empty audio array")
    return audio_array


def _maybe_cuda_synchronize(device: str) -> None:
    if str(device).startswith("cuda") and torch.cuda.is_available():
        torch.cuda.synchronize()


def _load_model_and_validate_speaker(
    *,
    model_path: Path,
    speaker: str | None,
    device: str,
    dtype: str,
    attn_implementation: str,
    max_seq_len: int,
    faster_model_cls: Any | None,
) -> tuple[Any, dict[str, Any], str, list[str], float]:
    contract = validate_exported_model_dir(model_path, expected_speaker=speaker)
    resolved_speaker = infer_speaker_name(model_path, explicit_speaker=speaker, require_single=speaker is None)
    faster_model_cls = faster_model_cls or load_faster_model_class()
    assert faster_model_cls is not None

    dtype_value = parse_dtype(dtype)
    load_started = time.perf_counter()
    model = faster_model_cls.from_pretrained(
        str(model_path),
        device=device,
        dtype=dtype_value,
        attn_implementation=attn_implementation,
        max_seq_len=max_seq_len,
    )
    load_elapsed = time.perf_counter() - load_started

    supported_speakers_getter = getattr(getattr(model, "model", None), "get_supported_speakers", None)
    raw_supported_speakers = supported_speakers_getter() if callable(supported_speakers_getter) else None
    if isinstance(raw_supported_speakers, (list, tuple, set)):
        supported_speakers = [str(speaker_name) for speaker_name in raw_supported_speakers]
    else:
        supported_speakers = []
    if supported_speakers and resolved_speaker.lower() not in {speaker_name.lower() for speaker_name in supported_speakers}:
        raise RuntimeError(
            f"Resolved speaker {resolved_speaker!r} was not reported by faster-qwen3-tts: {supported_speakers}"
        )

    return model, contract, resolved_speaker, supported_speakers, load_elapsed


def run_streaming_smoke_test(
    model_dir: str | Path,
    *,
    text: str,
    output_wav: str | Path,
    speaker: str | None = None,
    language: str = "Auto",
    instruct: str | None = None,
    device: str = "cuda",
    dtype: str = "bf16",
    attn_implementation: str = "sdpa",
    max_seq_len: int = 2048,
    max_new_tokens: int = 2048,
    temperature: float = 0.9,
    top_k: int = 50,
    top_p: float = 1.0,
    repetition_penalty: float = 1.05,
    greedy: bool = False,
    summary_json: str | Path | None = None,
    faster_model_cls: Any | None = None,
    audio_writer: Any = None,
    chunk_size: int = 8,
    warmup_max_new_tokens: int = 20,
) -> dict[str, Any]:
    model_path = Path(model_dir)
    audio_writer = audio_writer or sf.write
    model, contract, resolved_speaker, supported_speakers, load_elapsed = _load_model_and_validate_speaker(
        model_path=model_path,
        speaker=speaker,
        device=device,
        dtype=dtype,
        attn_implementation=attn_implementation,
        max_seq_len=max_seq_len,
        faster_model_cls=faster_model_cls,
    )

    warmup_text = text[: max(1, min(len(text), 32))]
    if warmup_max_new_tokens > 0:
        _ = model.generate_custom_voice(
            text=warmup_text,
            speaker=resolved_speaker,
            language=language,
            instruct=instruct,
            max_new_tokens=warmup_max_new_tokens,
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
            do_sample=not greedy,
            repetition_penalty=repetition_penalty,
        )
        _maybe_cuda_synchronize(device)

    _maybe_cuda_synchronize(device)
    t0 = time.perf_counter()
    generator = model.generate_custom_voice_streaming(
        text=text,
        speaker=resolved_speaker,
        language=language,
        instruct=instruct,
        chunk_size=chunk_size,
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        top_k=top_k,
        top_p=top_p,
        do_sample=not greedy,
        repetition_penalty=repetition_penalty,
    )

    chunk_summaries: list[dict[str, Any]] = []
    chunks: list[np.ndarray] = []
    first_item = next(generator, None)
    _maybe_cuda_synchronize(device)
    if first_item is None:
        raise RuntimeError("Streaming smoke test returned no audio chunks")

    first_chunk, sample_rate, first_timing = first_item
    ttft_ms = (time.perf_counter() - t0) * 1000.0
    first_chunk_audio = _to_numpy_audio(first_chunk)
    chunks.append(first_chunk_audio)
    chunk_summaries.append(
        {
            "wall_time_s": round(ttft_ms / 1000.0, 6),
            "chunk_samples": int(first_chunk_audio.shape[0]),
            "timing": dict(first_timing),
        }
    )

    for audio_chunk, chunk_sr, timing in generator:
        if chunk_sr != sample_rate:
            raise RuntimeError(
                f"Streaming sample-rate mismatch across chunks: first={sample_rate}, current={chunk_sr}"
            )
        chunk_audio = _to_numpy_audio(audio_chunk)
        chunks.append(chunk_audio)
        chunk_summaries.append(
            {
                "wall_time_s": round(time.perf_counter() - t0, 6),
                "chunk_samples": int(chunk_audio.shape[0]),
                "timing": dict(timing),
            }
        )

    _maybe_cuda_synchronize(device)
    total_elapsed = time.perf_counter() - t0
    full_audio = np.concatenate(chunks) if chunks else np.zeros(1, dtype=np.float32)
    audio_duration_s = float(full_audio.shape[0]) / float(sample_rate)
    rtf = audio_duration_s / total_elapsed if total_elapsed > 0 else 0.0

    output_path = Path(output_wav)
    ensure_dir(output_path.parent)
    audio_writer(str(output_path), full_audio, sample_rate)

    summary_path = Path(summary_json) if summary_json else output_path.with_suffix(output_path.suffix + DEFAULT_SUMMARY_SUFFIX)
    summary = {
        "mode": "streaming",
        "model_dir": str(model_path.resolve()),
        "output_wav": str(output_path.resolve()),
        "speaker": resolved_speaker,
        "language": language,
        "text": text,
        "device": device,
        "dtype": dtype,
        "attn_implementation": attn_implementation,
        "max_seq_len": max_seq_len,
        "chunk_size": chunk_size,
        "load_elapsed_seconds": round(load_elapsed, 4),
        "warmup_max_new_tokens": warmup_max_new_tokens,
        "ttft_ms": round(ttft_ms, 2),
        "stream_total_seconds": round(total_elapsed, 4),
        "rtf": round(rtf, 4),
        "sample_rate": int(sample_rate),
        "num_audio_samples": int(full_audio.shape[0]),
        "audio_duration_seconds": round(audio_duration_s, 4),
        "chunk_count": len(chunks),
        "supported_speakers": supported_speakers,
        "validated_contract": contract,
        "chunk_summaries": chunk_summaries,
    }
    save_json(summary, summary_path)
    summary["summary_json"] = str(summary_path.resolve())
    return summary


def run_smoke_test(
    model_dir: str | Path,
    *,
    text: str,
    output_wav: str | Path,
    speaker: str | None = None,
    language: str = "Auto",
    instruct: str | None = None,
    device: str = "cuda",
    dtype: str = "bf16",
    attn_implementation: str = "sdpa",
    max_seq_len: int = 2048,
    max_new_tokens: int = 2048,
    temperature: float = 0.9,
    top_k: int = 50,
    top_p: float = 1.0,
    repetition_penalty: float = 1.05,
    greedy: bool = False,
    summary_json: str | Path | None = None,
    faster_model_cls: Any | None = None,
    audio_writer: Any = None,
) -> dict[str, Any]:
    model_path = Path(model_dir)
    audio_writer = audio_writer or sf.write
    model, contract, resolved_speaker, supported_speakers, load_elapsed = _load_model_and_validate_speaker(
        model_path=model_path,
        speaker=speaker,
        device=device,
        dtype=dtype,
        attn_implementation=attn_implementation,
        max_seq_len=max_seq_len,
        faster_model_cls=faster_model_cls,
    )

    generate_started = time.perf_counter()
    audio_list, sample_rate = model.generate_custom_voice(
        text=text,
        speaker=resolved_speaker,
        language=language,
        instruct=instruct,
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        top_k=top_k,
        top_p=top_p,
        do_sample=not greedy,
        repetition_penalty=repetition_penalty,
    )
    generate_elapsed = time.perf_counter() - generate_started

    if not audio_list:
        raise RuntimeError("Smoke test generation returned no audio items")
    audio = _to_numpy_audio(audio_list[0])

    output_path = Path(output_wav)
    ensure_dir(output_path.parent)
    audio_writer(str(output_path), audio, sample_rate)

    summary_path = Path(summary_json) if summary_json else output_path.with_suffix(output_path.suffix + DEFAULT_SUMMARY_SUFFIX)
    summary = {
        "mode": "non_streaming",
        "model_dir": str(model_path.resolve()),
        "output_wav": str(output_path.resolve()),
        "speaker": resolved_speaker,
        "language": language,
        "text": text,
        "device": device,
        "dtype": dtype,
        "attn_implementation": attn_implementation,
        "max_seq_len": max_seq_len,
        "load_elapsed_seconds": round(load_elapsed, 4),
        "generate_elapsed_seconds": round(generate_elapsed, 4),
        "sample_rate": int(sample_rate),
        "num_audio_samples": int(audio.shape[0]),
        "audio_duration_seconds": round(float(audio.shape[0]) / float(sample_rate), 4),
        "supported_speakers": supported_speakers,
        "validated_contract": contract,
    }
    save_json(summary, summary_path)
    summary["summary_json"] = str(summary_path.resolve())
    return summary


def parse_args() -> argparse.Namespace:
    return build_parser().parse_args()


def main() -> None:
    args = parse_args()
    if args.streaming:
        summary = run_streaming_smoke_test(
            model_dir=args.model_dir,
            text=args.text,
            output_wav=args.output_wav,
            speaker=args.speaker,
            language=args.language,
            instruct=args.instruct,
            device=args.device,
            dtype=args.dtype,
            attn_implementation=args.attn_implementation,
            max_seq_len=args.max_seq_len,
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
            top_k=args.top_k,
            top_p=args.top_p,
            repetition_penalty=args.repetition_penalty,
            greedy=args.greedy,
            summary_json=args.summary_json,
            chunk_size=args.chunk_size,
            warmup_max_new_tokens=args.warmup_max_new_tokens,
        )
    else:
        summary = run_smoke_test(
            model_dir=args.model_dir,
            text=args.text,
            output_wav=args.output_wav,
            speaker=args.speaker,
            language=args.language,
            instruct=args.instruct,
            device=args.device,
            dtype=args.dtype,
            attn_implementation=args.attn_implementation,
            max_seq_len=args.max_seq_len,
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
            top_k=args.top_k,
            top_p=args.top_p,
            repetition_penalty=args.repetition_penalty,
            greedy=args.greedy,
            summary_json=args.summary_json,
        )

    print(f"Smoke test completed: {summary['output_wav']}")
    print(f"Speaker: {summary['speaker']}")
    print(f"Duration: {summary['audio_duration_seconds']}s @ {summary['sample_rate']} Hz")
    if summary.get("mode") == "streaming":
        print(f"TTFT: {summary['ttft_ms']} ms")
        print(f"RTF: {summary['rtf']}")
    print(f"Summary: {summary['summary_json']}")


if __name__ == "__main__":
    main()
