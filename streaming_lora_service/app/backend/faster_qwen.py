from __future__ import annotations

import hashlib
import importlib
import json
import logging
import shutil
import time
from pathlib import Path
from typing import Any, Iterable

import numpy as np

from faster_qwen3_tts_integration.contracts import ContractError, validate_exported_model_dir
from faster_qwen3_tts_integration.scripts.export_merged_model import merge_bundle_to_local_model
from lora_finetuning.common import ensure_dir, load_json, parse_torch_dtype

from ..audio_utils import float_audio_to_pcm16le_bytes
from ..bundle_loader import infer_speaker_name, resolve_bundle_artifacts
from ..models import SynthesizedAudio
from ..runtime_session import RuntimeSession
from .base import BackendCapabilities, BackendLoadResult


logger = logging.getLogger(__name__)


def _load_faster_model_class() -> Any:
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


def _hash_file(path: Path, hasher: "hashlib._Hash") -> None:
    hasher.update(path.name.encode("utf-8"))
    with path.open("rb") as f:
        while True:
            chunk = f.read(1024 * 1024)
            if not chunk:
                break
            hasher.update(chunk)


def _compute_cache_key(
    *,
    bundle_dir: Path,
    artifacts: Any,
    config_patch: dict[str, Any],
    speaker_name: str,
    torch_dtype: str,
) -> str:
    hasher = hashlib.sha256()
    hasher.update(str(bundle_dir.resolve()).encode("utf-8"))
    hasher.update(str(artifacts.base_model).encode("utf-8"))
    hasher.update(torch_dtype.encode("utf-8"))
    hasher.update(speaker_name.encode("utf-8"))
    hasher.update(json.dumps(config_patch, ensure_ascii=False, sort_keys=True).encode("utf-8"))

    manifest_path = bundle_dir / "manifest.json"
    if manifest_path.exists():
        _hash_file(manifest_path, hasher)
    for path in (
        artifacts.config_patch_file,
        artifacts.speaker_patch_file,
        artifacts.adapter_dir / "adapter_config.json",
        artifacts.adapter_dir / "adapter_model.safetensors",
    ):
        if Path(path).exists():
            _hash_file(Path(path), hasher)

    base_config_path = Path(artifacts.base_model) / "config.json"
    if base_config_path.exists():
        _hash_file(base_config_path, hasher)

    return hasher.hexdigest()[:16]


def _resolve_cache_output_dir(
    *,
    bundle_dir: Path,
    cache_root: str | Path | None,
    cache_key: str,
) -> Path:
    root = Path(cache_root) if cache_root is not None else bundle_dir / ".cache" / "faster_merged"
    ensure_dir(root)
    safe_name = bundle_dir.name or "bundle"
    return root / f"{safe_name}_{cache_key}"


def _resolve_or_export_merged_model(
    *,
    bundle_dir: Path,
    cache_root: str | Path | None,
    base_model: str | None,
    speaker_name: str | None,
    device: str,
    torch_dtype: str,
    attn_implementation: str,
    local_files_only: bool,
) -> tuple[Path, bool, Any, str]:
    artifacts = resolve_bundle_artifacts(
        bundle_dir,
        base_model=base_model,
        speaker_name=speaker_name,
    )
    config_patch = load_json(artifacts.config_patch_file)
    resolved_speaker_name = infer_speaker_name(config_patch, artifacts.speaker_name)
    cache_key = _compute_cache_key(
        bundle_dir=bundle_dir,
        artifacts=artifacts,
        config_patch=config_patch,
        speaker_name=resolved_speaker_name,
        torch_dtype=torch_dtype,
    )
    output_dir = _resolve_cache_output_dir(bundle_dir=bundle_dir, cache_root=cache_root, cache_key=cache_key)

    cache_hit = False
    if output_dir.exists():
        try:
            validate_exported_model_dir(output_dir, expected_speaker=resolved_speaker_name)
            cache_hit = True
        except ContractError:
            shutil.rmtree(output_dir)

    if not cache_hit:
        merge_bundle_to_local_model(
            bundle_dir=bundle_dir,
            output_dir=output_dir,
            base_model=artifacts.base_model,
            speaker_name=resolved_speaker_name,
            device_map=device,
            torch_dtype=torch_dtype,
            attn_implementation=attn_implementation,
            local_files_only=local_files_only,
        )

    return output_dir, cache_hit, artifacts, resolved_speaker_name


def _extract_supported_speakers(model: Any) -> tuple[str, ...]:
    candidates = (
        getattr(model, "get_supported_speakers", None),
        getattr(getattr(model, "model", None), "get_supported_speakers", None),
    )
    for getter in candidates:
        if callable(getter):
            raw = getter()
            if isinstance(raw, (list, tuple, set)):
                return tuple(str(item) for item in raw)
    raw_supported = getattr(getattr(model, "model", None), "supported_speakers", None)
    if isinstance(raw_supported, (list, tuple, set)):
        return tuple(str(item) for item in raw_supported)
    return ()


def _infer_sample_rate(model: Any) -> int:
    direct = getattr(model, "sample_rate", None)
    if isinstance(direct, int) and direct > 0:
        return direct

    speech_tokenizer = getattr(getattr(model, "model", None), "speech_tokenizer", None)
    getter = getattr(speech_tokenizer, "get_output_sample_rate", None)
    if callable(getter):
        try:
            sample_rate = getter()
            if isinstance(sample_rate, (int, float, str)):
                return int(sample_rate)
        except Exception:
            pass
    return 24000


def _to_numpy_audio(audio: Any) -> np.ndarray:
    if hasattr(audio, "detach"):
        audio = audio.detach().cpu().numpy()
    elif hasattr(audio, "cpu") and hasattr(audio, "numpy"):
        audio = audio.cpu().numpy()
    return np.asarray(audio, dtype=np.float32).reshape(-1)


class FasterQwenSpeechBackend:
    kind = "faster"
    capabilities = BackendCapabilities()

    def __init__(
        self,
        model: Any,
        *,
        supported_speakers: tuple[str, ...],
        sample_rate: int,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        self.model = model
        self._supported_speakers = tuple(str(item) for item in supported_speakers)
        self._speaker_lookup = {speaker.lower(): speaker for speaker in self._supported_speakers}
        self.sample_rate = int(sample_rate)
        self.metadata = dict(metadata or {})

    def list_supported_speakers(self) -> tuple[str, ...]:
        return self._supported_speakers

    def _resolve_speaker_name(self, speaker: str) -> str:
        matched = self._speaker_lookup.get(str(speaker).lower())
        if matched is not None:
            return matched
        if not self._supported_speakers:
            return str(speaker)
        raise ValueError(
            f"Requested speaker {speaker!r} is not available in faster backend. Supported speakers: {list(self._supported_speakers)}"
        )

    def synthesize(
        self,
        *,
        text: str,
        speaker: str,
        language: str = "Auto",
        instruct: str | None = None,
    ) -> SynthesizedAudio:
        resolved_speaker = self._resolve_speaker_name(speaker)
        audio_list, sample_rate = self.model.generate_custom_voice(
            text=text,
            speaker=resolved_speaker,
            language=language,
            instruct=instruct,
        )
        if not audio_list:
            raise RuntimeError("faster-qwen3-tts returned no audio for synthesize()")
        audio_bytes = float_audio_to_pcm16le_bytes(_to_numpy_audio(audio_list[0]))
        return SynthesizedAudio(
            audio_bytes=audio_bytes,
            sample_rate=int(sample_rate),
            channels=1,
        )

    def stream_synthesize(
        self,
        *,
        text: str,
        speaker: str,
        language: str = "Auto",
        instruct: str | None = None,
        runtime_session: RuntimeSession | None = None,
        chunk_steps: int = 4,
        left_context_steps: int = 25,
        first_chunk_steps: int | None = 1,
        crossfade_samples: int = 0,
        runtime_session_sync_mode: str = "chunk",
    ) -> Iterable[bytes]:
        del left_context_steps, first_chunk_steps, crossfade_samples, runtime_session_sync_mode
        resolved_speaker = self._resolve_speaker_name(speaker)
        if runtime_session is not None:
            runtime_session.reset_generation_state()
        started_at = time.perf_counter()
        first_chunk_ready_ms: float | None = None
        emitted_chunks = 0
        total_decode_ms = 0.0
        total_prefill_ms = 0.0
        total_steps = 0

        generator = self.model.generate_custom_voice_streaming(
            text=text,
            speaker=resolved_speaker,
            language=language,
            instruct=instruct,
            chunk_size=max(1, int(chunk_steps)),
        )
        for audio_chunk, sample_rate, timing in generator:
            if int(sample_rate) != self.sample_rate:
                raise RuntimeError(
                    f"faster-qwen3-tts stream sample-rate mismatch: expected {self.sample_rate}, got {sample_rate}"
                )
            pcm_chunk = float_audio_to_pcm16le_bytes(_to_numpy_audio(audio_chunk))
            if not pcm_chunk:
                continue
            emitted_chunks += 1
            total_decode_ms += float(timing.get("decode_ms") or 0.0)
            total_prefill_ms += float(timing.get("prefill_ms") or 0.0)
            total_steps = max(total_steps, int(timing.get("total_steps_so_far") or 0))
            if first_chunk_ready_ms is None:
                first_chunk_ready_ms = (time.perf_counter() - started_at) * 1000.0
            if runtime_session is not None:
                runtime_session.state.last_generation_metrics = {
                    "backend": self.kind,
                    "generated_steps": total_steps,
                    "emitted_chunks": emitted_chunks,
                    "finish_reason": "streaming",
                    "first_chunk_ready_ms": round(first_chunk_ready_ms, 2) if first_chunk_ready_ms is not None else None,
                    "prefill_ms": round(total_prefill_ms, 2),
                    "total_decode_ms": round(total_decode_ms, 2),
                    "avg_decode_ms": round(total_decode_ms / emitted_chunks, 2) if emitted_chunks else None,
                    "chunk_timing": dict(timing),
                    "supports_state_resume": False,
                }
            yield pcm_chunk

        if runtime_session is not None:
            elapsed_ms = (time.perf_counter() - started_at) * 1000.0
            runtime_session.state.last_generation_metrics = {
                **runtime_session.state.last_generation_metrics,
                "backend": self.kind,
                "generated_steps": total_steps,
                "emitted_chunks": emitted_chunks,
                "finish_reason": "completed",
                "first_chunk_ready_ms": round(first_chunk_ready_ms, 2) if first_chunk_ready_ms is not None else None,
                "prefill_ms": round(total_prefill_ms, 2),
                "total_decode_ms": round(total_decode_ms, 2),
                "avg_decode_ms": round(total_decode_ms / emitted_chunks, 2) if emitted_chunks else None,
                "init_total_ms": round(first_chunk_ready_ms or 0.0, 2),
                "total_step_ms": round(elapsed_ms, 2),
                "avg_step_ms": round(elapsed_ms / total_steps, 2) if total_steps else None,
                "supports_state_resume": False,
            }


def load_faster_backend(
    bundle_dir: str | Path,
    *,
    base_model: str | None = None,
    speaker_name: str | None = None,
    device: str = "cuda:0",
    torch_dtype: str = "bfloat16",
    attn_implementation: str = "sdpa",
    local_files_only: bool = False,
    max_seq_len: int = 2048,
    cache_root: str | Path | None = None,
    faster_model_cls: Any | None = None,
) -> BackendLoadResult:
    bundle_path = Path(bundle_dir)
    merged_model_dir, cache_hit, artifacts, resolved_speaker_name = _resolve_or_export_merged_model(
        bundle_dir=bundle_path,
        cache_root=cache_root,
        base_model=base_model,
        speaker_name=speaker_name,
        device=device,
        torch_dtype=torch_dtype,
        attn_implementation=attn_implementation,
        local_files_only=local_files_only,
    )
    contract = validate_exported_model_dir(merged_model_dir, expected_speaker=resolved_speaker_name)
    faster_model_cls = faster_model_cls or _load_faster_model_class()
    if faster_model_cls is None:
        raise RuntimeError("Failed to resolve faster-qwen3-tts model class")
    dtype = parse_torch_dtype(torch_dtype)

    load_started_at = time.perf_counter()
    model = faster_model_cls.from_pretrained(
        str(merged_model_dir),
        device=device,
        dtype=dtype,
        attn_implementation=attn_implementation,
        max_seq_len=max_seq_len,
    )
    load_elapsed_ms = (time.perf_counter() - load_started_at) * 1000.0

    supported_speakers = _extract_supported_speakers(model) or tuple(contract.get("speaker_names") or ())
    if supported_speakers and resolved_speaker_name.lower() not in {speaker.lower() for speaker in supported_speakers}:
        raise RuntimeError(
            f"Resolved speaker {resolved_speaker_name!r} was not reported by faster-qwen3-tts: {list(supported_speakers)}"
        )

    backend = FasterQwenSpeechBackend(
        model,
        supported_speakers=supported_speakers or (resolved_speaker_name,),
        sample_rate=_infer_sample_rate(model),
        metadata={
            "cache_hit": cache_hit,
            "merged_model_dir": str(merged_model_dir.resolve()),
            "load_elapsed_ms": round(load_elapsed_ms, 2),
            "max_seq_len": max_seq_len,
        },
    )
    logger.warning(
        "loaded faster backend merged_model_dir=%s cache_hit=%s speaker=%s max_seq_len=%s load_elapsed_ms=%.2f",
        merged_model_dir,
        cache_hit,
        resolved_speaker_name,
        max_seq_len,
        load_elapsed_ms,
    )
    return BackendLoadResult(
        backend=backend,
        speaker_name=resolved_speaker_name,
        supported_speakers=backend.list_supported_speakers(),
        tts_model_type="custom_voice",
        loaded_bundle=None,
        bundle_artifacts=artifacts,
        metadata={
            "backend": backend.kind,
            "cache_hit": cache_hit,
            "merged_model_dir": str(merged_model_dir.resolve()),
            "max_seq_len": max_seq_len,
            "load_elapsed_ms": round(load_elapsed_ms, 2),
        },
    )
