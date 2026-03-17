from __future__ import annotations

import argparse
import asyncio
import base64
from dataclasses import dataclass, field
import json
import logging
from pathlib import Path
import time
from typing import Any
from uuid import uuid4

from fastapi import FastAPI, HTTPException, Request, WebSocket, WebSocketDisconnect
from fastapi.responses import Response, StreamingResponse
from pydantic import BaseModel, Field

from .backend import (
    BackendLoadResult,
    SpeechBackend,
    NativeQwenSpeechBackend,
    load_faster_backend,
    load_native_backend,
)
from .audio_utils import pcm16le_bytes_to_wav_bytes
from .bundle_loader import BundleLoader
from .models import LoadedBundle, SessionOptions, SynthesizedAudio
from .qwen_compat_ws import QwenRealtimeProtocolAdapter
from .runtime_session import RuntimeSession
from .step_streamer import AudioStepStreamer, AudioStepStreamerConfig
from .voice_registry import VoiceRegistry


logger = logging.getLogger(__name__)


def _resolve_attr_path(root: Any, path: str) -> Any | None:
    current = root
    if not path:
        return current
    for part in path.split("."):
        if current is None or not hasattr(current, part):
            return None
        current = getattr(current, part)
    return current


def _extract_config_attn_implementation(target: Any) -> str | None:
    config = getattr(target, "config", None)
    value = getattr(config, "_attn_implementation", None)
    return str(value) if value is not None else None


def _collect_attention_backend_snapshot(config: "RealtimeServerConfig", loaded_bundle: LoadedBundle) -> dict[str, Any]:
    model = loaded_bundle.qwen3tts.model
    snapshot: dict[str, Any] = {
        "requested_attn_implementation": config.attn_implementation,
        "compile_talker": config.compile_talker,
        "compile_mode": config.compile_mode if config.compile_talker else None,
        "compile_dynamic": config.compile_dynamic if config.compile_talker else None,
        "device_map": config.device_map,
        "torch_dtype": config.torch_dtype,
        "model_class": type(model).__name__,
        "model_config_attn_implementation": _extract_config_attn_implementation(model),
        "model_talker_compile_enabled": bool(getattr(model, "_talker_compile_enabled", False)),
        "model_talker_compile_mode": getattr(model, "_talker_compile_mode", None),
        "model_talker_compile_dynamic": getattr(model, "_talker_compile_dynamic", None),
    }

    component_paths = {
        "speech_tokenizer": "speech_tokenizer",
        "speech_tokenizer_decoder": "speech_tokenizer.decoder",
        "speech_tokenizer_decoder_dit": "speech_tokenizer.decoder.dit",
        "speech_tokenizer_decoder_bigvgan": "speech_tokenizer.decoder.bigvgan",
        "speech_tokenizer_dit": "speech_tokenizer.dit",
        "speech_tokenizer_bigvgan": "speech_tokenizer.bigvgan",
    }
    for name, path in component_paths.items():
        target = _resolve_attr_path(model, path)
        if target is None:
            continue
        snapshot[f"{name}_class"] = type(target).__name__
        attn_impl = _extract_config_attn_implementation(target)
        if attn_impl is not None:
            snapshot[f"{name}_attn_implementation"] = attn_impl

    try:
        import torch

        snapshot["cuda_available"] = bool(torch.cuda.is_available())
        if hasattr(torch.backends, "cuda"):
            snapshot["sdpa_runtime_flags"] = {
                "flash_sdp_enabled": bool(torch.backends.cuda.flash_sdp_enabled()),
                "mem_efficient_sdp_enabled": bool(torch.backends.cuda.mem_efficient_sdp_enabled()),
                "math_sdp_enabled": bool(torch.backends.cuda.math_sdp_enabled()),
            }
    except Exception as exc:  # pragma: no cover - diagnostic only
        snapshot["sdpa_runtime_flags_error"] = str(exc)

    return snapshot


def _log_attention_backend_snapshot(config: "RealtimeServerConfig", loaded_bundle: LoadedBundle) -> None:
    snapshot = _collect_attention_backend_snapshot(config, loaded_bundle)
    logger.warning(
        "attention_backend_snapshot=%s",
        json.dumps(snapshot, ensure_ascii=False, sort_keys=True),
    )


@dataclass
class RealtimeServerDependencies:
    voice_registry: VoiceRegistry
    public_model_alias: str
    default_voice_alias: str
    loaded_bundle: LoadedBundle | None = None
    backend: SpeechBackend | None = None
    audio_chunk_duration_ms: int = 320
    chunk_steps: int = 4
    left_context_steps: int = 25
    first_chunk_steps: int | None = 1
    crossfade_samples: int = 0
    samples_per_step: int = 1920
    trace_timing: bool = False
    runtime_session_sync_mode: str = "chunk"
    compile_talker: bool = False
    compile_mode: str = "reduce-overhead"
    compile_dynamic: bool = True
    backend_kind: str = "native"
    backend_metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.backend is None:
            if self.loaded_bundle is None:
                raise ValueError("RealtimeServerDependencies requires either backend or loaded_bundle")
            self.backend = NativeQwenSpeechBackend(self.loaded_bundle)
        if not self.backend_kind:
            self.backend_kind = getattr(self.backend, "kind", "native")


@dataclass(frozen=True)
class StoredAudioAsset:
    audio_id: str
    media_type: str
    payload: bytes


class InMemoryAudioStore:
    def __init__(self) -> None:
        self._assets: dict[str, StoredAudioAsset] = {}

    def save(self, payload: bytes, *, media_type: str) -> StoredAudioAsset:
        audio_id = f"audio_{uuid4().hex}"
        asset = StoredAudioAsset(audio_id=audio_id, media_type=media_type, payload=payload)
        self._assets[audio_id] = asset
        return asset

    def get(self, audio_id: str) -> StoredAudioAsset:
        if audio_id not in self._assets:
            raise KeyError(audio_id)
        return self._assets[audio_id]


class TTSRequest(BaseModel):
    model: str
    text: str = Field(min_length=1)
    voice: str
    language_type: str = "Auto"
    instructions: str = ""
    optimize_instructions: bool = False
    stream: bool = False


@dataclass(frozen=True)
class RealtimeServerConfig:
    bundle_dir: Path
    backend: str = "native"
    public_model_alias: str = "qwen3-tts-flash-realtime"
    default_voice_alias: str = "default"
    voice_registry_file: Path | None = None
    host: str = "127.0.0.1"
    port: int = 9000
    ws_path: str = "/api-ws/v1/realtime"
    health_path: str = "/healthz"
    audio_chunk_duration_ms: int = 320
    chunk_steps: int = 4
    left_context_steps: int = 25
    first_chunk_steps: int | None = 1
    crossfade_samples: int = 0
    samples_per_step: int = 1920
    device_map: str = "cuda:0"
    torch_dtype: str = "bfloat16"
    attn_implementation: str = "sdpa"
    local_files_only: bool = False
    trace_timing: bool = False
    trace_attention_backend: bool = False
    runtime_session_sync_mode: str = "chunk"
    compile_talker: bool = False
    compile_mode: str = "reduce-overhead"
    compile_dynamic: bool = True
    faster_merged_cache_dir: Path | None = None
    faster_max_seq_len: int = 2048


class BundleSpeechService:
    def __init__(self, deps: RealtimeServerDependencies, *, audio_store: InMemoryAudioStore | None = None) -> None:
        self.deps = deps
        self.audio_store = audio_store or InMemoryAudioStore()
        self.step_streamer = AudioStepStreamer(
            AudioStepStreamerConfig(
                samples_per_step=self.deps.samples_per_step,
                chunk_steps=self.deps.chunk_steps,
                left_context_steps=self.deps.left_context_steps,
                crossfade_samples=self.deps.crossfade_samples,
            )
        )

    @property
    def public_model_alias(self) -> str:
        return self.deps.public_model_alias

    @property
    def default_voice_alias(self) -> str:
        return self.deps.default_voice_alias

    @property
    def sample_rate(self) -> int:
        return int(getattr(self.deps.backend, "sample_rate", 24000))

    def build_initial_session_options(self) -> SessionOptions:
        return SessionOptions(
            model=self.public_model_alias,
            voice=self.default_voice_alias,
            sample_rate=self.sample_rate,
        )

    def create_protocol_adapter(self) -> QwenRealtimeProtocolAdapter:
        return QwenRealtimeProtocolAdapter(
            voice_registry=self.deps.voice_registry,
            synthesize_callback=self.synthesize,
            stream_synthesize_callback=self.stream_synthesize,
            audio_chunker=self.iter_stream_chunks,
            audio_chunk_duration_ms=self.deps.audio_chunk_duration_ms,
        )

    def list_voices(self) -> list[dict[str, Any]]:
        return [
            {
                "voice": profile.voice_alias,
                "speaker_name": profile.speaker_name,
                "language_type": profile.language_type,
                "description": profile.description,
                "supported_models": list(profile.supported_models),
            }
            for profile in self.deps.voice_registry.list_profiles()
        ]

    def synthesize(self, session, text: str) -> SynthesizedAudio:
        profile = self.deps.voice_registry.resolve(session.options.voice, model=session.options.model)
        return self.deps.backend.synthesize(
            text=text,
            language=session.options.language_type,
            speaker=profile.speaker_name,
            instruct=session.options.instructions or None,
        )

    def build_runtime_session(
        self,
        *,
        model: str,
        voice: str,
        language_type: str,
        instructions: str = "",
        session_id: str | None = None,
    ) -> RuntimeSession:
        options = SessionOptions(
            model=model,
            voice=voice,
            language_type=language_type,
            instructions=instructions,
        )
        return RuntimeSession(session_id=session_id or f"http_{uuid4().hex}", options=options)

    def synthesize_http(self, *, text: str, model: str, voice: str, language_type: str, instructions: str = "") -> SynthesizedAudio:
        runtime_session = self.build_runtime_session(
            model=model,
            voice=voice,
            language_type=language_type,
            instructions=instructions,
        )
        return self.synthesize(runtime_session, text)

    def stream_synthesize(self, session, text: str):
        profile = self.deps.voice_registry.resolve(session.options.voice, model=session.options.model)
        stream_started_at = time.perf_counter()
        generator = self.deps.backend.stream_synthesize(
            text=text,
            language=session.options.language_type,
            speaker=profile.speaker_name,
            instruct=session.options.instructions or None,
            runtime_session=session,
            chunk_steps=self.deps.chunk_steps,
            left_context_steps=self.deps.left_context_steps,
            first_chunk_steps=self.deps.first_chunk_steps,
            crossfade_samples=self.deps.crossfade_samples,
            runtime_session_sync_mode=self.deps.runtime_session_sync_mode,
        )
        first_chunk_at: float | None = None
        total_audio_bytes = 0
        try:
            for chunk in generator:
                if first_chunk_at is None and chunk:
                    first_chunk_at = time.perf_counter()
                total_audio_bytes += len(chunk)
                yield chunk
        finally:
            if self.deps.trace_timing:
                total_elapsed_ms = (time.perf_counter() - stream_started_at) * 1000.0
                runtime_metrics = dict(session.state.last_generation_metrics)
                logger.info(
                    "stream_timing session=%s voice=%s text_chars=%s init_ms=%.2f first_chunk_ms=%s total_ms=%.2f audio_bytes=%s generated_steps=%s emitted_chunks=%s avg_forward_ms=%s avg_decode_ms=%s avg_state_sync_ms=%s metrics=%s",
                    getattr(session, "session_id", "unknown"),
                    session.options.voice,
                    len(text),
                    runtime_metrics.get("init_total_ms") or 0.0,
                    round((first_chunk_at - stream_started_at) * 1000.0, 2) if first_chunk_at is not None else None,
                    total_elapsed_ms,
                    total_audio_bytes,
                    runtime_metrics.get("generated_steps"),
                    runtime_metrics.get("emitted_chunks"),
                    runtime_metrics.get("avg_forward_ms"),
                    runtime_metrics.get("avg_decode_ms"),
                    runtime_metrics.get("avg_state_sync_ms"),
                    runtime_metrics,
                )

    def stream_synthesize_http(self, *, text: str, model: str, voice: str, language_type: str, instructions: str = ""):
        runtime_session = self.build_runtime_session(
            model=model,
            voice=voice,
            language_type=language_type,
            instructions=instructions,
        )
        yield from self.stream_synthesize(runtime_session, text)

    def store_wav_asset(self, synthesized: SynthesizedAudio) -> StoredAudioAsset:
        wav_bytes = pcm16le_bytes_to_wav_bytes(
            synthesized.audio_bytes,
            sample_rate=synthesized.sample_rate,
            channels=synthesized.channels,
        )
        return self.audio_store.save(wav_bytes, media_type="audio/wav")

    def iter_stream_chunks(self, synthesized: SynthesizedAudio) -> list[bytes]:
        return self.step_streamer.iter_audio_chunks(synthesized)


def build_voice_registry(config: RealtimeServerConfig, loaded_bundle: Any) -> VoiceRegistry:
    source_speaker_name = getattr(loaded_bundle, "speaker_name", None)
    supported_speakers = getattr(loaded_bundle, "supported_speakers", None)
    if supported_speakers is None:
        qwen3tts = getattr(loaded_bundle, "qwen3tts", None)
        model = getattr(qwen3tts, "model", None)
        supported_speakers = getattr(model, "supported_speakers", None)

    if config.voice_registry_file is not None:
        registry = VoiceRegistry.from_file(config.voice_registry_file)
    else:
        registry = VoiceRegistry.single_voice(
        voice_alias=config.default_voice_alias,
        speaker_name=source_speaker_name,
        model_alias=config.public_model_alias,
        description="Default voice derived from the loaded LoRA bundle",
        )

    if supported_speakers is not None:
        supported = {str(item) for item in supported_speakers}
        invalid_profiles = [
            profile
            for profile in registry.list_profiles()
            if profile.speaker_name not in supported
        ]
        if invalid_profiles:
            invalid_summary = ", ".join(
                f"{profile.voice_alias}->{profile.speaker_name}" for profile in invalid_profiles
            )
            raise ValueError(
                "Voice registry configured speaker names that are not available in the loaded bundle: "
                f"{invalid_summary}. Supported speakers: {sorted(supported)}"
            )

    return registry


def build_dependencies(
    config: RealtimeServerConfig,
    *,
    bundle_loader: BundleLoader | None = None,
) -> RealtimeServerDependencies:
    load_result: BackendLoadResult
    if config.backend == "native":
        load_result = load_native_backend(
            config.bundle_dir,
            device_map=config.device_map,
            torch_dtype=config.torch_dtype,
            attn_implementation=config.attn_implementation,
            local_files_only=config.local_files_only,
            compile_talker=config.compile_talker,
            compile_mode=config.compile_mode,
            compile_dynamic=config.compile_dynamic,
            bundle_loader=bundle_loader,
        )
        if config.trace_attention_backend and load_result.loaded_bundle is not None:
            _log_attention_backend_snapshot(config, load_result.loaded_bundle)
    elif config.backend == "faster":
        if config.trace_attention_backend:
            logger.warning("trace_attention_backend is not supported for backend=faster; ignoring request")
        if config.compile_talker:
            logger.warning("compile_talker is native-only and will be ignored for backend=faster")
        if config.runtime_session_sync_mode != "chunk":
            logger.warning(
                "runtime_session_sync_mode=%s is not implemented for backend=faster; each commit will synthesize from scratch",
                config.runtime_session_sync_mode,
            )
        load_result = load_faster_backend(
            config.bundle_dir,
            device=config.device_map,
            torch_dtype=config.torch_dtype,
            attn_implementation=config.attn_implementation,
            local_files_only=config.local_files_only,
            max_seq_len=config.faster_max_seq_len,
            cache_root=config.faster_merged_cache_dir,
        )
    else:
        raise ValueError(f"Unsupported backend: {config.backend}")

    voice_registry = build_voice_registry(config, load_result)
    return RealtimeServerDependencies(
        loaded_bundle=load_result.loaded_bundle,
        backend=load_result.backend,
        voice_registry=voice_registry,
        public_model_alias=config.public_model_alias,
        default_voice_alias=voice_registry.default_voice or config.default_voice_alias,
        audio_chunk_duration_ms=config.audio_chunk_duration_ms,
        chunk_steps=config.chunk_steps,
        left_context_steps=config.left_context_steps,
        first_chunk_steps=config.first_chunk_steps,
        crossfade_samples=config.crossfade_samples,
        samples_per_step=config.samples_per_step,
        trace_timing=config.trace_timing,
        runtime_session_sync_mode=config.runtime_session_sync_mode,
        compile_talker=config.compile_talker,
        compile_mode=config.compile_mode,
        compile_dynamic=config.compile_dynamic,
        backend_kind=load_result.backend.kind,
        backend_metadata=load_result.metadata,
    )


def build_protocol_adapter(deps: RealtimeServerDependencies) -> QwenRealtimeProtocolAdapter:
    """Create the Qwen-compatible protocol adapter for the currently loaded service dependencies."""
    return BundleSpeechService(deps).create_protocol_adapter()


def create_app(
    *,
    config: RealtimeServerConfig | None = None,
    service: Any = None,
):
    if service is None:
        if config is None:
            raise ValueError("Either config or service must be provided")
        service = BundleSpeechService(build_dependencies(config))

    ws_path = config.ws_path if config is not None else "/api-ws/v1/realtime"
    health_path = config.health_path if config is not None else "/healthz"

    app = FastAPI(title="Qwen-Compatible Realtime TTS MVP")

    def fallback_stream_chunks(synthesized: SynthesizedAudio) -> list[bytes]:
        chunk_size_bytes = max(1, int(synthesized.sample_rate * 0.32)) * max(1, synthesized.channels) * 2
        return [
            synthesized.audio_bytes[idx : idx + chunk_size_bytes]
            for idx in range(0, len(synthesized.audio_bytes), chunk_size_bytes)
        ]

    def build_tts_response(request_id: str, audio_id: str, audio_url: str, *, characters: int, audio_data: str = "", finish_reason: str = "stop") -> dict[str, Any]:
        return {
            "status_code": 200,
            "request_id": request_id,
            "code": "",
            "message": "",
            "output": {
                "text": None,
                "finish_reason": finish_reason,
                "choices": None,
                "audio": {
                    "data": audio_data,
                    "url": audio_url,
                    "id": audio_id,
                },
            },
            "usage": {
                "characters": characters,
            },
        }

    @app.get(health_path)
    async def health() -> dict[str, Any]:
        return {
            "status": "ok",
            "model": service.public_model_alias,
            "default_voice": service.default_voice_alias,
        }

    @app.get("/v1/voices")
    async def list_voices() -> dict[str, Any]:
        return {
            "default_voice": service.default_voice_alias,
            "data": service.list_voices(),
        }

    @app.get("/v1/audio/{audio_id}")
    async def get_audio(audio_id: str) -> Response:
        try:
            asset = service.audio_store.get(audio_id)
        except KeyError as exc:
            raise HTTPException(status_code=404, detail=f"Unknown audio id: {audio_id}") from exc
        return Response(content=asset.payload, media_type=asset.media_type)

    @app.post("/v1/tts")
    @app.post("/v1/audio/speech")
    @app.post("/api/v1/services/aigc/multimodal-generation/generation")
    async def tts(request: Request, payload: TTSRequest):
        request_id = f"req_{uuid4().hex}"

        if not payload.stream:
            synthesized = service.synthesize_http(
                text=payload.text,
                model=payload.model,
                voice=payload.voice,
                language_type=payload.language_type,
                instructions=payload.instructions,
            )
            asset = service.store_wav_asset(synthesized)
            audio_url = str(request.url_for("get_audio", audio_id=asset.audio_id))
            return build_tts_response(
                request_id,
                asset.audio_id,
                audio_url,
                characters=len(payload.text),
            )

        async def stream_generator():
            if hasattr(service, "stream_synthesize_http"):
                chunk_iterable = service.stream_synthesize_http(
                    text=payload.text,
                    model=payload.model,
                    voice=payload.voice,
                    language_type=payload.language_type,
                    instructions=payload.instructions,
                )
                stream_sample_rate = int(getattr(service, "sample_rate", 24000))
            elif hasattr(service, "iter_stream_chunks"):
                synthesized = service.synthesize_http(
                    text=payload.text,
                    model=payload.model,
                    voice=payload.voice,
                    language_type=payload.language_type,
                    instructions=payload.instructions,
                )
                chunk_iterable = service.iter_stream_chunks(synthesized)
                stream_sample_rate = synthesized.sample_rate
            else:
                synthesized = service.synthesize_http(
                    text=payload.text,
                    model=payload.model,
                    voice=payload.voice,
                    language_type=payload.language_type,
                    instructions=payload.instructions,
                )
                chunk_iterable = fallback_stream_chunks(synthesized)
                stream_sample_rate = synthesized.sample_rate

            collected_audio = bytearray()

            for chunk in chunk_iterable:
                collected_audio.extend(chunk)
                yield json.dumps(
                    build_tts_response(
                        request_id,
                        "",
                        "",
                        characters=len(payload.text),
                        audio_data=base64.b64encode(chunk).decode("ascii"),
                        finish_reason="null",
                    ),
                    ensure_ascii=False,
                ) + "\n"
                await asyncio.sleep(0)

            asset = service.audio_store.save(
                pcm16le_bytes_to_wav_bytes(bytes(collected_audio), sample_rate=stream_sample_rate),
                media_type="audio/wav",
            )
            audio_url = str(request.url_for("get_audio", audio_id=asset.audio_id))
            yield json.dumps(
                build_tts_response(
                    request_id,
                    asset.audio_id,
                    audio_url,
                    characters=len(payload.text),
                ),
                ensure_ascii=False,
            ) + "\n"
            await asyncio.sleep(0)

        return StreamingResponse(stream_generator(), media_type="application/x-ndjson")

    @app.websocket(ws_path)
    async def realtime_endpoint(websocket: WebSocket) -> None:
        await websocket.accept()
        adapter = service.create_protocol_adapter()
        for event in adapter.open_connection(service.build_initial_session_options()):
            await websocket.send_json(event)
            await asyncio.sleep(0)

        try:
            while True:
                event = await websocket.receive_json()
                for response_event in adapter.iter_events(event):
                    await websocket.send_json(response_event)
                    await asyncio.sleep(0)
        except WebSocketDisconnect:
            return

    return app


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Serve a Qwen-compatible realtime TTS MVP backed by a default LoRA bundle")
    parser.add_argument("--bundle_dir", required=True)
    parser.add_argument("--backend", default="native", choices=("native", "faster"))
    parser.add_argument("--public_model_alias", default="qwen3-tts-flash-realtime")
    parser.add_argument("--default_voice_alias", default="default")
    parser.add_argument("--voice_registry_file", default=None)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=9000)
    parser.add_argument("--ws_path", default="/api-ws/v1/realtime")
    parser.add_argument("--health_path", default="/healthz")
    parser.add_argument("--audio_chunk_duration_ms", type=int, default=320)
    parser.add_argument("--chunk_steps", type=int, default=4)
    parser.add_argument("--left_context_steps", type=int, default=25)
    parser.add_argument("--first_chunk_steps", type=int, default=1)
    parser.add_argument("--crossfade_samples", type=int, default=0)
    parser.add_argument("--samples_per_step", type=int, default=1920)
    parser.add_argument("--device_map", default="cuda:0")
    parser.add_argument("--torch_dtype", default="bfloat16")
    parser.add_argument("--attn_implementation", default="sdpa")
    parser.add_argument("--local_files_only", action="store_true")
    parser.add_argument("--trace_timing", action="store_true")
    parser.add_argument("--trace_attention_backend", action="store_true")
    parser.add_argument(
        "--runtime_session_sync_mode",
        choices=("step", "chunk", "final"),
        default="chunk",
        help="How often to bind RuntimeSession generation state during streaming.",
    )
    parser.add_argument("--compile_talker", action="store_true", help="Experimental: wrap qwen3tts.model.talker with torch.compile.")
    parser.add_argument(
        "--compile_mode",
        default="reduce-overhead",
        choices=("default", "reduce-overhead", "max-autotune"),
        help="torch.compile mode used when --compile_talker is enabled.",
    )
    parser.add_argument(
        "--compile_dynamic",
        action="store_true",
        help="Enable dynamic shape support when compiling the talker module.",
    )
    parser.add_argument(
        "--faster_merged_cache_dir",
        default=None,
        help="Optional cache root for bundle->merged model exports used by backend=faster.",
    )
    parser.add_argument(
        "--faster_max_seq_len",
        type=int,
        default=2048,
        help="Static cache sequence length passed to faster-qwen3-tts when backend=faster.",
    )
    return parser


def main() -> None:
    import uvicorn

    args = build_parser().parse_args()
    config = RealtimeServerConfig(
        bundle_dir=Path(args.bundle_dir),
        backend=args.backend,
        public_model_alias=args.public_model_alias,
        default_voice_alias=args.default_voice_alias,
        voice_registry_file=Path(args.voice_registry_file) if args.voice_registry_file else None,
        host=args.host,
        port=args.port,
        ws_path=args.ws_path,
        health_path=args.health_path,
        audio_chunk_duration_ms=args.audio_chunk_duration_ms,
        chunk_steps=args.chunk_steps,
        left_context_steps=args.left_context_steps,
        first_chunk_steps=args.first_chunk_steps,
        crossfade_samples=args.crossfade_samples,
        samples_per_step=args.samples_per_step,
        device_map=args.device_map,
        torch_dtype=args.torch_dtype,
        attn_implementation=args.attn_implementation,
        local_files_only=args.local_files_only,
        trace_timing=args.trace_timing,
        trace_attention_backend=args.trace_attention_backend,
        runtime_session_sync_mode=args.runtime_session_sync_mode,
        compile_talker=args.compile_talker,
        compile_mode=args.compile_mode,
        compile_dynamic=args.compile_dynamic,
        faster_merged_cache_dir=Path(args.faster_merged_cache_dir) if args.faster_merged_cache_dir else None,
        faster_max_seq_len=args.faster_max_seq_len,
    )
    logging.basicConfig(level=logging.INFO if config.trace_timing else logging.WARNING)
    app = create_app(config=config)
    uvicorn.run(app, host=config.host, port=config.port)


if __name__ == "__main__":
    main()
