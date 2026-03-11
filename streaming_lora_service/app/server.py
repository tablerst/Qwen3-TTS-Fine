from __future__ import annotations

import argparse
import base64
from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any
from uuid import uuid4

from fastapi import FastAPI, HTTPException, Request, WebSocket, WebSocketDisconnect
from fastapi.responses import Response, StreamingResponse
from pydantic import BaseModel, Field

from .audio_utils import float_audio_to_pcm16le_bytes, pcm16le_bytes_to_wav_bytes
from .bundle_loader import BundleLoader
from .models import LoadedBundle, SessionOptions, SynthesizedAudio
from .qwen_compat_ws import QwenRealtimeProtocolAdapter
from .runtime_session import RuntimeSession
from .streaming_generator import StreamingCustomVoiceGenerator
from .step_generator import generate_custom_voice_step_aware
from .step_streamer import AudioStepStreamer, AudioStepStreamerConfig
from .voice_registry import VoiceRegistry


@dataclass
class RealtimeServerDependencies:
    loaded_bundle: LoadedBundle
    voice_registry: VoiceRegistry
    public_model_alias: str
    default_voice_alias: str
    audio_chunk_duration_ms: int = 320
    chunk_steps: int = 4
    left_context_steps: int = 25
    samples_per_step: int = 1920


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
    samples_per_step: int = 1920
    device_map: str = "cuda:0"
    torch_dtype: str = "bfloat16"
    attn_implementation: str = "sdpa"
    local_files_only: bool = False


class BundleSpeechService:
    def __init__(self, deps: RealtimeServerDependencies, *, audio_store: InMemoryAudioStore | None = None) -> None:
        self.deps = deps
        self.audio_store = audio_store or InMemoryAudioStore()
        self.step_streamer = AudioStepStreamer(
            AudioStepStreamerConfig(
                samples_per_step=self.deps.samples_per_step,
                chunk_steps=self.deps.chunk_steps,
                left_context_steps=self.deps.left_context_steps,
            )
        )

    @property
    def public_model_alias(self) -> str:
        return self.deps.public_model_alias

    @property
    def default_voice_alias(self) -> str:
        return self.deps.default_voice_alias

    def build_initial_session_options(self) -> SessionOptions:
        return SessionOptions(
            model=self.public_model_alias,
            voice=self.default_voice_alias,
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
        return generate_custom_voice_step_aware(
            self.deps.loaded_bundle.qwen3tts,
            text=text,
            language=session.options.language_type,
            speaker=profile.speaker_name,
            instruct=session.options.instructions or None,
        )

    def synthesize_http(self, *, text: str, model: str, voice: str, language_type: str, instructions: str = "") -> SynthesizedAudio:
        options = SessionOptions(
            model=model,
            voice=voice,
            language_type=language_type,
            instructions=instructions,
        )
        runtime_session = RuntimeSession(session_id=f"http_{uuid4().hex}", options=options)
        return self.synthesize(runtime_session, text)

    def stream_synthesize(self, session, text: str):
        profile = self.deps.voice_registry.resolve(session.options.voice, model=session.options.model)
        generator = StreamingCustomVoiceGenerator(
            self.deps.loaded_bundle.qwen3tts,
            text=text,
            language=session.options.language_type,
            speaker=profile.speaker_name,
            instruct=session.options.instructions or None,
            runtime_session=session,
            chunk_steps=self.deps.chunk_steps,
            left_context_steps=self.deps.left_context_steps,
        )
        yield from generator.iter_audio_chunks()

    def stream_synthesize_http(self, *, text: str, model: str, voice: str, language_type: str, instructions: str = ""):
        options = SessionOptions(
            model=model,
            voice=voice,
            language_type=language_type,
            instructions=instructions,
        )
        runtime_session = RuntimeSession(session_id=f"http_{uuid4().hex}", options=options)
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


def build_voice_registry(config: RealtimeServerConfig, loaded_bundle: LoadedBundle) -> VoiceRegistry:
    if config.voice_registry_file is not None:
        registry = VoiceRegistry.from_file(config.voice_registry_file)
    else:
        registry = VoiceRegistry.single_voice(
        voice_alias=config.default_voice_alias,
        speaker_name=loaded_bundle.speaker_name,
        model_alias=config.public_model_alias,
        description="Default voice derived from the loaded LoRA bundle",
        )

    supported_speakers = getattr(loaded_bundle.qwen3tts.model, "supported_speakers", None)
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
    loader = bundle_loader or BundleLoader()
    loaded_bundle = loader.load(
        config.bundle_dir,
        device_map=config.device_map,
        torch_dtype=config.torch_dtype,
        attn_implementation=config.attn_implementation,
        local_files_only=config.local_files_only,
    )
    voice_registry = build_voice_registry(config, loaded_bundle)
    return RealtimeServerDependencies(
        loaded_bundle=loaded_bundle,
        voice_registry=voice_registry,
        public_model_alias=config.public_model_alias,
        default_voice_alias=voice_registry.default_voice or config.default_voice_alias,
        audio_chunk_duration_ms=config.audio_chunk_duration_ms,
        chunk_steps=config.chunk_steps,
        left_context_steps=config.left_context_steps,
        samples_per_step=config.samples_per_step,
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
                stream_sample_rate = 24000
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

        return StreamingResponse(stream_generator(), media_type="application/x-ndjson")

    @app.websocket(ws_path)
    async def realtime_endpoint(websocket: WebSocket) -> None:
        await websocket.accept()
        adapter = service.create_protocol_adapter()
        for event in adapter.open_connection(service.build_initial_session_options()):
            await websocket.send_json(event)

        try:
            while True:
                event = await websocket.receive_json()
                for response_event in adapter.iter_events(event):
                    await websocket.send_json(response_event)
        except WebSocketDisconnect:
            return

    return app


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Serve a Qwen-compatible realtime TTS MVP backed by a default LoRA bundle")
    parser.add_argument("--bundle_dir", required=True)
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
    parser.add_argument("--samples_per_step", type=int, default=1920)
    parser.add_argument("--device_map", default="cuda:0")
    parser.add_argument("--torch_dtype", default="bfloat16")
    parser.add_argument("--attn_implementation", default="sdpa")
    parser.add_argument("--local_files_only", action="store_true")
    return parser


def main() -> None:
    import uvicorn

    args = build_parser().parse_args()
    config = RealtimeServerConfig(
        bundle_dir=Path(args.bundle_dir),
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
        samples_per_step=args.samples_per_step,
        device_map=args.device_map,
        torch_dtype=args.torch_dtype,
        attn_implementation=args.attn_implementation,
        local_files_only=args.local_files_only,
    )
    app = create_app(config=config)
    uvicorn.run(app, host=config.host, port=config.port)


if __name__ == "__main__":
    main()
