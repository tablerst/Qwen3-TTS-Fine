from __future__ import annotations

from typing import Iterable

from ..bundle_loader import BundleLoader
from ..models import LoadedBundle, SynthesizedAudio
from ..runtime_session import RuntimeSession
from ..step_generator import generate_custom_voice_step_aware
from ..streaming_generator import StreamingCustomVoiceGenerator
from .base import BackendCapabilities, BackendLoadResult


class NativeQwenSpeechBackend:
    kind = "native"
    capabilities = BackendCapabilities(
        supports_state_resume=True,
        supports_runtime_session_sync_mode=True,
        supports_compile_talker=True,
        supports_trace_attention_backend=True,
    )

    def __init__(self, loaded_bundle: LoadedBundle) -> None:
        self.loaded_bundle = loaded_bundle
        self.qwen3tts = loaded_bundle.qwen3tts
        self.sample_rate = self._infer_sample_rate()

    def _infer_sample_rate(self) -> int:
        speech_tokenizer = getattr(self.qwen3tts.model, "speech_tokenizer", None)
        getter = getattr(speech_tokenizer, "get_output_sample_rate", None)
        if callable(getter):
            sample_rate = getter()
            if isinstance(sample_rate, (int, float, str)):
                return int(sample_rate)
        return 24000

    def list_supported_speakers(self) -> tuple[str, ...]:
        raw = getattr(self.qwen3tts.model, "supported_speakers", None) or [self.loaded_bundle.speaker_name]
        return tuple(str(item) for item in raw)

    def synthesize(
        self,
        *,
        text: str,
        speaker: str,
        language: str = "Auto",
        instruct: str | None = None,
    ) -> SynthesizedAudio:
        return generate_custom_voice_step_aware(
            self.qwen3tts,
            text=text,
            language=language,
            speaker=speaker,
            instruct=instruct,
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
        generator = StreamingCustomVoiceGenerator(
            self.qwen3tts,
            text=text,
            language=language,
            speaker=speaker,
            instruct=instruct,
            runtime_session=runtime_session,
            chunk_steps=chunk_steps,
            left_context_steps=left_context_steps,
            first_chunk_steps=first_chunk_steps,
            crossfade_samples=crossfade_samples,
            runtime_session_sync_mode=runtime_session_sync_mode,
        )
        for chunk in generator.iter_audio_chunks():
            if chunk:
                yield chunk


def load_native_backend(
    bundle_dir: str,
    *,
    base_model: str | None = None,
    adapter_dir: str | None = None,
    config_patch_file: str | None = None,
    speaker_patch_file: str | None = None,
    speaker_name: str | None = None,
    device_map: str = "cuda:0",
    torch_dtype: str = "bfloat16",
    attn_implementation: str = "sdpa",
    local_files_only: bool = False,
    compile_talker: bool = False,
    compile_mode: str = "reduce-overhead",
    compile_dynamic: bool = True,
    bundle_loader: BundleLoader | None = None,
) -> BackendLoadResult:
    loader = bundle_loader or BundleLoader()
    loaded_bundle = loader.load(
        bundle_dir,
        base_model=base_model,
        adapter_dir=adapter_dir,
        config_patch_file=config_patch_file,
        speaker_patch_file=speaker_patch_file,
        speaker_name=speaker_name,
        device_map=device_map,
        torch_dtype=torch_dtype,
        attn_implementation=attn_implementation,
        local_files_only=local_files_only,
        compile_talker=compile_talker,
        compile_mode=compile_mode,
        compile_dynamic=compile_dynamic,
    )
    backend = NativeQwenSpeechBackend(loaded_bundle)
    return BackendLoadResult(
        backend=backend,
        speaker_name=loaded_bundle.speaker_name,
        supported_speakers=backend.list_supported_speakers(),
        tts_model_type=loaded_bundle.tts_model_type,
        loaded_bundle=loaded_bundle,
        bundle_artifacts=loaded_bundle.artifacts,
        metadata={
            "backend": backend.kind,
            "speaker_id": loaded_bundle.speaker_id,
            "compile_talker": bool(getattr(loaded_bundle.qwen3tts.model, "_talker_compile_enabled", False)),
        },
    )
