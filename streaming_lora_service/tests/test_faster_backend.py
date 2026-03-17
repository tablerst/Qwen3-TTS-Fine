from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import torch

from lora_finetuning.common import build_bundle_manifest, make_config_patch, save_json, save_speaker_patch
from streaming_lora_service.app.backend import BackendCapabilities, BackendLoadResult
from streaming_lora_service.app.backend.faster_qwen import FasterQwenSpeechBackend, load_faster_backend
from streaming_lora_service.app.models import SessionOptions, SynthesizedAudio
from streaming_lora_service.app.runtime_session import RuntimeSession
from streaming_lora_service.app.server import RealtimeServerConfig, build_dependencies, build_parser


class FakeFastInnerModel:
    supported_speakers = ["inference_speaker"]

    @staticmethod
    def get_supported_speakers():
        return ["inference_speaker"]


class FakeFasterQwen3TTS:
    load_calls: list[tuple[tuple[object, ...], dict[str, object]]] = []

    def __init__(self) -> None:
        self.model = FakeFastInnerModel()
        self.generate_calls: list[dict[str, object]] = []
        self.sample_rate = 24000

    @classmethod
    def from_pretrained(cls, *args, **kwargs) -> "FakeFasterQwen3TTS":
        instance = cls()
        cls.load_calls.append((args, kwargs))
        return instance

    def generate_custom_voice(self, **kwargs):
        self.generate_calls.append(kwargs)
        return [np.linspace(-0.1, 0.1, num=160, dtype=np.float32)], 24000

    def generate_custom_voice_streaming(self, **kwargs):
        self.generate_calls.append({"streaming": True, **kwargs})
        yield np.full(80, 0.1, dtype=np.float32), 24000, {
            "chunk_index": 0,
            "chunk_steps": 8,
            "prefill_ms": 11.5,
            "decode_ms": 3.0,
            "total_steps_so_far": 8,
            "is_final": False,
        }
        yield np.full(80, 0.2, dtype=np.float32), 24000, {
            "chunk_index": 1,
            "chunk_steps": 4,
            "prefill_ms": 0.0,
            "decode_ms": 2.0,
            "total_steps_so_far": 12,
            "is_final": True,
        }


class FakeBackend:
    kind = "faster"
    sample_rate = 24000
    capabilities = BackendCapabilities()

    def list_supported_speakers(self) -> tuple[str, ...]:
        return ("inference_speaker",)

    def synthesize(self, *, text: str, speaker: str, language: str = "Auto", instruct: str | None = None) -> SynthesizedAudio:
        del text, speaker, language, instruct
        return SynthesizedAudio(audio_bytes=b"\x00\x00", sample_rate=24000)

    def stream_synthesize(self, **kwargs):
        del kwargs
        yield b"chunk"


class FasterBackendTests(unittest.TestCase):
    def make_temp_dir(self) -> Path:
        temp_dir = tempfile.TemporaryDirectory()
        self.addCleanup(temp_dir.cleanup)
        return Path(temp_dir.name)

    def build_bundle(self) -> Path:
        bundle_dir = self.make_temp_dir() / "bundle"
        adapter_dir = bundle_dir / "adapter"
        adapter_dir.mkdir(parents=True, exist_ok=True)
        (adapter_dir / "adapter_model.safetensors").write_bytes(b"stub")
        (adapter_dir / "adapter_config.json").write_text("{}", encoding="utf-8")
        save_json(make_config_patch("inference_speaker", 3000), bundle_dir / "config_patch.json")
        save_speaker_patch(bundle_dir / "speaker_embedding.safetensors", 3000, torch.randn(8))
        save_json(
            build_bundle_manifest(
                base_model_path="models/Qwen3-TTS-12Hz-1.7B-Base",
                speaker_name="inference_speaker",
                speaker_id=3000,
            ),
            bundle_dir / "manifest.json",
        )
        return bundle_dir

    @staticmethod
    def write_minimal_exported_model(output_dir: str | Path) -> dict[str, object]:
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        config = {
            "tts_model_type": "custom_voice",
            "talker_config": {
                "spk_id": {"inference_speaker": 3000},
                "spk_is_dialect": {"inference_speaker": False},
            },
        }
        (output_path / "config.json").write_text(json.dumps(config), encoding="utf-8")
        (output_path / "model.safetensors").write_bytes(b"stub")
        (output_path / "tokenizer_config.json").write_text("{}", encoding="utf-8")
        return {
            "output_dir": str(output_path),
            "resolved_speaker_name": "inference_speaker",
        }

    def test_load_faster_backend_exports_once_and_then_reuses_cache(self) -> None:
        bundle_dir = self.build_bundle()
        cache_root = self.make_temp_dir() / "cache"
        merge_calls: list[Path] = []

        def fake_merge_bundle_to_local_model(*, output_dir: str | Path, **kwargs):
            del kwargs
            merge_calls.append(Path(output_dir))
            return self.write_minimal_exported_model(output_dir)

        with patch(
            "streaming_lora_service.app.backend.faster_qwen.merge_bundle_to_local_model",
            side_effect=fake_merge_bundle_to_local_model,
        ):
            first = load_faster_backend(
                bundle_dir,
                device="cpu",
                torch_dtype="float32",
                cache_root=cache_root,
                faster_model_cls=FakeFasterQwen3TTS,
            )
            second = load_faster_backend(
                bundle_dir,
                device="cpu",
                torch_dtype="float32",
                cache_root=cache_root,
                faster_model_cls=FakeFasterQwen3TTS,
            )

        self.assertEqual(len(merge_calls), 1)
        self.assertEqual(first.metadata["cache_hit"], False)
        self.assertEqual(second.metadata["cache_hit"], True)
        self.assertEqual(first.metadata["merged_model_dir"], second.metadata["merged_model_dir"])
        self.assertEqual(first.speaker_name, "inference_speaker")
        self.assertEqual(first.supported_speakers, ("inference_speaker",))

    def test_faster_backend_stream_synthesize_yields_pcm_and_updates_metrics(self) -> None:
        backend = FasterQwenSpeechBackend(
            FakeFasterQwen3TTS(),
            supported_speakers=("inference_speaker",),
            sample_rate=24000,
        )
        runtime_session = RuntimeSession(
            "sess_test",
            SessionOptions(model="qwen3-tts-flash-realtime", voice="default"),
        )

        chunks = list(
            backend.stream_synthesize(
                text="你好，faster backend。",
                speaker="INFERENCE_SPEAKER",
                language="Chinese",
                runtime_session=runtime_session,
                chunk_steps=8,
            )
        )

        self.assertEqual(len(chunks), 2)
        self.assertTrue(all(isinstance(chunk, bytes) and chunk for chunk in chunks))
        self.assertEqual(runtime_session.state.last_generation_metrics["backend"], "faster")
        self.assertEqual(runtime_session.state.last_generation_metrics["emitted_chunks"], 2)
        self.assertEqual(runtime_session.state.last_generation_metrics["generated_steps"], 12)
        self.assertEqual(runtime_session.state.last_generation_metrics["finish_reason"], "completed")
        self.assertFalse(runtime_session.state.last_generation_metrics["supports_state_resume"])

    def test_build_dependencies_accepts_faster_backend_load_result(self) -> None:
        config = RealtimeServerConfig(bundle_dir=self.make_temp_dir(), backend="faster")
        load_result = BackendLoadResult(
            backend=FakeBackend(),
            speaker_name="inference_speaker",
            supported_speakers=("inference_speaker",),
            tts_model_type="custom_voice",
            metadata={"backend": "faster"},
        )

        with patch("streaming_lora_service.app.server.load_faster_backend", return_value=load_result):
            deps = build_dependencies(config)

        self.assertEqual(deps.backend_kind, "faster")
        self.assertEqual(deps.backend_metadata["backend"], "faster")
        self.assertEqual(deps.voice_registry.default_voice, "default")
        self.assertEqual(deps.voice_registry.resolve("default", model=config.public_model_alias).speaker_name, "inference_speaker")

    def test_server_defaults_now_point_to_faster_backend(self) -> None:
        config = RealtimeServerConfig(bundle_dir=self.make_temp_dir())
        parser = build_parser()
        args = parser.parse_args(["--bundle_dir", "/tmp/demo_bundle"])

        self.assertEqual(config.backend, "faster")
        self.assertEqual(args.backend, "faster")


if __name__ == "__main__":
    unittest.main()
