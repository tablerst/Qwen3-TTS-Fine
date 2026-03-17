from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import torch
from torch import nn

from faster_qwen3_tts_integration.contracts import (
    ContractError,
    infer_speaker_name,
    validate_exported_model_dir,
)
from faster_qwen3_tts_integration.scripts.export_merged_model import (
    SUMMARY_FILENAME as EXPORT_SUMMARY_FILENAME,
    merge_bundle_to_local_model,
)
from faster_qwen3_tts_integration.scripts.smoke_test import run_smoke_test, run_streaming_smoke_test
from streaming_lora_service.app.models import BundleArtifacts


class FakeProcessor:
    def save_pretrained(self, output_dir: str) -> None:
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        (output_path / "tokenizer_config.json").write_text("{}", encoding="utf-8")


class FakeMergedModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.dummy_weight = nn.Parameter(torch.zeros(1))
        self.config = SimpleNamespace(
            tts_model_type="custom_voice",
            talker_config=SimpleNamespace(spk_id={"smoke_speaker": 3000}, spk_is_dialect={}),
        )
        self.tts_model_type = "custom_voice"
        self.eval_called = False
        self.merge_called = False

    def eval(self) -> "FakeMergedModel":
        self.eval_called = True
        return self

    def merge(self) -> None:
        self.merge_called = True

    @property
    def merged(self) -> bool:
        return self.merge_called


class FakeQwen3TTSModel:
    def __init__(self) -> None:
        self.model = FakeMergedModel()
        self.processor = FakeProcessor()
        self.load_args: tuple[object, ...] = ()
        self.load_kwargs: dict[str, object] = {}

    @classmethod
    def from_pretrained(cls, *args, **kwargs) -> "FakeQwen3TTSModel":
        instance = cls()
        instance.load_args = args
        instance.load_kwargs = kwargs
        return instance


class FakeFastInnerModel:
    def get_supported_speakers(self):
        return ["smoke_speaker"]


class FakeFasterQwen3TTS:
    last_instance: "FakeFasterQwen3TTS | None" = None

    def __init__(self) -> None:
        self.model = FakeFastInnerModel()
        self.generate_calls: list[dict[str, object]] = []
        self.load_args: tuple[object, ...] = ()
        self.load_kwargs: dict[str, object] = {}

    @classmethod
    def from_pretrained(cls, *args, **kwargs) -> "FakeFasterQwen3TTS":
        instance = cls()
        instance.load_args = args
        instance.load_kwargs = kwargs
        cls.last_instance = instance
        return instance

    def generate_custom_voice(self, **kwargs):
        self.generate_calls.append(kwargs)
        return [np.zeros(320, dtype=np.float32)], 24000

    def generate_custom_voice_streaming(self, **kwargs):
        self.generate_calls.append({"streaming": True, **kwargs})
        yield np.zeros(160, dtype=np.float32), 24000, {
            "chunk_index": 0,
            "chunk_steps": 8,
            "prefill_ms": 12.5,
            "decode_ms": 3.2,
            "total_steps_so_far": 8,
            "is_final": False,
        }
        yield np.zeros(160, dtype=np.float32), 24000, {
            "chunk_index": 1,
            "chunk_steps": 4,
            "prefill_ms": 0.0,
            "decode_ms": 1.8,
            "total_steps_so_far": 12,
            "is_final": True,
        }


class FasterIntegrationContractTests(unittest.TestCase):
    def make_temp_dir(self) -> Path:
        temp_dir = tempfile.TemporaryDirectory()
        self.addCleanup(temp_dir.cleanup)
        return Path(temp_dir.name)

    def make_minimal_exported_model_dir(self, root: Path, *, speakers: dict[str, int] | None = None) -> Path:
        speakers = speakers or {"smoke_speaker": 3000}
        root.mkdir(parents=True, exist_ok=True)
        config = {
            "tts_model_type": "custom_voice",
            "talker_config": {
                "spk_id": speakers,
                "spk_is_dialect": {name: False for name in speakers},
            },
        }
        (root / "config.json").write_text(json.dumps(config), encoding="utf-8")
        (root / "model.safetensors").write_bytes(b"stub")
        (root / "tokenizer_config.json").write_text("{}", encoding="utf-8")
        return root

    def test_validate_exported_model_dir_accepts_minimal_custom_voice_layout(self) -> None:
        model_dir = self.make_minimal_exported_model_dir(self.make_temp_dir() / "merged_model")

        summary = validate_exported_model_dir(model_dir)

        self.assertEqual(summary["tts_model_type"], "custom_voice")
        self.assertEqual(summary["resolved_speaker"], "smoke_speaker")
        self.assertEqual(summary["speaker_count"], 1)

    def test_infer_speaker_name_requires_explicit_choice_for_multi_speaker_model(self) -> None:
        model_dir = self.make_minimal_exported_model_dir(
            self.make_temp_dir() / "merged_model",
            speakers={"alpha": 3000, "beta": 3001},
        )

        with self.assertRaisesRegex(ContractError, "multiple speakers"):
            infer_speaker_name(model_dir)

        self.assertEqual(infer_speaker_name(model_dir, explicit_speaker="beta", require_single=False), "beta")

    def test_merge_bundle_to_local_model_writes_summary_and_manifest(self) -> None:
        work_dir = self.make_temp_dir()
        bundle_dir = work_dir / "bundle"
        adapter_dir = bundle_dir / "adapter"
        adapter_dir.mkdir(parents=True, exist_ok=True)
        config_patch_file = bundle_dir / "config_patch.json"
        speaker_patch_file = bundle_dir / "speaker_embedding.safetensors"
        output_dir = work_dir / "merged_output"

        config_patch = {
            "tts_model_type": "custom_voice",
            "talker_config": {
                "spk_id": {"smoke_speaker": 3000},
                "spk_is_dialect": {"smoke_speaker": False},
            },
        }
        config_patch_file.write_text(json.dumps(config_patch), encoding="utf-8")
        speaker_patch_file.write_bytes(b"stub")

        artifacts = BundleArtifacts(
            bundle_dir=bundle_dir,
            base_model="models/Qwen3-TTS-12Hz-1.7B-Base",
            adapter_dir=adapter_dir,
            config_patch_file=config_patch_file,
            speaker_patch_file=speaker_patch_file,
            speaker_name="smoke_speaker",
            manifest={"base_model_path": "models/Qwen3-TTS-12Hz-1.7B-Base", "speaker_name": "smoke_speaker"},
        )

        with patch(
            "faster_qwen3_tts_integration.scripts.export_merged_model.resolve_bundle_artifacts",
            return_value=artifacts,
        ), patch(
            "faster_qwen3_tts_integration.scripts.export_merged_model.load_lora_adapter",
            return_value=None,
        ), patch(
            "faster_qwen3_tts_integration.scripts.export_merged_model.apply_speaker_patch",
            return_value=3000,
        ):
            summary = merge_bundle_to_local_model(
                bundle_dir=bundle_dir,
                output_dir=output_dir,
                model_factory=FakeQwen3TTSModel,
            )

        self.assertEqual(summary["resolved_speaker_name"], "smoke_speaker")
        self.assertGreater(summary["merged_layer_count"], 0)
        self.assertTrue((output_dir / EXPORT_SUMMARY_FILENAME).exists())
        self.assertTrue((output_dir / "source_bundle_manifest.json").exists())
        self.assertTrue((output_dir / "model.safetensors").exists())
        self.assertTrue((output_dir / "tokenizer_config.json").exists())

    def test_run_smoke_test_writes_audio_and_summary(self) -> None:
        work_dir = self.make_temp_dir()
        model_dir = self.make_minimal_exported_model_dir(work_dir / "merged_model")
        output_wav = work_dir / "samples" / "smoke.wav"
        audio_writes: list[tuple[str, int, int]] = []

        def fake_audio_writer(path: str, audio: np.ndarray, sample_rate: int) -> None:
            audio_writes.append((path, int(audio.shape[0]), sample_rate))

        summary = run_smoke_test(
            model_dir=model_dir,
            text="最小烟雾测试。",
            output_wav=output_wav,
            speaker="smoke_speaker",
            faster_model_cls=FakeFasterQwen3TTS,
            audio_writer=fake_audio_writer,
        )

        self.assertEqual(summary["speaker"], "smoke_speaker")
        self.assertEqual(summary["sample_rate"], 24000)
        self.assertEqual(summary["num_audio_samples"], 320)
        self.assertEqual(len(audio_writes), 1)
        self.assertTrue(Path(summary["summary_json"]).exists())
        self.assertIsNotNone(FakeFasterQwen3TTS.last_instance)
        assert FakeFasterQwen3TTS.last_instance is not None
        self.assertEqual(FakeFasterQwen3TTS.last_instance.generate_calls[0]["speaker"], "smoke_speaker")

    def test_run_streaming_smoke_test_records_ttft_and_rtf(self) -> None:
        work_dir = self.make_temp_dir()
        model_dir = self.make_minimal_exported_model_dir(work_dir / "merged_model")
        output_wav = work_dir / "samples" / "streaming_smoke.wav"
        audio_writes: list[tuple[str, int, int]] = []

        def fake_audio_writer(path: str, audio: np.ndarray, sample_rate: int) -> None:
            audio_writes.append((path, int(audio.shape[0]), sample_rate))

        summary = run_streaming_smoke_test(
            model_dir=model_dir,
            text="流式烟雾测试。",
            output_wav=output_wav,
            speaker="smoke_speaker",
            faster_model_cls=FakeFasterQwen3TTS,
            audio_writer=fake_audio_writer,
            warmup_max_new_tokens=4,
            chunk_size=8,
        )

        self.assertEqual(summary["mode"], "streaming")
        self.assertGreaterEqual(summary["ttft_ms"], 0.0)
        self.assertGreater(summary["rtf"], 0.0)
        self.assertEqual(summary["chunk_count"], 2)
        self.assertEqual(summary["num_audio_samples"], 320)
        self.assertEqual(len(audio_writes), 1)
        self.assertTrue(Path(summary["summary_json"]).exists())


if __name__ == "__main__":
    unittest.main()
