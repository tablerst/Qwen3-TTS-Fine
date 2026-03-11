from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import torch

from lora_finetuning.common import build_bundle_manifest, make_config_patch, save_json, save_speaker_patch
from streaming_lora_service.app.bundle_loader import BundleLoader, resolve_bundle_artifacts


class FakeInferenceModel:
    def __init__(self) -> None:
        self.config = SimpleNamespace(
            tts_model_type="base",
            talker_config=SimpleNamespace(spk_id={}, spk_is_dialect={}),
        )
        self.tts_model_type = "base"
        self.talker = SimpleNamespace(
            model=SimpleNamespace(
                codec_embedding=SimpleNamespace(weight=torch.zeros(4001, 8, dtype=torch.float32))
            )
        )
        self.supported_speakers: list[str] = []
        self.eval_called = False

    def eval(self) -> "FakeInferenceModel":
        self.eval_called = True
        return self


class FakeQwen3TTS:
    def __init__(self) -> None:
        self.model = FakeInferenceModel()


class FakeFactory:
    call_args = None

    @classmethod
    def from_pretrained(cls, base_model: str, **kwargs):
        cls.call_args = (base_model, kwargs)
        return FakeQwen3TTS()


class BundleLoaderTests(unittest.TestCase):
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

    def test_resolve_bundle_artifacts_uses_manifest_defaults(self) -> None:
        bundle_dir = self.build_bundle()

        artifacts = resolve_bundle_artifacts(bundle_dir)

        self.assertEqual(artifacts.base_model, "models/Qwen3-TTS-12Hz-1.7B-Base")
        self.assertEqual(artifacts.adapter_dir, bundle_dir / "adapter")
        self.assertEqual(artifacts.config_patch_file, bundle_dir / "config_patch.json")
        self.assertEqual(artifacts.speaker_patch_file, bundle_dir / "speaker_embedding.safetensors")
        self.assertEqual(artifacts.speaker_name, "inference_speaker")

    def test_load_bundle_applies_patch_and_returns_loaded_bundle(self) -> None:
        bundle_dir = self.build_bundle()
        loader = BundleLoader(model_factory=FakeFactory)

        with patch("streaming_lora_service.app.bundle_loader.load_lora_adapter") as load_lora_adapter:
            loaded = loader.load(bundle_dir, device_map="cpu", torch_dtype="float32")

        self.assertIsNotNone(FakeFactory.call_args)
        if FakeFactory.call_args is None:
            self.fail("FakeFactory.call_args should have been populated by from_pretrained")
        base_model, call_kwargs = FakeFactory.call_args
        self.assertEqual(base_model, "models/Qwen3-TTS-12Hz-1.7B-Base")
        self.assertEqual(call_kwargs["device_map"], "cpu")
        self.assertEqual(call_kwargs["dtype"], torch.float32)
        load_lora_adapter.assert_called_once_with(loaded.qwen3tts.model, bundle_dir / "adapter")
        self.assertEqual(loaded.speaker_name, "inference_speaker")
        self.assertEqual(loaded.speaker_id, 3000)
        self.assertEqual(loaded.tts_model_type, "custom_voice")
        self.assertTrue(loaded.qwen3tts.model.eval_called)
        self.assertIn("inference_speaker", loaded.qwen3tts.model.supported_speakers)


if __name__ == "__main__":
    unittest.main()
