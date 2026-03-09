from __future__ import annotations

import json
import sys
import tempfile
import unittest
from argparse import Namespace
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import torch
from torch import nn

from lora_finetuning import common, export_custom_voice, inference_with_lora, sft_12hz_lora


class DummyLoRAModule(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.q_proj = nn.Linear(8, 8, bias=False)
        self.k_proj = nn.Linear(8, 8, bias=False)
        self.v_proj = nn.Linear(8, 8, bias=False)
        self.o_proj = nn.Linear(8, 8, bias=False)
        self.gate_proj = nn.Linear(8, 8, bias=False)
        self.up_proj = nn.Linear(8, 8, bias=False)
        self.down_proj = nn.Linear(8, 8, bias=False)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        hidden = self.q_proj(inputs)
        hidden = self.k_proj(hidden)
        hidden = self.v_proj(hidden)
        hidden = self.o_proj(hidden)
        return hidden


class FakeInferenceModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
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
        self.generate_calls: list[dict[str, object]] = []

    def generate_custom_voice(self, **kwargs):
        self.generate_calls.append(kwargs)
        return [np.zeros(240, dtype=np.float32)], 24000


class LoRASmokeTests(unittest.TestCase):
    def make_temp_dir(self) -> Path:
        temp_dir = tempfile.TemporaryDirectory()
        self.addCleanup(temp_dir.cleanup)
        return Path(temp_dir.name)

    def build_dummy_lora_model(self) -> tuple[nn.Module, object]:
        model = DummyLoRAModule()
        lora_config = common.build_lora_config(
            r=4,
            alpha=8,
            dropout=0.0,
            target_modules=common.DEFAULT_TARGET_MODULES,
        )
        trainable_params, total_params, enabled_extra = common.inject_lora(model, lora_config)
        self.assertGreater(trainable_params, 0)
        self.assertGreater(total_params, trainable_params)
        self.assertEqual(enabled_extra, [])
        return model, lora_config

    def test_resolve_config_prefers_cli_over_yaml(self) -> None:
        work_dir = self.make_temp_dir()
        config_path = work_dir / "smoke_config.yaml"
        config_path.write_text(
            """
model:
  base_model: models/from-yaml
training:
  batch_size: 2
lora:
  target_modules:
    - q_proj
    - o_proj
data:
  speaker_name: yaml_speaker
artifacts:
  output_root: outputs/from-yaml
            """.strip(),
            encoding="utf-8",
        )

        args = Namespace(
            config=str(config_path),
            base_model=None,
            train_jsonl=None,
            output_root=None,
            speaker_name="cli_speaker",
            speaker_id=3001,
            torch_dtype="fp16",
            attn_implementation=None,
            local_files_only=True,
            batch_size=1,
            gradient_accumulation_steps=None,
            learning_rate=None,
            weight_decay=None,
            num_epochs=None,
            max_grad_norm=None,
            mixed_precision=None,
            logging_steps=None,
            sub_talker_loss_weight=None,
            lora_r=None,
            lora_alpha=None,
            lora_dropout=None,
            lora_bias=None,
            target_modules=["q_proj", "v_proj"],
            extra_trainable_modules=["linear_fc1"],
        )

        resolved = sft_12hz_lora.resolve_config(args)

        self.assertEqual(resolved.base_model, "models/from-yaml")
        self.assertEqual(resolved.batch_size, 1)
        self.assertEqual(resolved.speaker_name, "cli_speaker")
        self.assertEqual(resolved.speaker_id, 3001)
        self.assertEqual(resolved.torch_dtype, "fp16")
        self.assertEqual(resolved.target_modules, ["q_proj", "v_proj"])
        self.assertEqual(resolved.extra_trainable_modules, ["linear_fc1"])
        self.assertTrue(resolved.local_files_only)

    def test_save_artifacts_writes_expected_training_outputs(self) -> None:
        output_dir = self.make_temp_dir() / "artifacts"
        model, lora_config = self.build_dummy_lora_model()
        resolved = sft_12hz_lora.ResolvedConfig(
            base_model="models/Qwen3-TTS-12Hz-1.7B-Base",
            train_jsonl="resources/segments/train_with_codes_qwen3tts.jsonl",
            output_root=str(output_dir),
            speaker_name="smoke_speaker",
            speaker_id=3000,
        )
        metrics = {"epoch_losses": [{"epoch": 0, "avg_loss": 0.1234, "num_steps": 1}]}
        speaker_embedding = torch.randn(8)

        sft_12hz_lora.save_artifacts(output_dir, model, lora_config, speaker_embedding, resolved, metrics)

        self.assertTrue((output_dir / "adapter" / "adapter_model.safetensors").exists())
        self.assertTrue((output_dir / "adapter" / "adapter_config.json").exists())
        self.assertTrue((output_dir / "config_patch.json").exists())
        self.assertTrue((output_dir / "train_args.json").exists())
        self.assertTrue((output_dir / "metrics.json").exists())
        self.assertTrue((output_dir / "speaker_embedding.safetensors").exists())

        speaker_id, loaded_embedding = common.load_speaker_patch(output_dir / "speaker_embedding.safetensors")
        self.assertEqual(speaker_id, 3000)
        self.assertEqual(tuple(loaded_embedding.shape), (8,))

        config_patch = common.load_json(output_dir / "config_patch.json")
        self.assertEqual(config_patch["tts_model_type"], "custom_voice")
        self.assertEqual(config_patch["talker_config"]["spk_id"]["smoke_speaker"], 3000)

    def test_export_custom_voice_bundles_artifacts_and_manifest(self) -> None:
        source_dir = self.make_temp_dir() / "train_outputs"
        output_dir = self.make_temp_dir() / "bundle"
        model, lora_config = self.build_dummy_lora_model()
        resolved = sft_12hz_lora.ResolvedConfig(
            base_model="models/Qwen3-TTS-12Hz-1.7B-Base",
            train_jsonl="resources/segments/train_with_codes_qwen3tts.jsonl",
            output_root=str(source_dir),
            speaker_name="bundle_speaker",
            speaker_id=3002,
        )
        sft_12hz_lora.save_artifacts(
            source_dir,
            model,
            lora_config,
            torch.randn(8),
            resolved,
            metrics={"ok": True},
        )

        argv = [
            "export_custom_voice.py",
            "--base_model",
            "models/Qwen3-TTS-12Hz-1.7B-Base",
            "--source_dir",
            str(source_dir),
            "--output_dir",
            str(output_dir),
        ]
        with patch.object(sys, "argv", argv):
            export_custom_voice.main()

        self.assertTrue((output_dir / "adapter" / "adapter_model.safetensors").exists())
        self.assertTrue((output_dir / "config_patch.json").exists())
        self.assertTrue((output_dir / "speaker_embedding.safetensors").exists())
        self.assertTrue((output_dir / "manifest.json").exists())

        manifest = common.load_json(output_dir / "manifest.json")
        self.assertEqual(manifest["base_model_path"], "models/Qwen3-TTS-12Hz-1.7B-Base")
        self.assertEqual(manifest["speaker_name"], "bundle_speaker")
        self.assertEqual(manifest["speaker_id"], 3002)
        self.assertEqual(manifest["adapter_dir"], "adapter")

    def test_inference_main_loads_bundle_and_writes_output(self) -> None:
        bundle_dir = self.make_temp_dir() / "bundle"
        adapter_dir = bundle_dir / "adapter"
        adapter_dir.mkdir(parents=True, exist_ok=True)
        (adapter_dir / "adapter_model.safetensors").write_bytes(b"stub")
        (adapter_dir / "adapter_config.json").write_text("{}", encoding="utf-8")

        common.save_json(common.make_config_patch("inference_speaker", 3000), bundle_dir / "config_patch.json")
        common.save_speaker_patch(bundle_dir / "speaker_embedding.safetensors", 3000, torch.randn(8))
        common.save_json(
            common.build_bundle_manifest(
                base_model_path="models/Qwen3-TTS-12Hz-1.7B-Base",
                speaker_name="inference_speaker",
                speaker_id=3000,
            ),
            bundle_dir / "manifest.json",
        )

        fake_qwen3tts = FakeQwen3TTS()
        output_wav = bundle_dir / "outputs" / "smoke.wav"
        argv = [
            "inference_with_lora.py",
            "--bundle_dir",
            str(bundle_dir),
            "--text",
            "这是一次 LoRA smoke test。",
            "--output_wav",
            str(output_wav),
        ]

        with patch("lora_finetuning.inference_with_lora.Qwen3TTSModel.from_pretrained", return_value=fake_qwen3tts) as from_pretrained, \
             patch("lora_finetuning.inference_with_lora.load_lora_adapter") as load_lora_adapter, \
             patch("lora_finetuning.inference_with_lora.sf.write") as sf_write:
            with patch.object(sys, "argv", argv):
                inference_with_lora.main()

        from_pretrained.assert_called_once()
        call_kwargs = from_pretrained.call_args.kwargs
        self.assertEqual(call_kwargs["device_map"], "cuda:0")
        self.assertEqual(call_kwargs["attn_implementation"], "sdpa")
        self.assertEqual(call_kwargs["dtype"], torch.bfloat16)

        load_lora_adapter.assert_called_once_with(fake_qwen3tts.model, bundle_dir / "adapter")
        self.assertTrue(fake_qwen3tts.model.eval_called)
        self.assertEqual(len(fake_qwen3tts.generate_calls), 1)
        self.assertEqual(fake_qwen3tts.generate_calls[0]["speaker"], "inference_speaker")
        self.assertEqual(fake_qwen3tts.generate_calls[0]["text"], "这是一次 LoRA smoke test。")
        self.assertIn("inference_speaker", fake_qwen3tts.model.supported_speakers)
        self.assertEqual(fake_qwen3tts.model.config.tts_model_type, "custom_voice")
        self.assertEqual(fake_qwen3tts.model.talker.model.codec_embedding.weight[3000].shape[0], 8)

        sf_write.assert_called_once()
        self.assertEqual(Path(sf_write.call_args.args[0]), output_wav)
        self.assertEqual(sf_write.call_args.args[2], 24000)

    def test_load_train_data_rejects_empty_jsonl(self) -> None:
        empty_jsonl = self.make_temp_dir() / "empty.jsonl"
        empty_jsonl.write_text("", encoding="utf-8")

        with self.assertRaisesRegex(ValueError, "Training jsonl is empty"):
            sft_12hz_lora.load_train_data(empty_jsonl)


if __name__ == "__main__":
    unittest.main()
