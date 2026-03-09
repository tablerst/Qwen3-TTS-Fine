from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

CURRENT_DIR = Path(__file__).resolve().parent
REPO_ROOT = CURRENT_DIR.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import torch
from accelerate import Accelerator
from torch.optim import AdamW
from torch.utils.data import DataLoader
from transformers import AutoConfig

from finetuning.dataset import TTSDataset
from lora_finetuning.common import (DEFAULT_TARGET_MODULES,
                                    PHASE2_OPTIONAL_MODULES,
                                    build_lora_config, count_parameters,
                                    ensure_dir,
                                    extract_target_speaker_embedding,
                                    inject_lora, load_yaml_config,
                                    make_config_patch,
                                    normalize_string_list,
                                    parse_torch_dtype, resolve_setting,
                                    save_json, save_lora_adapter,
                                    save_speaker_patch)
from qwen_tts import Qwen3TTSModel


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="LoRA fine-tuning for Qwen3-TTS 12Hz Base models")
    parser.add_argument("--config", type=str, default=None, help="Optional YAML config file")

    parser.add_argument("--base_model", type=str, default=None)
    parser.add_argument("--train_jsonl", type=str, default=None)
    parser.add_argument("--output_root", type=str, default=None)
    parser.add_argument("--speaker_name", type=str, default=None)
    parser.add_argument("--speaker_id", type=int, default=None)

    parser.add_argument("--torch_dtype", type=str, default=None)
    parser.add_argument("--attn_implementation", type=str, default=None)
    parser.add_argument("--local_files_only", action="store_true")

    parser.add_argument("--batch_size", type=int, default=None)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=None)
    parser.add_argument("--learning_rate", type=float, default=None)
    parser.add_argument("--weight_decay", type=float, default=None)
    parser.add_argument("--num_epochs", type=int, default=None)
    parser.add_argument("--max_grad_norm", type=float, default=None)
    parser.add_argument("--mixed_precision", type=str, default=None)
    parser.add_argument("--logging_steps", type=int, default=None)
    parser.add_argument("--sub_talker_loss_weight", type=float, default=None)

    parser.add_argument("--lora_r", type=int, default=None)
    parser.add_argument("--lora_alpha", type=int, default=None)
    parser.add_argument("--lora_dropout", type=float, default=None)
    parser.add_argument("--lora_bias", type=str, default=None)
    parser.add_argument("--target_modules", nargs="*", default=None)
    parser.add_argument("--extra_trainable_modules", nargs="*", default=None)

    return parser.parse_args()


class ResolvedConfig(dict):
    def __getattr__(self, item: str) -> Any:
        return self[item]


def resolve_config(args: argparse.Namespace) -> ResolvedConfig:
    cfg = load_yaml_config(args.config)

    base_model = resolve_setting(args.base_model, cfg, "model", "base_model", "Qwen/Qwen3-TTS-12Hz-0.6B-Base")
    train_jsonl = resolve_setting(args.train_jsonl, cfg, "data", "train_jsonl", "train_with_codes.jsonl")
    output_root = resolve_setting(args.output_root, cfg, "artifacts", "output_root", "outputs/lora_single_speaker")
    speaker_name = resolve_setting(args.speaker_name, cfg, "data", "speaker_name", "speaker_test")
    speaker_id = resolve_setting(args.speaker_id, cfg, None, "speaker_id", 3000)

    target_modules = normalize_string_list(
        args.target_modules if args.target_modules else resolve_setting(None, cfg, "lora", "target_modules", DEFAULT_TARGET_MODULES),
        DEFAULT_TARGET_MODULES,
    )
    extra_trainable_modules = normalize_string_list(
        args.extra_trainable_modules if args.extra_trainable_modules else resolve_setting(None, cfg, "lora", "phase2_optional_modules", []),
        [],
    )

    resolved = ResolvedConfig(
        base_model=base_model,
        train_jsonl=train_jsonl,
        output_root=output_root,
        speaker_name=speaker_name,
        speaker_id=int(speaker_id),
        torch_dtype=resolve_setting(args.torch_dtype, cfg, "model", "dtype", "bfloat16"),
        attn_implementation=resolve_setting(args.attn_implementation, cfg, "model", "attn_implementation", "sdpa"),
        batch_size=int(resolve_setting(args.batch_size, cfg, "training", "batch_size", 2)),
        gradient_accumulation_steps=int(resolve_setting(args.gradient_accumulation_steps, cfg, "training", "gradient_accumulation_steps", 8)),
        learning_rate=float(resolve_setting(args.learning_rate, cfg, "training", "learning_rate", 1e-4)),
        weight_decay=float(resolve_setting(args.weight_decay, cfg, "training", "weight_decay", 0.0)),
        num_epochs=int(resolve_setting(args.num_epochs, cfg, "training", "num_epochs", 5)),
        max_grad_norm=float(resolve_setting(args.max_grad_norm, cfg, "training", "max_grad_norm", 1.0)),
        mixed_precision=resolve_setting(args.mixed_precision, cfg, "training", "mixed_precision", "bf16"),
        logging_steps=int(resolve_setting(args.logging_steps, cfg, "training", "logging_steps", 10)),
        sub_talker_loss_weight=float(resolve_setting(args.sub_talker_loss_weight, cfg, None, "sub_talker_loss_weight", 0.3)),
        lora_r=int(resolve_setting(args.lora_r, cfg, "lora", "r", 16)),
        lora_alpha=int(resolve_setting(args.lora_alpha, cfg, "lora", "alpha", 32)),
        lora_dropout=float(resolve_setting(args.lora_dropout, cfg, "lora", "dropout", 0.05)),
        lora_bias=resolve_setting(args.lora_bias, cfg, "lora", "bias", "none"),
        target_modules=target_modules,
        extra_trainable_modules=extra_trainable_modules,
        local_files_only=bool(args.local_files_only),
        config_path=args.config,
    )
    return resolved


def load_train_data(train_jsonl: str | Path) -> list[dict[str, Any]]:
    with open(train_jsonl, "r", encoding="utf-8") as f:
        records = [json.loads(line) for line in f if line.strip()]
    if not records:
        raise ValueError(f"Training jsonl is empty: {train_jsonl}")
    return records


def save_artifacts(
    output_dir: Path,
    unwrapped_model: torch.nn.Module,
    lora_config,
    speaker_embedding: torch.Tensor,
    resolved: ResolvedConfig,
    metrics: dict[str, Any],
) -> None:
    output_dir = ensure_dir(output_dir)
    save_lora_adapter(unwrapped_model, output_dir / "adapter", lora_config)
    save_speaker_patch(output_dir / "speaker_embedding.safetensors", resolved.speaker_id, speaker_embedding)
    save_json(make_config_patch(resolved.speaker_name, resolved.speaker_id), output_dir / "config_patch.json")
    save_json(dict(resolved), output_dir / "train_args.json")
    save_json(metrics, output_dir / "metrics.json")


def main() -> None:
    args = parse_args()
    resolved = resolve_config(args)
    torch_dtype = parse_torch_dtype(resolved.torch_dtype)

    output_root = ensure_dir(resolved.output_root)
    train_data = load_train_data(resolved.train_jsonl)

    accelerator = Accelerator(
        gradient_accumulation_steps=resolved.gradient_accumulation_steps,
        mixed_precision=resolved.mixed_precision,
    )

    qwen3tts = Qwen3TTSModel.from_pretrained(
        resolved.base_model,
        dtype=torch_dtype,
        attn_implementation=resolved.attn_implementation,
        local_files_only=resolved.local_files_only,
    )
    config = AutoConfig.from_pretrained(resolved.base_model, local_files_only=resolved.local_files_only)

    speaker_embedding = extract_target_speaker_embedding(qwen3tts, train_data[0]["ref_audio"])

    lora_config = build_lora_config(
        r=resolved.lora_r,
        alpha=resolved.lora_alpha,
        dropout=resolved.lora_dropout,
        target_modules=resolved.target_modules,
        bias=resolved.lora_bias,
    )
    trainable_params, total_params, enabled_extra = inject_lora(
        qwen3tts.model,
        lora_config,
        extra_trainable_modules=resolved.extra_trainable_modules,
    )
    if trainable_params == 0:
        raise RuntimeError(
            "No trainable parameters were enabled. Please check target_modules / extra_trainable_modules."
        )

    dataset = TTSDataset(train_data, qwen3tts.processor, config)
    dataloader = DataLoader(
        dataset,
        batch_size=resolved.batch_size,
        shuffle=True,
        collate_fn=dataset.collate_fn,
    )

    optimizer = AdamW(
        [param for param in qwen3tts.model.parameters() if param.requires_grad],
        lr=resolved.learning_rate,
        weight_decay=resolved.weight_decay,
    )

    model, optimizer, dataloader = accelerator.prepare(qwen3tts.model, optimizer, dataloader)

    metrics: dict[str, Any] = {
        "base_model": resolved.base_model,
        "speaker_name": resolved.speaker_name,
        "speaker_id": resolved.speaker_id,
        "trainable_params": trainable_params,
        "total_params": total_params,
        "trainable_ratio": trainable_params / total_params,
        "target_modules": resolved.target_modules,
        "extra_trainable_modules": enabled_extra,
        "step_losses": [],
        "epoch_losses": [],
    }
    save_json(dict(resolved), output_root / "train_args.json")

    accelerator.print(
        f"LoRA training started | trainable params: {trainable_params:,} / {total_params:,} "
        f"({trainable_params / total_params:.4%})"
    )
    accelerator.print(f"Target modules: {resolved.target_modules}")
    if enabled_extra:
        accelerator.print(f"Extra trainable parameters enabled: {len(enabled_extra)}")
    if resolved.extra_trainable_modules:
        accelerator.print(f"Phase-2 optional modules requested: {resolved.extra_trainable_modules}")
    else:
        accelerator.print(f"Phase-2 optional modules available but disabled by default: {PHASE2_OPTIONAL_MODULES}")

    model.train()

    for epoch in range(resolved.num_epochs):
        epoch_losses: list[float] = []

        for step, batch in enumerate(dataloader):
            with accelerator.accumulate(model):
                input_ids = batch["input_ids"]
                codec_ids = batch["codec_ids"]
                ref_mels = batch["ref_mels"]
                text_embedding_mask = batch["text_embedding_mask"]
                codec_embedding_mask = batch["codec_embedding_mask"]
                attention_mask = batch["attention_mask"]
                codec_0_labels = batch["codec_0_labels"]
                codec_mask = batch["codec_mask"]

                speaker_embedding_batch = model.speaker_encoder(ref_mels.to(model.device).to(model.dtype)).detach()

                input_text_ids = input_ids[:, :, 0]
                input_codec_ids = input_ids[:, :, 1]

                input_text_embedding = model.talker.model.text_embedding(input_text_ids) * text_embedding_mask
                input_codec_embedding = model.talker.model.codec_embedding(input_codec_ids) * codec_embedding_mask
                input_codec_embedding[:, 6, :] = speaker_embedding_batch

                input_embeddings = input_text_embedding + input_codec_embedding

                for group_idx in range(1, 16):
                    codec_i_embedding = model.talker.code_predictor.get_input_embeddings()[group_idx - 1](
                        codec_ids[:, :, group_idx]
                    )
                    codec_i_embedding = codec_i_embedding * codec_mask.unsqueeze(-1)
                    input_embeddings = input_embeddings + codec_i_embedding

                outputs = model.talker(
                    inputs_embeds=input_embeddings[:, :-1, :],
                    attention_mask=attention_mask[:, :-1],
                    labels=codec_0_labels[:, 1:],
                    output_hidden_states=True,
                )

                hidden_states = outputs.hidden_states[0][-1]
                talker_hidden_states = hidden_states[codec_mask[:, 1:]]
                talker_codec_ids = codec_ids[codec_mask]

                _, sub_talker_loss = model.talker.forward_sub_talker_finetune(
                    talker_codec_ids,
                    talker_hidden_states,
                )

                loss = outputs.loss + resolved.sub_talker_loss_weight * sub_talker_loss
                accelerator.backward(loss)

                if accelerator.sync_gradients:
                    accelerator.clip_grad_norm_(
                        [param for param in model.parameters() if param.requires_grad],
                        resolved.max_grad_norm,
                    )

                optimizer.step()
                optimizer.zero_grad()

            loss_value = float(loss.detach().item())
            epoch_losses.append(loss_value)

            if step % resolved.logging_steps == 0:
                accelerator.print(f"Epoch {epoch} | Step {step} | Loss: {loss_value:.4f}")
                metrics["step_losses"].append(
                    {
                        "epoch": epoch,
                        "step": step,
                        "loss": loss_value,
                    }
                )

        avg_epoch_loss = sum(epoch_losses) / max(len(epoch_losses), 1)
        metrics["epoch_losses"].append(
            {
                "epoch": epoch,
                "avg_loss": avg_epoch_loss,
                "num_steps": len(epoch_losses),
            }
        )
        accelerator.print(f"Epoch {epoch} completed | avg_loss={avg_epoch_loss:.4f}")

        accelerator.wait_for_everyone()
        if accelerator.is_main_process:
            unwrapped_model = accelerator.unwrap_model(model)
            save_artifacts(output_root / f"checkpoint-epoch-{epoch}", unwrapped_model, lora_config, speaker_embedding, resolved, metrics)
            save_artifacts(output_root, unwrapped_model, lora_config, speaker_embedding, resolved, metrics)

    accelerator.print(f"LoRA fine-tuning finished. Artifacts saved to: {output_root}")


if __name__ == "__main__":
    main()
