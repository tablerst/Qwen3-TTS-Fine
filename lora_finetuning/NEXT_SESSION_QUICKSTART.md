# 新会话快速接力说明

如果在一个新的聊天会话中继续这个项目，优先让新会话知道以下事实。

---

## 当前实验结论快照

- 当前**单语最像**候选：
	- `outputs/lora_formal_single_speaker_1p7b_timbre_transfer_20260310_bundle_best_refcand8`
- 当前**多语主线最佳**候选：
	- `outputs/lora_candidate8_multilingual_warmstart_1p7b_20260310_v2_bundle_best`
- `v3` 已实际执行：
	- 训练输出：`outputs/lora_candidate8_multilingual_warmstart_1p7b_20260311_v3`
	- bundleTest：`outputs/lora_candidate8_multilingual_warmstart_1p7b_20260311_v3_bundleTest`
- `v3` 的重要结论：
	- ZH 从 `3` 条增到 `5` 条后，`validation loss` 继续下降
	- 但**主观听感不如 `v2`**
	- 因此当前不要把 `v3` 当成默认最佳版本

---

## 当前已完成状态

### 环境

- 当前工作区：`E:\JetBrains\Pycharm\Qwen3-TTS-Fine`
- 已配置 Python 虚拟环境：`.venv`
- 已切换到 CUDA 版 PyTorch
- 当前 GPU 可用：`RTX 4070 Ti SUPER`

### LoRA 脚本

已经落地：

- `lora_finetuning/common.py`
- `lora_finetuning/convert_resources_manifest.py`
- `lora_finetuning/sft_12hz_lora.py`
- `lora_finetuning/export_custom_voice.py`
- `lora_finetuning/inference_with_lora.py`

### 本地模型

已经下载到工作区：

- `models/Qwen3-TTS-12Hz-1.7B-Base`
- `models/Qwen3-TTS-Tokenizer-12Hz`

> 注意：不要直接用 `Qwen3-TTS-12Hz-1.7B-CustomVoice` 当微调基座。
> 当前仓库与官方说明都指向 **12Hz Base 系列** 作为微调入口。

### 数据

`resources` 里的原始数据已经处理完成：

- 原始 manifest：`resources/segments/prompt24k_manifest.jsonl`
- 训练清单：`resources/segments/train_raw_qwen3tts.jsonl`
- 已编码清单：`resources/segments/train_with_codes_qwen3tts.jsonl`
- 共享参考音频：`resources/reference/yachiyo_prompt_24k.wav`

---

## 下一步最值得做的事

优先顺序建议如下：

1. 先对比 `v2` 与 `v3` 的 JA / ZH / EN 试听，确认主观差异点
2. 如果继续做多语增强，请从 `v2` 而不是 `v3` 重新设计下一轮
3. 每新增一轮实验后，立即更新 `EXPERIMENT_CATALOG.md`
4. 如果要继续做中文增强，优先减少变量，不要同时大改数据配比和训练策略

---

## 新会话推荐先读的文件

- `lora_finetuning/EXPERIMENT_CATALOG.md`
- `lora_finetuning/README.md`
- `lora_finetuning/IMPLEMENTATION_PLAN.md`
- `lora_finetuning/configs/single_speaker_small_data.yaml`

---

## 建议的新会话首个目标

最适合的开场请求通常是：

- “帮我基于 `v2` 和 `v3` 的试听结论设计一个更稳的 `v4`”
- “帮我把最新实验结果补登记到 `EXPERIMENT_CATALOG.md`”
- “帮我从 `v2/best-checkpoint` 再做一轮更保守的中文增强实验”

这样新会话可以最少重复上下文，直接进入执行阶段。
