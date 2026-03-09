# 新会话快速接力说明

如果在一个新的聊天会话中继续这个项目，优先让新会话知道以下事实。

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

1. 检查 `train_with_codes_qwen3tts.jsonl` 的数据质量
2. 给 `sft_12hz_lora.py` 确定一版首跑参数
3. 启动一次最小 LoRA 训练
4. 如果训练正常，再补导出与推理验证

---

## 新会话推荐先读的文件

- `lora_finetuning/README.md`
- `lora_finetuning/IMPLEMENTATION_PLAN.md`
- `lora_finetuning/configs/single_speaker_small_data.yaml`

---

## 建议的新会话首个目标

最适合的开场请求通常是：

- “帮我给这 89 条数据定一版 LoRA 首跑参数”
- “帮我直接启动第一次 LoRA 训练”
- “帮我先做训练/验证切分，再开训”

这样新会话可以最少重复上下文，直接进入执行阶段。
