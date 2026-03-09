# LoRA 微调实施方案

本文档给出基于当前仓库实现 LoRA 微调的工程拆分方案，目标是：

- 不破坏现有 `finetuning/` 全参流程；
- 新增一条单独的 LoRA 训练链路；
- 输出 adapter + speaker patch，而不是整份大 checkpoint；
- 尽量复用现有的数据准备逻辑。

---

## A. 建议的新文件布局

```text
lora_finetuning/
  README.md
  IMPLEMENTATION_PLAN.md
  NEXT_SESSION_QUICKSTART.md
  configs/
    single_speaker_small_data.yaml
  common.py
  convert_resources_manifest.py
  sft_12hz_lora.py
  merge_lora.py                    # 未来新增，可选
  export_custom_voice.py
  inference_with_lora.py
```

---

## A.1 当前工作区已完成项

这份实施方案最初是为“未来落地”准备的，但当前仓库已经完成了以下部分：

- 已实现：
  - `common.py`
  - `convert_resources_manifest.py`
  - `sft_12hz_lora.py`
  - `export_custom_voice.py`
  - `inference_with_lora.py`
- 已准备本地模型：
  - `models/Qwen3-TTS-12Hz-1.7B-Base`
  - `models/Qwen3-TTS-Tokenizer-12Hz`
- 已准备训练数据：
  - `resources/segments/train_raw_qwen3tts.jsonl`
  - `resources/segments/train_with_codes_qwen3tts.jsonl`
  - `resources/reference/yachiyo_prompt_24k.wav`

因此本文档现在更适合作为：

- 已实现脚本的设计说明；
- 新会话继续扩展功能时的工程参考；
- 后续增加 `merge_lora.py`、验证集评估和自动化脚本时的路线图。

---

## B. 拆解当前全参脚本与 LoRA 脚本的差异

现有脚本：`finetuning/sft_12hz.py`

核心逻辑：

1. 读取 `train_with_codes.jsonl`
2. 构造 `TTSDataset`
3. 组装 `input_embeddings`
4. 调 `model.talker(...)`
5. 再计算 `sub_talker_loss`
6. `AdamW(qwen3tts.model.parameters())`
7. 保存整份 `model.safetensors`

LoRA 版建议改为：

1. **加载 base model**；
2. **冻结全部原始参数**；
3. 用 `peft.LoraConfig` 包装目标模块；
4. 优化器只接收：
   - LoRA 参数
   - （可选）少量额外 trainable module
5. 训练流程和 loss 逻辑尽量保持不变；
6. 保存：
   - adapter 权重
   - speaker embedding patch
   - config patch
   - train args

---

## C. 推荐的 PEFT 接入方式

### 推荐依赖

当前仓库已经接入并安装：

- `peft`

### 推荐的 `target_modules`

第一版建议只用名字匹配，不用手写每层完整路径。

推荐列表：

```python
[
  "q_proj",
  "k_proj",
  "v_proj",
  "o_proj",
  "gate_proj",
  "up_proj",
  "down_proj",
]
```

原因：

- 当前代码中的 talker 与 code predictor 都使用这批标准线性层命名；
- PEFT 会自动递归匹配；
- 兼容 `talker.model.layers.*` 与 `talker.code_predictor.model.layers.*`。

### 第二阶段可选 `target_modules`

```python
[
  "linear_fc1",
  "linear_fc2",
]
```

这些对应 `talker.text_projection`，建议只在第一阶段效果不够时再加。

---

## D. 参数冻结策略

### 默认冻结

- `speaker_encoder`
- `talker.model.text_embedding`
- `talker.model.codec_embedding`
- `talker.code_predictor.model.codec_embedding`
- `talker.codec_head`
- `talker.code_predictor.lm_head`

### 默认训练

- 所有 LoRA adapter 参数

### 可选训练（第二阶段）

- `talker.text_projection`
- `talker.code_predictor.small_to_mtp_projection`

---

## E. 与 speaker 适配相关的建议

### 推荐做法

不要为新 speaker 训练整个 embedding 矩阵，而是：

1. 从固定 `ref_audio` 提取 speaker embedding；
2. 写入一个固定的虚拟 speaker id（例如 `3000`）；
3. 单独导出 speaker embedding；
4. 推理时再把这行 embedding 注回去。

### 产物建议

#### `config_patch.json`

建议至少包含：

```json
{
  "tts_model_type": "custom_voice",
  "talker_config": {
    "spk_id": {
      "speaker_name": 3000
    },
    "spk_is_dialect": {
      "speaker_name": false
    }
  }
}
```

#### `speaker_embedding.safetensors`

建议保存：

- `speaker_id`
- `speaker_name`
- `embedding`

如果后续想做多说话人 adapter，也更容易扩展。

---

## F. 推荐训练阶段划分

### Phase 1：最小 LoRA

目标：先验证这条路线可用。

- 基座：`0.6B-Base`
- target modules：attention + MLP
- speaker encoder：冻结
- text projection：冻结
- 输出头：冻结

### Phase 2：增强适配

当出现以下情况时进入第二阶段：

- 音色偏移不够明显；
- 情绪/语气适配太弱；
- loss 已降但主观试听提升有限。

操作：

- 给 `text_projection.linear_fc1/linear_fc2` 也加 LoRA；
- 或放开 `small_to_mtp_projection`。

### Phase 3：导出与部署

- 保存 adapter
- 保存 config patch
- 保存 speaker patch
- 增加单独推理脚本，支持 base + adapter + patch 方式加载

---

## G. 训练脚本最重要的改动点

未来在 `sft_12hz_lora.py` 中，建议重点修改这些地方：

### 1) 模型加载

当前全参脚本：

- `Qwen3TTSModel.from_pretrained(..., attn_implementation="flash_attention_2")`

LoRA 版建议：

- Windows 下默认用 `attn_implementation="sdpa"`
- 保留 `torch_dtype=torch.bfloat16`

### 2) 参数冻结

在包 LoRA 之前先把所有参数 `requires_grad = False`。

### 3) LoRA 包装

仅包装 `qwen3tts.model.talker` 或整个 `qwen3tts.model` 中的目标线性层。

更建议包装整个 `qwen3tts.model`，但通过 `target_modules` 精确限制落点。

### 4) 优化器

优化器只吃：

- `p.requires_grad == True` 的参数

### 5) 保存逻辑

不要复制整份 base model 目录后再重写完整 `model.safetensors`。

LoRA 版建议保存：

- `adapter_model.safetensors`
- `adapter_config.json`
- `speaker_embedding.safetensors`
- `config_patch.json`
- `train_args.json`

---

## H. 推理侧需要补的能力

后续为了真正可用，推理侧至少要支持两种方式：

### 方式 1：运行时叠加

- 加载 base model
- 注入 LoRA adapter
- 应用 `config_patch`
- 写入 speaker embedding row

优点：

- 产物小
- 便于切换 speaker / adapter

### 方式 2：导出 merged 目录（可选）

- 把 LoRA merge 回 base
- 生成完整 custom voice checkpoint

优点：

- 部署简单

缺点：

- checkpoint 重新变大

建议先做方式 1。

---

## I. 推荐优先级

### 第一优先级

- `sft_12hz_lora.py`
- `export_custom_voice.py`
- `inference_with_lora.py`

### 第二优先级

- `merge_lora.py`
- 更多配置模板
- 验证集评估脚本

### 当前未完成但适合下一会话接手的项

- 为 `sft_12hz_lora.py` 补充更细的训练参数文档；
- 增加 `merge_lora.py`；
- 增加一份最小可运行的本地训练命令示例；
- 给 `resources` 数据加训练/验证切分脚本；
- 增加试听样本导出与 loss 记录可视化。

---

## J. 实施时的风险点

1. **target_modules 过多**：LoRA 参数会变大，小数据下不稳定。
2. **仍然训练 embedding / head**：会偏离 LoRA 轻量路线。
3. **speaker row 没单独管理**：推理时 adapter 生效但 speaker id 对不上。
4. **继续硬绑 flash-attn**：在当前 Windows 环境下会增加无谓摩擦。
5. **没有验证集试听**：很容易 loss 好看、音频不理想。

---

## K. 建议的里程碑

### M1

完成 LoRA 训练脚本，可跑通 1 个小样本实验。

### M2

完成 adapter + speaker patch 的推理加载。

### M3

完成一键导出 custom voice 目录。

### M4

加入验证集与试听对比，形成稳定 recipe。
