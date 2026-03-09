# Qwen3-TTS LoRA 微调规划

本目录用于沉淀 **Qwen3-TTS 12Hz Base 单说话人 LoRA 微调** 的方案设计、实施步骤和默认配置模板。

当前仓库内置的 `finetuning/sft_12hz.py` 是全参数 / 近全参数 SFT 方案；这里给出的是一条更适合 **小数据、单说话人、显存受限** 场景的 LoRA 设计路线。

---

## 0. 当前工作区已完成状态

截至目前，这个工作区已经完成了下面这些准备工作：

- 已落地 LoRA 相关脚本：
   - `lora_finetuning/sft_12hz_lora.py`
   - `lora_finetuning/export_custom_voice.py`
   - `lora_finetuning/inference_with_lora.py`
   - `lora_finetuning/convert_resources_manifest.py`
   - `lora_finetuning/common.py`
- 已下载本地模型：
   - `models/Qwen3-TTS-12Hz-1.7B-Base`
   - `models/Qwen3-TTS-Tokenizer-12Hz`
- 已将 `resources/segments/prompt24k_manifest.jsonl` 转成 Qwen3-TTS 可训练格式：
   - `resources/segments/train_raw_qwen3tts.jsonl`
   - `resources/segments/train_with_codes_qwen3tts.jsonl`
- 已复制共享参考音频到工作区：
   - `resources/reference/yachiyo_prompt_24k.wav`

也就是说，当前仓库已经不处于“纯方案阶段”，而是进入了**可以直接开始 LoRA 训练实验**的状态。

---

## 0.1 一个关键结论：不要直接用 CustomVoice 当微调基座

虽然用户最初想过直接使用 `Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice`，但根据当前仓库说明和官方微调文档，**单说话人微调入口是 12Hz Base 系列，不是 CustomVoice 系列**。

因此当前工作区已经按这个结论完成了本地准备：

- 使用 `Qwen3-TTS-12Hz-1.7B-Base` 作为本地实验基座；
- 使用 `Qwen3-TTS-Tokenizer-12Hz` 做 `audio_codes` 预处理。

---

## 1. 为什么这里更适合 LoRA

对当前项目来说，小数据场景下直接全参 SFT 的主要风险是：

- 容易过拟合；
- checkpoint 大；
- 训练/回滚成本高；
- 要反复试学习率和 epoch；
- 对 Windows + 单卡环境不够友好。

LoRA 的目标则是：

- **冻结大部分底座参数**，只训练少量低秩增量；
- 优先保留内容建模能力，只让模型学习目标说话人的风格偏移；
- 降低显存和存储压力；
- 让小数据试验更快更稳。

---

## 2. 结合当前代码结构的推荐 LoRA 路线

从 `qwen_tts/core/models/modeling_qwen3_tts.py` 与 `finetuning/sft_12hz.py` 来看，LoRA 最适合接在 **talker 主干** 与 **sub-talker / code predictor 主干** 的线性层上。

### 推荐第一阶段接入的模块

#### 主 talker

- `talker.model.layers.*.self_attn.q_proj`
- `talker.model.layers.*.self_attn.k_proj`
- `talker.model.layers.*.self_attn.v_proj`
- `talker.model.layers.*.self_attn.o_proj`
- `talker.model.layers.*.mlp.gate_proj`
- `talker.model.layers.*.mlp.up_proj`
- `talker.model.layers.*.mlp.down_proj`

#### sub-talker / code predictor

- `talker.code_predictor.model.layers.*.self_attn.q_proj`
- `talker.code_predictor.model.layers.*.self_attn.k_proj`
- `talker.code_predictor.model.layers.*.self_attn.v_proj`
- `talker.code_predictor.model.layers.*.self_attn.o_proj`
- `talker.code_predictor.model.layers.*.mlp.gate_proj`
- `talker.code_predictor.model.layers.*.mlp.up_proj`
- `talker.code_predictor.model.layers.*.mlp.down_proj`

### 第二阶段可选增强模块

如果第一阶段出现 **loss 降不动**、目标音色偏移不足、说话风格变化不明显，可以再追加：

- `talker.text_projection.linear_fc1`
- `talker.text_projection.linear_fc2`
- `talker.code_predictor.small_to_mtp_projection`（若不是 `Identity`）

---

## 3. 不建议第一版就动的部分

### `speaker_encoder`

当前全参脚本里：

- `speaker_encoder(...)` 的输出在训练时被 `detach()`；
- 导出 checkpoint 时还会把 `speaker_encoder` 权重从 `state_dict` 中移除；
- 实际上它更像一个固定说话人特征提取器，而不是本次 LoRA 的主要学习对象。

**建议：继续冻结 `speaker_encoder`。**

### Embedding / 输出头

第一版不建议直接全量训练这些模块：

- `talker.model.text_embedding`
- `talker.model.codec_embedding`
- `talker.code_predictor.model.codec_embedding`
- `talker.codec_head`
- `talker.code_predictor.lm_head`

原因：

- 参数量大；
- 小数据下更容易过拟合；
- 会削弱 LoRA 的轻量化意义。

如果后续验证发现输出发音稳定性或细节明显不足，再考虑有选择地放开部分模块。

---

## 4. 说话人信息建议怎么处理

当前全参脚本会把新说话人写入：

- `talker_config.spk_id[speaker_name] = 3000`
- `talker.model.codec_embedding.weight[3000] = target_speaker_embedding`

LoRA 方案建议保持这一思路，但**不要把整个 embedding 层一起训练**，而是：

1. 用固定 `ref_audio` 提取目标说话人 embedding；
2. 单独保存一个 `speaker_embedding.safetensors`（或同等格式）；
3. 单独保存一个 `config_patch.json`，只补充：
   - `tts_model_type = custom_voice`
   - `talker_config.spk_id`
   - `talker_config.spk_is_dialect`
4. 推理时加载：
   - 基座模型
   - LoRA adapter
   - speaker embedding patch
   - config patch

这样可以避免为了一个 speaker row 去保存整个大 embedding 权重。

---

## 5. 推荐训练策略（适合你当前这种小数据场景）

### 首选基座

优先从：

- `Qwen/Qwen3-TTS-12Hz-0.6B-Base`

开始，而不是直接上 1.7B。

原因：

- 显存更友好；
- 小数据更容易调；
- trial-and-error 成本低。

### 当前工作区的实际选择

从方法论上，我仍然更推荐先从 `0.6B-Base` 起步；但考虑到当前用户已经明确希望先准备 `1.7B` 路线，因此工作区里**已经实际下载并准备好的是本地 `1.7B-Base`**：

- `models/Qwen3-TTS-12Hz-1.7B-Base`

如果后续在显存、速度或过拟合上发现压力较大，再回切 `0.6B-Base` 会更稳。

### 推荐流程

1. 继续沿用现有 `prepare_data.py` 生成 `audio_codes`；
2. 新增 LoRA 训练脚本，例如：`lora_finetuning/sft_12hz_lora.py`；
3. 训练时仅开启 LoRA 参数；
4. `speaker_encoder` 冻结；
5. speaker row 单独导出；
6. 导出 adapter + config patch；
7. 推理前再把这些增量挂回基座模型。

### Windows / 当前环境建议

由于当前环境没有使用 `flash-attn`，建议 LoRA 训练时：

- 使用 `attn_implementation="sdpa"` 或默认 eager/sdpa；
- `dtype=torch.bfloat16` 优先；
- 显存吃紧时降低 batch、提高 gradient accumulation。

---

## 6. 对小数据的默认超参建议

### 小数据定义（经验分档）

- **极小数据**：< 20 分钟
- **小数据**：20 ~ 60 分钟
- **中等数据**：60 ~ 180 分钟

### 建议起手配置

#### 极小 / 小数据

- 基座：`0.6B-Base`
- LoRA rank：`16`
- LoRA alpha：`32`
- LoRA dropout：`0.05`
- batch size：`1 ~ 2`
- gradient accumulation：`8 ~ 32`
- lr：`1e-4`
- epoch：`3 ~ 8`
- warmup ratio：`0.03 ~ 0.05`
- weight decay：`0.0`

#### 中等数据

- LoRA rank：`32`
- LoRA alpha：`64`
- lr：`5e-5 ~ 1e-4`
- epoch：`2 ~ 5`

### 训练时重点观察

- 训练 loss 是否快速下降后震荡；
- 试听是否出现“像目标音色但内容发虚”；
- 是否出现文本照读正常但情感/韵律过拟合；
- 验证集文本是否泛化正常。

---

## 7. 推荐的落地产物

建议 LoRA 路线最终导出以下产物：

- `adapter/`：LoRA adapter 权重
- `speaker_embedding.safetensors`：目标 speaker row / embedding
- `config_patch.json`：custom speaker 配置补丁
- `train_args.json`：训练超参记录
- `metrics.json`：loss / 试听记录

---

## 8. 建议的实施顺序

1. 先复用现有数据准备脚本；
2. 新建 LoRA 训练脚本；
3. 先只打 attention + MLP 的 LoRA；
4. 跑最小样本验证；
5. 如果音色迁移不够，再加 `text_projection`；
6. 最后再决定要不要放开更大的模块。

---

## 9. 这个目录接下来可以继续放什么

当前目录已经落地了这些核心脚本：

- `sft_12hz_lora.py`：LoRA 训练脚本
- `export_custom_voice.py`：导出 custom voice bundle
- `inference_with_lora.py`：加载 base + adapter + speaker patch 进行推理

此外，数据桥接脚本也已落地：

- `convert_resources_manifest.py`：把 `resources` 内的原始 manifest 转成 Qwen3-TTS 所需的 `train_raw.jsonl`

后续如果要继续补齐整套工具链，建议再加入：

- `merge_lora.py`：将 adapter 合并到基座（可选）

当前目录已经从“方案说明”进入“可开始试跑”的状态；建议优先先跑训练脚本，再按需要补 `merge_lora.py`。

---

## 10. 新会话快速接手说明

如果在一个全新的会话中继续工作，优先记住这几个关键输入：

### 本地模型路径

- Base 模型：`models/Qwen3-TTS-12Hz-1.7B-Base`
- Tokenizer：`models/Qwen3-TTS-Tokenizer-12Hz`

### 训练数据路径

- 原始训练清单：`resources/segments/train_raw_qwen3tts.jsonl`
- 已编码训练清单：`resources/segments/train_with_codes_qwen3tts.jsonl`
- 共享参考音频：`resources/reference/yachiyo_prompt_24k.wav`

### 建议的新会话第一步

优先做以下三件事之一：

1. 检查 `train_with_codes_qwen3tts.jsonl` 的样本数与文本质量；
2. 设定 `sft_12hz_lora.py` 的首版训练参数；
3. 直接启动一次最小 LoRA 训练实验。

为了减少上下文损耗，也建议新会话优先阅读：

- `lora_finetuning/README.md`
- `lora_finetuning/IMPLEMENTATION_PLAN.md`
- `lora_finetuning/NEXT_SESSION_QUICKSTART.md`

---

## 11. 推荐补上的 LoRA smoke 测试

当前这条 LoRA 链路已经包含：

- `sft_12hz_lora.py`：训练并产出 adapter / speaker patch / config patch
- `export_custom_voice.py`：把训练产物打成 bundle
- `inference_with_lora.py`：加载 base + adapter + patch 做推理

因此最实用的一版 smoke 测试，不建议默认直接加载 1.7B 真模型跑一轮，而建议分成两层：

### 第一层：轻量编排 smoke test（默认）

目标：

- 验证训练产物目录结构是否正确；
- 验证 bundle 导出是否完整；
- 验证推理脚本是否能正确解析 bundle、加载 patch、调用生成、写出 wav。

这层测试已经适合落成仓库内的自动化检查，默认不依赖真实大模型权重前向推理。

当前仓库推荐入口：

- `tests/test_lora_smoke.py`

推荐执行方式：

- `python -m unittest tests.test_lora_smoke -v`

如果在当前 Windows 工作区里直接跑本地虚拟环境，也可以用：

- `.venv\Scripts\python.exe -m unittest tests.test_lora_smoke -v`

### 第二层：人工真机 smoke（按需）

当第一层通过后，再按需补一次人工真机验证：

1. 用最小训练参数跑 1 个 epoch / 极少 step；
2. 导出 bundle；
3. 用 `inference_with_lora.py` 生成 1 条 wav；
4. 核对产物存在、脚本日志正常、音频可播放。

这样可以把“代码接线没问题”和“真实模型闭环没问题”分开验证，避免每次改一行小逻辑都要重新扛完整大模型测试。