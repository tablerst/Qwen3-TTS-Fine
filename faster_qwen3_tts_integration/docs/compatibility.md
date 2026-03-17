# compatibility notes

## 当前已确认的前提

这条集成路径依赖以下事实同时成立：

1. 当前 LoRA bundle 不只是 adapter。
2. `config_patch.json` 必须一起固化进最终模型 config。
3. `speaker_embedding.safetensors` 必须写入 `codec_embedding` 对应权重行。
4. 最终模型需要表现为 `custom_voice`。
5. `talker_config.spk_id` 必须能解析出目标 speaker。

## 为什么不能只 merge adapter

如果只 merge adapter，通常还缺：

- `tts_model_type: custom_voice`
- `talker_config.spk_id`
- 对应 speaker embedding 的 baked-in 权重

这种情况下最常见的问题是：

- 模型可以加载，但不走 `custom_voice` 路径
- speaker 找不到
- 能生成，但音色与预期严重偏离

## 当前导出路径

当前首版导出脚本采用：

1. 解析 bundle（复用 `resolve_bundle_artifacts(...)`）
2. 加载 base model
3. 用 `PeftModel.from_pretrained(...).merge_and_unload()` 合并 LoRA
4. 应用 `apply_config_patch(...)`
5. 应用 `apply_speaker_patch(...)`
6. 保存为本地 merged 模型目录

之所以不直接复用 `BundleLoader.load()`，是因为它走的是“动态注入 adapter + 运行时 patch”的服务路径，而这里需要的是一个**静态可移动的 merged 目录**。

## 当前合同校验点

`contracts.py` 当前要求导出目录至少满足：

- `config.json` 存在
- 权重文件存在（`model.safetensors` / sharded index / `pytorch_model.bin`）
- processor/tokenizer 文件存在
- `tts_model_type == "custom_voice"`
- `talker_config.spk_id` 含 speaker

## 可能失败模式

### 1. merged 导出失败

常见原因：

- bundle 缺文件
- base model 路径无效
- PEFT merge 无法完成

### 2. 合同测试失败

常见原因：

- 没保存 processor/tokenizer
- config 没写回 `custom_voice`
- speaker mapping 丢失

### 3. faster smoke test 加载失败

常见原因：

- 外部 `faster-qwen3-tts` 未安装
- 外部仓库依赖的 `qwen_tts` 版本和当前工作区结构假设不一致
- 导出的本地目录虽然是 HF 风格，但仍缺它运行所需文件

### 4. faster smoke test 生成失败

常见原因：

- speaker 不在 `get_supported_speakers()` 中
- 实际没有走 `custom_voice`
- faster 侧对某些 config 字段有额外假设

### 5. 生成成功但音色跑偏

这类问题一般不是 faster 本身先有错，而是：

- speaker patch 没真正写进权重
- config patch 缺字段
- 使用了错误 speaker 名称

## Phase 2 扩展前提

只有在以下条件都满足后，才建议进入 Phase 2：

- merged 导出稳定
- 合同测试稳定
- faster smoke test 可以稳定产出非空音频
- 主观听感没有明显偏离当前 `inference_with_lora.py` 基线

届时再考虑：

- `configs/`
- benchmark 脚本
- 流式脚本
- 服务端封装

并优先参考 `vllm_omni_service/` 现有的 `configs/` / `scripts/` / `benchmarks/` 组织习惯，而不是再长一套第四种风格。
