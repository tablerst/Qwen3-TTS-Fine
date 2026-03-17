# faster_qwen3_tts_integration

`faster_qwen3_tts_integration/` 是当前仓库对外部 `faster-qwen3-tts` 的**本地适配层**，目标是验证：

1. 当前仓库导出的 custom voice bundle
2. 是否能够物化为一个完整的 merged 本地模型目录
3. 并被外部 `faster-qwen3-tts` 直接加载做最小 `custom_voice` 推理

## 为什么目录名不是 `faster_qwen3_tts/`

外部依赖本身的 Python 导入名就是 `faster_qwen3_tts`。
如果当前仓库根目录也创建同名目录，运行脚本时会优先导入本地目录，遮蔽外部包，导致：

- `from faster_qwen3_tts import FasterQwen3TTS` 导入混乱
- smoke test 误导入当前仓库代码而不是外部实现

因此这里使用 `faster_qwen3_tts_integration/` 作为实现目录名。

## Phase 1 范围

当前只做最小兼容性闭环：

- bundle → merged 本地模型目录
- merged 本地模型目录 → faster smoke test
- 契约测试与兼容性文档

当前**不做**：

- 服务化封装
- WebSocket / HTTP
- benchmark
- 训练链路改造
- 上游 `faster-qwen3-tts` 源码 vendoring

## 目录结构

- `scripts/export_merged_model.py`：把 bundle 导出为完整 merged 本地模型目录
- `scripts/smoke_test.py`：调用外部 `faster-qwen3-tts` 做最小 `custom_voice` 验证
- `contracts.py`：merged 模型目录的最小合同校验
- `tests/test_smoke_contract.py`：合同与脚本层的轻量测试
- `docs/compatibility.md`：兼容性前提、失败模式与后续扩展条件

## 输入输出契约

### 输入

优先输入一个 bundle 目录，内部至少包含：

- `adapter/`
- `config_patch.json`
- `speaker_embedding.safetensors`
- `manifest.json`

### 输出

导出脚本输出一个**完整可移动的 merged 模型目录**，并附带：

- `merged_export_summary.json`
- `source_bundle_manifest.json`

smoke test 输出：

- 一条 WAV 音频
- 一份 smoke summary JSON

## 成功标准

导出的 merged 模型目录必须满足：

- 能被 `from_pretrained(local_dir)` 正常读取
- `tts_model_type == "custom_voice"`
- `talker_config.spk_id` 含目标 speaker
- speaker embedding 已 baked 进权重
- 外部 `faster-qwen3-tts` 能完成一次最小 `generate_custom_voice(...)`

## 当前脚本边界

### `export_merged_model.py`

职责：

- 解析 bundle
- 加载 base model
- 通过 PEFT merge LoRA
- 应用 config patch
- 应用 speaker patch
- 保存本地 merged 模型目录

不负责：

- benchmark
- 服务启动
- 流式生成

### `smoke_test.py`

职责：

- 加载 merged 本地模型目录
- 自动解析单 speaker custom voice 配置
- 调外部 `FasterQwen3TTS.from_pretrained(...)`
- 跑一次最小 custom voice 生成

不负责：

- 动态 patch bundle
- 多轮性能测试
- 流式/服务侧验证

## 依赖说明

本目录默认依赖当前仓库已有：

- `lora_finetuning`
- `streaming_lora_service`
- `qwen_tts`

而 `smoke_test.py` 还要求运行环境中已安装外部 `faster-qwen3-tts`。若未安装，脚本会给出明确的导入错误提示。

## 运行方式

这两个脚本当前按**模块**方式运行，避免包内相对导入和外部 `faster_qwen3_tts` 包名产生额外干扰：

- `python -m faster_qwen3_tts_integration.scripts.export_merged_model ...`
- `python -m faster_qwen3_tts_integration.scripts.smoke_test ...`

不建议直接用文件路径执行 `scripts/export_merged_model.py` 或 `scripts/smoke_test.py`。
