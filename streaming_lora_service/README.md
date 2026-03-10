# Streaming LoRA Service

本目录用于承载 **Qwen3-TTS 真双向流式服务 + LoRA bundle 加载** 的整体实施方案、协议设计和后续代码落地点。

当前阶段先以 **方案收敛与文档落地** 为主，不直接改动现有推理主链路；后续开发建议优先在这里推进，而不是把流式逻辑散落到 `qwen_tts/cli/demo.py` 或 `lora_finetuning/` 中。

## 目标

构建一个支持以下能力的独立服务层：

- **真正的双向流式 WebSocket 会话**
  - 客户端持续发送文本增量
  - 服务端持续返回音频 chunk
  - 支持 `flush / stop / cancel`
- **加载 LoRA 自定义音色 bundle**
  - 基座模型
  - LoRA adapter
  - `config_patch.json`
  - `speaker_embedding.safetensors`
- **优先支持 `custom_voice` 场景**
  - 先完成稳定版本
  - 后续再扩展 `voice_design` / `voice_clone`

## 为什么要单独开目录

现有仓库已经具备：

- 一次性离线推理：`qwen_tts/inference/qwen3_tts_model.py`
- 本地 Web UI：`qwen_tts/cli/demo.py`
- LoRA bundle 推理：`lora_finetuning/inference_with_lora.py`

但还没有一套真正适合生产化流式服务演进的独立结构。新目录的作用是：

1. 把“流式服务”与“训练/导出脚本”解耦；
2. 让 LoRA bundle 加载逻辑与 WebSocket 服务逻辑在同一个领域内收敛；
3. 给后续代码、测试、压测、协议兼容留下明确位置。

## 与现有代码的关系

这条方案不会推翻现有实现，而是复用下面这些关键能力：

- `Qwen3TTSModel.from_pretrained(...)`
- `Qwen3TTSForConditionalGeneration.generate(...)` 中的 prompt / prefill 逻辑
- `Qwen3TTSTalkerForConditionalGeneration.forward(...)` 中的逐步生成状态
- `Qwen3TTSTokenizerV2Decoder.chunked_decode(...)` 的按块解码能力
- `lora_finetuning/common.py` 中的 LoRA / speaker patch 加载逻辑

## V1 范围

V1 明确聚焦：

- 模型类型：`custom_voice`
- 接口协议：WebSocket
- LoRA 加载方式：优先支持 `--bundle_dir`
- 输出格式：PCM16 单声道 chunk
- 运行方式：单模型常驻 + 多会话排队/限流

V1 暂不承诺：

- `voice_clone` 真流式
- 麦克风音频上行流式输入
- HTTP/REST 与 WebSocket 双栈同时上线
- 多模型热切换
- 横向扩容与分布式调度

## 建议后续代码结构

后续如果开始写代码，建议优先在本目录中补以下子结构：

- `streaming_lora_service/app/`
  - `bundle_loader.py`
  - `session.py`
  - `incremental_decoder.py`
  - `sampler.py`
  - `server.py`
- `streaming_lora_service/tests/`
  - `test_bundle_loader.py`
  - `test_session_smoke.py`
  - `test_protocol_smoke.py`

## 先读哪几份文档

建议按顺序阅读：

1. `docs/IMPLEMENTATION_PLAN.md`
2. `docs/ARCHITECTURE.md`
3. `docs/WEBSOCKET_PROTOCOL.md`
4. `docs/DELIVERY_CHECKLIST.md`

## 当前结论

**这件事可以做，而且应当做成一个独立的流式服务域。**

最关键的原则只有一条：

> 不要只在现有 `generate_custom_voice()` 外面套 WebSocket；
> 必须下沉到逐步生成循环与增量解码层，才能落成真正的双向流式。
