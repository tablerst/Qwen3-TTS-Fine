# Implementation Plan

## 1. 目标定义

本项目要落地的是：

- **真双向流式**：客户端持续上行文本增量，服务端持续下行音频块；
- **LoRA 可插拔**：服务可加载现有 `bundle_dir` 产物；
- **优先落地 `custom_voice`**：先稳定闭环，再扩展其他模式。

这里的“真双向流式”不等于“WebSocket 包了一层一次性推理”。真正需要满足的是：

1. 输入可以分段提交；
2. 模型生成可以逐步推进；
3. 音频可以持续按块输出；
4. 同一连接中会话状态保持连续；
5. 用户可在会话中发送 `flush / stop / cancel`。

## 2. 现有仓库可复用能力

### 2.1 LoRA bundle 加载链路

仓库里已经有可复用的 LoRA 推理链：

- `lora_finetuning/inference_with_lora.py`
- `lora_finetuning/common.py`

现有 bundle 推理闭环是：

1. `Qwen3TTSModel.from_pretrained(base_model, ...)`
2. `load_lora_adapter(qwen3tts.model, adapter_dir)`
3. `apply_config_patch(qwen3tts.model, config_patch)`
4. `apply_speaker_patch(qwen3tts.model, speaker_patch_file)`
5. 调用 `generate_custom_voice(...)`

这说明后续服务应优先支持：

- `--bundle_dir`
- 读取 `manifest.json`
- 自动找到 `adapter/`、`config_patch.json`、`speaker_embedding.safetensors`

### 2.2 生成状态的底层基础

在 `qwen_tts/core/models/modeling_qwen3_tts.py` 中，`Qwen3TTSTalkerForConditionalGeneration.forward(...)` 已维护：

- `past_key_values`
- `past_hidden`
- `generation_step`
- `trailing_text_hidden`
- `tts_pad_embed`

这说明底层已经有“逐步推进生成”的结构基础。

### 2.3 增量解码基础

12Hz tokenizer decoder 已提供：

- `Qwen3TTSTokenizerV2Decoder.chunked_decode(...)`

此外：

- `output_sample_rate = 24000`
- `decode_upsample_rate = 1920`

约等于每个 codec step 对应 $1920 / 24000 = 0.08s$，即约 80ms 音频。

## 3. 不应该怎么做

以下方案不建议采用：

### 3.1 只包一层 WebSocket

即：

- 收到完整文本
- 调现有 `generate_custom_voice()`
- 最后一次性返回整段 wav

这只是“WebSocket 传输”，不是真正的流式生成。

### 3.2 把所有逻辑写进 `qwen_tts/cli/demo.py`

`demo.py` 是 UI demo，不适合作为长期服务化演进主入口。

### 3.3 第一版就追求全场景支持

不要在 V1 同时做：

- `custom_voice`
- `voice_design`
- `voice_clone`
- 麦克风上行流式音频
- HTTP + WS 双栈

这会让调试成本直接起飞。

## 4. V1 落地范围

### 4.1 支持的能力

- WebSocket 服务
- `custom_voice`
- LoRA bundle 加载
- 单文本会话持续增量输入
- 增量返回 PCM16 音频 chunk
- `flush / stop / cancel`
- 基础指标输出（TTFB / chunk 数 / codec step 数）

### 4.2 明确不做的能力

- `voice_clone` 真流式
- 多 bundle 热切换
- 分布式 worker 池
- 浏览器端 SDK
- 复杂鉴权系统

## 5. 建议目录结构

建议后续代码最终落到如下结构：

```text
streaming_lora_service/
  README.md
  docs/
    IMPLEMENTATION_PLAN.md
    ARCHITECTURE.md
    WEBSOCKET_PROTOCOL.md
    DELIVERY_CHECKLIST.md
  app/
    __init__.py
    bundle_loader.py
    prompt_builder.py
    session.py
    incremental_decoder.py
    sampler.py
    server.py
    models.py
  tests/
    test_bundle_loader.py
    test_session_smoke.py
    test_protocol_smoke.py
```

## 6. 分阶段实施

### Phase 0：设计冻结

目标：把技术边界、协议、V1 scope 固定下来。

完成标志：

- 本目录文档齐全；
- 确认 V1 只做 `custom_voice + bundle_dir + WebSocket`。

### Phase 1：Bundle Loader

目标：把 LoRA bundle 的加载从脚本式调用整理为服务可复用组件。

建议新增：

- `bundle_loader.py`

核心职责：

- 解析 `manifest.json`
- 加载 base model
- 注入 LoRA adapter
- 应用 config patch
- 应用 speaker patch
- 返回已就绪的 `Qwen3TTSModel`

完成标志：

- 能完全复用现有 bundle 推理结果；
- 可单测验证 `speaker_name`、`speaker_id`、`tts_model_type` 是否正确注入。

### Phase 2：Prompt Builder + Streaming Session

目标：从一次性 `generate_custom_voice()` 中拆出可流式复用的 prompt 构建和状态容器。

建议新增：

- `prompt_builder.py`
- `session.py`

核心职责：

- 从文本 / 语言 / speaker / instruct 构造 prefill 输入；
- 保存生成缓存态；
- 接收文本增量；
- 触发 step 级生成。

完成标志：

- 可以在 Python 侧不经 WebSocket，单独推进一次 session；
- 每次调用返回新增 codec step。

### Phase 3：Incremental Decoder

目标：实现“新增 codec -> 新增音频 chunk”而不是全量重复 decode。

建议新增：

- `incremental_decoder.py`

设计原则：

- 保留左上下文；
- 仅对最近窗口做 decode；
- 丢弃 overlap 区域；
- 返回新增 PCM16。

完成标志：

- 同一批 codec 连续追加时，能稳定输出无明显接缝的增量音频；
- 可以配置 `chunk_steps` 与 `left_context_steps`。

### Phase 4：WebSocket Server

目标：把 session 暴露成双向流式协议。

建议新增：

- `server.py`

推荐技术栈：

- `FastAPI` + WebSocket
- 或 `Starlette` 直接起 WebSocket 服务

完成标志：

- 客户端可发送 `session.start / text.delta / flush / stop / cancel`
- 服务端可发送 `audio.chunk / metrics / completed / error`

### Phase 5：测试与压测

目标：让这套服务不是“能跑一次”，而是“能反复改还能活”。

必须补：

- bundle loader smoke test
- session step test
- protocol smoke test
- cancel / flush / end-to-end 测试

建议补：

- 单机会话限流测试
- chunk 大小和 TTFB 实验记录

## 7. LoRA 兼容性要求

### 7.1 首版仅支持 bundle_dir

为了减少分支复杂度，V1 服务应优先只支持：

- `--bundle_dir <bundle>`

而不是同时支持：

- `--base_model`
- `--adapter_dir`
- `--config_patch_file`
- `--speaker_patch_file`

后者可以保留为内部调试能力，但不应成为 V1 主入口。

### 7.2 服务启动时加载，而不是每请求加载

建议模型加载策略：

- 服务进程启动时加载一次 bundle
- 后续请求只复用常驻模型

原因：

- 避免重复注入 LoRA
- 避免重复 patch speaker embedding
- 降低首请求抖动

### 7.3 Speaker 与 model type 必须在启动期校验

启动阶段应明确检查：

- `tts_model_type == custom_voice`
- `speaker_name` 是否存在于 `supported_speakers`
- `speaker_id` 是否成功 patch 到 `codec_embedding`

## 8. 关键设计决策

### 8.1 为什么 V1 只做 custom_voice

因为它最简单：

- 无参考音频在线解析
- 无 ref text / ref code 会话态
- LoRA bundle 也天然对齐 custom voice

### 8.2 为什么输出选 PCM16 chunk

因为相较 base64 wav：

- 体积更小
- 前端更容易实时播
- 延迟更低

### 8.3 为什么不用现有 generate_* 直接封装

因为现有接口返回的是：

- `Tuple[List[np.ndarray], int]`

这是整段音频返回范式，不适合真流式。

## 9. 风险与缓解

### 风险 1：逐步生成与离线整段生成结果不完全一致

缓解：

- V1 允许与离线结果存在轻微差异；
- 优先保证稳定 chunk 输出与可中断性。

### 风险 2：chunk 过小导致解码开销过高

缓解：

- 默认 `chunk_steps` 不低于 4；
- 默认 `left_context_steps` 约 20~30。

### 风险 3：文本分词边界不稳定

缓解：

- 引入稳定前缀提交策略；
- flush 时再强制提交尾部未稳定文本。

### 风险 4：Windows 环境吞吐不稳定

缓解：

- V1 先做单模型串行生成；
- 会话层增加排队与限流；
- 不承诺高并发。

## 10. 第一版建议 CLI

后续建议新增入口：

```text
qwen-tts-stream-serve --bundle_dir <path> --host 0.0.0.0 --port 9000
```

建议参数：

- `--bundle_dir`
- `--host`
- `--port`
- `--device_map`
- `--torch_dtype`
- `--attn_implementation`
- `--local_files_only`
- `--chunk_steps`
- `--left_context_steps`
- `--max_sessions`

## 11. 交付优先级

### 必须先做

1. bundle loader
2. custom_voice session
3. 增量 decoder
4. websocket 协议
5. smoke tests

### 可以第二批做

1. metrics 输出
2. 会话排队器
3. 多 bundle 注册表
4. 浏览器 demo

## 12. 本阶段结论

结论很明确：

- **LoRA 加载路径已经成熟**；
- **真正缺的是流式 session 与增量 decoder**；
- **V1 最合理的落地方向是 custom_voice + bundle_dir + WebSocket**。
