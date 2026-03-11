# Implementation Plan

## 1. 目标定义

本项目要落地的是：

- **公开接口官方风格化**：从第一版开始直接采用 Qwen-compatible 对外契约；
- **内部 LoRA 默认挂载**：LoRA bundle 由服务部署侧决定，不暴露为客户端必选参数；
- **真流式 runtime**：客户端持续输入文本，服务端持续输出音频块，而不是 WebSocket 包一层一次性推理；
- **优先落地 `custom_voice`**：先把最稳定的闭环做扎实，再扩其他模式。

这里的“Qwen-compatible”重点是：

1. 事件命名与字段命名兼容官方风格；
2. 对外文档与调用心智尽量靠近官方；
3. 但内部仍然由本仓库自己的 LoRA + session + decoder 实现负责。

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

这说明服务内部完全可以继续围绕：

- `--bundle_dir`
- `manifest.json`
- `adapter/`
- `config_patch.json`
- `speaker_embedding.safetensors`

完成默认模型装配。

### 2.2 生成状态的底层基础

在 `qwen_tts/core/models/modeling_qwen3_tts.py` 中，`Qwen3TTSTalkerForConditionalGeneration.forward(...)` 已维护：

- `past_key_values`
- `past_hidden`
- `generation_step`
- `trailing_text_hidden`
- `tts_pad_embed`

这说明底层已经有“逐步推进生成”的状态基础。

### 2.3 增量解码基础

12Hz tokenizer decoder 已提供：

- `Qwen3TTSTokenizerV2Decoder.chunked_decode(...)`

此外：

- `output_sample_rate = 24000`
- `decode_upsample_rate = 1920`

约等于每个 codec step 对应 $1920 / 24000 = 0.08s$，即约 80ms 音频。

## 3. 不应该怎么做

以下方案不建议采用：

### 3.1 不要继续扩展旧的公开自定义协议

即不要继续把下面这些事件当成公开 API 主标准：

- `session.start`
- `text.delta`
- `text.flush`
- `session.stop`
- `session.cancel`

它们可以保留为内部/native 语义，但不应继续对第三方公开推广。

### 3.2 不要把 LoRA 装配参数暴露给普通调用方

V1 公开客户端不应被要求理解：

- `bundle_dir`
- `adapter_dir`
- `config_patch_file`
- `speaker_patch_file`

这些应当是**服务部署时配置**，而不是常规 API 调用参数。

### 3.3 不要把所有逻辑写进 `qwen_tts/cli/demo.py`

`demo.py` 是 UI demo，不适合作为长期服务化入口。

### 3.4 不要为了“像官方”去强行复制官方内部实现

V1 需要兼容的是：

- 对外契约
- 交互节奏
- 事件命名

而不是让内部代码结构也强行变成官方 SDK 的样子。

## 4. V1 落地范围

### 4.1 V1 必做能力

- Qwen-compatible Realtime WebSocket 公开接口
- `custom_voice` 场景优先
- 服务启动时默认加载一个 LoRA bundle
- 对外通过 `model` / `voice` 暴露稳定别名
- 增量返回 `response.audio.delta`（Base64 音频块）
- `server_commit` / `commit` 模式
- 基础指标输出（TTFB / chunk 数 / codec step 数）

### 4.2 V1 可共享但非阻塞能力

- 同一套 `model` / `voice` 映射供 HTTP TTS 兼容层复用
- 同一套 voice registry 供后续 SDK / 网关复用

### 4.3 V1 明确不做的能力

- 对外继续维护旧的自定义 WebSocket 协议
- 把 `bundle_dir` 暴露为普通调用方参数
- `voice_clone` 真流式
- 多 bundle 热切换
- 分布式 worker 池
- 完整 DashScope 路径级兼容

## 5. 建议目录结构

建议后续代码最终落到如下结构：

```text
streaming_lora_service/
  README.md
  docs/
    QWEN_COMPAT_API.md
    QWEN_COMPATIBILITY_MATRIX.md
    IMPLEMENTATION_PLAN.md
    ARCHITECTURE.md
    WEBSOCKET_PROTOCOL.md
    DELIVERY_CHECKLIST.md
  app/
    __init__.py
    bundle_loader.py
    voice_registry.py
    runtime_session.py
    incremental_decoder.py
    qwen_compat_ws.py
    qwen_compat_http.py
    server.py
    models.py
  tests/
    test_bundle_loader.py
    test_voice_registry.py
    test_runtime_session_smoke.py
    test_realtime_protocol_smoke.py
```

## 6. 分阶段实施

### Phase 0：设计冻结

目标：把“公开协议官方风格化 + 内部 LoRA 默认挂载”这个方向固定下来。

完成标志：

- 文档明确废弃旧公开自定义协议；
- 文档明确 `model` / `voice` 是公开别名；
- 文档明确 `bundle_dir` 属于部署侧参数。

### Phase 1：Bundle Loader + Voice Registry

目标：把 LoRA bundle 的部署期装配与公开 voice/model 别名解析整理成服务组件。

建议新增：

- `bundle_loader.py`
- `voice_registry.py`

核心职责：

- 解析 `manifest.json`
- 加载 base model
- 注入 LoRA adapter
- 应用 config patch
- 应用 speaker patch
- 返回已就绪的 `Qwen3TTSModel`
- 把公开 `voice` / `model` 映射到 bundle / speaker profile

完成标志：

- 默认 bundle 可在服务启动时一次性加载；
- 外部不需要看到 bundle 细节；
- 可单测验证 `voice -> speaker/profile` 映射是否正确。

### Phase 2：Runtime Session

目标：从一次性 `generate_custom_voice()` 中拆出可流式复用的 session 容器。

建议新增：

- `runtime_session.py`

核心职责：

- 维护 `append / commit / finish` 语义；
- 保存生成缓存态；
- 接收文本增量；
- 推进 step 级生成。

完成标志：

- 可以在 Python 侧不经 WebSocket 单独推进一次 session；
- 每次调用可返回新增 codec step。

### Phase 3：Incremental Decoder

目标：实现“新增 codec -> 新增音频 delta”，而不是全量重复 decode。

建议新增：

- `incremental_decoder.py`

设计原则：

- 保留左上下文；
- 仅对最近窗口做 decode；
- 丢弃 overlap 区域；
- 返回新增 PCM 字节；
- 由兼容层负责编码为 `response.audio.delta` 所需的 Base64。

完成标志：

- 同一批 codec 连续追加时，能稳定输出无明显接缝的增量音频；
- 可以配置 `chunk_steps` 与 `left_context_steps`。

### Phase 4：Qwen-Compatible Realtime Adapter

目标：把内部 runtime 暴露成官方风格的 Realtime 事件流。

建议新增：

- `qwen_compat_ws.py`

推荐技术栈：

- `FastAPI` + WebSocket
- 或 `Starlette` 直接起 WebSocket 服务

完成标志：

- 客户端可发送 `session.update / input_text_buffer.append / input_text_buffer.commit / input_text_buffer.clear / session.finish`
- 服务端可发送 `session.created / session.updated / response.audio.delta / response.done / session.finished / error`

### Phase 5：HTTP 兼容层（可并行或次阶段）

目标：让非实时场景复用同一套公开 `model` / `voice` 语义。

建议新增：

- `qwen_compat_http.py`

说明：

- 当前 MVP 已在 `server.py` 中直接提供 HTTP 兼容层；
- 后续如 HTTP 逻辑继续复杂化，再单独拆分出 `qwen_compat_http.py`。

### Phase 6：测试与验证

必须补：

- bundle loader smoke test
- voice registry test
- runtime session step test
- Realtime 协议 smoke test
- HTTP TTS smoke test
- 默认 LoRA bundle 真实模型人工试听验证

建议补：

- 单机会话限流测试
- chunk 大小和 TTFB 实验记录
- `server_commit` 与 `commit` 模式行为差异测试

## 6.1 当前已完成的骨架实现（2026-03-11）

当前已经完成第一批可测试骨架，不再只是文档设计：

- `bundle_loader.py`
- `voice_registry.py`
- `runtime_session.py`
- `incremental_decoder.py`
- `qwen_compat_ws.py`
- `server.py`
- `audio_utils.py`

这些模块当前已经覆盖：

- bundle 解析与默认 LoRA 加载链路封装
- 公开 `voice` 别名解析
- 文本缓冲区 / commit / finish 状态机
- 增量解码窗口规划与 overlap 裁剪
- Qwen-compatible Realtime 事件适配
- FastAPI WebSocket MVP 服务入口
- 音频 PCM16 转换与分块下发

同时已经补齐并通过第一批单元测试：

- bundle loader smoke
- voice registry
- runtime session
- incremental decoder
- protocol smoke
- server smoke

## 6.1.1 当前 MVP 的实现方式

当前 MVP 已经完成“对外可用”的官方风格兼容层，但实现策略需要明确说明：

- 当前音频输出仍然基于现有 `generate_custom_voice()` 一次性生成；
- 然后再转换为 PCM16 并拆成多个 `response.audio.delta` 返回；
- HTTP 非流式接口返回官方风格 JSON，并提供本地可下载音频 URL；
- HTTP 流式接口返回 `application/x-ndjson`，在字段命名上保持官方风格；
- 因此它已经满足**兼容接口 MVP** 的目标；
- 但还没有达到最终目标中的**step-level 真流式生成**。

这个取舍是刻意的：

- 先把对外契约、服务启动、默认 LoRA 装配、联调路径做通；
- 再在同一套公开协议下替换底层生成内核。

## 6.2 下一阶段建议顺序

建议下一阶段按下面顺序继续推进：

1. **step-level 真流式生成**
  - 用真实 prompt builder / generation state 替换当前的一次性生成分块下发策略
2. **真实 incremental decoder 接入**
  - 让 `incremental_decoder` 管理 codec ring buffer，而不是仅做窗口规划
3. **指标记录与调优**
  - 记录 TTFB、chunk 粒度、一次性生成与未来真流式差异
4. **真实 bundle 人工试听回归**
  - 校验默认 LoRA bundle 的可懂度、稳定性与接口行为一致性
5. **可选 HTTP 兼容层**
  - 在共享 `voice` / `model` 语义的前提下补充非实时接口

## 7. LoRA 兼容性要求

### 7.1 `bundle_dir` 是部署参数，不是公开 API 参数

V1 服务应优先只支持：

- 服务启动时 `--bundle_dir <bundle>`

而不是让普通客户端在请求里传：

- `base_model`
- `adapter_dir`
- `config_patch_file`
- `speaker_patch_file`

### 7.2 服务启动时加载，而不是每请求加载

建议模型加载策略：

- 服务进程启动时加载一次默认 bundle；
- 后续请求只复用常驻模型。

原因：

- 避免重复注入 LoRA；
- 避免重复 patch speaker embedding；
- 降低首请求抖动；
- 让第三方客户端不必承担内部装配复杂度。

### 7.3 启动期必须校验 public alias 与实际模型的一致性

启动阶段应明确检查：

- `tts_model_type == custom_voice`
- 默认 speaker 已存在于 `supported_speakers`
- `voice_registry` 中的公开 `voice` 能正确映射到实际 profile
- 对外暴露的 `model` 别名与实际服务能力匹配

## 8. 关键设计决策

### 8.1 为什么 V1 先做公开协议兼容，而不是继续扩展旧协议

因为接口一旦对外发布，就会形成历史债。

当前服务尚未真正落地，是把公开契约一次定对的最佳时机。

### 8.2 为什么 LoRA 细节不暴露给客户端

因为第三方更关心：

- `model` 是什么
- `voice` 用什么
- 文本怎么送
- 音频怎么收

而不是想理解我们的 bundle 装配链路。

### 8.3 为什么公开输出选 `response.audio.delta`

因为相较“自定义 JSON meta + 二进制 PCM 帧”：

- 更接近官方文档；
- 更方便 SDK 与浏览器处理；
- 更适合作为面向第三方的统一公开协议。

## 9. 风险与缓解

### 风险 1：逐步生成与离线整段生成结果不完全一致

缓解：

- V1 允许与离线结果存在轻微差异；
- 优先保证稳定 chunk 输出与可中断性。

### 风险 2：官方风格外观与本地能力范围不完全重合

缓解：

- 通过兼容矩阵明确“完全兼容 / 部分兼容 / 暂不支持”；
- 避免对外宣称“完全等同官方云服务”。

### 风险 3：公开 `voice` 命名与内部 speaker 命名强绑定，后期难演进

缓解：

- 通过独立 `voice_registry` 做别名层；
- 不让公开 voice 直接等于内部文件名。

### 风险 4：Windows 环境吞吐不稳定

缓解：

- V1 先做单模型串行生成；
- 会话层增加排队与限流；
- 不承诺高并发。

## 10. 第一版建议 CLI

后续建议新增入口：

```text
qwen-tts-realtime-serve --bundle_dir <path> --host 0.0.0.0 --port 9000
```

建议参数：

- `--bundle_dir`
- `--host`
- `--port`
- `--device_map`
- `--torch_dtype`
- `--attn_implementation`
- `--local_files_only`
- `--public_model_alias`
- `--default_voice_alias`
- `--chunk_steps`
- `--left_context_steps`
- `--max_sessions`

## 11. 本阶段结论

结论很明确：

- **LoRA 加载路径已经成熟**；
- **真正缺的是 runtime session、增量 decoder 与公开协议适配层**；
- **V1 最合理的落地方向是：官方风格的公开 Realtime 接口 + 服务端默认 LoRA 挂载。**
