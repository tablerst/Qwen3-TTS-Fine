# Streaming LoRA Service

本目录用于承载 **面向第三方的 Qwen-compatible TTS / Realtime TTS 服务设计** 与后续代码落地点。

当前阶段仍以**文档冻结与接口收敛**为主，但方向已经明确：

> **不再继续扩展旧的自定义公开 WebSocket 协议；**
> **从第一版开始，直接采用 Qwen3-TTS / Qwen-TTS-Realtime 风格的对外接口。**

这意味着：

- **对外**：字段名、事件名、交互节奏尽量向官方 Qwen3-TTS / Realtime 看齐；
- **对内**：仍然复用本仓库已有的 `Qwen3TTSModel`、LoRA bundle、speaker patch 与真流式生成链路；
- **部署侧**：默认 LoRA bundle 由服务启动配置决定，而不是暴露给第三方调用方的心智负担。

## 设计原则

### 1. 对外官方风格，内部自有实现

服务对外暴露的协议目标是：

- 请求字段尽量兼容官方 Qwen3-TTS：如 `model`、`voice`、`language_type`、`instructions`；
- Realtime 事件尽量兼容官方 Qwen-TTS-Realtime：如 `session.update`、`input_text_buffer.append`、`response.audio.delta`；
- 文档心智、错误模型、示例代码尽量让第三方“一眼会用”。

但内部不要求照搬官方实现：

- 仍由服务进程启动时默认挂载 LoRA bundle；
- 仍由内部 runtime 负责真流式 step 推进与增量解码；
- 仍由自有 voice registry 把公开 `voice` 映射到 bundle / speaker profile。

### 2. LoRA 是服务端内部能力，不是客户端接口负担

第三方调用时不应被迫理解：

- `bundle_dir`
- `adapter_dir`
- `config_patch.json`
- `speaker_embedding.safetensors`

这些内容属于**部署与服务内部装配细节**，不属于 V1 公开 API。

### 3. 真流式优先于“只长得像流式”

这里的目标不是“把一次性生成套上一层 WebSocket”，而是：

- 输入可增量追加；
- 生成可 step 级推进；
- 音频可持续下发；
- 会话状态可持续维护；
- 对外协议虽然官方风格，但内部仍然是真流式实现。

## V1 定位

V1 面向的是：

- **Qwen-compatible Realtime WebSocket** 公开接口；
- **Qwen-compatible HTTP TTS** 公开接口；
- `custom_voice` 场景优先；
- 服务启动时默认加载一个 LoRA bundle；
- 对外通过 `model` / `voice` 暴露稳定别名；
- 下行音频采用官方风格的 `response.audio.delta`（Base64 音频块）。

V1 暂不追求：

- 完整 DashScope 路径级兼容；
- 把 LoRA bundle 参数暴露为客户端必填项；
- 一开始就支持所有 `voice_clone` / `voice_design` 实时能力；
- 多 bundle 热切换与复杂多租户路由；
- 二进制 PCM 帧作为公开主协议。

## 为什么要单独开目录

现有仓库已经具备：

- 一次性离线推理：`qwen_tts/inference/qwen3_tts_model.py`
- 本地 Web UI：`qwen_tts/cli/demo.py`
- LoRA bundle 推理：`lora_finetuning/inference_with_lora.py`

但还没有一套真正适合**对外服务化**的独立结构。新目录的作用是：

1. 把“公开 API 设计”与“训练/导出脚本”解耦；
2. 把 LoRA bundle 装配逻辑与 Realtime 协议适配逻辑收敛在同一域；
3. 给后续代码、测试、兼容矩阵、SDK 示例、压测留出明确位置。

## 与现有代码的关系

这条方案不会推翻现有实现，而是复用下面这些关键能力：

- `Qwen3TTSModel.from_pretrained(...)`
- `Qwen3TTSForConditionalGeneration.generate(...)` 中已有的 prompt / prefill 逻辑
- `Qwen3TTSTalkerForConditionalGeneration.forward(...)` 中维护的增量生成状态
- `Qwen3TTSTokenizerV2Decoder.chunked_decode(...)` 的按块解码能力
- `lora_finetuning/common.py` 中的 LoRA / speaker patch 加载逻辑

## 建议后续代码结构

后续如果开始写代码，建议优先在本目录中补以下子结构：

- `streaming_lora_service/app/`
  - `bundle_loader.py`
  - `voice_registry.py`
  - `runtime_session.py`
  - `incremental_decoder.py`
  - `qwen_compat_ws.py`
  - `qwen_compat_http.py`
  - `server.py`
- `streaming_lora_service/tests/`
  - `test_bundle_loader.py`
  - `test_voice_registry.py`
  - `test_protocol_smoke.py`
  - `test_runtime_session.py`

## 当前实现进度（2026-03-11）

当前目录已经落下第一批**可运行的代码骨架**，不再只是纯文档：

- `app/models.py`：公共数据结构与会话配置模型
- `app/voice_registry.py`：公开 `voice` 别名解析
- `app/bundle_loader.py`：默认 LoRA bundle 解析与加载骨架
- `app/runtime_session.py`：文本缓冲区 / commit / finish 状态机骨架
- `app/incremental_decoder.py`：增量解码规划与 overlap 裁剪逻辑骨架
- `app/qwen_compat_ws.py`：Qwen-compatible Realtime 事件适配器 MVP
- `app/server.py`：可启动的 FastAPI WebSocket 服务 MVP
- `app/audio_utils.py`：浮点音频转 PCM16 与 chunk 切分工具
- `configs/voice_registry.example.json`：公开 `voice` 映射配置示例

同时已经补齐第一批单元测试：

- `tests/test_bundle_loader.py`
- `tests/test_voice_registry.py`
- `tests/test_runtime_session.py`
- `tests/test_incremental_decoder.py`
- `tests/test_protocol_smoke.py`
- `tests/test_server_smoke.py`
- `tests/test_http_tts_smoke.py`

并已完成一次本地验证：

- `python -m unittest discover -s streaming_lora_service/tests -v`
- 结果：**18 个测试全部通过**

## 当前 MVP 能力边界

当前已经完成的是一个**可运行的兼容层 MVP**：

- 服务启动时加载默认 LoRA bundle
- 通过官方风格 WebSocket 事件对外提供能力
- 通过官方风格 HTTP 请求/响应字段对外提供能力
- 支持 `session.update / input_text_buffer.append / input_text_buffer.commit / input_text_buffer.clear / session.finish`
- 支持 `response.audio.delta / response.audio.done / response.done / session.finished`
- 提供 `/healthz` 与 `/v1/voices` 两个基础 HTTP 端点
- 提供 HTTP TTS 路由：`/v1/tts`、`/v1/audio/speech`、`/api/v1/services/aigc/multimodal-generation/generation`
- 提供音频下载路由：`/v1/audio/{audio_id}`
- 支持从 JSON / YAML voice registry 文件加载公开 `voice` 别名映射

当前**尚未完成**的部分也需要明确说明：

- `response.audio.delta` 目前是基于现有 `generate_custom_voice()` 一次性生成后的**分块下发 MVP**；
- 还不是严格意义上的 step-level 真流式生成；
- 真正的 prompt builder / step generator / 增量 codec 级生成仍是下一阶段工作。

也就是说：

> **当前 MVP 已经可用、可跑、可联调；**
> **但它是“官方风格兼容优先”的 MVP，不是最终版真流式内核。**

## 如何启动 MVP

安装完依赖并确保默认 LoRA bundle 可用后，可直接启动：

```text
qwen-tts-realtime-serve --bundle_dir <path_to_bundle>
```

常见参数示例：

```text
qwen-tts-realtime-serve --bundle_dir outputs/lora_formal_single_speaker_1p7b_bundle --public_model_alias qwen3-tts-flash-realtime --default_voice_alias yachiyo_formal --voice_registry_file streaming_lora_service/configs/voice_registry.example.json --host 127.0.0.1 --port 9000
```

默认端点：

- WebSocket：`/api-ws/v1/realtime`
- 健康检查：`/healthz`
- Voice 列表：`/v1/voices`
- HTTP TTS：`/v1/tts`
- HTTP TTS（兼容别名）：`/v1/audio/speech`
- HTTP TTS（官方风格兼容别名）：`/api/v1/services/aigc/multimodal-generation/generation`
- 音频下载：`/v1/audio/{audio_id}`

## 下一阶段建议

下一阶段建议优先补下面 4 件事：

1. `qwen_compat_ws.py` 从一次性生成分块下发升级到真正 step-level 音频下发
2. `runtime_session` 接真实 prompt builder / step generator
3. `incremental_decoder` 接真实 codec ring buffer
4. 增加真实 bundle 的端到端联调与试听回归

## 先读哪几份文档

建议按顺序阅读：

1. `docs/QWEN_COMPAT_API.md`
2. `docs/QWEN_COMPATIBILITY_MATRIX.md`
3. `docs/IMPLEMENTATION_PLAN.md`
4. `docs/ARCHITECTURE.md`
5. `docs/WEBSOCKET_PROTOCOL.md`
6. `docs/DELIVERY_CHECKLIST.md`

## 当前结论

**这件事应该直接按 Qwen-compatible 的正式公开接口落地。**

最关键的原则只有一条：

> 公开协议从第一版开始就做官方风格；
> LoRA 默认挂载、bundle 路由、speaker patch 下沉到服务内部；
> 真流式 runtime 仍由我们自己的实现负责。
