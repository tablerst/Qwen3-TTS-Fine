# Qwen-Compatible API Design

## 1. 文档定位

本文定义本项目面向第三方的**正式公开接口方向**。

目标是提供一套：

- **看起来像官方 Qwen3-TTS / Qwen-TTS-Realtime** 的调用方式；
- **内部默认挂载我们自己的 LoRA bundle** 的服务实现；
- **对第三方足够简单** 的公开心智模型。

简言之：

> 外观按官方风格；
> 引擎仍是我们的 LoRA + 真流式实现。

## 2. 兼容范围

V1 的兼容范围定义为：

### 2.1 强兼容项

- 公开字段命名
- Realtime 事件命名
- HTTP TTS 请求/响应字段命名
- `server_commit` / `commit` 模式语义
- `response.audio.delta` 的输出形态
- `model` / `voice` / `language_type` / `instructions` 的基础语义

### 2.2 部分兼容项

- 官方模型名：支持公开别名兼容，但底层为本地实现
- 官方音色名：仅在有对应 registry 映射时兼容
- HTTP 非实时接口：建议共享同一套字段体系，但可晚于 Realtime 落地

### 2.3 非目标

- 完整 DashScope 路径与鉴权行为的逐项兼容
- 官方云端全部模型与音色全集
- 官方云端专属音色注册流程

## 3. 对外与对内的边界

### 3.1 对外暴露什么

第三方调用方应只需要理解：

- `model`
- `voice`
- `language_type`
- `instructions`
- WebSocket 事件或 HTTP 请求

### 3.2 不对外暴露什么

普通客户端不应需要理解：

- `bundle_dir`
- `adapter_dir`
- `config_patch.json`
- `speaker_embedding.safetensors`
- speaker patch 注入细节

这些属于服务端装配与部署细节。

## 4. 公开对象模型

### 4.1 `model`

`model` 是对外公开的**能力别名**，不要求与本地模型目录名一一对应。

建议 V1 先支持稳定的公开别名，例如：

- `qwen3-tts-flash-realtime`
- `qwen3-tts-instruct-flash-realtime`
- `qwen3-tts-flash`

实际由服务端映射到：

- 当前加载的 base model
- 当前默认 LoRA bundle
- 当前可用的 prompt / instruct / speaker 策略

### 4.2 `voice`

`voice` 是对外公开的**音色别名**，不必直接等于内部 speaker 名或文件名。

推荐通过 `voice_registry` 做别名映射，例如：

- `yachiyo_formal`
- `yachiyo_timbre_transfer`
- `default_female_cn`

这样未来更换 bundle 或 speaker patch 时，第三方调用接口不需要大改。

### 4.3 `language_type`

沿用官方风格：

- `Auto`
- `Chinese`
- `English`
- 其他后续支持语种

### 4.4 `instructions`

沿用官方指令控制语义：

- 用自然语言描述声音风格与表达方式；
- 是否生效取决于当前公开 `model` 是否支持 instruct 风格控制。

## 5. Realtime WebSocket（V1 必做）

V1 的主公开接口为 Realtime WebSocket。

### 5.1 客户端事件

- `session.update`
- `input_text_buffer.append`
- `input_text_buffer.commit`
- `input_text_buffer.clear`
- `session.finish`

### 5.2 服务端事件

- `session.created`
- `session.updated`
- `input_text_buffer.committed`
- `input_text_buffer.cleared`
- `response.created`
- `response.output_item.added`
- `response.content_part.added`
- `response.audio.delta`
- `response.audio.done`
- `response.content_part.done`
- `response.output_item.done`
- `response.done`
- `session.finished`
- `error`

### 5.3 关键约束

- `response.audio.delta` 为公开主输出形态；
- 当前 WebSocket Realtime V1 **只接受** `response_format="pcm"` 与 `sample_rate=24000`；不支持的值会直接返回 `error`，而不是静默降级；
- 普通调用不暴露 `bundle_dir`；
- V1 不再把旧自定义事件作为主协议继续维护。
- 当前 `response.created` / `response.done` 中已补充 `modalities`、`conversation_id` 等更接近官方风格的字段。

## 6. HTTP TTS（建议共享同一套语义）

当前 MVP 已经提供 HTTP TTS 兼容入口，并共享同一套公开字段：

- `model`
- `text`
- `voice`
- `language_type`
- `instructions`
- `optimize_instructions`
- `stream`

这样可以确保：

- HTTP 与 Realtime 使用同一套 `voice` / `model` 认知；
- 不会出现两套风格完全不同的外部接口。

当前提供的 HTTP 路由：

- `POST /v1/tts`
- `POST /v1/audio/speech`
- `POST /api/v1/services/aigc/multimodal-generation/generation`

当前提供的音频下载路由：

- `GET /v1/audio/{audio_id}`

### 6.1 HTTP 请求示例

```json
{
  "model": "qwen3-tts-flash-realtime",
  "text": "你好，欢迎体验 HTTP 兼容接口。",
  "voice": "yachiyo_formal",
  "language_type": "Chinese",
  "instructions": "",
  "optimize_instructions": false,
  "stream": false
}
```

### 6.2 HTTP 非流式响应示例

```json
{
  "status_code": 200,
  "request_id": "req_xxx",
  "code": "",
  "message": "",
  "output": {
    "text": null,
    "finish_reason": "stop",
    "choices": null,
    "audio": {
      "data": "",
      "url": "http://127.0.0.1:9000/v1/audio/audio_xxx",
      "id": "audio_xxx"
    }
  },
  "usage": {
    "characters": 14
  }
}
```

### 6.3 HTTP 流式响应说明

当 `stream=true` 时，当前 MVP 返回 `application/x-ndjson`：

- 中间块包含 `output.audio.data`（Base64 音频片段）
- 中间块当前固定为 **24kHz / 单声道 / PCM16** 片段
- 最后一块返回 `finish_reason = "stop"`

## 7. 推荐示例

### 7.1 Realtime 初始化

```json
{
  "event_id": "event_123",
  "type": "session.update",
  "session": {
    "model": "qwen3-tts-flash-realtime",
    "voice": "yachiyo_formal",
    "language_type": "Chinese",
    "mode": "server_commit",
    "response_format": "pcm",
    "sample_rate": 24000,
    "instructions": "",
    "optimize_instructions": false
  }
}
```

### 7.2 Realtime 增量文本

```json
{
  "event_id": "event_124",
  "type": "input_text_buffer.append",
  "text": "大家好，欢迎来到今天的演示。"
}
```

### 7.3 Realtime 音频输出

```json
{
  "event_id": "event_srv_007",
  "type": "response.audio.delta",
  "response_id": "resp_001",
  "item_id": "item_001",
  "output_index": 0,
  "content_index": 0,
  "delta": "<base64-encoded-audio-chunk>"
}
```

## 8. 部署侧配置建议

V1 建议把下面这些参数限定在服务部署侧：

- `--bundle_dir`
- `--public_model_alias`
- `--default_voice_alias`
- `--voice_registry_file`（可选）
- `--chunk_steps`
- `--left_context_steps`

这样第三方客户端调用时只需要关注公开契约，不用理解服务装配过程。

## 9. 最终原则

这个目录后续所有设计与代码实现，都应遵守以下原则：

1. **公开 API 从第一版开始就是官方风格兼容接口；**
2. **LoRA 默认挂载属于服务端行为，不增加第三方心智负担；**
3. **内部实现可自由演进，但公开 `model` / `voice` / event contract 尽量保持稳定。**
