# Qwen-Compatible Realtime WebSocket Protocol

## 1. 协议定位

本协议是本项目面向第三方的 **Qwen-compatible Realtime TTS** 公开协议。

协议目标不是继续维护旧的自定义事件：

- `session.start`
- `text.delta`
- `text.flush`
- `session.stop`
- `session.cancel`

而是从第一版开始直接采用**官方 Qwen-TTS-Realtime 风格**的事件与字段命名。

## 2. 兼容目标

V1 的兼容目标是：

- **事件级兼容优先**：优先兼容官方事件名、字段名和会话节奏；
- **语义级兼容优先**：优先兼容 `server_commit` / `commit` 两种使用模式；
- **实现级不强绑定**：内部 runtime 仍由本仓库自己的 LoRA + 流式生成链路实现；
- **部署级 LoRA 隐藏**：客户端不需要理解 `bundle_dir`、`adapter_dir` 等内部细节。

V1 不要求：

- 完整 DashScope 路径级兼容；
- 完整鉴权头 / SDK 行为的逐字节兼容；
- 把内部二进制 PCM 通道暴露为公开协议主形态。

## 3. 连接模型

- 一个 WebSocket 连接对应一个活跃 session；
- 一个 session 在 V1 中只绑定一个公开 `model` 与一个公开 `voice`；
- `model` / `voice` 是**对外稳定别名**，内部由服务解析到默认 LoRA bundle 与 speaker profile；
- 调用方无需感知 bundle 路由细节。

## 4. 客户端 -> 服务端事件

所有客户端事件使用 JSON 文本帧。

### 4.1 `session.update`

用途：初始化或更新会话配置。

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

说明：

- `model` / `voice` 为对外公开标识，而不是内部文件路径；
- 服务端可将其映射到默认 LoRA bundle 与目标 speaker；
- 常规第三方调用不需要知道 `bundle_dir`；
- 当前 WebSocket Realtime V1 **仅支持** `response_format=pcm`、`sample_rate=24000`；
- 若客户端提交其他值，服务端会直接返回 `error`，避免出现“会话看起来更新成功，但实际下行音频仍是 24k PCM16”的假兼容。

### 4.2 `input_text_buffer.append`

用途：向文本缓冲区追加待合成文本。

```json
{
  "event_id": "event_124",
  "type": "input_text_buffer.append",
  "text": "你好，今天我们来介绍新产品。"
}
```

说明：

- 在 `server_commit` 模式下，服务端负责判断切分与触发时机；
- 在 `commit` 模式下，仅追加文本，不立即触发合成。

### 4.3 `input_text_buffer.commit`

用途：提交当前缓冲区文本。

```json
{
  "event_id": "event_125",
  "type": "input_text_buffer.commit"
}
```

说明：

- `server_commit` 模式下可视作“立即把当前尾部文本说出来”；
- `commit` 模式下必须显式发送该事件，才会触发音频生成流程。

### 4.4 `input_text_buffer.clear`

用途：清空当前尚未提交的文本缓冲区。

```json
{
  "event_id": "event_126",
  "type": "input_text_buffer.clear"
}
```

### 4.5 `session.finish`

用途：通知服务端不会再有后续文本输入。

```json
{
  "event_id": "event_127",
  "type": "session.finish"
}
```

语义：

- 服务端尽量把剩余已生成音频发送完；
- 然后返回 `response.done` 与 `session.finished`；
- 公开协议不再使用自定义 `session.stop` / `session.cancel` 作为主事件名。

## 5. 服务端 -> 客户端事件

### 5.1 `session.created`

连接建立后，服务端返回默认或当前会话配置。

```json
{
  "event_id": "event_srv_001",
  "type": "session.created",
  "session": {
    "object": "realtime.session",
    "id": "sess_001",
    "model": "qwen3-tts-flash-realtime",
    "voice": "yachiyo_formal",
    "mode": "server_commit",
    "response_format": "pcm",
    "sample_rate": 24000
  }
}
```

### 5.2 `session.updated`

服务端成功应用 `session.update` 后返回。

```json
{
  "event_id": "event_srv_002",
  "type": "session.updated",
  "session": {
    "object": "realtime.session",
    "id": "sess_001",
    "model": "qwen3-tts-flash-realtime",
    "voice": "yachiyo_formal",
    "language_type": "Chinese",
    "mode": "server_commit",
    "response_format": "pcm",
    "sample_rate": 24000
  }
}
```

### 5.3 `input_text_buffer.committed`

服务端确认一次文本提交。

```json
{
  "event_id": "event_srv_003",
  "type": "input_text_buffer.committed",
  "item_id": "item_001"
}
```

### 5.4 `response.created`

服务端开始生成一轮音频响应时返回。

```json
{
  "event_id": "event_srv_004",
  "type": "response.created",
  "response": {
    "id": "resp_001",
    "object": "realtime.response",
    "status": "in_progress",
    "voice": "yachiyo_formal",
    "output": []
  }
}
```

### 5.5 `response.output_item.added`

当新的输出 item 建立时返回。

```json
{
  "event_id": "event_srv_005",
  "type": "response.output_item.added",
  "response_id": "resp_001",
  "output_index": 0,
  "item": {
    "id": "item_001",
    "object": "realtime.item",
    "type": "message",
    "status": "in_progress",
    "role": "assistant",
    "content": []
  }
}
```

### 5.6 `response.content_part.added`

当新的内容 part（通常是音频 part）建立时返回。

```json
{
  "event_id": "event_srv_006",
  "type": "response.content_part.added",
  "response_id": "resp_001",
  "item_id": "item_001",
  "output_index": 0,
  "content_index": 0,
  "part": {
    "type": "audio",
    "text": ""
  }
}
```

### 5.7 `response.audio.delta`

当模型产生新增音频片段时返回。

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

说明：

- V1 公开协议主形态为 **Base64 音频块**；
- 当前实现固定为 **24kHz 单声道 PCM16** 对应的字节流编码结果；
- 不再把 `audio.chunk.meta + binary pcm` 作为公开协议标准；
- 若未来保留二进制 PCM 通道，应作为内部/native 扩展能力，不纳入主公开协议。

### 5.8 `response.audio.done`

当前音频内容 part 已完成。

```json
{
  "event_id": "event_srv_008",
  "type": "response.audio.done",
  "response_id": "resp_001",
  "item_id": "item_001",
  "output_index": 0,
  "content_index": 0
}
```

### 5.9 `response.content_part.done`

```json
{
  "event_id": "event_srv_009",
  "type": "response.content_part.done",
  "response_id": "resp_001",
  "item_id": "item_001",
  "output_index": 0,
  "content_index": 0,
  "part": {
    "type": "audio",
    "text": ""
  }
}
```

### 5.10 `response.output_item.done`

```json
{
  "event_id": "event_srv_010",
  "type": "response.output_item.done",
  "response_id": "resp_001",
  "output_index": 0,
  "item": {
    "id": "item_001",
    "object": "realtime.item",
    "type": "message",
    "status": "completed",
    "role": "assistant",
    "content": [
      {
        "type": "audio",
        "text": ""
      }
    ]
  }
}
```

### 5.11 `response.done`

一轮响应完成时返回。

```json
{
  "event_id": "event_srv_011",
  "type": "response.done",
  "response": {
    "id": "resp_001",
    "object": "realtime.response",
    "status": "completed",
    "voice": "yachiyo_formal",
    "output": [
      {
        "id": "item_001",
        "object": "realtime.item",
        "type": "message",
        "status": "completed",
        "role": "assistant",
        "content": [
          {
            "type": "audio",
            "text": ""
          }
        ]
      }
    ],
    "usage": {
      "characters": 24
    }
  }
}
```

### 5.12 `session.finished`

会话生命周期结束时返回。

```json
{
  "event_id": "event_srv_012",
  "type": "session.finished"
}
```

### 5.13 `error`

出现客户端输入错误或服务端错误时返回。

```json
{
  "event_id": "event_srv_err_001",
  "type": "error",
  "error": {
    "code": "invalid_value",
    "message": "Unsupported voice: yachiyo_unknown"
  }
}
```

## 6. 模式说明

### 6.1 `server_commit`

- 客户端只需持续发送 `input_text_buffer.append`；
- 服务端根据内部 commit policy 判断文本切分与生成时机；
- 若客户端主动发送 `input_text_buffer.commit`，表示“立即把当前尾部文本交给模型”。

### 6.2 `commit`

- 客户端先发送一个或多个 `input_text_buffer.append`；
- 仅当发送 `input_text_buffer.commit` 后，才正式触发音频生成；
- 适用于需要业务侧精细控制语音节奏的场景。

## 7. 与内部 runtime 的映射关系

| 对外兼容事件 | 内部语义 |
| --- | --- |
| `session.update` | 创建/更新 session 配置，解析 `model` 与 `voice` 到 bundle / speaker profile |
| `input_text_buffer.append` | `runtime.append_text(...)` |
| `input_text_buffer.commit` | `runtime.commit()` |
| `input_text_buffer.clear` | `runtime.clear_pending_text()` |
| `session.finish` | `runtime.finish()` |
| `response.audio.delta` | `incremental_decoder` 输出的新增音频字节经 Base64 编码 |

也就是说：

- **公开接口看起来像官方；**
- **内部仍然是我们自己的 LoRA + session + decoder 实现。**

## 8. 推荐时序

### 8.1 `server_commit` 正常流程

```text
client -> session.update
server -> session.created
server -> session.updated
client -> input_text_buffer.append
client -> input_text_buffer.append
server -> response.created
server -> response.output_item.added
server -> response.content_part.added
server -> response.audio.delta
server -> response.audio.delta
client -> session.finish
server -> response.audio.done
server -> response.content_part.done
server -> response.output_item.done
server -> response.done
server -> session.finished
```

### 8.2 `commit` 正常流程

```text
client -> session.update
server -> session.created
server -> session.updated
client -> input_text_buffer.append
client -> input_text_buffer.append
client -> input_text_buffer.commit
server -> input_text_buffer.committed
server -> response.created
server -> response.audio.delta
server -> response.done
client -> session.finish
server -> session.finished
```

## 9. V1 边界

V1 不建议支持：

- 在同一连接内切换 bundle；
- 在同一连接内切换 voice profile；
- 上行参考音频流；
- 多模态输入；
- 公开暴露 `bundle_dir`、`adapter_dir` 等部署参数。

## 10. 公开协议废弃项

下列旧事件不再作为公开协议继续扩展：

- `session.start`
- `text.delta`
- `text.flush`
- `session.stop`
- `session.cancel`
- `audio.chunk.meta`

如果未来保留这些能力，应当：

- 只作为内部/native 协议；
- 或在兼容层内转译为官方风格事件；
- 不再作为面向第三方的主文档标准。
