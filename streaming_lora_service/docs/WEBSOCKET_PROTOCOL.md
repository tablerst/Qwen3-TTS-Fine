# WebSocket Protocol

## 1. 设计目标

协议目标是支持：

- 持续上行文本增量
- 持续下行音频 chunk
- 明确的会话生命周期
- 可观测的错误与指标

V1 协议默认围绕 `custom_voice + bundle_dir` 设计。

## 2. 连接模型

- 一个 WebSocket 连接对应一个活跃 session
- 一次连接只服务一个 speaker profile
- 如果后续需要多 session 复用，建议另开 v2 协议，而不是直接把 V1 搅浑

## 3. 客户端 -> 服务端消息

所有控制消息使用 JSON 文本帧。

### 3.1 `session.start`

用途：初始化会话。

```json
{
  "type": "session.start",
  "request_id": "req-001",
  "audio": {
    "format": "pcm_s16le",
    "sample_rate": 24000
  },
  "voice": {
    "speaker": "inference_speaker",
    "language": "Auto",
    "instruct": ""
  },
  "generation": {
    "chunk_steps": 4,
    "max_new_tokens": 2048,
    "temperature": 0.9,
    "top_k": 50,
    "top_p": 1.0,
    "repetition_penalty": 1.05
  }
}
```

说明：

- `speaker` 默认应与 bundle 内 speaker 一致；
- V1 可以只允许 bundle 中唯一 speaker；
- `chunk_steps` 可选，服务端可做裁剪与兜底。

### 3.2 `text.delta`

用途：提交文本增量。

```json
{
  "type": "text.delta",
  "text": "你好，今天"
}
```

说明：

- 服务端收到后追加到 `raw_text_buffer`；
- 不保证立刻全部提交给模型；
- 由 commit policy 决定稳定前缀。

### 3.3 `text.flush`

用途：强制提交当前未稳定尾部。

```json
{
  "type": "text.flush"
}
```

适用场景：

- 用户停顿
- 当前一轮输入结束
- 需要尽快让模型把手头文本说完

### 3.4 `session.stop`

用途：优雅结束会话。

```json
{
  "type": "session.stop"
}
```

语义：

- 服务端尽量发送已生成但尚未发出的尾部音频；
- 然后返回 `session.completed`。

### 3.5 `session.cancel`

用途：立即终止会话。

```json
{
  "type": "session.cancel"
}
```

语义：

- 立即停止生成；
- 可丢弃未发音频；
- 返回 `session.cancelled`。

## 4. 服务端 -> 客户端消息

### 4.1 `session.ready`

```json
{
  "type": "session.ready",
  "session_id": "sess-001",
  "audio": {
    "format": "pcm_s16le",
    "sample_rate": 24000,
    "channels": 1
  },
  "voice": {
    "speaker": "inference_speaker",
    "language": "Auto"
  }
}
```

### 4.2 `audio.chunk.meta`

建议使用一个 JSON 元信息帧，紧接着一个二进制帧。

元信息示例：

```json
{
  "type": "audio.chunk.meta",
  "seq": 1,
  "samples": 7680,
  "sample_rate": 24000,
  "is_final": false
}
```

紧随其后的二进制帧：

- 内容：PCM16 little-endian
- 长度：`samples * 2` bytes

### 4.3 `metrics`

```json
{
  "type": "metrics",
  "ttfb_ms": 420,
  "generated_steps": 12,
  "emitted_chunks": 3
}
```

说明：

- V1 可以低频发送，例如首包后一次、会话结束时一次；
- 不建议每个 step 都发，太吵。

### 4.4 `session.completed`

```json
{
  "type": "session.completed",
  "reason": "stop",
  "generated_steps": 36,
  "emitted_chunks": 9
}
```

### 4.5 `session.cancelled`

```json
{
  "type": "session.cancelled",
  "reason": "client_cancel"
}
```

### 4.6 `session.error`

```json
{
  "type": "session.error",
  "code": "INVALID_MESSAGE",
  "message": "Missing field: type"
}
```

## 5. 错误码建议

建议至少包含：

- `INVALID_MESSAGE`
- `INVALID_STATE`
- `UNSUPPORTED_SPEAKER`
- `UNSUPPORTED_LANGUAGE`
- `MODEL_NOT_READY`
- `SESSION_ALREADY_STARTED`
- `SESSION_NOT_STARTED`
- `GENERATION_FAILED`
- `DECODE_FAILED`
- `SERVER_OVERLOADED`

## 6. 时序建议

### 6.1 正常流程

```text
client -> session.start
server -> session.ready
client -> text.delta
client -> text.delta
server -> audio.chunk.meta
server -> <binary pcm>
client -> text.flush
server -> audio.chunk.meta
server -> <binary pcm>
client -> session.stop
server -> session.completed
```

### 6.2 取消流程

```text
client -> session.start
server -> session.ready
client -> text.delta
server -> audio.chunk.meta
server -> <binary pcm>
client -> session.cancel
server -> session.cancelled
```

## 7. 服务端策略建议

### 7.1 文本提交策略

服务端不要盲目把每个 `text.delta` 都原样立即喂给模型，应维护：

- `raw_text_buffer`
- `committed_text`
- `pending_text_tail`

推荐策略：

- 标点优先提交
- 英文按空格/单词边界提交
- 中文允许更细粒度提交
- `flush` 时强制提交尾部

### 7.2 chunk 下发策略

建议：

- 至少积累 `chunk_steps >= 4`
- 或收到 `flush`
- 或检测到 eos

再执行一次增量 decode 并下发。

## 8. V1 协议边界

V1 不建议支持：

- 一条连接内切换 speaker
- 一条连接内切换 bundle
- 上行音频参考流
- 多模态输入

## 9. 前端对接建议

浏览器侧建议：

- JSON 帧用来解析控制信息；
- 二进制 PCM chunk 进入播放缓冲；
- 建议客户端自己维护 `seq` 连续性检查；
- 出现断序时可记录错误但不中断整条连接。
