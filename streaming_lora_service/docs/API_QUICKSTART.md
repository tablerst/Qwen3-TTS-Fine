# API Quickstart

本文面向准备联调 `streaming_lora_service` 的调用方，提供一份可以直接照抄的启动与调用说明。

## 1. 当前联调实例信息

建议联调时使用下面这组服务参数：

- `bundle_dir`: `outputs/lora_formal_single_speaker_1p7b_timbre_transfer_20260310_bundle_best_refcand8`
- `voice_registry_file`: `streaming_lora_service/configs/voice_registry.example.json`
- `public_model_alias`: `qwen3-tts-flash-realtime`
- `default_voice_alias`: `yachiyo_formal`
- `host`: `0.0.0.0`
- `port`: `9010`

本机访问地址：

- HTTP Base URL: `http://127.0.0.1:9010`
- WebSocket URL: `ws://127.0.0.1:9010/api-ws/v1/realtime`

如果需要局域网其他机器联调，请把 `127.0.0.1` 替换成服务机实际 IP，并确认防火墙已放行 `9010` 端口。

## 2. 可用接口一览

### 2.1 基础接口

- `GET /healthz`
- `GET /v1/voices`
- `GET /v1/audio/{audio_id}`

### 2.2 HTTP TTS 接口

以下三个路径当前等价：

- `POST /v1/tts`
- `POST /v1/audio/speech`
- `POST /api/v1/services/aigc/multimodal-generation/generation`

### 2.3 WebSocket Realtime 接口

- `WS /api-ws/v1/realtime`

## 3. HTTP 请求字段

所有 HTTP TTS 接口共享相同请求字段：

- `model`: 公开模型别名，当前使用 `qwen3-tts-flash-realtime`
- `text`: 待合成文本
- `voice`: 公开音色别名，当前示例使用 `yachiyo_formal`
- `language_type`: `Auto` / `Chinese` / `Japanese` 等
- `instructions`: 风格指令
- `optimize_instructions`: 当前已接字段，默认 `false`
- `stream`: 是否启用流式输出

推荐联调请求体：

```json
{
  "model": "qwen3-tts-flash-realtime",
  "text": "你好，欢迎使用实时语音服务。",
  "voice": "yachiyo_formal",
  "language_type": "Chinese",
  "instructions": "正式，平静，清晰。",
  "optimize_instructions": false,
  "stream": false
}
```

## 4. HTTP 非流式调用示例

### 4.1 curl

```bash
curl -X POST "http://127.0.0.1:9010/v1/tts" \
  -H "Content-Type: application/json" \
  -d '{
    "model": "qwen3-tts-flash-realtime",
    "text": "你好，欢迎使用实时语音服务。",
    "voice": "yachiyo_formal",
    "language_type": "Chinese",
    "instructions": "正式，平静，清晰。",
    "optimize_instructions": false,
    "stream": false
  }'
```

### 4.2 Python

```python
import requests

payload = {
    "model": "qwen3-tts-flash-realtime",
    "text": "你好，欢迎使用 HTTP 接口。",
    "voice": "yachiyo_formal",
    "language_type": "Chinese",
    "instructions": "正式，平静，清晰。",
    "optimize_instructions": False,
    "stream": False,
}

resp = requests.post("http://127.0.0.1:9010/v1/tts", json=payload, timeout=600)
resp.raise_for_status()
result = resp.json()
print(result)

audio_url = result["output"]["audio"]["url"]
wav_resp = requests.get(audio_url, timeout=600)
wav_resp.raise_for_status()
with open("demo.wav", "wb") as f:
    f.write(wav_resp.content)
```

### 4.3 非流式响应示例

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
      "url": "http://127.0.0.1:9010/v1/audio/audio_xxx",
      "id": "audio_xxx"
    }
  },
  "usage": {
    "characters": 14
  }
}
```

## 5. HTTP 流式调用示例

当 `stream=true` 时，服务返回 `application/x-ndjson`。

- 中间多行 JSON：`output.audio.data` 为 Base64 PCM 音频块
- 中间多行 JSON 当前固定为 **24kHz / 单声道 / PCM16** 音频块
- 最后一行 JSON：`finish_reason="stop"`，并包含完整音频的 `audio.id` / `audio.url`

### 5.1 curl

```bash
curl -N -X POST "http://127.0.0.1:9010/v1/tts" \
  -H "Content-Type: application/json" \
  -d '{
    "model": "qwen3-tts-flash-realtime",
    "text": "你好，这是一次流式语音输出测试。",
    "voice": "yachiyo_formal",
    "language_type": "Chinese",
    "instructions": "正式，平静，清晰。",
    "optimize_instructions": false,
    "stream": true
  }'
```

### 5.2 Python

```python
import base64
import json
import requests

payload = {
    "model": "qwen3-tts-flash-realtime",
    "text": "你好，这是流式返回测试。",
    "voice": "yachiyo_formal",
    "language_type": "Chinese",
    "instructions": "正式，平静，清晰。",
    "optimize_instructions": False,
    "stream": True,
}

pcm_chunks = []
with requests.post("http://127.0.0.1:9010/v1/tts", json=payload, stream=True, timeout=600) as resp:
    resp.raise_for_status()
    for line in resp.iter_lines(decode_unicode=True):
        if not line:
            continue
        item = json.loads(line)
        audio_b64 = item["output"]["audio"]["data"]
        if audio_b64:
            pcm_chunks.append(base64.b64decode(audio_b64))
        else:
            print("final:", item)

with open("stream_audio.pcm", "wb") as f:
    for chunk in pcm_chunks:
        f.write(chunk)
```

## 6. WebSocket Realtime 调用示例

先注意两件事：

- `ws://127.0.0.1:9010/api-ws/v1/realtime` 必须使用真正的 WebSocket 客户端连接；
- 不能直接在浏览器地址栏里打开，也不能用普通 `curl http://...` 或 HTTP `GET` 去访问，否则通常会看到 `404` 或 Upgrade 失败提示。

### 6.1 客户端典型时序

1. 建立连接 `ws://127.0.0.1:9010/api-ws/v1/realtime`
2. 服务端先推送 `session.created`
3. 客户端发送 `session.update`
4. 客户端发送 `input_text_buffer.append`
5. 客户端发送 `input_text_buffer.commit`
6. 服务端持续推送 `response.audio.delta`
7. 服务端结束时推送 `response.audio.done`、`response.done`

### 6.2 Python 示例

```python
import asyncio
import base64
import json
import websockets

async def main():
    uri = "ws://127.0.0.1:9010/api-ws/v1/realtime"
    pcm_chunks = []

    async with websockets.connect(uri, max_size=None) as ws:
        init_event = await ws.recv()
        print("init:", init_event)

        await ws.send(json.dumps({
            "type": "session.update",
            "session": {
                "model": "qwen3-tts-flash-realtime",
                "voice": "yachiyo_formal",
                "language_type": "Chinese",
                "mode": "commit",
                "response_format": "pcm",
                "sample_rate": 24000,
                "instructions": "正式，平静，清晰。",
                "optimize_instructions": False
            }
        }))

        print("updated:", await ws.recv())

        await ws.send(json.dumps({
            "type": "input_text_buffer.append",
            "text": "你好，欢迎使用 WebSocket 实时语音接口。"
        }))
        await ws.send(json.dumps({"type": "input_text_buffer.commit"}))

        while True:
            raw = await ws.recv()
            event = json.loads(raw)
            print(event["type"])
            if event["type"] == "response.audio.delta":
                pcm_chunks.append(base64.b64decode(event["delta"]))
            if event["type"] == "response.done":
                break

    with open("realtime_audio.pcm", "wb") as f:
        for chunk in pcm_chunks:
            f.write(chunk)

asyncio.run(main())
```

### 6.3 `session.update` 示例

```json
{
  "type": "session.update",
  "session": {
    "model": "qwen3-tts-flash-realtime",
    "voice": "yachiyo_formal",
    "language_type": "Chinese",
    "mode": "commit",
    "response_format": "pcm",
    "sample_rate": 24000,
    "instructions": "正式，平静，清晰。",
    "optimize_instructions": false
  }
}
```

## 7. 联调建议

推荐按下面顺序接入：

1. 先打 `GET /healthz`，确认服务在线
2. 再打 `GET /v1/voices`，确认 `voice` 别名可用
3. 先联调 HTTP 非流式
4. 再联调 HTTP 流式
5. 最后联调 WebSocket Realtime

## 8. 当前已知注意事项

- 当前服务默认返回单声道 `24000 Hz` 音频
- WebSocket `session.update` 当前只接受 `response_format="pcm"` 与 `sample_rate=24000`；传其他值会直接返回 `error`
- 流式中间块是 Base64 编码的 PCM16 音频片段，不是每块都带 WAV 文件头
- HTTP `stream=true` 的中间 PCM 片段与最后 `audio.url` 下载到的 WAV 属于同一份音频内容；如需试听，优先下载最终 WAV
- 完整 WAV 建议通过最后一条响应中的 `audio.url` 下载
- 如果启动日志提示 `No supported WebSocket library detected`，说明当前环境缺少 `websockets` / `wsproto`；安装 `websockets` 后请重启服务进程
- 当前版本已具备真实流式公开能力，但仍属于可联调 MVP，生产化还需补鉴权、限流、监控与反向代理
