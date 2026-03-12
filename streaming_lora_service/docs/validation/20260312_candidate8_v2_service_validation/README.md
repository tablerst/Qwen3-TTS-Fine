# Candidate8 v2 Service Validation

验证时间：2026-03-12

目标 bundle：`outputs/lora_candidate8_multilingual_warmstart_1p7b_20260310_v2_bundle_best`

运行方式：

- 使用 `python -m streaming_lora_service.quality_regression`
- 使用 bundle 自带 speaker（`default_voice_alias=yachiyo_candidate8_v2`）
- 未使用 `voice_registry.example.json`，因为该示例 registry 的 `speaker_name=yachiyo_formal` 与该 bundle 不匹配

## Summary

- `case_count = 2`
- `warning_count = 0`
- `warning_case_count = 0`

说明：当前两条默认回归 case（中文正式、日文正式）均通过基础服务验证。

## 关键观察

### 1. Realtime/Streaming 路径稳定

- `http_streaming_runtime` 与 `websocket_realtime` 两条路径完全对齐
- 两条路径都正常 `finish_reason = "eos"`
- 未出现过生成、异常拉长、WS/HTTP streaming 不一致等问题

### 2. v2 bundle 的中文服务结果是正常的

中文样本：

- `offline_non_streaming = 7.84s`
- `http_non_streaming = 8.48s`
- `http_streaming_runtime = 7.92s`
- `websocket_realtime = 7.92s`

解读：

- 服务的流式/Realtime 路径已经稳定
- 中文场景下 `http_non_streaming` 与 streaming 路径仍存在轻微模式差异（约 `0.56s`）
- 这更像当前 `non_streaming_mode` 语义差异，而不是稳定性 bug

### 3. 日文路径对齐更紧

日文样本：

- `offline_non_streaming = 6.40s`
- `http_non_streaming = 6.64s`
- `http_streaming_runtime = 6.64s`
- `websocket_realtime = 6.64s`

解读：

- 日文场景下服务三条路径已基本一致
- 与离线也只有轻微差异

## 产物

- `metrics.json`
- `zh_formal/offline_non_streaming.wav`
- `zh_formal/http_non_streaming.wav`
- `zh_formal/http_streaming_runtime.wav`
- `zh_formal/websocket_realtime.wav`
- `ja_formal/offline_non_streaming.wav`
- `ja_formal/http_non_streaming.wav`
- `ja_formal/http_streaming_runtime.wav`
- `ja_formal/websocket_realtime.wav`

## 当前结论

如果你的主要目标是：

- WebSocket Realtime 服务可用
- HTTP streaming 服务可用
- 音质与连贯性正常
- 不再出现重复词/严重拉长/失真

那么这个 `candidate8 v2` bundle 当前已经通过这轮服务验证。

## 当前定位建议

从 2026-03-12 起，可以把：

- `outputs/lora_candidate8_multilingual_warmstart_1p7b_20260310_v2_bundle_best`

视为 **当前服务候选主线 bundle**。

推荐配套：

- `streaming_lora_service/configs/voice_registry.candidate8_v2.json`

这样既能使用显式 alias：

- `yachiyo_candidate8_v2`

也能兼容历史 alias：

- `yachiyo_formal`

如果后续还要继续收敛：

- 最值得看的不是 streaming 稳定性，而是**中文场景下 HTTP 非流式与 streaming 模式差异是否要继续统一**。
