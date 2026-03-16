# candidate8_v2_ws_v1

- 日期：2026-03-16
- 目标：首次 WebSocket benchmark
- 服务地址：`ws://127.0.0.1:9010/api-ws/v1/realtime`
- HTTP Base URL：`http://127.0.0.1:9010`
- bundle：`outputs/lora_candidate8_multilingual_warmstart_1p7b_20260310_v2_bundle_best`
- voice registry：`streaming_lora_service/configs/voice_registry.candidate8_v2.json`
- voice：`yachiyo_candidate8_v2`
- 模式：`commit`
- 采样率：`24000`
- 响应格式：`pcm`
- warmup：`1`
- measured iterations：`5`
- case：`zh_formal`、`ja_formal`

## 总览

- success / failure：`10 / 0`
- TTFT：`p50=302.9562 ms`，`p95=401.2292 ms`
- total_elapsed_ms：`p50=23939.392 ms`，`mean=23772.3872 ms`
- audio_duration_s：`mean=8.752 s`
- RTF：`mean=2.7308`，`p95=3.1649`

## 分 case 摘要

### zh_formal

- TTFT：`median=293.8948 ms`
- total_elapsed_ms：`median=23791.0872 ms`
- audio_duration_s：`median=9.2 s`
- RTF：`median=2.586`

### ja_formal

- TTFT：`median=342.3073 ms`
- total_elapsed_ms：`median=24093.6044 ms`
- audio_duration_s：`median=8.4 s`
- RTF：`median=2.7381`

## 产物

- `ws_benchmark_metrics.json`
- `audio/`
- `server.log`
