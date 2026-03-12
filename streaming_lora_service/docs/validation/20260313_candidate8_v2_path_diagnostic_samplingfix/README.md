# Candidate8 v2 Path Diagnostic after Sampling Fixes

验证时间：2026-03-13

目标 bundle：`outputs/lora_candidate8_multilingual_warmstart_1p7b_20260310_v2_bundle_best`

产物：`metrics.json`

## 本轮修复背景

在此前诊断中，已先后定位并修复了两个 sampling fidelity 问题：

1. 采样前未将 `next_token_logits` 统一转成 `float32`
2. 重复 token 被重复施加 `repetition_penalty`

本目录记录修复后的完整五路对照：

- `offline_non_streaming`
- `http_non_streaming`
- `streaming_sampler_full_decode`
- `http_streaming_runtime`
- `websocket_realtime`

## Summary

- `case_count = 2`
- `warning_count = 0`
- `warning_case_count = 0`

## 关键结果

### 中文 `zh_formal`

- `http_non_streaming = 8.48s / 106 steps`
- `streaming_sampler_full_decode = 8.48s / 106 steps`
- `http_streaming_runtime = 8.48s / 106 steps`
- `websocket_realtime = 8.48s / 106 steps`

codec 对比：

- `http_non_vs_streaming_sampler_full_decode.identical = true`
- `streaming_sampler_full_decode_vs_http_streaming_runtime.identical = true`
- `http_streaming_runtime_vs_websocket_realtime.identical = true`

### 日文 `ja_formal`

- `http_non_streaming = 6.64s / 83 steps`
- `streaming_sampler_full_decode = 6.64s / 83 steps`
- `http_streaming_runtime = 6.64s / 83 steps`
- `websocket_realtime = 6.64s / 83 steps`

codec 对比：

- `http_non_vs_streaming_sampler_full_decode.identical = true`
- `streaming_sampler_full_decode_vs_http_streaming_runtime.identical = true`
- `http_streaming_runtime_vs_websocket_realtime.identical = true`

## 当前结论

candidate8 v2 在默认回归 case 上，服务内部几条主要路径已经完成对齐：

- 服务 full-generate 非流式
- 服务 step-level sampler full decode
- HTTP runtime streaming
- WebSocket realtime

因此当前剩余的 offline/online 差异，不再优先解释为在线流式 fidelity 问题，而更应解释为：

- `offline_non_streaming` 使用 `non_streaming_mode=True`
- 服务主线使用 `non_streaming_mode=False`

也就是说：

> 这轮修复后，candidate8 v2 的服务路径已经回到“模式内一致”；
> 剩余差异主要属于 offline 模式定义差异，而不是在线路径继续跑偏。
