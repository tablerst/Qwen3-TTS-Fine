# Candidate8 v2 Path Diagnostic

验证时间：2026-03-12

目标 bundle：`outputs/lora_candidate8_multilingual_warmstart_1p7b_20260310_v2_bundle_best`

运行方式：

- `python -m streaming_lora_service.quality_regression`
- `voice_registry_file = streaming_lora_service/configs/voice_registry.candidate8_v2.json`
- `default_voice_alias = yachiyo_candidate8_v2`
- `seed = 1234`

## 本次新增对照路径

在原有 `offline_non_streaming / http_non_streaming / http_streaming_runtime / websocket_realtime` 的基础上，新增：

- `streaming_sampler_full_decode`
  - 与在线流式共享同一套 step-level sampler
  - 但在采样结束后整段 full decode

因此这份产物可以拆开看：

1. `http_non_streaming` vs `streaming_sampler_full_decode`
   - 观察 full-generate 与 step-level sampler 是否分叉
   - 这两条路径当前都基于 `non_streaming_mode=False` 的 streaming-compatible prompt 语义
2. `streaming_sampler_full_decode` vs `http_streaming_runtime`
   - 观察增量解码是否继续引入额外损耗
3. `http_streaming_runtime` vs `websocket_realtime`
   - 观察 HTTP/WS 在线路径是否一致

## Summary

- `case_count = 2`
- `warning_count = 0`
- `warning_case_count = 0`

## 关键结论

### 1. 在线 sampler 与在线解码链路基本一致

中文：

- `streaming_sampler_full_decode = 7.92s / 99 steps`
- `http_streaming_runtime = 7.92s / 99 steps`
- `websocket_realtime = 7.92s / 99 steps`

日文：

- `streaming_sampler_full_decode = 6.64s / 83 steps`
- `http_streaming_runtime = 6.64s / 83 steps`
- `websocket_realtime = 6.64s / 83 steps`

说明：当前 HTTP 流式与 WS Realtime 并没有额外改写 codec 轨迹；两者都严格跟随 step-level sampler。

### 2. 中文的主要分叉发生得很早

`http_non_streaming` vs `streaming_sampler_full_decode`：

- `reference_codec_steps = 106`
- `candidate_codec_steps = 99`
- `shared_prefix_steps = 7`
- `first_divergence_step = 7`

第一处分叉 step 的 token 仅第 1 个 group 已经不同：

- `http_non_streaming`: `1681, 1540, 1723, 204, 1097, 1668, 1980, 1759, 2033, 484, 802, 1412, 904, 284, 1272, 1890`
- `streaming_sampler_full_decode`: `1530, 1540, 1723, 204, 1097, 1668, 1980, 1759, 2033, 484, 802, 1412, 904, 284, 1272, 1890`

解读：中文 case 中，step-level sampler 不是到尾段才轻微漂移，而是在很早阶段就已经与 `http_non_streaming` 产生 codec 分叉。

### 3. 日文主要在后段才开始分叉

`http_non_streaming` vs `streaming_sampler_full_decode`：

- `reference_codec_steps = 83`
- `candidate_codec_steps = 83`
- `shared_prefix_steps = 67`
- `first_divergence_step = 67`

第一处分叉 step 同样主要是第 1 个 group 不同：

- `http_non_streaming`: `546, 53, 187, 1293, 1535, 850, 612, 1786, 1698, 512, 1406, 385, 530, 1384, 649, 1495`
- `streaming_sampler_full_decode`: `1129, 53, 187, 1293, 1535, 850, 612, 1786, 1698, 512, 1406, 385, 530, 1384, 649, 1495`

解读：日文 case 中两条路径前缀大体一致，分叉明显晚于中文，因此主观差异通常也更轻。

### 4. 当前主因判断

这份产物强化了同一个判断：

- `candidate8 v2` 的在线/离线差异主因更像 **sampler / prompt-path divergence**
- 增量解码仍可能带来次级损耗，但不是当前主矛盾

更进一步地说：

- `http_non_streaming` 与 `streaming_sampler_full_decode` 已经不是“不同 prompt 模式”之间的对比
- 它们共享同一套 `non_streaming_mode=False` prompt 语义
- 因此当前观察到的中文第 7 步分叉，更像是 **official `talker.generate(...)` 与手写 step loop 之间的采样实现差异**

## 产物

- `metrics.json`
- `zh_formal/*.wav`
- `ja_formal/*.wav`

建议后续优先继续对齐：

- `http_non_streaming` vs `streaming_sampler_full_decode`

而不是先去怀疑：

- WebSocket 包装层
- HTTP chunking
- LoRA 未加载
