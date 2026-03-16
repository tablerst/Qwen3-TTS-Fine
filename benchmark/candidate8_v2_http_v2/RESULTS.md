# candidate8_v2_http_v2 benchmark results

## Service config

- service: `streaming_lora_service.app.server`
- bundle: `outputs/lora_candidate8_multilingual_warmstart_1p7b_20260310_v2_bundle_best`
- voice registry: `streaming_lora_service/configs/voice_registry.candidate8_v2.json`
- host: `127.0.0.1:45460`
- attention backend: `sdpa`
- runtime session sync mode: `chunk`
- trace timing: enabled

## Artifacts

- HTTP benchmark (same-text zh_formal): `http_benchmark_zh_formal_metrics.json`
- Server trace log: `server_trace.log`
- Saved audio: `audio_zh_formal/`

## Same-text zh_formal benchmark

Text:

`你好，欢迎使用 Qwen3-TTS WebSocket benchmark。这是一段用于测量首包时延与实时因子的中文测试文本。`

Measured config:

- iterations: `3`
- warmup: `1`
- endpoint: `/v1/tts`
- transport: `HTTP streaming`

Summary from `http_benchmark_zh_formal_metrics.json`:

- response headers p50: `2.1036 ms`
- TTFT p50: `299.3127 ms`
- total elapsed p50: `25187.7927 ms`
- RTF mean: `2.7105`
- audio duration mean: `9.3333 s`
- audio chunks mean: `30.3333`

## Service-side trace observations

For the three measured runs (excluding warmup), `server_trace.log` shows:

- `avg_forward_ms`: `201.26`, `198.15`, `193.79`
- `avg_decode_ms`: `36.34`, `35.44`, `33.44`
- `avg_state_sync_ms`: around `0.04~0.05`
- `runtime_session_sync_mode`: always `chunk`

Interpretation:

- HTTP streaming 路径下，主要时间仍然在 talker 单步 forward；
- decode 比 WS 同口径 benchmark 稍高，但仍明显不是主瓶颈；
- state sync 仍然很小，说明前两轮实现层减税继续有效。

## Comparison against previous live zh_formal HTTP baseline

Reference: `outputs/http_benchmark_20260316_live.json` (`zh_formal`, 5 measured runs, warmup 1)

Previous summary:

- TTFT p50: `307.5309 ms`
- total elapsed p50: `25293.6040 ms`
- RTF mean: `2.8151`

Current summary:

- TTFT p50: `299.3127 ms`
- total elapsed p50: `25187.7927 ms`
- RTF mean: `2.7105`

Directional change:

- TTFT p50 improved by about `8.22 ms` (~`2.7%`)
- total elapsed p50 improved by about `105.81 ms` (~`0.4%`)
- RTF mean improved by about `0.1046` (~`3.7%`)

## Notes

- Previous baseline used `5` measured runs; current run used `3` measured runs.
- HTTP path improvement is present, but weaker than the WS same-text improvement from `benchmark/candidate8_v2_ws_v2/RESULTS.md`.
- Combined with service trace, this suggests the remaining ceiling is increasingly dominated by model forward cost rather than session sync or decode-window packing.
