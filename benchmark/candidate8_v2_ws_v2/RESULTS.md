# candidate8_v2_ws_v2 benchmark results

## Service config

- service: `streaming_lora_service.app.server`
- bundle: `outputs/lora_candidate8_multilingual_warmstart_1p7b_20260310_v2_bundle_best`
- voice registry: `streaming_lora_service/configs/voice_registry.candidate8_v2.json`
- host: `127.0.0.1:47348`
- attention backend: `sdpa`
- runtime session sync mode: `chunk`
- trace timing: enabled

## Artifacts

- WS benchmark (custom text): `ws_benchmark_metrics.json`
- WS benchmark (same-text zh_formal comparison): `ws_benchmark_zh_formal_metrics.json`
- Server trace log: `server_trace.log`
- Saved audio: `audio/`, `audio_zh_formal/`

## Same-text zh_formal benchmark

Text:

`你好，欢迎使用 Qwen3-TTS WebSocket benchmark。这是一段用于测量首包时延与实时因子的中文测试文本。`

Measured config:

- iterations: `3`
- warmup: `1`
- transport: `WebSocket`

Summary from `ws_benchmark_zh_formal_metrics.json`:

- TTFT p50: `302.8361 ms`
- TTFT mean: `308.2095 ms`
- total elapsed p50: `22527.2135 ms`
- RTF mean: `2.5030`
- audio duration mean: `9.0933 s`
- audio chunks mean: `29.3333`

## Service-side trace observations

For the three measured `zh_formal` runs (excluding warmup), `server_trace.log` shows:

- `avg_forward_ms`: `183.36`, `181.47`, `183.76`
- `avg_decode_ms`: `32.60`, `31.72`, `32.50`
- `avg_state_sync_ms`: all around `0.04`
- `runtime_session_sync_mode`: always `chunk`

Interpretation:

- 当前单步 `talker` forward 仍是主成本；
- decode 已经降到大约每 chunk `32 ms` 左右；
- session sync 开销已经很小，不再像主要瓶颈。

## Comparison against previous live zh_formal baseline

Reference: `outputs/ws_benchmark_20260316_live.json` (`zh_formal`, 5 measured runs, warmup 1)

Previous p50/mean summary:

- TTFT p50: `317.2640 ms`
- total elapsed p50: `24273.7458 ms`
- RTF mean: `2.8078`

Current same-text summary:

- TTFT p50: `302.8361 ms`
- total elapsed p50: `22527.2135 ms`
- RTF mean: `2.5030`

Directional change:

- TTFT p50 improved by about `14.43 ms` (~`4.5%`)
- total elapsed p50 improved by about `1746.53 ms` (~`7.2%`)
- RTF mean improved by about `0.3048` (~`10.9%`)

## Caveats

- Previous baseline used `5` measured runs; current run used `3` measured runs.
- Comparison is only intended for same-text `zh_formal` directionality, not as a final statistically stable report.
- Warmup run still shows much higher `init_ms` / `first_chunk_ms`; comparisons above use measured runs only.
