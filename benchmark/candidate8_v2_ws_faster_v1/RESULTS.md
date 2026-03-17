# candidate8_v2_ws_faster_v1 benchmark results

## Service config

- service: `streaming_lora_service.app.server`
- backend: `faster`
- bundle: `outputs/lora_candidate8_multilingual_warmstart_1p7b_20260310_v2_bundle_best`
- voice registry: `streaming_lora_service/configs/voice_registry.candidate8_v2.json`
- host: `127.0.0.1:47349`
- merged cache dir: `benchmark/candidate8_v2_ws_faster_v1/faster_cache`
- local files only: `true`
- trace timing: enabled

## Artifacts

- WS benchmark (same-text zh_formal): `ws_benchmark_zh_formal_metrics.json`
- Server trace log (first boot): `server_trace.log`
- Server trace log (cache-hit restart): `server_trace_restart.log`
- Saved audio: `audio_zh_formal/`

## Same-text zh_formal benchmark

Text:

`你好，欢迎使用 Qwen3-TTS WebSocket benchmark。这是一段用于测量首包时延与实时因子的中文测试文本。`

Measured config:

- iterations: `3`
- warmup: `1`
- transport: `WebSocket`

Summary from `ws_benchmark_zh_formal_metrics.json`:

- TTFT p50: `233.3014 ms`
- TTFT mean: `234.0469 ms`
- total elapsed p50: `4409.5522 ms`
- RTF mean: `0.4632`
- audio duration mean: `9.5467 s`
- audio chunks mean: `30.3333`

## Real-service startup observations

From `server_trace.log`:

- first boot used `cache_hit=False`
- merged model loaded from `benchmark/candidate8_v2_ws_faster_v1/faster_cache/lora_candidate8_multilingual_warmstart_1p7b_20260310_v2_bundle_best_861355669d541076`
- faster model load elapsed: about `2218.71 ms`
- first warmup WS request triggered CUDA graph capture
- warmup `first_chunk_ready_ms`: about `7255.30 ms`

From `server_trace_restart.log`:

- restart used `cache_hit=True`
- merged export cache was reused successfully
- faster model load elapsed on restart: about `2573.99 ms`

## Service-side trace observations

For the three measured runs (excluding warmup), `server_trace.log` shows:

- `first_chunk_ready_ms`: `236.19`, `229.88`, `231.95`
- `generated_steps`: `109`, `131`, `118`
- `emitted_chunks`: `28`, `33`, `30`
- `avg_decode_ms`: `64.22`, `104.98`, `104.23`
- `supports_state_resume`: always `False`

Interpretation:

- 当前 first chunk 已进入约 `230 ms` 量级；
- measured runs 总时长大幅低于 native baseline；
- faster backend 当前不暴露 native 路径那种 `avg_forward_ms` / state-sync 细粒度指标；
- 当前第一版仍是“协议兼容优先”，每次 commit 独立生成，不做 KV 恢复。

## Comparison against native same-text zh_formal baseline

Reference: `benchmark/candidate8_v2_ws_v2/ws_benchmark_zh_formal_metrics.json`

Native summary:

- TTFT p50: `302.8361 ms`
- total elapsed p50: `22527.2135 ms`
- RTF mean: `2.5030`

Current faster summary:

- TTFT p50: `233.3014 ms`
- total elapsed p50: `4409.5522 ms`
- RTF mean: `0.4632`

Directional change:

- TTFT p50 improved by about `69.53 ms` (~`23.0%`)
- total elapsed p50 improved by about `18117.66 ms` (~`80.4%`)
- RTF mean improved by about `2.0398` (~`81.5%` lower, lower is better here)

## Caveats

- faster and native outputs are not bit-identical; audio duration also differs slightly between runs.
- faster first warmup request includes CUDA graph capture cost and should not be mixed into measured summaries.
- current faster service path focuses on WS protocol compatibility, not internal state-machine equivalence.
