# candidate8_v2_ws_compile_v1 compile experiment

## Goal

验证在当前 `streaming_lora_service` + `candidate8_v2` 主线 bundle 上，是否可以通过对 `qwen3tts.model.talker` 开启实验性 `torch.compile` 获得 WS streaming 改善。

## Service config

- service: `streaming_lora_service.app.server`
- bundle: `outputs/lora_candidate8_multilingual_warmstart_1p7b_20260310_v2_bundle_best`
- voice registry: `streaming_lora_service/configs/voice_registry.candidate8_v2.json`
- host: `127.0.0.1:47054`
- attention backend: `sdpa`
- runtime session sync mode: `chunk`
- compile target: `qwen3tts.model.talker`
- compile mode: `reduce-overhead`
- compile dynamic: `true`
- trace timing: enabled

## Benchmark config

- transport: `WebSocket`
- text: same-text `zh_formal`
- iterations: `2`
- warmup: `1`

## Outcome

**Current experiment is not production-viable in this runtime.**

The benchmark did not complete any successful measured runs:

- success count: `0`
- failure count: `2`

Artifacts:

- benchmark JSON: `ws_benchmark_zh_formal_metrics.json`
- server log: `server_trace.log`

## Observed failures

### First attempt symptom

After a very large prefill/compile stall, the service failed with a compiled CUDAGraph output reuse error around `past_hidden`:

- `init_total_ms` observed around `167297.60 ms`
- first chunk was never produced
- error mentioned: `accessing tensor output of CUDAGraphs that has been overwritten by a subsequent run`

Mitigation was attempted by:

- calling `torch.compiler.cudagraph_mark_step_begin()` before compiled talker invocations
- cloning reusable tensor outputs such as `past_hidden`

### Second attempt symptom

After the mitigation, the service still failed during generation tracing inside Transformers generation utilities:

- error: `torch._dynamo.exc.UncapturedHigherOrderOpError`
- message includes:
  - `Cond doesn't work unless it is captured completely with torch.compile`
  - `Encountered aliasing during higher order op tracing`
- failing path points into `transformers/generation/utils.py` and `torch.cond`

This indicates the current compiled talker path is blocked not just by state reuse, but by a deeper incompatibility between this generation path and `torch.compile` in the present runtime stack.

## Conclusion

For the current environment and code path:

- experimental compile of `qwen3tts.model.talker` is **not a good next-step optimization path**;
- it adds major warmup/compile cost;
- it currently breaks WS streaming correctness before any meaningful throughput comparison can be made.

## Recommended next step

Return focus to:

1. deeper forward profiling without `torch.compile`
2. isolating model-internal hotspots within talker forward
3. only revisiting compile after upstream/runtime compatibility changes materially
