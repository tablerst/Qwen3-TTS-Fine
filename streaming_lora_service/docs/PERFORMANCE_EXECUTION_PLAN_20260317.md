# Non-Attention Performance Execution Plan (2026-03-17)

## Goal

在已经确认 `sdpa` 比 `flash_attention_2` 更适合当前短 token 流式场景之后，下一轮优化聚焦于 **不改模型权重、不改采样逻辑、不改公开协议** 的实现层提速。

本轮执行目标：

1. 降低流式生成每 step 的 Python 状态同步税；
2. 消除 `attention_mask` 每 step 递增式重分配；
3. 保持 `response.audio.delta` / HTTP NDJSON 兼容；
4. 保持现有质量回归基线可复用；
5. 为后续 decode/window 优化与 `torch.compile` 试验打基础。

## Constraints

- `sdpa` 已在当前环境验证优于 `flash_attention_2`；
- 当前优先保证“同一请求内稳定跑完”，**不要求中途强恢复**；
- WebSocket / HTTP 对外协议必须保持与当前 Qwen-compatible 输出兼容；
- 质量回归仍需以 `quality_regression.py`、HTTP runtime、WebSocket realtime 三路对照为准。

## Phase 1: Observability Baseline

目标：把性能账单拆出来，避免“改了很多，只知道快了/慢了，但不知道为什么”。

### Deliverables

- 扩展 `streaming_generator.py` 的生成指标，至少能区分：
  - prefill
  - single-step forward
  - state sync
  - decode
- 扩展 `server.py` 的 trace 日志，保留现有 `trace_timing` 开关；
- 继续复用 `ws_benchmark.py` / `http_streaming_benchmark.py` 做前后对照。

### Status update

本阶段已完成：

- `StreamingCustomVoiceGenerator` 已新增 `first_forward_ms / total_forward_ms / avg_forward_ms`；
- `state sync` 与 `decode` 的总耗时 / 平均耗时已进入 runtime metrics snapshot；
- `server.py` 的 `trace_timing` 日志现在会直接打印 `generated_steps / emitted_chunks / avg_forward_ms / avg_decode_ms / avg_state_sync_ms`；
- `quality_regression.py` 的 `PathMetrics` 已开始保存 streaming 路径的 `runtime_metrics`；
- 新 benchmark 产物已落盘到 `benchmark/candidate8_v2_ws_v2/`。

## Phase 2: Runtime Session Sync Throttling

目标：把 `RuntimeSession.bind_generation_state()` 从“每 step 都做完整绑定”改成“只在有意义的时候同步”。

### Proposed Default

- 初始化时同步一次；
- 每次真正产出音频 chunk 时同步一次；
- 结束时强制同步一次；
- 提供兼容模式，允许保留旧的 step 级同步行为以便调试/验证。

### Why

当前每次同步都会复制：

- `generated_codes`
- `sampled_tokens`
- `metrics`

在短 token 流式场景下，这些复制的性价比偏低。

## Phase 3: Attention Mask Preallocation

目标：避免当前 `_run_single_step()` 中的：

- `torch.ones_like(...)`
- `torch.cat([...])`

### Proposed Direction

- 在 prefill / restore 阶段一次性准备足够长度的全 1 mask buffer；
- step 时仅更新逻辑长度并切片传入 `talker.forward(...)`；
- 不改变当前 `attention_mask` 与 `cache_position` 的契约。

## Phase 4: Decode Window Packing

目标：减少 codec ring buffer -> decode window 之间的 Python 对象重建。

### Current Hot Path

- `tuple(islice(...))`
- `list(codec_window)`
- `torch.stack(...)`

### Follow-up

本阶段已完成第一步收敛：

- `StreamingCustomVoiceGenerator` 不再把同一份 codec step 同时写进 `generated_codes` 和 `IncrementalAudioDecoder` 的内部 ring buffer；
- 流式 decode 改为使用 `IncrementalAudioDecoder.decode(total_steps, decode_fn(start, end), ...)`；
- 生成器内部新增预分配 `generated_code_buffer`，按 step 范围切片直接传给 speech tokenizer decode。

这样当前已经消掉了：

- 生成器侧的重复 codec 缓存；
- `decode_buffered()` 路径上的 `tuple(islice(...)) -> list(...) -> torch.stack(...)` 这一串窗口打包开销。

后续仍可继续观察：

- `generated_codes` 列表是否还需要进一步瘦身；
- resume 场景下 `generated_code_buffer` 的重建是否值得再优化。

## Validation

每完成一个阶段，至少执行：

1. `tests/test_streaming_generator.py`
2. `tests/test_runtime_session.py`
3. `tests/test_incremental_decoder.py`
4. `tests/test_server_smoke.py`

并对比：

- TTFT
- total elapsed
- RTF
- `generated_steps`
- `emitted_chunks`
- `first_emitted_step`

## Success Criteria

若本轮完成后出现以下任一结果，即可认为方向有效：

- TTFT 下降；
- 总耗时下降；
- RTF 稳定下降；
- 无新增 waveform / codec 回归告警；
- WebSocket / HTTP 兼容契约保持不变。

## Explicit Non-Goals

本轮不包含：

- 量化；
- 降低 left context 以换速度；
- 改公开协议为二进制帧；
- 重新评估 `flash_attention_2` / `FA3` / `FA4`；
- 在未稳定 shape 前强推 `torch.compile` 默认开启。

## Experimental compile status

截至 2026-03-17，服务已支持通过启动参数开启实验性 `torch.compile`：

- `--compile_talker`
- `--compile_mode <default|reduce-overhead|max-autotune>`
- `--compile_dynamic`

当前策略仍然是：

- 默认关闭；
- 仅编译 `qwen3tts.model.talker`；
- 配合 benchmark 与 `trace_timing` 做 A/B，而不是直接切成默认生产配置。

### 2026-03-17 same-text WS compile experiment result

已在 `benchmark/candidate8_v2_ws_compile_v1/` 对 `candidate8_v2` 主线 bundle 做过一轮实验：

- 开关：`--compile_talker --compile_mode reduce-overhead --compile_dynamic`
- 传输：WebSocket
- 文本：same-text `zh_formal`

当前结果：

- measured runs 成功数为 `0`
- 首轮曾出现极大的 compile/prefill stall（`init_total_ms` 约 `167s`）
- 先后命中了：
  - CUDAGraph 输出复用覆盖错误
  - `torch._dynamo.exc.UncapturedHigherOrderOpError`（Transformers generation 内部 `torch.cond` aliasing）

当前判断：

- **在现有 runtime / Transformers / Qwen3-TTS 这条生成路径上，compile 不是下一步主推方向**；
- 若未来要重试，应优先等待上游 runtime/graph 兼容性变化，或进一步隔离更小的可编译子路径，而不是直接编译整个 `talker` 主调用路径。
