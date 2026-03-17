# Streaming LoRA Service

本目录用于承载 **面向第三方的 Qwen-compatible TTS / Realtime TTS 服务设计** 与后续代码落地点。

当前阶段仍以**文档冻结与接口收敛**为主，但方向已经明确：

> **不再继续扩展旧的自定义公开 WebSocket 协议；**
> **从第一版开始，直接采用 Qwen3-TTS / Qwen-TTS-Realtime 风格的对外接口。**

这意味着：

- **对外**：字段名、事件名、交互节奏尽量向官方 Qwen3-TTS / Realtime 看齐；
- **对内**：仍然复用本仓库已有的 `Qwen3TTSModel`、LoRA bundle、speaker patch 与真流式生成链路；
- **部署侧**：默认 LoRA bundle 由服务启动配置决定，而不是暴露给第三方调用方的心智负担。

## 设计原则

### 1. 对外官方风格，内部自有实现

服务对外暴露的协议目标是：

- 请求字段尽量兼容官方 Qwen3-TTS：如 `model`、`voice`、`language_type`、`instructions`；
- Realtime 事件尽量兼容官方 Qwen-TTS-Realtime：如 `session.update`、`input_text_buffer.append`、`response.audio.delta`；
- 文档心智、错误模型、示例代码尽量让第三方“一眼会用”。

但内部不要求照搬官方实现：

- 仍由服务进程启动时默认挂载 LoRA bundle；
- 仍由内部 runtime 负责真流式 step 推进与增量解码；
- 仍由自有 voice registry 把公开 `voice` 映射到 bundle / speaker profile。

### 2. LoRA 是服务端内部能力，不是客户端接口负担

第三方调用时不应被迫理解：

- `bundle_dir`
- `adapter_dir`
- `config_patch.json`
- `speaker_embedding.safetensors`

这些内容属于**部署与服务内部装配细节**，不属于 V1 公开 API。

### 3. 真流式优先于“只长得像流式”

这里的目标不是“把一次性生成套上一层 WebSocket”，而是：

- 输入可增量追加；
- 生成可 step 级推进；
- 音频可持续下发；
- 会话状态可持续维护；
- 对外协议虽然官方风格，但内部仍然是真流式实现。

## V1 定位

V1 面向的是：

- **Qwen-compatible Realtime WebSocket** 公开接口；
- **Qwen-compatible HTTP TTS** 公开接口；
- `custom_voice` 场景优先；
- 服务启动时默认加载一个 LoRA bundle；
- 对外通过 `model` / `voice` 暴露稳定别名；
- 下行音频采用官方风格的 `response.audio.delta`（Base64 音频块）。

V1 暂不追求：

- 完整 DashScope 路径级兼容；
- 把 LoRA bundle 参数暴露为客户端必填项；
- 一开始就支持所有 `voice_clone` / `voice_design` 实时能力；
- 多 bundle 热切换与复杂多租户路由；
- 二进制 PCM 帧作为公开主协议。

## 为什么要单独开目录

现有仓库已经具备：

- 一次性离线推理：`qwen_tts/inference/qwen3_tts_model.py`
- 本地 Web UI：`qwen_tts/cli/demo.py`
- LoRA bundle 推理：`lora_finetuning/inference_with_lora.py`

但还没有一套真正适合**对外服务化**的独立结构。新目录的作用是：

1. 把“公开 API 设计”与“训练/导出脚本”解耦；
2. 把 LoRA bundle 装配逻辑与 Realtime 协议适配逻辑收敛在同一域；
3. 给后续代码、测试、兼容矩阵、SDK 示例、压测留出明确位置。

## 与现有代码的关系

这条方案不会推翻现有实现，而是复用下面这些关键能力：

- `Qwen3TTSModel.from_pretrained(...)`
- `Qwen3TTSForConditionalGeneration.generate(...)` 中已有的 prompt / prefill 逻辑
- `Qwen3TTSTalkerForConditionalGeneration.forward(...)` 中维护的增量生成状态
- `Qwen3TTSTokenizerV2Decoder.chunked_decode(...)` 的按块解码能力
- `lora_finetuning/common.py` 中的 LoRA / speaker patch 加载逻辑

## 建议后续代码结构

后续如果开始写代码，建议优先在本目录中补以下子结构：

- `streaming_lora_service/app/`
  - `bundle_loader.py`
  - `voice_registry.py`
  - `runtime_session.py`
  - `incremental_decoder.py`
  - `qwen_compat_ws.py`
  - `qwen_compat_http.py`
  - `server.py`
- `streaming_lora_service/tests/`
  - `test_bundle_loader.py`
  - `test_voice_registry.py`
  - `test_protocol_smoke.py`
  - `test_runtime_session.py`

## 当前实现进度（2026-03-11）

当前目录已经落下第一批**可运行的代码骨架**，不再只是纯文档：

- `app/models.py`：公共数据结构与会话配置模型
- `app/voice_registry.py`：公开 `voice` 别名解析
- `app/bundle_loader.py`：默认 LoRA bundle 解析与加载骨架
- `app/runtime_session.py`：文本缓冲区 / commit / finish 状态机骨架
- `app/incremental_decoder.py`：增量解码规划与 overlap 裁剪逻辑骨架
- `app/prompt_builder.py`：`custom_voice` 场景的 talker prefill / sampling 配置构建
- `app/streaming_generator.py`：stateful step-level codec 生成与增量音频输出
- `app/qwen_compat_ws.py`：Qwen-compatible Realtime 事件适配器 MVP
- `app/server.py`：可启动的 FastAPI WebSocket 服务 MVP
- `app/audio_utils.py`：浮点音频转 PCM16 与 chunk 切分工具
- `configs/voice_registry.example.json`：公开 `voice` 映射配置示例

同时已经补齐第一批单元测试：

- `tests/test_bundle_loader.py`
- `tests/test_voice_registry.py`
- `tests/test_runtime_session.py`
- `tests/test_incremental_decoder.py`
- `tests/test_prompt_builder.py`
- `tests/test_protocol_smoke.py`
- `tests/test_protocol_streaming.py`
- `tests/test_server_smoke.py`
- `tests/test_http_tts_smoke.py`
- `tests/test_streaming_generator.py`

并已完成一次本地验证：

- `python -m unittest discover -s streaming_lora_service/tests -v`
- 结果：**37 个测试全部通过**

并已补充一轮默认 bundle 的真实端到端验证产物：

- 指标记录：`docs/validation/20260311_real_bundle/metrics.json`
- 试听样本：`docs/validation/20260311_real_bundle/sample_01_zh_formal.wav`
- 试听样本：`docs/validation/20260311_real_bundle/sample_02_ja_formal.wav`

## 当前服务候选主线 bundle（2026-03-12）

当前更推荐作为**服务候选主线**的 bundle 是：

- `outputs/lora_candidate8_multilingual_warmstart_1p7b_20260310_v2_bundle_best`

对应验证产物：

- `docs/validation/20260312_candidate8_v2_service_validation/metrics.json`

当前结论：

- HTTP streaming / WebSocket Realtime 已通过基础服务验证
- 中日文默认回归 case `warning_count = 0`
- 更适合作为当前多语服务候选方向

推荐搭配的 voice registry：

- `streaming_lora_service/configs/voice_registry.candidate8_v2.json`

其中：

- 规范 alias：`yachiyo_candidate8_v2`
- 兼容 alias：`yachiyo_formal`

## 当前 MVP 能力边界

当前已经完成的是一个**可运行的兼容层 MVP**：

- 服务启动时加载默认 LoRA bundle
- 通过官方风格 WebSocket 事件对外提供能力
- 通过官方风格 HTTP 请求/响应字段对外提供能力
- 支持 `session.update / input_text_buffer.append / input_text_buffer.commit / input_text_buffer.clear / session.finish`
- 支持服务端 `input_text_buffer.cleared` 事件
- 支持 `response.audio.delta / response.audio.done / response.done / session.finished`
- WebSocket Realtime V1 明确收敛为 **24kHz / mono / PCM16** 输出，`session.update` 中若传入不支持的 `response_format` / `sample_rate` 将直接返回 `error`
- 提供 `/healthz` 与 `/v1/voices` 两个基础 HTTP 端点
- 提供 HTTP TTS 路由：`/v1/tts`、`/v1/audio/speech`、`/api/v1/services/aigc/multimodal-generation/generation`
- 提供音频下载路由：`/v1/audio/{audio_id}`
- 支持从 JSON / YAML voice registry 文件加载公开 `voice` 别名映射
- `custom_voice` 场景已具备初版 step-level 流式内核：`prompt_builder -> StreamingCustomVoiceGenerator -> IncrementalAudioDecoder -> response.audio.delta`
- WebSocket Realtime 与 HTTP `stream=true` 已可复用同一条 stateful step generator 链路
- 流式生成指标已记录 `finish_reason`（`eos` / `length`），便于排查“重复词 / 拉长 / 未正常停下”问题

当前**尚未完成**的部分也需要明确说明：

- `runtime_session` 已可绑定真实生成状态，并持久保存 `past_key_values / past_hidden / generated_codes / incremental_decoder`，为后续跨 append/commit 复用打底；
- `incremental_decoder` 已升级为 codec ring buffer + overlap decode 组合；
- 默认 LoRA bundle 的真实端到端 TTFB / chunk 粒度指标已经记录，但**人工试听结论**仍需最终确认。
- 当前 runtime session 的状态复用仍偏向**同一段文本生成中的恢复**；真正跨 `append/commit` 的 continuation 仍未宣称完成。

## 2026-03-17 性能推进：非 attention 热路径第一轮落地

在已经确认当前短 token 流式场景下 `sdpa` 优于 `flash_attention_2` 后，服务侧开始转向“**不碰模型质量、优先砍实现层白税**”的路线。

本轮已先落两项：

- `StreamingCustomVoiceGenerator` 新增 `runtime_session_sync_mode`，默认从旧的“每 step 绑定状态”收敛到 **`chunk` 粒度同步**；
- `_run_single_step()` 不再每 step 用 `torch.cat(...)` 增长 `attention_mask`，而是改为**预分配 buffer + 按当前长度切片**。

随后又继续推进了一步 decode 热路径优化：

- 生成器不再把 codec step 同时写入 `generated_codes` 和 decoder 内部 ring buffer；
- 增量解码改为直接基于 generator 侧预分配的 `generated_code_buffer` 按 step 范围切片 decode。

这两项的目标都不是改变采样或音频结果，而是减少：

- Python 列表 / 字典复制；
- `attention_mask` 线性增长带来的张量重分配；
- codec step 的重复缓存与 decode window 打包；
- 短 token 场景下每 step 的额外实现层开销。

说明：

- 默认模式现在更偏向“同一请求内稳定跑完”；
- 如果你需要保留旧的逐 step 调试 / 恢复行为，可显式传 `runtime_session_sync_mode="step"`，或在服务启动时加 `--runtime_session_sync_mode step`；
- 若只希望在结束时同步 session，可使用 `final`。
- 如果要开始试 `torch.compile`，当前服务已提供实验开关 `--compile_talker`，默认仍关闭；建议仅在已经拿到稳定 benchmark 基线后再做 A/B。

对应执行方案文档见：

- `docs/PERFORMANCE_EXECUTION_PLAN_20260317.md`

## 2026-03-12 修复更新：step-level 流式过生成

本轮已修复一个会直接导致流式路径严重过生成的根因：

- `StreamingCustomVoiceGenerator` 在单步生成时，原先把**旧的** `attention_mask` 传给 `talker.forward(...)`
- 这会让“当前新采样 token 已经作为输入参与生成，但 mask 还没把它算进去”
- 修复后改为：**先扩展 `attention_mask`，再执行该 step 的 talker forward**

修复效果已通过真实 bundle 三路对照验证：

- `docs/validation/20260312_compare_fix1/metrics.json`
- 中文样本：流式从约 `27.84s` 回落到 `7.76s`
- 日文样本：流式从约 `22.08s` 回落到 `6.56s`
- `http_streaming_runtime` 与 `websocket_realtime` 现在和 `http_non_streaming` 对齐，告警数降为 `0`

也就是说：

> **当前 MVP 已经可用、可跑、可联调；**
> **并且 `custom_voice` 的初版真流式内核已经接进公开服务路径；**
> **但它还不是“全部优化项都完成”的最终稳态版本。**

## 如何启动 MVP

安装完依赖并确保默认 LoRA bundle 可用后，可直接启动：

```text
python -m streaming_lora_service.app.server --bundle_dir <path_to_bundle>
```

常见参数示例：

```text
python -m streaming_lora_service.app.server --bundle_dir outputs/lora_candidate8_multilingual_warmstart_1p7b_20260310_v2_bundle_best --public_model_alias qwen3-tts-flash-realtime --default_voice_alias yachiyo_candidate8_v2 --voice_registry_file streaming_lora_service/configs/voice_registry.candidate8_v2.json --host 0.0.0.0 --port 9010 --local_files_only
```

如果你想继续把 `streaming_lora_service` 作为唯一对外服务层，但把 `faster-qwen3-tts` 接成 backend，当前可直接这样启动：

```text
python -m streaming_lora_service.app.server --backend faster --bundle_dir outputs/lora_candidate8_multilingual_warmstart_1p7b_20260310_v2_bundle_best --public_model_alias qwen3-tts-flash-realtime --default_voice_alias yachiyo_candidate8_v2 --voice_registry_file streaming_lora_service/configs/voice_registry.candidate8_v2.json --host 0.0.0.0 --port 9010 --local_files_only
```

说明：

- `backend=faster` 仍然以 `bundle_dir` 为启动输入；服务启动时会自动执行或复用 `bundle -> merged model -> FasterQwen3TTS` 这条加载链路；
- 如需控制 merged 导出缓存目录，可额外传 `--faster_merged_cache_dir <path>`；
- 如需控制 faster 静态 cache 长度，可额外传 `--faster_max_seq_len 2048`；
- 当前第一版重点保证 **WebSocket `/api-ws/v1/realtime` 协议兼容**；内部状态机不追求与 native streaming 路径完全等价。

如果你想切换 runtime session 同步策略，可额外加：

```text
--runtime_session_sync_mode chunk
```

支持值：

- `step`：每个 generation step 都同步（兼容旧行为，调试友好，性能最保守）
- `chunk`：仅在产出音频 chunk 时同步，外加初始化/结束强制同步（当前默认）
- `final`：只在初始化和结束时同步

如果要实验 `torch.compile`：

```text
--compile_talker --compile_mode reduce-overhead --compile_dynamic
```

说明：

- 当前只对 `qwen3tts.model.talker` 提供 opt-in compile；
- 这是实验路径，不会默认开启；
- 建议始终结合 benchmark 与 `trace_timing` 一起看，不要只看首轮 warmup 结果。

补充：在 `backend=faster` 模式下，`--compile_talker`、`--trace_attention_backend` 与 `--runtime_session_sync_mode` 不再驱动 native 内部状态机；服务会保留协议行为，但每次 commit 会从头生成，不做 KV / session state 恢复。

说明：

- 当前仓库已经验证 `python -m streaming_lora_service.app.server ...` 可以直接启动服务；
- 如果只做本机联调，可把 `--host 0.0.0.0` 改回 `127.0.0.1`；
- 如果要给局域网其他机器联调，保留 `--host 0.0.0.0` 并放行对应端口。

默认端点：

- WebSocket：`/api-ws/v1/realtime`
- 健康检查：`/healthz`
- Voice 列表：`/v1/voices`
- HTTP TTS：`/v1/tts`
- HTTP TTS（兼容别名）：`/v1/audio/speech`
- HTTP TTS（官方风格兼容别名）：`/api/v1/services/aigc/multimodal-generation/generation`
- 音频下载：`/v1/audio/{audio_id}`

## 下一阶段建议

下一阶段建议优先补下面 3 件事：

1. 继续优化 decode window / codec ring buffer 的打包路径，减少 `tuple/list/stack` 往返
2. 增加默认 bundle 的人工试听结论与主观质量记录
3. 在 shape 更稳定后，再评估 `torch.compile` 是否值得作为实验开关引入

如果当前目标是“对外服务先稳定跑起来”，优先建议围绕 `candidate8 v2` 继续扩展样本和试听结论，而不是回到旧的单语 refcand8 版本重新做服务口径。

## 新增：质量回归对照工具

当前仓库已经提供一套可直接运行的诊断对照工具，用来比较：

- 离线 `generate_custom_voice(...)`
- 服务侧 HTTP 非流式路径
- `streaming_sampler_full_decode` 诊断路径（与流式 sampler 相同 codec 生成，但在末尾整段 full decode）
- 服务侧真流式 runtime / WebSocket Realtime 路径

可执行方式：

```text
qwen-tts-streaming-validate --bundle_dir <path_to_bundle> --voice_registry_file <path_to_voice_registry> --output_dir <path_to_output_dir>
```

或：

```text
python -m streaming_lora_service.quality_regression --bundle_dir <path_to_bundle> --voice_registry_file <path_to_voice_registry> --output_dir <path_to_output_dir>
```

工具会输出：

- 多路音频的 `duration_s / total_audio_bytes / elapsed_ms`
- 流式路径的 `ttfb_ms / delta_chunks / generated_steps / finish_reason`
- 用于拆分“流式采样分叉”和“增量解码损耗”的附加诊断 WAV
- codec 级诊断：`shared_prefix_steps / first_divergence_step / step_tokens`
- 自动告警：如流式时长显著高于离线、`finish_reason != eos`、HTTP/WS 输出大小不一致
- 每个 case 对应的 WAV 文件与汇总 `metrics.json`

## 新增：WS 时延 benchmark 工具

如果你已经把服务跑起来，当前仓库还提供了一个**面向外部 WebSocket 端点**的轻量 benchmark：

- 入口：`qwen-tts-ws-benchmark`
- 模块：`python -m streaming_lora_service.ws_benchmark`

它和 `quality_regression.py` 的定位不同：

- `quality_regression.py` 更偏**离线 / HTTP / WS 三路质量与路径对照**；
- `ws_benchmark.py` 更偏**对已启动 WS 服务做真实联机时延测量**。

当前脚本会测：

- `TTFT`：从发送 `input_text_buffer.commit` 到收到首个 `response.audio.delta` 的耗时
- `response_created_ms`：从 `commit` 到 `response.created` 的耗时
- `total_elapsed_ms`：从 `commit` 到 `response.done` 的总耗时
- `audio_duration_s`：收到的 PCM 音频总时长
- `RTF`：`total_elapsed_s / audio_duration_s`
- `audio_chunks / total_audio_bytes`

说明：

- 对这个纯音频 WS 接口来说，`TTFT` 可以理解为“首个音频增量块可用时间”；
- 如果你更习惯叫 `TTFB` / `TTFA`，这里语义上是同一件事：**首个 `response.audio.delta` 到达时间**。

### benchmark 数据落盘约定

- 根目录统一使用 `benchmark/` 保存每次 benchmark 的原始产物；
- 单次运行目录命名为 `benchmark/<summary>_vN/`；
- `<summary>` 建议使用小写下划线，表达本轮 benchmark 的主题，例如 `candidate8_v2_ws`、`candidate8_v2_http`、`candidate8_v2_ws_load`；
- `vN` 从 `v1` 开始递增；同主题重跑时创建新版本目录，不覆盖旧数据；
- `streaming_lora_service/docs/validation/` 继续用于沉淀经过整理的验证结论，不作为高频 benchmark 原始数据的默认落点。

示例：

```text
qwen-tts-ws-benchmark --ws_url ws://127.0.0.1:9010/api-ws/v1/realtime --iterations 5 --warmup 1 --output_path benchmark/candidate8_v2_ws_v1/ws_benchmark_metrics.json --save_audio_dir benchmark/candidate8_v2_ws_v1/audio
```

如果要指定文本：

```text
qwen-tts-ws-benchmark --ws_url ws://127.0.0.1:9010/api-ws/v1/realtime --voice yachiyo_candidate8_v2 --text "你好，这是一次 WebSocket benchmark 测试。" --language_type Chinese --instructions "正式，平静，清晰。"
```

补充说明：

- 若不传 `--voice`，脚本会先调用 `/healthz` 与 `/v1/voices` 做 preflight，并自动选择一个兼容 voice；
- 若服务对外宣告地址是 `0.0.0.0`，客户端实际调用时请改成 `127.0.0.1` 或服务机真实 IP；
- 可通过 `--save_audio_dir` 把每轮 benchmark 收到的音频另存为 WAV，便于试听与排查异常。

## 新增：HTTP streaming benchmark 工具

如果你想专门测 HTTP `stream=true` 路径，当前仓库也已经补了一份独立脚本：

- 入口：`qwen-tts-http-benchmark`
- 模块：`python -m streaming_lora_service.http_streaming_benchmark`

当前脚本会测：

- `response_headers_ms`：发起 POST 到收到 HTTP 响应头
- `TTFT`：发起 POST 到收到**首个带音频数据**的 NDJSON 行
- `final_chunk_ms`：发起 POST 到收到最终 `finish_reason="stop"` 的尾行
- `total_elapsed_ms`
- `audio_duration_s`
- `RTF`
- `audio_chunks / total_audio_bytes`

示例：

```text
qwen-tts-http-benchmark --http_base_url http://127.0.0.1:9010 --endpoint /v1/tts --iterations 5 --warmup 1 --output_path benchmark/candidate8_v2_http_v1/http_benchmark_metrics.json --save_audio_dir benchmark/candidate8_v2_http_v1/audio
```

单条文本示例：

```text
qwen-tts-http-benchmark --http_base_url http://127.0.0.1:9010 --voice yachiyo_candidate8_v2 --text "你好，这是一次 HTTP streaming benchmark 测试。" --language_type Chinese --instructions "正式，平静，清晰。"
```

说明：

- `TTFT` 在 HTTP streaming 场景下定义为：**首个包含 `output.audio.data` 的 NDJSON 音频块到达时间**；
- 中间块是 Base64 PCM16，尾块包含 `audio.id / audio.url`；
- 也支持 `--save_audio_dir` 把流式过程中拼起来的 PCM 直接落成 WAV。

## 新增：并发压测工具

如果你需要压服务在**并发下**的首包和吞吐，当前仓库也已经补了一份 load test：

- 入口：`qwen-tts-load-test`
- 模块：`python -m streaming_lora_service.concurrent_benchmark`

支持两种 transport：

- `--transport http-streaming`
- `--transport ws`

会输出：

- `success_count / failure_count`
- `wall_time_s`
- `throughput_rps`
- `audio_seconds_per_wall_second`
- 并发场景下的 `TTFT p50/p95`
- 并发场景下的 `total_elapsed_ms p50/p95`
- 并发场景下的 `RTF mean/p50/p95`

HTTP 并发示例：

```text
qwen-tts-load-test --transport http-streaming --http_base_url http://127.0.0.1:9010 --concurrency 4 --requests 8 --text "你好，这是一次并发 HTTP streaming 压测。" --output_path benchmark/candidate8_v2_http_load_v1/http_load_test_metrics.json
```

WS 并发示例：

```text
qwen-tts-load-test --transport ws --ws_url ws://127.0.0.1:9010/api-ws/v1/realtime --concurrency 4 --requests 8 --text "你好，这是一次并发 WebSocket 压测。" --output_path benchmark/candidate8_v2_ws_load_v1/ws_load_test_metrics.json
```

建议：

- 先用 `--warmup 1` 或 `2` 预热，再看正式数据；
- 真要看极限吞吐，建议固定单条文本并拉长 `requests`；
- 若本机测试环境开了系统代理，脚本的 preflight 已默认避开环境代理，但实际压测地址仍建议显式写 `127.0.0.1` 或真实 IP。

## 先读哪几份文档

建议按顺序阅读：

1. `docs/QWEN_COMPAT_API.md`
2. `docs/QWEN_COMPATIBILITY_MATRIX.md`
3. `docs/IMPLEMENTATION_PLAN.md`
4. `docs/ARCHITECTURE.md`
5. `docs/WEBSOCKET_PROTOCOL.md`
6. `docs/DELIVERY_CHECKLIST.md`

## 当前结论

**这件事应该直接按 Qwen-compatible 的正式公开接口落地。**

最关键的原则只有一条：

> 公开协议从第一版开始就做官方风格；
> LoRA 默认挂载、bundle 路由、speaker patch 下沉到服务内部；
> 真流式 runtime 仍由我们自己的实现负责。
