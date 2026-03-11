# Delivery Checklist

## Phase 0：文档冻结

- [x] 新建独立目录 `streaming_lora_service/`
- [x] 写入总体说明 `README.md`
- [x] 写入 Qwen-compatible 外部接口说明 `docs/QWEN_COMPAT_API.md`
- [x] 写入兼容矩阵 `docs/QWEN_COMPATIBILITY_MATRIX.md`
- [x] 写入实施方案 `docs/IMPLEMENTATION_PLAN.md`
- [x] 写入架构设计 `docs/ARCHITECTURE.md`
- [x] 写入 Realtime 协议设计 `docs/WEBSOCKET_PROTOCOL.md`
- [x] 明确放弃继续扩展旧的自定义公开 WebSocket 协议
- [x] 明确 LoRA bundle 为服务部署参数，而非普通客户端参数
- [x] 明确从第一版开始采用官方风格兼容接口

## Phase 1：Bundle Loader + Voice Registry

- [x] 抽离 `bundle_dir` 解析组件
- [x] 抽离 base + adapter + patch 加载组件
- [x] 抽离 `voice_registry`（公开 `voice` -> 内部 profile）
- [x] 为 bundle loader 补 smoke test
- [x] 为 voice registry 补单测
- [x] 校验 `tts_model_type == custom_voice`
- [x] 校验默认 speaker 已注入 `supported_speakers`
- [x] 校验公开 `model` / `voice` 别名与实际能力一致

## Phase 2：Runtime Session

- [ ] 抽离 prompt builder
- [x] 实现 session state 容器
- [ ] 实现 step 级生成接口
- [x] 支持 `append_text()`
- [x] 支持 `commit()`
- [x] 支持 `clear_pending_text()`
- [x] 支持 `finish()`

## Phase 3：Incremental Decoder

- [ ] 维护 codec ring buffer
- [x] 支持左上下文 overlap decode
- [x] 输出新增音频字节
- [x] 允许配置 `chunk_steps`
- [x] 允许配置 `left_context_steps`
- [x] 输出可供 `response.audio.delta` 使用的稳定音频块

## Phase 4：Qwen-Compatible Realtime 服务

- [x] 启动服务 CLI
- [x] 支持 `session.update`
- [x] 支持 `input_text_buffer.append`
- [x] 支持 `input_text_buffer.commit`
- [x] 支持 `input_text_buffer.clear`
- [x] 支持 `session.finish`
- [x] 下发 `session.created / session.updated`
- [x] 下发 `response.created`
- [x] 下发 `response.audio.delta`
- [x] 下发 `response.audio.done / response.done / session.finished`
- [x] 下发 `error`

## Phase 4.5：Qwen-Compatible HTTP TTS 服务

- [x] 支持 `POST /v1/tts`
- [x] 支持 `POST /v1/audio/speech`
- [x] 支持 `POST /api/v1/services/aigc/multimodal-generation/generation`
- [x] 支持 `model / text / voice / language_type / instructions / optimize_instructions / stream`
- [x] 支持非流式官方风格 JSON 响应
- [x] 支持流式 `application/x-ndjson` 输出
- [x] 支持音频下载路由 `/v1/audio/{audio_id}`

## Phase 5：验证

- [x] 单元测试通过
- [x] bundle smoke 测试通过
- [x] voice registry 测试通过
- [x] runtime session smoke 测试通过
- [x] Realtime 协议 smoke 测试通过
- [x] 服务级 WebSocket smoke 测试通过
- [x] HTTP TTS smoke 测试通过
- [ ] 默认 LoRA bundle 真实模型人工试听通过
- [ ] 记录首包延迟和默认 chunk 粒度表现

## 当前代码骨架（已落地）

- [x] `app/models.py`
- [x] `app/voice_registry.py`
- [x] `app/bundle_loader.py`
- [x] `app/runtime_session.py`
- [x] `app/incremental_decoder.py`
- [x] `app/qwen_compat_ws.py`
- [x] `app/audio_utils.py`
- [x] `app/server.py`
- [x] `configs/voice_registry.example.json`
- [x] `tests/test_bundle_loader.py`
- [x] `tests/test_voice_registry.py`
- [x] `tests/test_runtime_session.py`
- [x] `tests/test_incremental_decoder.py`
- [x] `tests/test_protocol_smoke.py`
- [x] `tests/test_server_smoke.py`
- [x] `tests/test_http_tts_smoke.py`

## 下一阶段（建议优先级）

1. [ ] `qwen_compat_ws.py` 升级为 step-level 真流式 `response.audio.delta`
2. [ ] `runtime_session` 接真实 prompt builder / step generator
3. [ ] `incremental_decoder` 接真实 codec ring buffer
4. [ ] 默认 bundle 人工试听回归
5. [ ] 记录 TTFB / chunk 粒度指标

## 当前 MVP 说明

- [x] 官方风格兼容接口 MVP 已完成
- [x] 默认 LoRA bundle 启动期加载已完成
- [x] 可运行 WebSocket 服务 MVP 已完成
- [x] 可运行 HTTP TTS 服务 MVP 已完成
- [x] `response.audio.delta` 分块下发 MVP 已完成
- [ ] 真正的 step-level 流式内核仍未完成

## 首批默认配置建议

- `public_model_alias`: `qwen3-tts-flash-realtime`
- `transport`: `websocket`
- `response_format`: `pcm`
- `sample_rate`: `24000`
- `chunk_steps`: `4`
- `left_context_steps`: `25`
- `attn_implementation`: `sdpa`
- `torch_dtype`: `bfloat16`
- `deployment_bundle`: `--bundle_dir <path>`

## 暂不纳入 V1 的项目

- [ ] 公开暴露 `bundle_dir` 给普通客户端
- [ ] 继续维护旧的 `session.start / text.delta / text.flush` 主协议
- [ ] `voice_clone` 真流式
- [ ] `voice_design` 真流式
- [ ] 多 bundle 热切换
- [ ] 完整 DashScope 路径级兼容
- [ ] 浏览器 demo UI
- [ ] 集群化调度
