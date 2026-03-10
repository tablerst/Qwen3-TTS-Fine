# Delivery Checklist

## Phase 0：文档冻结

- [x] 新建独立目录 `streaming_lora_service/`
- [x] 写入总体说明 `README.md`
- [x] 写入实施方案 `docs/IMPLEMENTATION_PLAN.md`
- [x] 写入架构设计 `docs/ARCHITECTURE.md`
- [x] 写入协议设计 `docs/WEBSOCKET_PROTOCOL.md`
- [x] 明确 V1 只做 `custom_voice + LoRA bundle + WebSocket`

## Phase 1：Bundle Loader

- [ ] 抽离 `bundle_dir` 解析组件
- [ ] 抽离 base + adapter + patch 加载组件
- [ ] 为 bundle loader 补 smoke test
- [ ] 校验 `tts_model_type == custom_voice`
- [ ] 校验 bundle speaker 已注入 `supported_speakers`

## Phase 2：Streaming Session

- [ ] 抽离 prompt builder
- [ ] 实现 session state 容器
- [ ] 实现 step 级生成接口
- [ ] 支持 `append_text()`
- [ ] 支持 `flush()`
- [ ] 支持 `stop()` / `cancel()`

## Phase 3：Incremental Decoder

- [ ] 维护 codec ring buffer
- [ ] 支持左上下文 overlap decode
- [ ] 输出新增 PCM16 chunk
- [ ] 允许配置 `chunk_steps`
- [ ] 允许配置 `left_context_steps`

## Phase 4：WebSocket 服务

- [ ] 启动服务 CLI
- [ ] 支持 `session.start`
- [ ] 支持 `text.delta`
- [ ] 支持 `text.flush`
- [ ] 支持 `session.stop`
- [ ] 支持 `session.cancel`
- [ ] 下发 `audio.chunk.meta + binary pcm`
- [ ] 下发 `completed / cancelled / error`

## Phase 5：验证

- [ ] 单元测试通过
- [ ] bundle smoke 测试通过
- [ ] session smoke 测试通过
- [ ] WebSocket 协议 smoke 测试通过
- [ ] LoRA bundle 真实模型人工试听通过
- [ ] 记录首包延迟和默认 chunk 粒度表现

## 首批默认配置建议

- `model_type`: `custom_voice`
- `transport`: `websocket`
- `audio_format`: `pcm_s16le`
- `sample_rate`: `24000`
- `chunk_steps`: `4`
- `left_context_steps`: `25`
- `attn_implementation`: `sdpa`
- `torch_dtype`: `bfloat16`

## 暂不纳入 V1 的项目

- [ ] `voice_clone` 真流式
- [ ] `voice_design` 真流式
- [ ] HTTP REST 接口
- [ ] 多 bundle 热切换
- [ ] 浏览器 demo UI
- [ ] 集群化调度
