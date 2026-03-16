# 决策记录

## 目标

在当前仓库中新增一条 `vllm-omni` 方案，用于承接：

- 基模；
- LoRA；
- 官方在线服务；
- HTTP / WebSocket 全流式能力。

## 首版已定决策

### 0. 串行延迟优先于并发吞吐

当前本机的真实使用模式以：

- 单请求串行为主；
- 调用端将切句后的请求做 `2~3` 路轻并发为辅；
- 基本不存在高并发压测式流量。

因此本目录首轮优化优先级明确为：

1. 单请求 `TTFA / TTFT`；
2. 单请求 `RTF`；
3. warmup 后第二次及后续请求的稳定性；
4. 低并发（`2~3`）是否出现明显回退。

这意味着首版 benchmark、脚本和文档都应默认围绕**串行基线**组织，而不是先追求大并发吞吐数字。

### 1. 单实例固定 1 个 LoRA

首版不做：

- 请求级动态切 LoRA；
- 同一进程多 LoRA 路由；
- 多租户 voice registry。

原因：

- 官方文档明确每个服务实例在启动时加载一个模型；
- `Qwen3-TTS + LoRA` 在线服务没有看到官方成文示例；
- 先把单实例跑通，比先做复杂路由更符合 PoC 目标。

### 2. 对外直接复用官方 API

首版直接采用：

- `POST /v1/audio/speech`
- `WS /v1/audio/speech/stream`

不复用当前仓库 `streaming_lora_service` 自己的公开协议层。

原因：

- 可以最大限度降低维护面；
- 避免为兼容性验证引入额外变量；
- 等 PoC 跑通后，再决定是否需要自定义外层适配服务。

### 3. 先做兼容性闸门，再决定模型产物形式

候选路线按优先级为：

1. `vllm-omni` 直接消费本地基模目录 + PEFT LoRA 结构；
2. 将当前 bundle 整理为 `vllm-omni` 可直接识别的本地模型目录；
3. 将 LoRA merge 回基模后，生成单独本地模型目录。

当前尚无证据表明第 1 条对 `Qwen3-TTS` 在线服务一定成立，因此不能直接把它写死为最终方案。

## 已验证的上游事实

### 在线服务

官方文档已确认 `Qwen3-TTS` 支持：

- `vllm serve ... --omni`
- `POST /v1/audio/speech`
- `GET /v1/audio/voices`
- `POST /v1/audio/voices`
- `WS /v1/audio/speech/stream`

### 流式约束

- `stream=true` 要求 `response_format="pcm"`
- streaming 不支持 `speed`
- 依赖 stage config 中 `async_chunk: true`

### WebSocket streaming 的版本边界

- 你当前仓库先前验证到的 `WS stream_audio=true` 缺少 `_generate_pcm_chunks`，且该失败在本机上是**稳定复现**的；
- 上游 PR `vllm-project/vllm-omni#1719` 已合并，用于补齐 `Qwen3-TTS` 的 WebSocket streaming audio output；
- 该 PR 的能力边界是：
	- WebSocket 仍然按**句子**组织输出；
	- 每个句子在 `audio.start` 与 `audio.done` 之间发送多个 PCM 二进制 chunk；
	- 不是跨句连续的任意粒度音频流。

因此，本目录后续对 WS 能力的陈述必须写成：

- **当前本地已验证版本**：`stream_audio=true` 不可用；
- **包含 PR #1719 的上游版本**：应支持“句内 chunk 的 PCM 流式输出”；
- 是否真正可用，以本地安装版本和回归验证结果为准。

### Base / ICL 路径的版本门槛

- 上游 PR `vllm-project/vllm-omni#1731` 修复了 `Base ICL` 在 async-chunk 多阶段流水线中的 `ref_code` 解码上下文缺失问题；
- 在较旧版本上，即使 HTTP / WS 能返回音频，也可能在首段出现噪声、前缀不稳或质量异常；
- 因此若要把 `Base` 的 voice cloning 结果纳入正式对比，应优先确认本地版本是否已经包含 `#1731`。

### stage config 关键信息

上游 `qwen3_tts.yaml` 中已经出现并值得保留的关键信息包括：

- `async_chunk: true`
- shared memory connector 开启 `codec_streaming: true`
- `codec_chunk_frames: 25`
- `codec_left_context_frames: 25`
- stage 0 为 talker，stage 1 为 code2wav

## 当前仓库可复用资产

### LoRA 产物结构

来自：`lora_finetuning/export_custom_voice.py`

结构包括：

- `manifest.json`
- `adapter/`
- `config_patch.json`
- `speaker_embedding.safetensors`

### LoRA 注入与 speaker patch

来自：`lora_finetuning/common.py`

关键函数：

- `load_lora_adapter(...)`
- `apply_config_patch(...)`
- `apply_speaker_patch(...)`

### bundle 解析经验

来自：`streaming_lora_service/app/bundle_loader.py`

关键点：

- 现有 bundle 已支持 manifest 解析；
- base model 路径已有本地解析与回退逻辑；
- 当前 bundle 最终解析为 `custom_voice` 运行时语义。

## 明确不做的事情

首版不做：

- 把 `vllm_omni_service/` 接入根 `pyproject.toml`；
- 编写新的 FastAPI server；
- 为 `streaming_lora_service` 提供双栈兼容；
- 承诺生产级吞吐或并发指标。

补充说明：

- 首版不会围绕高并发做系统级优化；
- `2~3` 路轻并发只用于检查串行优化是否产生明显副作用，不作为主要 KPI。

## 最近一步的现实结论

当前仓库虚拟环境中：

- `vllm`: 未安装
- `vllm_omni`: 未安装

因此当前目录里先交付：

- 文档；
- stage config 同步脚本；
- artifact 规范化脚本；
- HTTP / WS 探针；
- 启动封装。

真正的 `vllm-omni` 兼容性实验，需要安装依赖后再继续。
